import torch
import torch.nn as nn


# Hyperparameter search space (Table S2).
SEARCH_SPACE = {
    'learning_rate': {'low': 1e-05, 'high': 0.005, 'log': True},
    'weight_decay': {'low': 1e-08, 'high': 0.001, 'log': True},
    'dropout': {'low': 0.05, 'high': 0.5},
    'stem_channels': [8, 10, 12, 14, 16],
    'feature_channels': [24, 28, 32, 36, 40],
    'deep_channels': [48, 50, 64, 70, 80],
    'hidden_dim': [32, 48, 56, 64],
    'raw_channel_reduction': [1, 2, 3, 4],
    'wavelength_reduction': [2, 4, 6, 8, 10],
    'feature_channel_reduction': [2, 3, 4],
    'optimizer': ['Adam', 'AdamW'],
    'use_scheduler': [False, True],
}


def make_group_norm(num_channels: int, preferred_groups=(8, 4, 2, 1)):
    """Use the first group count in preferred_groups that divides num_channels."""
    for g in preferred_groups:
        if g <= num_channels and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)

    return nn.GroupNorm(1, num_channels)


class ConvGNAct(nn.Module):
    """Conv1d, group normalization and activation with same-length padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()

        padding = kernel_size // 2

        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.SiLU(inplace=True)

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            make_group_norm(out_channels),
            act,
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Two convolutions with a residual connection and optional dropout."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.gn1 = make_group_norm(channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.gn2 = make_group_norm(channels)
        self.act2 = nn.SiLU(inplace=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = self.act2(out)

        return out


class RawChannelAttention(nn.Module):
    """Sigmoid attention over preprocessing methods.

    For input (B, C, L), returns weighted spectra (B, C, L) and weights (B, C, 1).
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()

        hidden = max(1, channels // reduction)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = x.mean(dim=2)  # (B, C)
        weights = self.mlp(pooled)  # (B, C)
        weights = weights.unsqueeze(-1)  # (B, C, 1)

        out = x * weights  # (B, C, L)

        return out, weights


class WavelengthAttention(nn.Module):
    """Sigmoid attention over wavelengths in the shallow feature map.

    For input (B, C, L), returns features (B, C, L) and weights (B, 1, L).
    """

    def __init__(self, length: int, reduction: int = 8):
        super().__init__()

        hidden = max(4, length // reduction)

        self.mlp = nn.Sequential(
            nn.Linear(length, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, length),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, L)
        weights = self.mlp(pooled)  # (B, L)
        weights = weights.unsqueeze(1)  # (B, 1, L)

        out = x * weights  # (B, C, L)

        return out, weights


class FeatureChannelAttention(nn.Module):
    """Sigmoid attention over learned feature channels.

    For input (B, C, L), returns features (B, C, L) and weights (B, C, 1).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        hidden = max(4, channels // reduction)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = x.mean(dim=2)  # (B, C)
        weights = self.mlp(pooled)  # (B, C)
        weights = weights.unsqueeze(-1)  # (B, C, 1)

        out = x * weights  # (B, C, L)

        return out, weights


class GatedFusion(nn.Module):
    """Blend two (B, C, L) feature maps with a learned elementwise gate."""

    def __init__(self, channels: int):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=True), nn.Sigmoid()
        )

    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)
        gate = self.gate(cat)

        out = gate * x1 + (1.0 - gate) * x2

        return out, gate


class HybridDualAttentionSpectralNet(nn.Module):
    """IDASNet for spectral regression.

    Inputs have shape (batch, preprocessing methods, wavelengths). Outputs
    contain predictions and method, wavelength, feature-channel and fusion
    weights used by the manuscript attention and sensitivity analyses.
    """

    def __init__(
        self,
        input_channels: int = 5,
        length: int = 151,
        num_outputs: int = 1,
        stem_channels: int = 16,
        feature_channels: int = 32,
        deep_channels: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        raw_channel_reduction: int = 2,
        wavelength_reduction: int = 8,
        feature_channel_reduction: int = 4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.length = length
        self.num_outputs = num_outputs

        self.raw_channel_attention = RawChannelAttention(
            channels=input_channels, reduction=raw_channel_reduction
        )

        self.raw_to_stem = ConvGNAct(
            input_channels, stem_channels, kernel_size=1, activation="silu"
        )

        self.stem_conv = ConvGNAct(
            stem_channels, feature_channels, kernel_size=5, activation="silu"
        )

        self.local_branch = ConvGNAct(
            input_channels, feature_channels, kernel_size=3, activation="silu"
        )

        self.shallow_fusion = GatedFusion(feature_channels)

        self.wavelength_attention = WavelengthAttention(
            length=length, reduction=wavelength_reduction
        )

        self.feature_channel_attention = FeatureChannelAttention(
            channels=feature_channels, reduction=feature_channel_reduction
        )

        self.conv2 = ConvGNAct(feature_channels, deep_channels, kernel_size=5, activation="silu")

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.res1 = ResidualConvBlock(deep_channels, kernel_size=3, dropout=dropout * 0.5)

        self.conv3 = ConvGNAct(deep_channels, deep_channels * 2, kernel_size=3, activation="silu")

        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.res2 = ResidualConvBlock(deep_channels * 2, kernel_size=3, dropout=dropout * 0.5)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        pooled_dim = (deep_channels * 2) * 2

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize convolutions with Kaiming normal and linear layers with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Return predictions and attention weights for (B, input_channels, length)."""

        x_raw_att, raw_channel_weights = self.raw_channel_attention(x)

        stem_feat = self.raw_to_stem(x_raw_att)
        stem_feat = self.stem_conv(stem_feat)

        local_feat = self.local_branch(x_raw_att)

        shallow_feat, fusion_gate = self.shallow_fusion(stem_feat, local_feat)

        wavelength_attended_feature, wavelength_weights = self.wavelength_attention(shallow_feat)

        shallow_feat, feature_channel_weights = self.feature_channel_attention(
            wavelength_attended_feature
        )

        feat = self.conv2(shallow_feat)
        feat = self.pool2(feat)

        feat = self.res1(feat)

        feat = self.conv3(feat)
        feat = self.pool3(feat)

        feat = self.res2(feat)

        avg_feat = self.avg_pool(feat)
        max_feat = self.max_pool(feat)

        pooled_feat = torch.cat([avg_feat, max_feat], dim=1)

        pred = self.head(pooled_feat)

        outputs = {
            "pred": pred,
            "raw_channel_weights": raw_channel_weights,
            "wavelength_weights": wavelength_weights,
            "feature_channel_weights": feature_channel_weights,
            "fusion_gate": fusion_gate,
        }

        return outputs
