"""Attention-free convolutional baselines for spectral regression."""
import torch
from torch import nn


# Hyperparameter search space (Table S2).
SEARCH_SPACE = {
    'learning_rate': {'low': 1e-05, 'high': 0.005, 'log': True},
    'weight_decay': {'low': 1e-08, 'high': 0.001, 'log': True},
    'dropout': {'low': 0.05, 'high': 0.5},
    'base_channels': [16, 24, 32, 48],
    'kernel_size': [3, 5, 7],
    'hidden_dim': [32, 64, 96],
    'optimizer': ['Adam', 'AdamW'],
    'use_scheduler': [False, True],
}


class StandardCNN1D(nn.Module):
    """Three Conv-BN-ReLU blocks, global average pooling and a regression head.

    All five preprocessing methods are presented as input channels. Convolution
    mixes these channels and extracts local spectral patterns. No method,
    feature-channel, or wavelength attention is used.
    """

    def __init__(
        self,
        channels: int,
        length: int,
        base_channels: int = 24,
        kernel_size: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels < 1 or length < 1:
            raise ValueError("Input channel count and spectral length must be positive")
        if base_channels < 1 or hidden_dim < 1:
            raise ValueError("Network widths must be positive")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be positive and odd")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_channels = int(channels)
        self.length = int(length)
        self.model_name = "CNN1D"
        widths = [base_channels, base_channels * 2, base_channels * 4]
        kernels = [kernel_size, 5, 3]
        blocks = []
        previous = channels
        for width, kernel in zip(widths, kernels):
            blocks.extend(
                [
                    nn.Conv1d(previous, width, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            previous = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(nn.Linear(widths[-1], hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.input_channels or x.shape[2] != self.length:
            raise ValueError(
                f"Expected (batch, {self.input_channels}, {self.length}); got {tuple(x.shape)}"
            )
        spectral_features = self.features(x)
        pooled = self.pool(spectral_features).squeeze(-1)
        embedding = self.projection(pooled)
        output = {"pred": self.regressor(self.dropout(embedding))}
        return output

class ResidualBlock1D(nn.Module):
    """Conventional two-convolution residual block with projection when needed."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))

class LightResNet1D(nn.Module):
    """A convolution stem and three residual blocks, not a pretrained ResNet.

    The last two blocks downsample the spectral axis by stride 2. Global average
    pooling makes the regression head compact. No attention modules are used.
    """

    def __init__(
        self,
        channels: int,
        length: int,
        base_channels: int = 24,
        kernel_size: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channels < 1 or length < 1:
            raise ValueError("Input channel count and spectral length must be positive")
        if base_channels < 1 or hidden_dim < 1:
            raise ValueError("Network widths must be positive")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be positive and odd")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_channels = int(channels)
        self.length = int(length)
        self.model_name = "ResNet1D"
        self.stem = nn.Sequential(
            nn.Conv1d(channels, base_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(base_channels, base_channels, kernel_size, stride=1),
            ResidualBlock1D(base_channels, base_channels * 2, kernel_size, stride=2),
            ResidualBlock1D(base_channels * 2, base_channels * 4, kernel_size, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(nn.Linear(base_channels * 4, hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.input_channels or x.shape[2] != self.length:
            raise ValueError(
                f"Expected (batch, {self.input_channels}, {self.length}); got {tuple(x.shape)}"
            )
        spectral_features = self.blocks(self.stem(x))
        pooled = self.pool(spectral_features).squeeze(-1)
        embedding = self.projection(pooled)
        output = {"pred": self.regressor(self.dropout(embedding))}
        return output
