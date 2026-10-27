"""Attention gates and model construction for three ablation settings."""

from __future__ import annotations


from torch import nn

from idasnet import evaluate as trainer

# Gate order: preprocessing method, wavelength, learned feature channel.
EXPERIMENTS = {
    "no_attention": ("identity", "identity", "identity"),
    "method_only": ("adaptive", "identity", "identity"),
    "wavelength_only": ("identity", "adaptive", "identity"),
}


class IdentityGate(nn.Module):
    """Pass features through and return attention weights of one."""

    def __init__(self, size: int, axis: int):
        super().__init__()
        self.size, self.axis = size, axis

    def forward(self, x):
        shape = (x.shape[0], self.size, 1) if self.axis == 1 else (x.shape[0], 1, self.size)
        weights = x.new_ones(shape)
        return x * weights, weights


def build_ablation_model(params, channels, length, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    model = trainer.build_model(params, channels, length)
    specifications = [
        ("raw_channel_attention", channels, 1),
        ("wavelength_attention", length, 2),
        ("feature_channel_attention", int(params["feature_channels"]), 1),
    ]
    for mode, (name, size, axis) in zip(EXPERIMENTS[experiment], specifications):
        if mode == "identity":
            setattr(model, name, IdentityGate(size, axis))
    model.ablation_experiment = experiment
    check_ablation_structure(model, experiment)
    return model


def check_ablation_structure(model, experiment):
    modes = EXPERIMENTS[experiment]
    for name, mode in zip(
        ("raw_channel_attention", "wavelength_attention", "feature_channel_attention"), modes
    ):
        gate = getattr(model, name)
        if mode == "identity":
            if not isinstance(gate, IdentityGate) or list(gate.parameters()):
                raise RuntimeError(f"Identity gate mismatch: {name}")
        elif isinstance(gate, IdentityGate):
            raise RuntimeError(f"Adaptive gate unexpectedly replaced: {name}")
    return {
        "experiment": experiment,
        "gate_modes": list(modes),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "gated_fusion_retained": True,
    }



