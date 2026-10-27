"""Portable prediction from explicitly supplied saved model checkpoints."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from data import NORMALIZATION
from idasnet.evaluate import build_model, normalize_spectra


def load_checkpoint(path, device="cpu", input_channels=5, length=151):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    params = checkpoint["params"]
    metadata = checkpoint.get("input_metadata", {})
    if metadata.get("normalization"):
        if metadata["normalization"] != NORMALIZATION:
            raise ValueError("Checkpoint normalization differs from the active protocol")
    plan = checkpoint.get("selection_plan")
    shape = plan["input_shape"] if plan else metadata.get("input_shape", [input_channels, length])
    if "ablation_experiment" in checkpoint:
        from experiments.models import build_ablation_model
        model = build_ablation_model(params, *shape, checkpoint["ablation_experiment"])
    else:
        model = build_model(params, *shape)
    model = model.to(device).eval()
    model._idasnet_selection_plan = plan
    model._idasnet_expected_input_shape = (plan["original_input_shape"] if plan else shape)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def predict_spectra(models, spectra, device="cpu", batch_size=64):
    if not models or batch_size < 1:
        raise ValueError("Supply at least one checkpoint and a positive batch size")
    values = torch.as_tensor(spectra, dtype=torch.float32)
    if values.ndim != 3 or len(values) < 1 or values.shape[-1] < 2:
        raise ValueError("Expected unnormalized spectra (berries, channels, wavelengths)")
    values = normalize_spectra(values)
    predictions = []
    for model in models:
        model_values = values
        expected = getattr(model, "_idasnet_expected_input_shape", list(values.shape[1:]))
        if list(values.shape[1:]) != list(expected):
            raise ValueError("Input dimensions differ from checkpoint metadata")
        plan = getattr(model, "_idasnet_selection_plan", None)
        if plan:
            model_values = values[:, plan["method_indices"], :][:, :, plan["wavelength_indices"]]
        batches = []
        with torch.inference_mode():
            for batch in model_values.split(batch_size):
                batches.append(model(batch.to(device))["pred"].reshape(-1).cpu().numpy())
        predictions.append(np.concatenate(batches))
    return np.mean(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--spectra", type=Path, required=True, help="NPZ with X, processing_methods and wavelengths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    with np.load(args.spectra, allow_pickle=False) as archive:
        x = archive["X"]
        if x.ndim != 3 or not len(x):
            raise ValueError("X must be a nonempty (berries, channels, wavelengths) tensor")
        models = []
        for path in args.checkpoints:
            model, checkpoint = load_checkpoint(path, args.device, *x.shape[1:])
            info = checkpoint.get("input_metadata", {})
            for key in ("processing_methods", "wavelengths"):
                if key not in archive or key not in info:
                    raise ValueError(f"Both input and checkpoint must provide {key} for safe channel/band alignment")
                if list(archive[key]) != info[key]:
                    raise ValueError(f"Checkpoint and spectra disagree: {key}")
            models.append(model)
        sample_ids = archive["sample_ids"] if "sample_ids" in archive else np.arange(len(x))
        predicted = predict_spectra(models, x, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": sample_ids, "predicted_TSS": predicted}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
