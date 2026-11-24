"""Evaluate zero-masking sensitivity of saved IDASNet outer-test predictions."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from data.folds import load_raw_dataset, load_outer_fold_manifest
from idasnet.evaluate import normalize_spectra, next_result_directory
from idasnet.inference import load_checkpoint
from idasnet.statistics import metrics


def perturbations(methods, wavelengths, window=5):
    if window < 1:
        raise ValueError("window must be positive")
    conditions = [{"condition": "original", "kind": "original"}]
    conditions += [{"condition": f"channel_{i}", "kind": "channel", "index": i, "method": method}
                   for i, method in enumerate(methods)]
    conditions += [{"condition": f"bands_{i}_{i+window-1}", "kind": "wavelength", "index": i,
                    "start_nm": wavelengths[i], "end_nm": wavelengths[i+window-1], "window": window}
                   for i in range(0, len(wavelengths)-window+1, window)]
    return conditions


def apply_mask(normalized, condition):
    values = normalized.clone()
    if condition["kind"] == "channel":
        values[:, condition["index"], :] = 0
    elif condition["kind"] == "wavelength":
        start = condition["index"]
        values[:, :, start:start+condition["window"]] = 0
    elif condition["kind"] != "original":
        raise ValueError("Unknown perturbation kind")
    return values


def summarize(predictions, conditions):
    rows = []
    for repeat, repeat_data in predictions.groupby("repeat"):
        original = repeat_data[repeat_data.condition == "original"].sort_values("sample_id")
        base_rmse = metrics(original.measured_TSS, original.predicted_TSS)["RMSE"]
        for condition in conditions:
            block = repeat_data[repeat_data.condition == condition["condition"]].sort_values("sample_id")
            if block.sample_id.tolist() != original.sample_id.tolist():
                raise ValueError("Masked and original OOF coverage differs")
            score = metrics(block.measured_TSS, block.predicted_TSS)["RMSE"]
            rows.append({"repeat": repeat, **condition, "RMSE": score,
                         "original_RMSE": base_rmse, "delta_RMSE": score-base_rmse})
    repeats = pd.DataFrame(rows)
    aggregate = repeats.groupby("condition", sort=False).delta_RMSE.agg(mean="mean", sd="std").reset_index()
    summary = pd.DataFrame(conditions).merge(aggregate, on="condition", validate="one_to_one")
    return repeats, summary


def run_analysis(run, output, device="cpu", batch_size=64):
    from idasnet.protocol import PROTOCOL_VERSION, load_run_config
    run, output = Path(run), Path(output)
    config = load_run_config(run / "run_config.json")
    if config.get("model_type", "IDASNet") != "IDASNet":
        raise ValueError("Sensitivity uses the full IDASNet run")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Run version does not match the active training protocol")
    dataset = load_raw_dataset(run / "dataset_snapshot.pt", config["target"])
    folds = load_outer_fold_manifest(run / "outer_folds.json", dataset)
    conditions = perturbations(dataset.processing_methods, dataset.wavelengths)
    records = []
    for fold in folds:
        indices = fold.test_indices
        x = normalize_spectra(dataset.X[indices])
        totals = {c["condition"]: np.zeros(len(indices)) for c in conditions}
        for seed in config["train_seeds"]:
            path = run / f"repeat_{fold.repeat+1:02d}/fold_{fold.fold+1:02d}/model_seed_{seed}.pth"
            model, _ = load_checkpoint(path, device, *x.shape[1:])
            for condition in conditions:
                z = apply_mask(x, condition)
                with torch.inference_mode():
                    p = np.concatenate([model(batch.to(device))["pred"].reshape(-1).cpu().numpy()
                                        for batch in z.split(batch_size)])
                totals[condition["condition"]] += p / len(config["train_seeds"])
            del model
        for name, predicted in totals.items():
            for position, index in enumerate(indices):
                records.append({"repeat": fold.repeat, "fold": fold.fold, "condition": name,
                    "sample_id": dataset.global_ids[index], "acquisition_id": dataset.acquisition_ids[index],
                    "measured_TSS": float(dataset.y[index]), "predicted_TSS": float(predicted[position])})
        print(f"Sensitivity repeat {fold.repeat+1}, fold {fold.fold+1} complete", flush=True)
    predictions = pd.DataFrame(records)
    # Confirm the unmodified checkpoint predictions reproduce the saved seed ensemble.
    original = predictions[predictions.condition == "original"]
    reference = pd.read_csv(run / "oof_predictions_seed_ensemble.csv", dtype={"sample_id": str})
    joined = original.merge(reference, on=["repeat", "sample_id"], validate="one_to_one")
    if len(joined) != len(reference) or not np.allclose(joined.predicted_TSS, joined.ensemble_predicted_TSS, atol=2e-5, rtol=2e-5):
        raise ValueError("Unmodified checkpoint predictions disagree with the saved OOF ensemble")
    repeats, summary = summarize(predictions, conditions)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "masked_oof_predictions.csv", index=False)
    repeats.to_csv(output / "sensitivity_by_repeat.csv", index=False)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    (output / "analysis_config.json").write_text(json.dumps({"source_run": str(run.resolve()),
        "train_seeds": config["train_seeds"], "retraining": False,
        "aggregation": "seed mean -> complete outer OOF per repeat -> delta RMSE -> repeat mean/sample SD",
        "wavelength_intervals": 30, "trailing_unmasked_wavelengths": dataset.wavelengths[150:]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=Path(__file__).resolve().parents[1] / "results/Sensitivity")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run_analysis(args.run_dir, next_result_directory(args.result_root), args.device, args.batch_size)


if __name__ == "__main__":
    main()
