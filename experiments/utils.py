"""Training, inference, statistics and file handling for attention experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import torch
from idasnet import evaluate as trainer
from idasnet.statistics import cluster_bootstrap_ci, berry_mean_predictions
from experiments.models import build_ablation_model

METRICS = ["R2", "RMSE", "MAE"]


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, data):
    Path(path).write_text(
        json.dumps(_plain(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def validate_manifest(dataset, manifest, repeats):
    """Check sample identity, NH7 separation and exact once-per-repeat coverage."""
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if payload["n_samples"] != len(dataset.X):
        raise ValueError("Manifest sample count differs from the selected raw dataset")
    available = sorted({f["repeat"] for f in payload["folds"]})
    if repeats < 1 or repeats > len(available):
        raise ValueError(f"outer_repeats must be between 1 and {len(available)}")
    folds = [f for f in payload["folds"] if f["repeat"] in available[:repeats]]
    n = len(dataset.X)
    seen_pairs = set()
    for f in folds:
        pair = (f["repeat"], f["fold"])
        if pair in seen_pairs:
            raise ValueError("Duplicate repeat/fold in manifest")
        seen_pairs.add(pair)
        train, test = np.asarray(f["train_indices"]), np.asarray(f["test_indices"])
        if (
            train.ndim != 1
            or test.ndim != 1
            or len(train) == 0
            or len(test) == 0
            or len(np.unique(train)) != len(train)
            or len(np.unique(test)) != len(test)
            or np.any(train < 0)
            or np.any(test < 0)
            or np.any(train >= n)
            or np.any(test >= n)
            or set(train) & set(test)
            or set(train) | set(test) != set(range(n))
        ):
            raise ValueError("Invalid or overlapping outer train/test indices")
        for split, indices in [("train", train), ("test", test)]:
            if dataset.global_ids[indices].tolist() != f[f"{split}_sample_ids"]:
                raise ValueError("Manifest sample IDs/order do not match the raw dataset")
            groups = sorted(set(dataset.acquisition_ids[indices]))
            if groups != sorted(f[f"{split}_groups"]):
                raise ValueError("Manifest acquisition IDs do not match the raw dataset")
        if set(dataset.acquisition_ids[train]) & set(dataset.acquisition_ids[test]):
            raise ValueError("NH7 leakage in outer fold")
    for repeat in available[:repeats]:
        coverage = np.zeros(n, dtype=int)
        for f in folds:
            if f["repeat"] == repeat:
                coverage[f["test_indices"]] += 1
        if not np.all(coverage == 1):
            raise ValueError("Every sample must be tested exactly once per outer repeat")
    return {**payload, "n_repeats": repeats, "folds": folds}


def normalized_inputs(dataset, train, other):
    """Normalize each requested spectrum independently; return the two subsets."""
    tx = trainer.normalize_spectra(dataset.X[torch.as_tensor(train, dtype=torch.long)])
    vx = trainer.normalize_spectra(dataset.X[torch.as_tensor(other, dtype=torch.long)])
    return (tx, vx)


def new_model(params, dataset, config, experiment):
    return build_ablation_model(params, dataset.X.shape[1], dataset.X.shape[2], experiment)


def seed_training_stream(seed):
    # Start a reproducible training/dropout stream after network construction.
    trainer.set_seed(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    squared_error = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x)["pred"].reshape(-1), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        squared_error += float(loss.detach().cpu()) * len(y)
        n += len(y)
    return squared_error / n


def collect_inference(model, x, y, indices, dataset, config, device):
    """Collect predictions and attention weights required by the experiments."""
    model.eval()
    parts = {}
    with torch.inference_mode():
        for start in range(0, len(x), config.batch_size):
            result = model(x[start : start + config.batch_size].to(device))
            for key, value in result.items():
                parts.setdefault(key, []).append(value.cpu().numpy())
    arrays = {key: np.concatenate(values) for key, values in parts.items()}
    arrays.update(
        sample_indices=np.asarray(indices, dtype=int),
        sample_ids=dataset.global_ids[indices],
        acquisition_ids=dataset.acquisition_ids[indices],
        measured_TSS=y.numpy(),
    )
    return arrays


def bootstrap_ci(ensemble, iterations, seed):
    return cluster_bootstrap_ci(ensemble, iterations, seed)


def diagnostic_tables(ensemble, directory):
    data = ensemble.copy()
    data["residual_predicted_minus_measured"] = data.ensemble_predicted_TSS - data.measured_TSS
    data["bland_altman_average"] = (data.ensemble_predicted_TSS + data.measured_TSS) / 2
    data["year"] = data.sample_id.str[:4]
    data.to_csv(directory / "diagnostic_data_by_repeat.csv", index=False)
    diagnostics, ranges, years = [], [], []
    low, high = np.quantile(data.drop_duplicates("sample_id").measured_TSS, [1 / 3, 2 / 3])
    for repeat, block in data.groupby("repeat"):
        error = block.residual_predicted_minus_measured
        slope, intercept = trainer._linear_fit(block.measured_TSS, block.ensemble_predicted_TSS)
        cs, ci = trainer._linear_fit(block.ensemble_predicted_TSS, block.measured_TSS)
        diagnostics.append(
            {
                "repeat": repeat,
                "bias": error.mean(),
                "error_sd": error.std(ddof=1),
                "lower_loa": error.mean() - 1.96 * error.std(ddof=1),
                "upper_loa": error.mean() + 1.96 * error.std(ddof=1),
                "prediction_on_measurement_slope": slope,
                "prediction_on_measurement_intercept": intercept,
                "calibration_measured_on_prediction_slope": cs,
                "calibration_measured_on_prediction_intercept": ci,
            }
        )
        for label, mask in [
            ("low", block.measured_TSS <= low),
            ("middle", (block.measured_TSS > low) & (block.measured_TSS <= high)),
            ("high", block.measured_TSS > high),
        ]:
            part = block[mask]
            if len(part):
                ranges.append(
                    {
                        "repeat": repeat,
                        "TSS_range": label,
                        "n": len(part),
                        "low_boundary": low,
                        "high_boundary": high,
                        "bias": part.residual_predicted_minus_measured.mean(),
                        **trainer.compute_metrics(part.measured_TSS, part.ensemble_predicted_TSS),
                    }
                )
        for year, part in block.groupby("year"):
            years.append(
                {
                    "repeat": repeat,
                    "year": year,
                    "n": len(part),
                    **trainer.compute_metrics(part.measured_TSS, part.ensemble_predicted_TSS),
                }
            )
    pd.DataFrame(diagnostics).to_csv(directory / "diagnostics_by_repeat.csv", index=False)
    pd.DataFrame(ranges).to_csv(directory / "metrics_by_TSS_range_and_repeat.csv", index=False)
    pd.DataFrame(years).to_csv(directory / "metrics_by_year_and_repeat.csv", index=False)
    averaged = berry_mean_predictions(ensemble)
    averaged["residual_predicted_minus_measured"] = averaged.ensemble_predicted_TSS - averaged.measured_TSS
    averaged["bland_altman_average"] = (averaged.ensemble_predicted_TSS + averaged.measured_TSS) / 2
    averaged.to_csv(directory / "berry_mean_predictions.csv", index=False)
    return averaged


def save_method_rank_data(attention, directory):
    """Keep tied attention scores tied when exporting method ranks."""
    ranked_rows, stability = [], []
    for split, part in attention.groupby("split"):
        wide = part.pivot(index=["repeat", "fold", "train_seed"], columns="index", values="mean")
        average_ranks = wide.rank(axis=1, ascending=False, method="average")
        minimum_ranks = wide.rank(axis=1, ascending=False, method="min")
        maximum_ranks = wide.rank(axis=1, ascending=False, method="max")
        for index in wide.columns:
            frequency = np.zeros(wide.shape[1])
            for lo, hi in zip(minimum_ranks[index], maximum_ranks[index]):
                frequency[int(lo) - 1 : int(hi)] += 1 / (hi - lo + 1)
            for position, value in enumerate(frequency / len(wide), start=1):
                ranked_rows.append(
                    {
                        "split": split,
                        "method_index": index,
                        "rank": position,
                        "frequency": value,
                        "tie_policy": "equal mass across tied positions",
                    }
                )
        for (repeat, fold), group in average_ranks.groupby(level=["repeat", "fold"]):
            ranks = group.to_numpy()
            values = [
                float(np.corrcoef(ranks[i], ranks[j])[0, 1])
                for i in range(len(ranks))
                for j in range(i + 1, len(ranks))
                if ranks[i].std() > 0 and ranks[j].std() > 0
            ]
            stability.append(
                {
                    "split": split,
                    "repeat": repeat,
                    "fold": fold,
                    "mean_pairwise_spearman": float(np.mean(values)) if values else np.nan,
                    "valid_seed_pairs": len(values),
                    "total_seed_pairs": len(ranks) * (len(ranks) - 1) // 2,
                    "note": "All-tied gate rankings have undefined correlation, not zero stability.",
                }
            )
    pd.DataFrame(ranked_rows).to_csv(directory / "method_rank_frequencies.csv", index=False)
    pd.DataFrame(stability).to_csv(directory / "method_rank_stability_by_fold.csv", index=False)


def same_dataset(first, second):
    return (torch.equal(first.X, second.X) and torch.equal(first.y, second.y)
            and first.processing_methods == second.processing_methods
            and first.wavelengths == second.wavelengths
            and all(np.array_equal(getattr(first, name), getattr(second, name))
                    for name in ("global_ids", "acquisition_ids", "dates")))
