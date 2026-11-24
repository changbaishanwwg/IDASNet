"""Whole-cluster bootstrap intervals and paired error comparisons.

Bootstrap draws are shared across models and repeats. Form prediction errors
before averaging over repeated predictions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 102017
METRICS = ("R2", "RMSE", "MAE")


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    error = p - y
    total = np.square(y - y.mean()).sum()
    return {"R2": 1 - np.square(error).sum() / total if total > 0 else np.nan,
            "RMSE": np.sqrt(np.square(error).mean()), "MAE": np.abs(error).mean()}


def canonical_predictions(frame):
    data = frame.copy()
    if "ensemble_predicted_TSS" not in data:
        data = data.rename(columns={"predicted_TSS": "ensemble_predicted_TSS"})
    required = ["repeat", "sample_id", "acquisition_id", "measured_TSS", "ensemble_predicted_TSS"]
    if not set(required).issubset(data.columns):
        raise ValueError(f"Prediction table needs {required}")
    if data.empty:
        raise ValueError("Empty prediction table")
    data["sample_id"] = data.sample_id.astype(str)
    data["acquisition_id"] = data.acquisition_id.astype(str)
    if data.duplicated(["repeat", "sample_id"]).any():
        raise ValueError("Expected one seed-ensemble prediction per repeat and berry")
    if not np.isfinite(data[["measured_TSS", "ensemble_predicted_TSS"]].to_numpy()).all():
        raise ValueError("Non-finite predictions or measurements")
    ids = sorted(data.sample_id.unique())
    for _, part in data.groupby("repeat"):
        if sorted(part.sample_id) != ids:
            raise ValueError("Every repeat must contain exactly the same berries")
    if (data.groupby("sample_id").measured_TSS.nunique() != 1).any() or (
        data.groupby("sample_id").acquisition_id.nunique() != 1
    ).any():
        raise ValueError("Measurement or cluster identity changes between repeats")
    return data.sort_values(["repeat", "sample_id"]).reset_index(drop=True)


def berry_mean_predictions(frame):
    data = canonical_predictions(frame)
    return data.groupby(["sample_id", "acquisition_id"], as_index=False).agg(
        measured_TSS=("measured_TSS", "first"),
        ensemble_predicted_TSS=("ensemble_predicted_TSS", "mean"))


def cluster_bootstrap_draws(frame, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED, include_slope=False):
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    data = canonical_predictions(frame)
    parts = [part.sort_values("sample_id") for _, part in data.groupby("repeat", sort=True)]
    if not parts:
        raise ValueError("Empty prediction table")
    first = parts[0]
    group_names = sorted(first.acquisition_id.unique())
    positions = [np.flatnonzero(first.acquisition_id.to_numpy() == group) for group in group_names]
    y = first.measured_TSS.to_numpy(dtype=float)
    predictions = np.stack([part.ensemble_predicted_TSS.to_numpy(dtype=float) for part in parts])
    names = (*METRICS, "slope") if include_slope else METRICS
    draws = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        indices = np.concatenate([positions[j] for j in rng.integers(len(positions), size=len(positions))])
        values = []
        for p in predictions:
            item = metrics(y[indices], p[indices])
            if include_slope:
                item["slope"] = float(np.polyfit(y[indices], p[indices], 1)[0]) if np.ptp(y[indices]) > 0 else np.nan
            values.append(item)
        for name in names:
            draws[name].append(float(np.mean([item[name] for item in values])))
    return {name: np.asarray(values, dtype=float) for name, values in draws.items()}


def cluster_bootstrap_ci(frame, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED, include_slope=False):
    draws = cluster_bootstrap_draws(frame, iterations, seed, include_slope)
    return {name: np.nanquantile(values, [0.025, 0.975]).tolist()
            for name, values in draws.items() if len(values)}


def paired_cluster_comparison(reference, baseline, model_name):
    left, right = canonical_predictions(reference), canonical_predictions(baseline)
    columns = ["repeat", "sample_id", "acquisition_id", "measured_TSS", "ensemble_predicted_TSS"]
    paired = left[columns].merge(right[columns], on=["repeat", "sample_id"],
        suffixes=("_a", "_b"), validate="one_to_one", how="outer", indicator=True)
    if not paired._merge.eq("both").all() or not paired.acquisition_id_a.eq(paired.acquisition_id_b).all() or not np.array_equal(paired.measured_TSS_a, paired.measured_TSS_b):
        raise ValueError("Model comparisons require identical berries, repeats, clusters and targets")
    errors = pd.DataFrame({"acquisition_id": paired.acquisition_id_a})
    for model in ("a", "b"):
        error = paired[f"ensemble_predicted_TSS_{model}"] - paired.measured_TSS_a
        errors[f"AE_{model}"] = error.abs()
        errors[f"SE_{model}"] = error ** 2
    clusters = errors.groupby("acquisition_id", sort=True).mean()
    rows = []
    for measure in ("AE", "SE"):
        difference = clusters[f"{measure}_a"] - clusters[f"{measure}_b"]
        p = 1.0 if np.all(difference.to_numpy() == 0) else float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue)
        rows.append({"model_a": "IDASNet", "model_b": model_name, "error_measure": measure,
                     "n_clusters": len(clusters), "mean_paired_error_difference": float(difference.mean()),
                     "unadjusted_p_value": p})
    return pd.DataFrame(rows), clusters.reset_index(), paired.drop(columns="_merge")


def save_paired_comparisons(predictions, reference_run, result_dir, model_name):
    reference = pd.read_csv(Path(reference_run) / "oof_predictions_seed_ensemble.csv")
    summary, clusters, paired = paired_cluster_comparison(reference, predictions, model_name)
    root = Path(result_dir)
    summary.to_csv(root / "paired_model_comparisons.csv", index=False)
    clusters.to_csv(root / "paired_cluster_errors.csv", index=False)
    paired.to_csv(root / "paired_oof_predictions.csv", index=False)
    return summary
