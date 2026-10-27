"""Train and evaluate IDASNet with acquisition-grouped nested cross-validation.

Inner GroupKFold splits select hyperparameters and training epochs using
TPE and Hyperband. Each selected model is refitted on the outer training
partition before predicting the held-out acquisitions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from data.folds import (
    GroupFold,
    RawDataset,
    build_repeated_group_folds,
    iter_inner_group_folds,
    load_raw_dataset,
    save_outer_fold_manifest,
)
from idasnet.model import HybridDualAttentionSpectralNet  # noqa: E402
from data import NORMALIZATION
from idasnet.protocol import PROTOCOL_VERSION, TRAINING, load_run_config, protocol_settings, suggest
from idasnet.statistics import cluster_bootstrap_ci, berry_mean_predictions, save_paired_comparisons


# Default paths and experiment settings
DEFAULT_RAW_PT = DATASET_DIR / "generated" / "raw.pt"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "IDASNet"
DEFAULT_TARGET = "TSS"

DEFAULT_OUTER_SPLITS = TRAINING["outer_splits"]
DEFAULT_OUTER_REPEATS = TRAINING["outer_repeats"]
DEFAULT_OUTER_BASE_SEED = TRAINING["outer_base_seed"]
DEFAULT_INNER_SPLITS = TRAINING["inner_splits"]
DEFAULT_N_TRIALS = TRAINING["n_trials"]
DEFAULT_INNER_MAX_EPOCHS = TRAINING["max_epochs"]
DEFAULT_EARLY_STOPPING_PATIENCE = TRAINING["patience"]
DEFAULT_MIN_FOLDS_BEFORE_PRUNING = 2
# Prespecified stochastic repetitions. The complete seed mean is the
# official prediction; training seeds are fixed before evaluation.
DEFAULT_TRAIN_SEEDS = list(TRAINING["train_seeds"])
DEFAULT_BATCH_SIZE = TRAINING["batch_size"]
DEFAULT_BOOTSTRAP_ITERATIONS = TRAINING["bootstrap_iterations"]
DEFAULT_SAVE_CHECKPOINTS = True
DEFAULT_PARALLEL_OUTER_FOLDS = 4
DEFAULT_CPU_THREADS_PER_WORKER = 1


@dataclass(frozen=True)
class RunConfig:
    raw_pt: str
    target: str
    outer_splits: int
    outer_repeats: int
    outer_base_seed: int
    inner_splits: int
    n_trials: int
    inner_max_epochs: int
    early_stopping_patience: int
    min_folds_before_pruning: int
    train_seeds: list[int]
    batch_size: int
    bootstrap_iterations: int
    save_checkpoints: bool
    parallel_outer_folds: int
    cpu_threads_per_worker: int
    device: str
    model_type: str = "IDASNet"
    reference_result: str | None = None
    protocol_version: str = PROTOCOL_VERSION


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_spectra(X: torch.Tensor, epsilon: float = NORMALIZATION["epsilon"]) -> torch.Tensor:
    """Z-score each sample/method spectrum independently over wavelengths.

    X has shape (samples, preprocessing methods, wavelengths). The sample
    standard deviation uses correction=1. No dataset-level statistics are fitted.
    """
    normalized = (X - X.mean(dim=-1, keepdim=True)) / (
        X.std(dim=-1, keepdim=True, correction=NORMALIZATION["std_correction"]) + epsilon
    )
    if not torch.isfinite(normalized).all():
        raise ValueError("Non-finite values after spectral Z-score normalization")
    return normalized


def next_result_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    result = root / str(max(existing, default=0) + 1)
    result.mkdir()
    return result


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
    }


def build_model(params: dict[str, Any], channels: int, length: int) -> nn.Module:
    if params.get("model_type", "IDASNet") != "IDASNet":
        from baselines.neural_models import StandardCNN1D, LightResNet1D
        families = {"CNN1D": StandardCNN1D, "ResNet1D": LightResNet1D}
        cls = families[params["model_type"]]
        return cls(channels, length, base_channels=int(params["base_channels"]),
                   kernel_size=int(params["kernel_size"]), hidden_dim=int(params["hidden_dim"]),
                   dropout=float(params["dropout"]))
    model = HybridDualAttentionSpectralNet(
        input_channels=channels,
        length=length,
        num_outputs=1,
        stem_channels=int(params["stem_channels"]),
        feature_channels=int(params["feature_channels"]),
        deep_channels=int(params["deep_channels"]),
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        raw_channel_reduction=int(params["raw_channel_reduction"]),
        wavelength_reduction=int(params["wavelength_reduction"]),
        feature_channel_reduction=int(params["feature_channel_reduction"]),
    )
    return model


def suggest_params(trial: optuna.Trial, model_type="IDASNet") -> dict[str, Any]:
    return suggest(trial, model_type)


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(X.float(), y.float()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _optimizer_and_scheduler(
    model: nn.Module,
    params: dict[str, Any],
    max_epochs: int,
) -> tuple[torch.optim.Optimizer, Any]:
    optimizer_class = torch.optim.AdamW if params["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = None
    if bool(params["use_scheduler"]):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))
    return optimizer, scheduler


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for X, _ in loader:
            output = model(X.to(device))["pred"]
            values.append(output.detach().cpu().reshape(-1).numpy())
    return np.concatenate(values)


def train_with_validation(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    valid_X: torch.Tensor,
    valid_y: torch.Tensor,
    params: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    """Train one inner-fold model and return its best validation MSE/epoch."""
    set_seed(seed)
    model = build_model(params, train_X.shape[1], train_X.shape[2]).to(device)
    optimizer, scheduler = _optimizer_and_scheduler(model, params, max_epochs)
    criterion = nn.MSELoss()
    train_loader = make_loader(train_X, train_y, batch_size, True, seed)
    valid_loader = make_loader(valid_X, valid_y, batch_size, False, seed)
    best_loss = math.inf
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            output = model(X)["pred"].reshape(-1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        predictions = predict(model, valid_loader, device)
        validation_loss = float(np.mean((predictions - valid_y.numpy()) ** 2))
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (best_loss, best_epoch)


def optimize_inner(
    dataset: RawDataset,
    outer_fold: GroupFold,
    config: RunConfig,
    fold_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], int, float]:
    """Run a fresh TPE+Hyperband study within one outer-training partition."""
    inner_folds = list(
        iter_inner_group_folds(
            outer_fold.train_indices, dataset.acquisition_ids, config.inner_splits
        )
    )
    actual_inner_folds = len(inner_folds)
    minimum_folds = min(config.min_folds_before_pruning, actual_inner_folds)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, config.model_type)
        fold_losses: list[float] = []
        fold_epochs: list[int] = []
        for inner_number, (train_indices, valid_indices) in enumerate(inner_folds, start=1):
            train_X = normalize_spectra(dataset.X[torch.as_tensor(train_indices, dtype=torch.long)])
            valid_X = normalize_spectra(dataset.X[torch.as_tensor(valid_indices, dtype=torch.long)])
            train_y = dataset.y[torch.as_tensor(train_indices, dtype=torch.long)]
            valid_y = dataset.y[torch.as_tensor(valid_indices, dtype=torch.long)]
            inner_seed = (
                config.outer_base_seed
                + outer_fold.repeat * 10000
                + outer_fold.fold * 1000
                + inner_number
            )
            loss, best_epoch = train_with_validation(
                train_X,
                train_y,
                valid_X,
                valid_y,
                params,
                inner_seed,
                config.inner_max_epochs,
                config.early_stopping_patience,
                config.batch_size,
                device,
            )
            fold_losses.append(loss)
            fold_epochs.append(best_epoch)
            cumulative_loss = float(np.mean(fold_losses))
            trial.report(cumulative_loss, step=inner_number)
            if inner_number >= minimum_folds and trial.should_prune():
                trial.set_user_attr("completed_fold_losses", fold_losses)
                trial.set_user_attr("completed_fold_best_epochs", fold_epochs)
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("fold_best_epochs", fold_epochs)
        trial.set_user_attr("mean_validation_loss", float(np.mean(fold_losses)))
        return float(np.mean(fold_losses))

    sampler_seed = config.outer_base_seed + outer_fold.repeat * 100 + outer_fold.fold
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=minimum_folds, max_resource=actual_inner_folds, reduction_factor=2
        ),
        study_name=f"{config.model_type}_repeat_{outer_fold.repeat + 1:02d}_fold_{outer_fold.fold + 1:02d}",
    )
    study.optimize(objective, n_trials=config.n_trials, n_jobs=1, gc_after_trial=True)
    if not study.trials or not any(
        (trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    ):
        raise RuntimeError("No inner hyperparameter trial completed all grouped folds")
    trials = study.trials_dataframe()
    trials.to_csv(fold_dir / "inner_optuna_trials.csv", index=False, encoding="utf-8-sig")
    best = study.best_trial
    best_epochs = [int(v) for v in best.user_attrs.get("fold_best_epochs", [])]
    selected_epochs = max(1, int(round(float(np.median(best_epochs))))) if best_epochs else 1
    best_params = dict(best.params)
    if config.model_type != "IDASNet":
        best_params["model_type"] = config.model_type
    payload = {
        "best_trial_number": best.number,
        "best_inner_mse": float(best.value),
        "best_params": best_params,
        "inner_fold_best_epochs": best_epochs,
        "selected_final_epochs": selected_epochs,
        "inner_folds": actual_inner_folds,
        "minimum_folds_before_pruning": minimum_folds,
    }
    (fold_dir / "inner_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (best_params, selected_epochs, float(best.value))


def train_final_model(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    """Refit on the outer training partition and record the learning curve."""
    set_seed(seed)
    model = build_model(params, train_X.shape[1], train_X.shape[2]).to(device)
    optimizer, scheduler = _optimizer_and_scheduler(model, params, epochs)
    criterion = nn.MSELoss()
    loader = make_loader(train_X, train_y, batch_size, True, seed)
    evaluation_loader = make_loader(train_X, train_y, batch_size, False, seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_squared_error = 0.0
        batch_sample_count = 0
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            output = model(X)["pred"].reshape(-1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            batch_squared_error += float(loss.detach().cpu()) * len(X)
            batch_sample_count += len(X)
        if scheduler is not None:
            scheduler.step()
        training_predictions = predict(model, evaluation_loader, device)
        training_metrics = compute_metrics(train_y.numpy(), training_predictions)
        history.append(
            {
                "epoch": epoch,
                "optimization_batch_MSE": batch_squared_error / batch_sample_count,
                "evaluation_training_R2": training_metrics["R2"],
                "evaluation_training_RMSE": training_metrics["RMSE"],
                "evaluation_training_MAE": training_metrics["MAE"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    return (model, history)


def collect_training_attention(
    model: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect attention weights from the outer training samples."""
    loader = make_loader(train_X, train_y, batch_size, False, 0)
    method_values: list[np.ndarray] = []
    wavelength_values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X, _ in loader:
            output = model(X.to(device))
            method_values.append(
                output["raw_channel_weights"].detach().cpu().reshape(len(X), -1).numpy()
            )
            wavelength_values.append(
                output["wavelength_weights"].detach().cpu().reshape(len(X), -1).numpy()
            )
    return np.concatenate(method_values).mean(axis=0), np.concatenate(wavelength_values).mean(
        axis=0
    )


def validate_oof_coverage(
    predictions: pd.DataFrame, dataset: RawDataset, config: RunConfig
) -> None:
    expected_ids = set(dataset.global_ids.tolist())
    for repeat in range(config.outer_repeats):
        for seed in config.train_seeds:
            block = predictions[
                (predictions["repeat"] == repeat) & (predictions["train_seed"] == seed)
            ]
            ids = block["sample_id"].astype(str).tolist()
            if len(ids) != len(dataset.X) or set(ids) != expected_ids or len(set(ids)) != len(ids):
                raise RuntimeError(f"Invalid OOF coverage for repeat={repeat}, train_seed={seed}")


def summarize_predictions(predictions, dataset, config, result_dir):
    """Official prespecified seed ensemble; no test-based seed selection."""
    validate_oof_coverage(predictions, dataset, config)
    predictions.to_csv(result_dir / "oof_predictions_by_seed.csv", index=False)
    keys = ["repeat", "fold", "outer_seed", "sample_index", "sample_id", "acquisition_id", "measured_TSS"]
    ensemble = predictions.groupby(keys, as_index=False, sort=False).predicted_TSS.mean().rename(
        columns={"predicted_TSS": "ensemble_predicted_TSS"})
    ensemble.to_csv(result_dir / "oof_predictions_seed_ensemble.csv", index=False)
    rows = []
    for repeat, part in ensemble.groupby("repeat"):
        error = part.ensemble_predicted_TSS - part.measured_TSS
        rows.append({"repeat": int(repeat), **compute_metrics(part.measured_TSS, part.ensemble_predicted_TSS),
                     "bias": float(error.mean())})
    scores = pd.DataFrame(rows)
    scores.to_csv(result_dir / "metrics_by_repeat_seed_ensemble.csv", index=False)
    summary = {"protocol_version": config.protocol_version, "model": config.model_type,
        "official_all_seed_ensemble_performance": {
            "seed_count": len(config.train_seeds), "seeds": config.train_seeds,
            **{m: {"mean": float(scores[m].mean()), "sd_across_repeats": float(scores[m].std(ddof=1))}
               for m in ("R2", "RMSE", "MAE", "bias")}},
        "bootstrap_note": "Fixed fitted models; shared whole-cluster draws across all repeats and models; no repeat resampling."}
    if config.bootstrap_iterations:
        summary["cluster_bootstrap_95_ci"] = cluster_bootstrap_ci(
            ensemble, config.bootstrap_iterations, TRAINING["bootstrap_seed"], include_slope=True)
    (result_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_posthoc_diagnostics(ensemble, result_dir)
    if config.reference_result:
        save_paired_comparisons(ensemble, config.reference_result, result_dir, config.model_type)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return slope and intercept, or NaN when a fit is not identifiable."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) <= 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def save_posthoc_diagnostics(
    ensemble: pd.DataFrame,
    result_dir: Path,
    output_prefix: str = "",
) -> None:
    """Export numerical diagnostics from outer-test predictions."""
    diagnostic_rows: list[dict[str, float | int]] = []
    for repeat, block in ensemble.groupby("repeat", sort=True):
        measured = block["measured_TSS"].to_numpy(dtype=np.float64)
        predicted = block["ensemble_predicted_TSS"].to_numpy(dtype=np.float64)
        error = predicted - measured
        prediction_slope, prediction_intercept = _linear_fit(measured, predicted)
        calibration_slope, calibration_intercept = _linear_fit(predicted, measured)
        residual_slope, residual_intercept = _linear_fit(measured, error)
        bias = float(error.mean())
        error_sd = float(error.std(ddof=1))
        diagnostic_rows.append(
            {
                "repeat": int(repeat),
                "n": int(len(block)),
                "bias_predicted_minus_measured": bias,
                "bias_sd": error_sd,
                "bland_altman_lower_loa": bias - 1.96 * error_sd,
                "bland_altman_upper_loa": bias + 1.96 * error_sd,
                "prediction_vs_measurement_slope": prediction_slope,
                "prediction_vs_measurement_intercept": prediction_intercept,
                "calibration_measured_on_prediction_slope": calibration_slope,
                "calibration_measured_on_prediction_intercept": calibration_intercept,
                "error_vs_measurement_slope": residual_slope,
                "error_vs_measurement_intercept": residual_intercept,
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(result_dir / f"{output_prefix}diagnostics_by_repeat.csv", index=False)
    diagnostics.describe().loc[["mean", "std"]].to_csv(
        result_dir / f"{output_prefix}diagnostics_mean_sd.csv"
    )

    # TSS tertiles are used for descriptive diagnostics.
    unique_samples = ensemble.drop_duplicates("sample_id")
    boundaries = np.quantile(unique_samples["measured_TSS"], [1 / 3, 2 / 3])
    low, high = (float(boundaries[0]), float(boundaries[1]))
    range_rows: list[dict[str, Any]] = []
    labels = [f"low_<=_{low:.3f}", f"middle_{low:.3f}_to_{high:.3f}", f"high_>_{high:.3f}"]
    for repeat, block in ensemble.groupby("repeat", sort=True):
        measured = block["measured_TSS"].to_numpy(dtype=np.float64)
        masks = [measured <= low, (measured > low) & (measured <= high), measured > high]
        for label, mask in zip(labels, masks, strict=True):
            selected = block.loc[mask]
            if selected.empty:
                continue
            range_rows.append(
                {
                    "repeat": int(repeat),
                    "TSS_range": label,
                    "lower_tertile_boundary": low,
                    "upper_tertile_boundary": high,
                    "n": int(len(selected)),
                    **compute_metrics(selected["measured_TSS"], selected["ensemble_predicted_TSS"]),
                    "bias_predicted_minus_measured": float(
                        (selected["ensemble_predicted_TSS"] - selected["measured_TSS"]).mean()
                    ),
                }
            )
    range_metrics = pd.DataFrame(range_rows)
    range_metrics.to_csv(
        result_dir / f"{output_prefix}metrics_by_TSS_range_and_repeat.csv",
        index=False,
    )
    range_metrics.groupby("TSS_range")[["R2", "RMSE", "MAE", "bias_predicted_minus_measured"]].agg(
        ["mean", "std"]
    ).to_csv(result_dir / f"{output_prefix}metrics_by_TSS_range_mean_sd.csv")

    ensemble = berry_mean_predictions(ensemble)
    measured = ensemble["measured_TSS"].to_numpy(dtype=np.float64)
    predicted = ensemble["ensemble_predicted_TSS"].to_numpy(dtype=np.float64)
    error = predicted - measured
    average = (predicted + measured) / 2.0

    # Export berry-level mean predictions and residuals.
    diagnostic_data = ensemble.copy()
    diagnostic_data["bland_altman_average"] = average
    diagnostic_data["prediction_error_predicted_minus_measured"] = error
    diagnostic_data.to_csv(
        result_dir / f"{output_prefix}berry_mean_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_attention_rankings(
    method_records: list[dict[str, Any]],
    wavelength_records: list[dict[str, Any]],
    result_dir: Path,
    method_seed_records: list[dict[str, Any]] | None = None,
    wavelength_seed_records: list[dict[str, Any]] | None = None,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    methods = pd.DataFrame(method_records)
    wavelengths = pd.DataFrame(wavelength_records)
    methods.to_csv(
        result_dir / "training_method_attention_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    wavelengths.to_csv(
        result_dir / "training_wavelength_attention_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    if method_seed_records is not None:
        pd.DataFrame(method_seed_records).to_csv(
            result_dir / "training_method_attention_by_fold_and_seed.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if wavelength_seed_records is not None:
        pd.DataFrame(wavelength_seed_records).to_csv(
            result_dir / "training_wavelength_attention_by_fold_and_seed.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not methods.empty:
        methods = methods.groupby(["repeat", "method_index", "method"], as_index=False).importance.mean()
        methods.to_csv(result_dir / "training_method_attention_by_repeat.csv", index=False)
        method_summary = (
            methods.groupby(["method_index", "method"])["importance"]
            .agg(
                mean="mean",
                std="std",
                median="median",
                minimum="min",
                maximum="max",
                n_repeats="count",
            )
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        method_summary.to_csv(
            result_dir / "training_method_attention_summary.csv", index=False, encoding="utf-8-sig"
        )
        method_summary_data = method_summary.rename(
            columns={
                "method": "Method",
                "mean": "Mean_Importance",
                "std": "SD_Across_Repeats",
                "median": "Median_Importance",
                "minimum": "Minimum_Importance",
                "maximum": "Maximum_Importance",
                "n_repeats": "N_Repeats",
            }
        )
        method_summary_data.to_csv(
            result_dir / "method_importance_mean.csv", index=False, encoding="utf-8-sig"
        )
    if not wavelengths.empty:
        wavelengths = wavelengths.groupby(["repeat", "wavelength_index", "wavelength_nm"], as_index=False).importance.mean()
        wavelengths.to_csv(result_dir / "training_wavelength_attention_by_repeat.csv", index=False)
        ranked = wavelengths.copy()
        ranked["rank_within_repeat"] = ranked.groupby(["repeat"])["importance"].rank(
            method="average", ascending=False
        )
        ranked["is_top20_within_repeat"] = (ranked["rank_within_repeat"] <= 20).astype(int)
        wavelength_summary = (
            ranked.groupby(["wavelength_index", "wavelength_nm"])
            .agg(
                mean=("importance", "mean"),
                std=("importance", "std"),
                median=("importance", "median"),
                minimum=("importance", "min"),
                maximum=("importance", "max"),
                n_repeats=("importance", "count"),
                mean_rank=("rank_within_repeat", "mean"),
                top20_repeat_count=("is_top20_within_repeat", "sum"),
            )
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        wavelength_summary["top20_repeat_frequency"] = (
            wavelength_summary["top20_repeat_count"] / wavelength_summary["n_repeats"]
        )
        wavelength_summary.to_csv(
            result_dir / "training_wavelength_attention_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        wavelength_summary_data = wavelength_summary.rename(
            columns={
                "wavelength_nm": "Wavelength_nm",
                "mean": "Mean_Importance",
                "std": "SD_Across_Repeats",
                "median": "Median_Importance",
                "minimum": "Minimum_Importance",
                "maximum": "Maximum_Importance",
                "n_repeats": "N_Repeats",
                "mean_rank": "Mean_Rank_Across_Repeats",
                "top20_repeat_count": "Top20_Repeat_Count",
                "top20_repeat_frequency": "Top20_Repeat_Frequency",
            }
        )
        wavelength_summary_data.to_csv(
            result_dir / "wavelength_importance_mean.csv", index=False, encoding="utf-8-sig"
        )
    metadata = {
        "interpretation": "Attention-derived importance, not causal feature importance.",
        "test_data_used_for_attention": False,
        "within_model_aggregation": "Mean attention across outer-training samples.",
        "within_fold_aggregation": "Mean across all prespecified final-training seeds.",
        "point_estimate": "Mean of repetition estimates; each repetition first averages its outer folds.",
        "variability": "Sample standard deviation across repetition means.",
        "outer_fold_dependence_note": (
            "Repeated partitions overlap in samples; the repetition SD is a descriptive "
            "stability measure, not an independent-sample confidence interval."
        ),
    }
    (result_dir / "attention_aggregation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_WORKER_DATASET: RawDataset | None = None
_WORKER_DATASET_KEY: tuple[str, str] | None = None


def _initialize_outer_worker(cpu_threads: int) -> None:
    """Configure one isolated process before it starts CUDA work."""
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _worker_dataset(config: RunConfig) -> RawDataset:
    """Load the small raw dataset once per worker process."""
    global _WORKER_DATASET, _WORKER_DATASET_KEY
    key = (config.raw_pt, config.target)
    if _WORKER_DATASET is None or _WORKER_DATASET_KEY != key:
        _WORKER_DATASET = load_raw_dataset(Path(config.raw_pt), config.target)
        _WORKER_DATASET_KEY = key
    return _WORKER_DATASET


def _outer_fold_directory(result_dir: Path, outer_fold: GroupFold) -> Path:
    return result_dir / f"repeat_{outer_fold.repeat + 1:02d}" / f"fold_{outer_fold.fold + 1:02d}"


def _outer_fold_output_paths(fold_dir: Path) -> dict[str, Path]:
    return {
        "predictions": fold_dir / "fold_predictions.csv",
        "methods": fold_dir / "fold_training_method_attention.csv",
        "wavelengths": fold_dir / "fold_training_wavelength_attention.csv",
        "methods_by_seed": fold_dir / "fold_training_method_attention_by_seed.csv",
        "wavelengths_by_seed": fold_dir / "fold_training_wavelength_attention_by_seed.csv",
        "selected": fold_dir / "selected_hyperparameters.json",
        "complete": fold_dir / "fold_complete.json",
    }


def _outer_fold_is_complete(fold_dir: Path) -> bool:
    paths = _outer_fold_output_paths(fold_dir)
    if paths["complete"].is_file():
        try:
            marker = json.loads(paths["complete"].read_text(encoding='utf-8'))
            if marker.get("model_type", "IDASNet") != "IDASNet":
                return marker.get("status") == "complete" and paths["predictions"].is_file() and paths["selected"].is_file()
        except (OSError, ValueError):
            return False
    required = (
        paths["predictions"],
        paths["methods"],
        paths["wavelengths"],
        paths["methods_by_seed"],
        paths["wavelengths_by_seed"],
        paths["selected"],
        paths["complete"],
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("status") == "complete"


def run_one_outer_fold(
    config: RunConfig, outer_fold: GroupFold, result_dir_text: str
) -> tuple[int, int]:
    """Execute one complete outer fold inside an isolated worker process."""
    result_dir = Path(result_dir_text)
    fold_dir = _outer_fold_directory(result_dir, outer_fold)
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = _outer_fold_output_paths(fold_dir)
    if _outer_fold_is_complete(fold_dir):
        return (outer_fold.repeat, outer_fold.fold)
    paths["complete"].unlink(missing_ok=True)
    prefix = f"R{outer_fold.repeat + 1:02d}-F{outer_fold.fold + 1:02d}"
    print(
        f"\n[{prefix}] starting in worker process; train groups={len(outer_fold.train_groups)}, test groups={len(outer_fold.test_groups)}",
        flush=True,
    )
    dataset = _worker_dataset(config)
    device = torch.device(config.device)
    best_params, final_epochs, best_inner_mse = optimize_inner(
        dataset, outer_fold, config, fold_dir, device
    )
    selected_record = {
        "repeat": outer_fold.repeat,
        "fold": outer_fold.fold,
        "best_inner_mse": best_inner_mse,
        "selected_final_epochs": final_epochs,
        **best_params,
    }
    train_X = normalize_spectra(
        dataset.X[torch.as_tensor(outer_fold.train_indices, dtype=torch.long)]
    )
    test_X = normalize_spectra(
        dataset.X[torch.as_tensor(outer_fold.test_indices, dtype=torch.long)]
    )
    train_y = dataset.y[torch.as_tensor(outer_fold.train_indices, dtype=torch.long)]
    test_y = dataset.y[torch.as_tensor(outer_fold.test_indices, dtype=torch.long)]
    test_loader = make_loader(test_X, test_y, config.batch_size, False, 0)
    fold_method_attention: list[np.ndarray] = []
    fold_wavelength_attention: list[np.ndarray] = []
    fold_prediction_records: list[dict[str, Any]] = []
    for train_seed in config.train_seeds:
        print(f"[{prefix}] final training seed={train_seed}, epochs={final_epochs}", flush=True)
        model, training_history = train_final_model(
            train_X, train_y, best_params, train_seed, final_epochs, config.batch_size, device
        )
        pd.DataFrame(training_history).to_csv(
            fold_dir / f"training_history_seed_{train_seed}.csv", index=False, encoding="utf-8-sig"
        )
        predicted = predict(model, test_loader, device)
        if config.model_type == "IDASNet":
            method_attention, wavelength_attention = collect_training_attention(
                model, train_X, train_y, config.batch_size, device)
            fold_method_attention.append(method_attention)
            fold_wavelength_attention.append(wavelength_attention)
        if config.save_checkpoints:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "params": best_params,
                    "input_metadata": {"input_shape": list(dataset.X.shape[1:]),
                        "processing_methods": dataset.processing_methods, "wavelengths": dataset.wavelengths,
                        "normalization": NORMALIZATION},
                    "epochs": final_epochs,
                    "train_seed": train_seed,
                    "repeat": outer_fold.repeat,
                    "fold": outer_fold.fold,
                },
                fold_dir / f"model_seed_{train_seed}.pth",
            )
        for local_index, sample_index in enumerate(outer_fold.test_indices):
            fold_prediction_records.append(
                {
                    "repeat": outer_fold.repeat,
                    "fold": outer_fold.fold,
                    "outer_seed": outer_fold.seed,
                    "train_seed": train_seed,
                    "sample_index": int(sample_index),
                    "sample_id": str(dataset.global_ids[sample_index]),
                    "acquisition_id": str(dataset.acquisition_ids[sample_index]),
                    "measured_TSS": float(test_y[local_index]),
                    "predicted_TSS": float(predicted[local_index]),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if config.model_type != "IDASNet":
        pd.DataFrame(fold_prediction_records).to_csv(paths["predictions"], index=False)
        paths["selected"].write_text(json.dumps(selected_record, indent=2), encoding="utf-8")
        paths["complete"].write_text(json.dumps({"status": "complete", "model_type": config.model_type}), encoding="utf-8")
        return outer_fold.repeat, outer_fold.fold
    mean_method = np.mean(np.stack(fold_method_attention), axis=0)
    mean_wavelength = np.mean(np.stack(fold_wavelength_attention), axis=0)
    method_names = dataset.processing_methods or [
        f"method_{index}" for index in range(len(mean_method))
    ]
    wavelength_values = dataset.wavelengths or list(range(len(mean_wavelength)))
    fold_method_records = [
        {
            "repeat": outer_fold.repeat,
            "fold": outer_fold.fold,
            "method_index": index,
            "method": method_names[index],
            "importance": float(value),
            "source": "outer_training_samples_only",
        }
        for index, value in enumerate(mean_method)
    ]
    fold_wavelength_records = [
        {
            "repeat": outer_fold.repeat,
            "fold": outer_fold.fold,
            "wavelength_index": index,
            "wavelength_nm": float(wavelength_values[index]),
            "importance": float(value),
            "source": "outer_training_samples_only",
        }
        for index, value in enumerate(mean_wavelength)
    ]
    fold_method_seed_records = [
        {
            "repeat": outer_fold.repeat,
            "fold": outer_fold.fold,
            "train_seed": int(train_seed),
            "method_index": index,
            "method": method_names[index],
            "importance": float(value),
            "source": "outer_training_samples_only",
        }
        for train_seed, attention in zip(config.train_seeds, fold_method_attention, strict=True)
        for index, value in enumerate(attention)
    ]
    fold_wavelength_seed_records = [
        {
            "repeat": outer_fold.repeat,
            "fold": outer_fold.fold,
            "train_seed": int(train_seed),
            "wavelength_index": index,
            "wavelength_nm": float(wavelength_values[index]),
            "importance": float(value),
            "source": "outer_training_samples_only",
        }
        for train_seed, attention in zip(config.train_seeds, fold_wavelength_attention, strict=True)
        for index, value in enumerate(attention)
    ]
    pd.DataFrame(fold_prediction_records).to_csv(
        paths["predictions"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_method_records).to_csv(paths["methods"], index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_wavelength_records).to_csv(
        paths["wavelengths"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_method_seed_records).to_csv(
        paths["methods_by_seed"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_wavelength_seed_records).to_csv(
        paths["wavelengths_by_seed"], index=False, encoding="utf-8-sig"
    )
    paths["selected"].write_text(
        json.dumps(selected_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker_tmp = fold_dir / "fold_complete.json.tmp"
    marker_tmp.write_text(
        json.dumps(
            {
                "status": "complete",
                "repeat": outer_fold.repeat,
                "fold": outer_fold.fold,
                "prediction_rows": len(fold_prediction_records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    marker_tmp.replace(paths["complete"])
    print(f"[{prefix}] completed", flush=True)
    return (outer_fold.repeat, outer_fold.fold)


def _collect_parallel_fold_outputs(
    folds: list[GroupFold],
    result_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    prediction_records: list[dict[str, Any]] = []
    method_records: list[dict[str, Any]] = []
    wavelength_records: list[dict[str, Any]] = []
    method_seed_records: list[dict[str, Any]] = []
    wavelength_seed_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for outer_fold in folds:
        fold_dir = _outer_fold_directory(result_dir, outer_fold)
        if not _outer_fold_is_complete(fold_dir):
            raise RuntimeError(f"Outer fold is incomplete after execution: {fold_dir}")
        paths = _outer_fold_output_paths(fold_dir)
        prediction_records.extend(pd.read_csv(paths["predictions"]).to_dict(orient="records"))
        if paths["methods"].is_file():
            method_records.extend(pd.read_csv(paths["methods"]).to_dict(orient="records"))
        if paths["wavelengths"].is_file():
            wavelength_records.extend(pd.read_csv(paths["wavelengths"]).to_dict(orient="records"))
        if paths["methods_by_seed"].is_file():
            method_seed_records.extend(pd.read_csv(paths["methods_by_seed"]).to_dict(orient="records"))
        if paths["wavelengths_by_seed"].is_file():
            wavelength_seed_records.extend(pd.read_csv(paths["wavelengths_by_seed"]).to_dict(orient="records"))
        selected_records.append(json.loads(paths["selected"].read_text(encoding="utf-8")))
    return (
        prediction_records,
        method_records,
        wavelength_records,
        method_seed_records,
        wavelength_seed_records,
        selected_records,
    )


def save_dataset_snapshot(raw_pt: Path, result_dir: Path) -> None:
    """Keep the exact input dataset with the run, including after relocation."""
    snapshot = result_dir / "dataset_snapshot.pt"
    content = raw_pt.read_bytes()
    if snapshot.is_file():
        if snapshot.read_bytes() != content:
            raise ValueError("Input dataset differs from this run's saved dataset snapshot")
    else:
        snapshot.write_bytes(content)


def run_parallel(config: RunConfig, result_dir: Path) -> None:
    """Run independent outer folds concurrently and aggregate in canonical order."""
    print("Input normalization: Z-score per sample and method over wavelengths", flush=True)
    settings = protocol_settings()
    settings_path = result_dir / "paper_protocol.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Saved search spaces or preprocessing settings differ from the current configuration")
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    dataset = load_raw_dataset(Path(config.raw_pt), config.target)
    save_dataset_snapshot(Path(config.raw_pt), result_dir)
    if config.reference_result:
        from data.folds import load_outer_fold_manifest
        reference = Path(config.reference_result)
        source = load_raw_dataset(reference / "dataset_snapshot.pt", config.target)
        if (not torch.equal(source.X, dataset.X) or not torch.equal(source.y, dataset.y)
            or source.processing_methods != dataset.processing_methods or source.wavelengths != dataset.wavelengths
            or not np.array_equal(source.acquisition_ids, dataset.acquisition_ids)
            or not np.array_equal(source.global_ids, dataset.global_ids)):
            raise ValueError("Reference run and baseline dataset differ")
        folds = load_outer_fold_manifest(reference / "outer_folds.json", dataset)
    else:
        folds = build_repeated_group_folds(dataset.acquisition_ids, n_splits=config.outer_splits,
            n_repeats=config.outer_repeats, base_seed=config.outer_base_seed)
    save_outer_fold_manifest(result_dir / "outer_folds.json", folds, dataset)
    (result_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending = [
        fold
        for fold in folds
        if not _outer_fold_is_complete(_outer_fold_directory(result_dir, fold))
    ]
    completed_count = len(folds) - len(pending)
    print(
        f"Outer-fold execution: {config.parallel_outer_folds} parallel workers; completed={completed_count}, pending={len(pending)}",
        flush=True,
    )
    if pending:
        workers = min(config.parallel_outer_folds, len(pending))
        if workers == 1:
            _initialize_outer_worker(config.cpu_threads_per_worker)
            for fold in pending:
                run_one_outer_fold(config, fold, str(result_dir.resolve()))
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_outer_worker,
                initargs=(config.cpu_threads_per_worker,),
            ) as executor:
                futures = {
                    executor.submit(
                        run_one_outer_fold, config, fold, str(result_dir.resolve())
                    ): fold
                    for fold in pending
                }
                for future in as_completed(futures):
                    fold = futures[future]
                    try:
                        repeat_index, fold_index = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"Parallel outer fold failed: repeat={fold.repeat + 1}, fold={fold.fold + 1}"
                        ) from error
                    print(
                        f"Main process received completed R{repeat_index + 1:02d}-F{fold_index + 1:02d}",
                        flush=True,
                    )
    predictions, methods, wavelengths, method_seed_records, wavelength_seed_records, selected = (
        _collect_parallel_fold_outputs(folds, result_dir)
    )
    pd.DataFrame(selected).to_csv(
        result_dir / "selected_hyperparameters_by_outer_fold.csv", index=False, encoding="utf-8-sig"
    )
    summarize_predictions(
        pd.DataFrame(predictions), dataset, config, result_dir
    )
    if config.model_type == "IDASNet":
        save_attention_rankings(methods, wavelengths, result_dir,
            method_seed_records=method_seed_records, wavelength_seed_records=wavelength_seed_records)
    (result_dir / "run_complete.json").write_text(json.dumps({"status": "complete", "protocol_version": config.protocol_version}))
    (result_dir / "paper_protocol.json").write_text(json.dumps(protocol_settings(), indent=2), encoding="utf-8")
    print(f"\nNested cross-validation completed: {result_dir}")


def parse_args(default_model="IDASNet") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pt", type=Path, default=DEFAULT_RAW_PT)
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "results" / default_model)
    parser.add_argument("--reference-result", type=Path, default=None if default_model == "IDASNet" else DEFAULT_RESULT_ROOT / "1")
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Continue a previously interrupted IDASNet evaluation result directory.",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--outer-splits", type=int, default=DEFAULT_OUTER_SPLITS)
    parser.add_argument("--outer-repeats", type=int, default=DEFAULT_OUTER_REPEATS)
    parser.add_argument("--outer-base-seed", type=int, default=DEFAULT_OUTER_BASE_SEED)
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--inner-max-epochs", type=int, default=DEFAULT_INNER_MAX_EPOCHS)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE
    )
    parser.add_argument(
        "--min-folds-before-pruning", type=int, default=DEFAULT_MIN_FOLDS_BEFORE_PRUNING
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_CHECKPOINTS,
        help="Save every final outer-fold model (default: enabled).",
    )
    parser.add_argument(
        "--parallel-outer-folds",
        type=int,
        default=DEFAULT_PARALLEL_OUTER_FOLDS,
        help="Number of independent outer folds trained concurrently.",
    )
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=DEFAULT_CPU_THREADS_PER_WORKER,
        help="PyTorch CPU threads allocated to each outer-fold worker.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main(default_model="IDASNet") -> None:
    args = parse_args(default_model)
    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is not available")
    if args.reference_result:
        source = load_run_config(args.reference_result / "run_config.json")
        args.raw_pt = args.reference_result / "dataset_snapshot.pt"
        for name in ("outer_splits", "outer_repeats", "outer_base_seed"):
            setattr(args, name, source[name])
        if list(args.train_seeds) != source["train_seeds"]:
            raise ValueError("Baseline and reference must use the same training seeds; match --train-seeds")
    config = RunConfig(
        model_type=default_model,
        reference_result=str(args.reference_result.resolve()) if args.reference_result else None,
        raw_pt=str(args.raw_pt.resolve()),
        target=args.target,
        outer_splits=args.outer_splits,
        outer_repeats=args.outer_repeats,
        outer_base_seed=args.outer_base_seed,
        inner_splits=args.inner_splits,
        n_trials=args.n_trials,
        inner_max_epochs=args.inner_max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_folds_before_pruning=args.min_folds_before_pruning,
        train_seeds=list(args.train_seeds),
        batch_size=args.batch_size,
        bootstrap_iterations=args.bootstrap_iterations,
        save_checkpoints=args.save_checkpoints,
        parallel_outer_folds=args.parallel_outer_folds,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        device=args.device,
    )
    if (
        config.outer_repeats < 1
        or config.n_trials < 1
        or (not config.train_seeds)
        or (config.parallel_outer_folds < 1)
        or (config.cpu_threads_per_worker < 1)
    ):
        raise ValueError(
            "outer_repeats, n_trials, train_seeds, parallel_outer_folds, and cpu_threads_per_worker must be positive/non-empty"
        )
    if args.resume_dir is None:
        result_dir = next_result_directory(args.result_root)
    else:
        result_dir = args.resume_dir.resolve()
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {result_dir}")
        existing_config_path = result_dir / "run_config.json"
        if existing_config_path.exists():
            existing_config = load_run_config(existing_config_path)
            if existing_config != asdict(config):
                raise ValueError(
                    "The requested configuration does not match the saved resume run_config.json"
                )
    run_parallel(config, result_dir)


if __name__ == "__main__":
    main()
