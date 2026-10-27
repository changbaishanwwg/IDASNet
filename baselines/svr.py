"""SVR regression on the saved IDASNet outer folds.

Uses shared input normalization, grouped inner search, prediction
and comparison with IDASNet.
"""

from __future__ import annotations

import os

# Limit native threads before importing numerical libraries (also in spawn workers).
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from idasnet.statistics import cluster_bootstrap_ci, berry_mean_predictions, save_paired_comparisons as shared_paired_comparisons
from baselines import SamplewiseMethodZScore
from idasnet.protocol import PROTOCOL_VERSION, TRAINING, load_run_config


MODEL_NAME = "SVR"
DEFAULT_RAW_PT = PROJECT_ROOT / "data" / "generated" / "raw.pt"
DEFAULT_IDASNET_RESULT_DIR = PROJECT_ROOT / "results" / "IDASNet" / "1"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "SVR"
DEFAULT_INNER_SPLITS = TRAINING["inner_splits"]
DEFAULT_N_TRIALS = TRAINING["n_trials"]
DEFAULT_OUTER_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_BOOTSTRAP_ITERATIONS = TRAINING["bootstrap_iterations"]
DEFAULT_SAVE_MODELS = True
# Hyperparameter search space (Table S2).
DEFAULT_SEARCH_SPACE = {
    'svr__kernel': ['rbf', 'linear', 'poly'],
    'svr__C': [0.1, 1, 5, 10, 30, 50, 100],
    'svr__epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
    'svr__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],
    'svr__degree': [2, 3],
}

DEFAULT_SEARCH_SEED_BASE = 2026002
DEFAULT_BOOTSTRAP_SEED = TRAINING["bootstrap_seed"]


@dataclass(frozen=True)
class RawDataset:
    X: torch.Tensor
    y: torch.Tensor
    global_ids: np.ndarray
    acquisition_ids: np.ndarray
    dates: np.ndarray
    processing_methods: list[str]
    wavelengths: list[float]


@dataclass(frozen=True)
class GroupFold:
    repeat: int
    fold: int
    seed: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def load_raw_dataset(path: Path, target: str = "TSS") -> RawDataset:
    """Load and validate the raw dataset required by nested CV."""
    data = torch.load(path, weights_only=False)
    if "X" not in data or "y" not in data or "metadata" not in data:
        raise ValueError(f"{path} is not a valid IDASNet raw dataset")
    if target not in data["y"]:
        raise ValueError(f"Target {target!r} is not available in {path}")

    X = torch.as_tensor(data["X"], dtype=torch.float32)
    y = torch.as_tensor(data["y"][target], dtype=torch.float32).reshape(-1)
    metadata = data["metadata"]
    required = ("global_ids", "acquisition_ids", "dates")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(
            f"Raw dataset metadata is missing {missing}; rebuild it with data/build.py"
        )

    global_ids = np.asarray(metadata["global_ids"], dtype=str)
    acquisition_ids = np.asarray(metadata["acquisition_ids"], dtype=str)
    dates = np.asarray(metadata["dates"], dtype=str)
    n = len(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (samples, methods, bands), got {tuple(X.shape)}")
    if any(len(values) != n for values in (y, global_ids, acquisition_ids, dates)):
        raise ValueError("Feature, target, and metadata lengths do not match")
    if not torch.isfinite(X).all() or not torch.isfinite(y).all():
        raise ValueError("Nested CV requires finite spectral features and target values")
    if len(np.unique(global_ids)) != n:
        raise ValueError("global_ids must be unique")
    if len(np.unique(acquisition_ids)) < 2:
        raise ValueError("At least two NH7 acquisitions are required")

    return RawDataset(
        X=X,
        y=y,
        global_ids=global_ids,
        acquisition_ids=acquisition_ids,
        dates=dates,
        processing_methods=[str(v) for v in data.get("processing_methods", [])],
        wavelengths=[float(v) for v in data.get("wavelengths", [])],
    )


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


@dataclass(frozen=True)
class BaselineConfig:
    raw_pt: str
    idasnet_result_dir: str
    inner_splits: int
    n_trials: int
    outer_workers: int
    bootstrap_iterations: int
    save_models: bool
    search_space: dict[str, list[Any]]
    protocol_version: str = PROTOCOL_VERSION


def next_result_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    result = root / str(max(existing, default=0) + 1)
    result.mkdir()
    return result


def load_saved_outer_folds(path: Path, dataset: RawDataset) -> list[GroupFold]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("n_samples") != len(dataset.X):
        raise ValueError("Outer-fold manifest sample count does not match the raw PT")
    folds: list[GroupFold] = []
    for record in payload.get("folds", []):
        train_indices = np.asarray(record["train_indices"], dtype=np.int64)
        test_indices = np.asarray(record["test_indices"], dtype=np.int64)
        if dataset.global_ids[test_indices].tolist() != record["test_sample_ids"]:
            raise ValueError("Outer-fold sample IDs do not match the raw PT ordering")
        train_groups = tuple(sorted(np.unique(dataset.acquisition_ids[train_indices])))
        test_groups = tuple(sorted(np.unique(dataset.acquisition_ids[test_indices])))
        if set(train_groups) & set(test_groups):
            raise RuntimeError("Acquisition leakage in saved outer-fold manifest")
        folds.append(
            GroupFold(
                repeat=int(record["repeat"]),
                fold=int(record["fold"]),
                seed=int(record["seed"]),
                train_indices=train_indices,
                test_indices=test_indices,
                train_groups=train_groups,
                test_groups=test_groups,
            )
        )
    if not folds:
        raise ValueError("No folds were found in the saved outer-fold manifest")
    repeats = sorted({fold.repeat for fold in folds})
    for repeat in repeats:
        coverage = np.zeros(len(dataset.X), dtype=np.int64)
        for fold in folds:
            if fold.repeat == repeat:
                coverage[fold.test_indices] += 1
        if not np.all(coverage == 1):
            raise RuntimeError(f"Invalid OOF coverage in saved repeat {repeat}")
    return sorted(folds, key=lambda fold: (fold.repeat, fold.fold))


def flatten_features(dataset: RawDataset) -> np.ndarray:
    X = dataset.X.detach().cpu().numpy().astype(np.float64, copy=False)
    return X.reshape(len(X), -1)


def _search_seed(fold: GroupFold) -> int:
    return DEFAULT_SEARCH_SEED_BASE + fold.repeat * 1000 + fold.fold * 10


def _search_definition(seed, minimum_inner_train_size, config, n_methods=5, n_wavelengths=151):
    space = dict(config.search_space)
    estimator = Pipeline([("svr", SVR())])
    n_iter = config.n_trials
    estimator = Pipeline(
        [("samplewise_zscore", SamplewiseMethodZScore(n_methods, n_wavelengths)), *estimator.steps]
    )
    return (estimator, space, n_iter)


def _model_fold_directory(result_dir: Path, fold: GroupFold) -> Path:
    return result_dir / f"repeat_{fold.repeat + 1:02d}" / f"fold_{fold.fold + 1:02d}"


def _model_fold_complete(directory: Path) -> bool:
    required = (
        directory / "fold_predictions.csv",
        directory / "selected_hyperparameters.json",
        directory / "inner_search_results.csv",
        directory / "fold_complete.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        marker = json.loads((directory / "fold_complete.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("status") == "complete"


_WORKER_DATASET: RawDataset | None = None


def _worker_dataset(config: BaselineConfig) -> RawDataset:
    global _WORKER_DATASET
    if _WORKER_DATASET is None:
        _WORKER_DATASET = load_raw_dataset(Path(config.raw_pt), "TSS")
    return _WORKER_DATASET


@threadpool_limits.wrap(limits=1)
def run_one_outer_fold(
    config: BaselineConfig,
    fold: GroupFold,
    result_dir_text: str,
) -> tuple[int, int]:
    torch.set_num_threads(1)
    dataset = _worker_dataset(config)
    X = flatten_features(dataset)
    y = dataset.y.detach().cpu().numpy().astype(np.float64)
    result_dir = Path(result_dir_text)
    train_indices = fold.train_indices
    test_indices = fold.test_indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    groups_train = dataset.acquisition_ids[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    actual_inner_splits = min(config.inner_splits, len(np.unique(groups_train)))
    inner_cv = list(
        GroupKFold(n_splits=actual_inner_splits).split(
            X_train,
            y_train,
            groups_train,
        )
    )
    minimum_inner_train_size = min(len(train_local) for train_local, _ in inner_cv)

    output_dir = _model_fold_directory(result_dir, fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _model_fold_complete(output_dir):
        return fold.repeat, fold.fold
    (output_dir / "fold_complete.json").unlink(missing_ok=True)
    seed = _search_seed(fold)
    estimator, parameter_space, n_iter = _search_definition(
        seed,
        minimum_inner_train_size,
        config,
        dataset.X.shape[1],
        dataset.X.shape[2],
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    print(
        f"[R{fold.repeat + 1:02d}-F{fold.fold + 1:02d} {MODEL_NAME}] "
        f"inner search: {n_iter} candidates",
        flush=True,
    )
    search.fit(X_train, y_train)
    predicted = np.asarray(search.predict(X_test), dtype=np.float64).reshape(-1)

    search_results = pd.DataFrame(search.cv_results_)
    retained_columns = [
        column
        for column in search_results.columns
        if column.startswith("param_")
        or column
        in {
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
            "std_fit_time",
        }
    ]
    search_results[retained_columns].sort_values("rank_test_score").to_csv(
        output_dir / "inner_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected = {
        "model": MODEL_NAME,
        "repeat": fold.repeat,
        "fold": fold.fold,
        "outer_seed": fold.seed,
        "inner_splits": actual_inner_splits,
        "search_seed": seed,
        "best_inner_mse": float(-search.best_score_),
        "best_params": search.best_params_,
    }
    (output_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_frame = pd.DataFrame(
        {
            "model": MODEL_NAME,
            "repeat": fold.repeat,
            "fold": fold.fold,
            "outer_seed": fold.seed,
            "sample_index": test_indices,
            "sample_id": dataset.global_ids[test_indices],
            "acquisition_id": dataset.acquisition_ids[test_indices],
            "measured_TSS": y_test,
            "predicted_TSS": predicted,
        }
    )
    prediction_frame.to_csv(
        output_dir / "fold_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if config.save_models:
        joblib.dump(
            {
                "model": search.best_estimator_,
                "selected": selected,
            },
            output_dir / "model.joblib",
        )
    marker_tmp = output_dir / "fold_complete.json.tmp"
    marker_tmp.write_text(
        json.dumps(
            {
                "status": "complete",
                "model": MODEL_NAME,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "prediction_rows": len(prediction_frame),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    marker_tmp.replace(output_dir / "fold_complete.json")
    return fold.repeat, fold.fold


def _validate_predictions(
    predictions: pd.DataFrame,
    dataset: RawDataset,
    repeats: list[int],
) -> None:
    expected_ids = set(dataset.global_ids.tolist())
    for repeat in repeats:
        block = predictions[predictions["repeat"] == repeat]
        ids = block["sample_id"].astype(str).tolist()
        if len(ids) != len(dataset.X) or set(ids) != expected_ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"Invalid OOF coverage for repeat {repeat}")


def summarize_model(
    predictions: pd.DataFrame,
    dataset: RawDataset,
    config: BaselineConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repeats = sorted(predictions["repeat"].unique().tolist())
    _validate_predictions(predictions, dataset, repeats)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    metric_rows = []
    for repeat, block in predictions.groupby("repeat", sort=True):
        metric_rows.append(
            {
                "model": MODEL_NAME,
                "repeat": int(repeat),
                **compute_metrics(block["measured_TSS"], block["predicted_TSS"]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    diagnostic_input = predictions.rename(columns={"predicted_TSS": "ensemble_predicted_TSS"})
    save_posthoc_diagnostics(
        diagnostic_input,
        output_dir,
    )
    summary: dict[str, Any] = {
        "model": MODEL_NAME,
        "outer_folds_source": str(Path(config.idasnet_result_dir) / "outer_folds.json"),
        "metrics": {
            metric: {
                "mean": float(metrics[metric].mean()),
                "sd_across_repeats": float(metrics[metric].std(ddof=1)),
            }
            for metric in ("R2", "RMSE", "MAE")
        },
    }
    if config.bootstrap_iterations > 0:
        summary["cluster_bootstrap_95_ci"] = cluster_bootstrap_ci(
            diagnostic_input,
            iterations=config.bootstrap_iterations,
            seed=DEFAULT_BOOTSTRAP_SEED,
        )
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def save_paired_comparisons(predictions, idasnet_result_dir, result_dir):
    return shared_paired_comparisons(predictions, idasnet_result_dir, result_dir, "SVR")


def run(config: BaselineConfig, result_dir: Path) -> None:
    dataset = load_raw_dataset(Path(config.raw_pt), "TSS")
    idasnet_result_dir = Path(config.idasnet_result_dir)
    reference = load_raw_dataset(idasnet_result_dir / "dataset_snapshot.pt", "TSS")
    if (
        reference.processing_methods != dataset.processing_methods
        or reference.wavelengths != dataset.wavelengths
        or not np.array_equal(reference.dates, dataset.dates)
        or not np.array_equal(reference.global_ids, dataset.global_ids)
        or not np.array_equal(reference.acquisition_ids, dataset.acquisition_ids)
        or (not torch.equal(reference.X, dataset.X))
        or (not torch.equal(reference.y, dataset.y))
    ):
        raise ValueError("Baseline data differ from the reference IDASNet data")
    print("Input normalization: Z-score per sample and method over wavelengths", flush=True)
    folds = load_saved_outer_folds(idasnet_result_dir / "outer_folds.json", dataset)
    (result_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending = [
        fold for fold in folds if not _model_fold_complete(_model_fold_directory(result_dir, fold))
    ]
    print(
        f"Baseline nested CV: model={MODEL_NAME}, folds={len(folds)}, pending={len(pending)}, workers={config.outer_workers}",
        flush=True,
    )
    if pending:
        worker_count = min(config.outer_workers, len(pending))
        if worker_count == 1:
            for fold in pending:
                run_one_outer_fold(config, fold, str(result_dir.resolve()))
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                futures = {
                    executor.submit(
                        run_one_outer_fold, config, fold, str(result_dir.resolve())
                    ): fold
                    for fold in pending
                }
                for future in as_completed(futures):
                    fold = futures[future]
                    future.result()
                    print(
                        f"Completed baseline outer R{fold.repeat + 1:02d}-F{fold.fold + 1:02d}",
                        flush=True,
                    )
    selected_rows: list[dict[str, Any]] = []
    prediction_parts = []
    for fold in folds:
        directory = _model_fold_directory(result_dir, fold)
        if not _model_fold_complete(directory):
            raise RuntimeError(f"Incomplete baseline fold: {directory}")
        prediction_parts.append(pd.read_csv(directory / "fold_predictions.csv"))
        selected_payload = json.loads(
            (directory / "selected_hyperparameters.json").read_text(encoding="utf-8")
        )
        selected_rows.append(
            {
                **{key: value for key, value in selected_payload.items() if key != "best_params"},
                **selected_payload["best_params"],
            }
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summarize_model(predictions, dataset, config, result_dir)
    pd.DataFrame(selected_rows).to_csv(
        result_dir / "selected_hyperparameters_by_outer_fold.csv", index=False, encoding="utf-8-sig"
    )
    save_paired_comparisons(predictions, idasnet_result_dir, result_dir)
    print(f"Baseline nested cross-validation completed: {result_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pt", type=Path, default=DEFAULT_RAW_PT)
    parser.add_argument("--idasnet-result-dir", type=Path, default=DEFAULT_IDASNET_RESULT_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--outer-workers", type=int, default=DEFAULT_OUTER_WORKERS)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument(
        "--save-models", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_MODELS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_pt == DEFAULT_RAW_PT:
        args.raw_pt = args.idasnet_result_dir / "dataset_snapshot.pt"
    config = BaselineConfig(
        raw_pt=str(args.raw_pt.resolve()),
        idasnet_result_dir=str(args.idasnet_result_dir.resolve()),
        inner_splits=args.inner_splits,
        n_trials=args.n_trials,
        outer_workers=args.outer_workers,
        bootstrap_iterations=args.bootstrap_iterations,
        save_models=args.save_models,
        search_space=DEFAULT_SEARCH_SPACE,
    )
    if config.inner_splits < 2 or config.n_trials < 1 or config.outer_workers < 1:
        raise ValueError("inner_splits, n_trials, and outer_workers must be positive")
    if args.resume_dir is None:
        result_dir = next_result_directory(args.result_root)
    else:
        result_dir = args.resume_dir.resolve()
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {result_dir}")
        saved_config = load_run_config(result_dir / "run_config.json")
        if saved_config != asdict(config):
            raise ValueError("Resume configuration does not match run_config.json")
    run(config, result_dir)


if __name__ == "__main__":
    main()
