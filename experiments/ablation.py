"""Compare three attention configurations on acquisition-grouped outer folds.

Each fold uses its saved IDASNet hyperparameters. A shared grouped holdout
selects training epochs, then each configuration is refitted from scratch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from data.folds import load_raw_dataset
from data import NORMALIZATION
from idasnet.protocol import TRAINING, load_run_config
from experiments import utils as common
from experiments.models import EXPERIMENTS, check_ablation_structure

trainer = common.trainer
save_json = common.save_json

# Experiment settings
DEFAULT_RAW_PT = ROOT / "data/generated/raw.pt"
DEFAULT_FOLD_MANIFEST = ROOT / "results/IDASNet/1/outer_folds.json"
DEFAULT_REFERENCE_RESULT = ROOT / "results/IDASNet/1"
DEFAULT_RESULT_ROOT = ROOT / "results/Ablation"
DEFAULT_TRAIN_SEEDS = list(TRAINING["train_seeds"])
DEFAULT_OUTER_REPEATS = TRAINING["outer_repeats"]
# Bootstrap seed base; outer folds are read from the saved manifest.
DEFAULT_BOOTSTRAP_SEED_BASE = 2026
# Fraction of outer-training acquisitions held out to select training epochs.
DEFAULT_VALIDATION_FRACTION = TRAINING["validation_fraction"]
# Base seed for choosing validation acquisitions.
DEFAULT_VALIDATION_SEED = 2026
# Base seed for model initialization and training during epoch selection.
DEFAULT_EPOCH_SELECTION_SEED = 72026
DEFAULT_MAX_EPOCHS = TRAINING["max_epochs"]
DEFAULT_PATIENCE = TRAINING["patience"]
DEFAULT_BATCH_SIZE = TRAINING["batch_size"]
DEFAULT_PARALLEL_OUTER_FOLDS = 4
DEFAULT_CPU_THREADS = 1
DEFAULT_BOOTSTRAP_ITERATIONS = TRAINING["bootstrap_iterations"]
DEFAULT_EXPERIMENTS = [
    "no_attention",
    "method_only",
    "wavelength_only",
]
DEFAULTS = dict(
    raw_pt=DEFAULT_RAW_PT,
    fold_manifest=DEFAULT_FOLD_MANIFEST,
    reference_result=DEFAULT_REFERENCE_RESULT,
    result_root=DEFAULT_RESULT_ROOT,
    train_seeds=DEFAULT_TRAIN_SEEDS,
    outer_repeats=DEFAULT_OUTER_REPEATS,
    bootstrap_seed_base=DEFAULT_BOOTSTRAP_SEED_BASE,
    validation_fraction=DEFAULT_VALIDATION_FRACTION,
    validation_seed=DEFAULT_VALIDATION_SEED,
    epoch_selection_seed=DEFAULT_EPOCH_SELECTION_SEED,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    batch_size=DEFAULT_BATCH_SIZE,
    parallel_outer_folds=DEFAULT_PARALLEL_OUTER_FOLDS,
    cpu_threads=DEFAULT_CPU_THREADS,
    bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
    experiments=DEFAULT_EXPERIMENTS,
)


@dataclass(frozen=True)
class FixedConfig:
    raw_pt: str
    fold_manifest: str
    reference_result: str
    target: str
    experiments: list[str]
    train_seeds: list[int]
    outer_repeats: int
    bootstrap_seed_base: int
    validation_fraction: float
    validation_seed: int
    epoch_selection_seed: int
    max_epochs: int
    patience: int
    batch_size: int
    parallel_outer_folds: int
    cpu_threads: int
    device: str
    bootstrap_iterations: int


def task_directory(root, experiment, fold):
    return Path(root) / experiment / f"repeat_{fold['repeat']+1:02d}" / f"fold_{fold['fold']+1:02d}"


def make_validation_split(dataset, fold, fraction, split_seed, selection_seed):
    """Use the same acquisition-level holdout for all three ablations."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    outer_train = np.asarray(fold["train_indices"], dtype=int)
    seed = split_seed + int(fold["repeat"]) * 10000 + int(fold["fold"]) * 1000
    groups = dataset.acquisition_ids[outer_train]
    if len(np.unique(groups)) < 2:
        raise ValueError("Need at least two outer-training NH7 groups for validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_local, validation_local = next(splitter.split(np.zeros(len(outer_train)), groups=groups))
    train, validation = outer_train[train_local], outer_train[validation_local]
    test = np.asarray(fold["test_indices"], dtype=int)
    if not len(train) or not len(validation) or set(train) | set(validation) != set(outer_train):
        raise ValueError("Invalid training/validation coverage")
    for a, b in ((train, validation), (train, test), (validation, test)):
        if set(a) & set(b) or set(dataset.acquisition_ids[a]) & set(dataset.acquisition_ids[b]):
            raise ValueError("NH7 leakage across training/validation/outer test")
    return {
        "repeat": int(fold["repeat"]),
        "fold": int(fold["fold"]),
        "split_seed": seed,
        "epoch_selection_seed": selection_seed
        + int(fold["repeat"]) * 10000
        + int(fold["fold"]) * 1000,
        "validation_group_fraction_requested": fraction,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "outer_test_indices": test.tolist(),
        "train_sample_ids": dataset.global_ids[train].tolist(),
        "validation_sample_ids": dataset.global_ids[validation].tolist(),
        "train_groups": sorted(set(dataset.acquisition_ids[train])),
        "validation_groups": sorted(set(dataset.acquisition_ids[validation])),
        "outer_test_groups": sorted(set(dataset.acquisition_ids[test])),
    }


def validate_parameter_source(dataset, config, manifest):
    """Check that each parameter record belongs to its outer training fold."""
    source = Path(config.reference_result)
    source_config_path = source / "run_config.json"
    source_manifest_path = source / "outer_folds.json"
    reference_config = load_run_config(source_config_path)
    if list(reference_config["train_seeds"]) != list(config.train_seeds):
        raise ValueError("Ablation training seeds differ from the reference IDASNet run")
    if reference_config.get("target") != config.target:
        raise ValueError("Hyperparameter source predicts another target")
    if int(reference_config["batch_size"]) != config.batch_size:
        raise ValueError("Batch size differs from the reused IDASNet training recipe")
    source_dataset = load_raw_dataset(source / "dataset_snapshot.pt", config.target)
    if (
        not torch.equal(dataset.X, source_dataset.X)
        or not torch.equal(dataset.y, source_dataset.y)
        or (not np.array_equal(dataset.global_ids, source_dataset.global_ids))
        or (not np.array_equal(dataset.acquisition_ids, source_dataset.acquisition_ids))
        or (not np.array_equal(dataset.dates, source_dataset.dates))
        or (dataset.processing_methods != source_dataset.processing_methods)
        or (dataset.wavelengths != source_dataset.wavelengths)
    ):
        raise ValueError("Reference search dataset differs from current samples/features/targets")
    source_folds = common.validate_manifest(
        source_dataset, source_manifest_path, config.outer_repeats
    )
    by_pair = {(f["repeat"], f["fold"]): f for f in source_folds["folds"]}
    parameters = []
    required = {
        "learning_rate",
        "weight_decay",
        "dropout",
        "stem_channels",
        "feature_channels",
        "deep_channels",
        "hidden_dim",
        "raw_channel_reduction",
        "wavelength_reduction",
        "feature_channel_reduction",
        "optimizer",
        "use_scheduler",
    }
    for fold in manifest["folds"]:
        pair = (fold["repeat"], fold["fold"])
        if pair not in by_pair:
            raise ValueError(f"Missing source outer fold: {pair}")
        for key in (
            "seed",
            "train_indices",
            "test_indices",
            "train_sample_ids",
            "test_sample_ids",
            "train_groups",
            "test_groups",
        ):
            if fold[key] != by_pair[pair][key]:
                raise ValueError(f"Source outer fold mismatch at {pair}: {key}")
        relative = f"repeat_{pair[0] + 1:02d}/fold_{pair[1] + 1:02d}/inner_best.json"
        path = source / relative
        item = json.loads(path.read_text(encoding="utf-8"))
        params = item["best_params"]
        if not required.issubset(params):
            raise ValueError(f"Incomplete source hyperparameters: {path}")
        if any(
            (not np.isfinite(float(params[k])) for k in required - {"optimizer", "use_scheduler"})
        ):
            raise ValueError("Non-finite source hyperparameter")
        parameters.append(
            {
                "repeat": pair[0],
                "fold": pair[1],
                "params": params,
                "source_file": str(path),
                "source_best_trial": item.get("best_trial_number"),
            }
        )
    return parameters


def evaluation_metrics(model, loader, target, device):
    """Calculate all learning-curve metrics from one inference pass."""
    prediction = trainer.predict(model, loader, device)
    y, z = target.numpy().astype(float), prediction.astype(float)
    mse = float(np.mean((z - y) ** 2))
    if not np.isfinite(mse):
        raise FloatingPointError("Non-finite evaluation loss")
    return {"MSE": mse, **trainer.compute_metrics(y, z), "bias": float(np.mean(z - y))}


def select_epoch(dataset, config, experiment, assignment, params, directory, device):
    """Select an epoch using one grouped validation fit per ablation and fold."""
    ti, vi = (assignment["train_indices"], assignment["validation_indices"])
    tx, vx = common.normalized_inputs(dataset, ti, vi)
    ty, vy = (dataset.y[ti], dataset.y[vi])
    torch.save(
        {
            "train_X": tx,
            "validation_X": vx,
            "train_y": ty,
            "validation_y": vy,
            "train_indices": ti,
            "validation_indices": vi,
        },
        directory / "epoch_selection_data.pt",
    )
    seed = assignment["epoch_selection_seed"]
    trainer.set_seed(seed)
    model = common.new_model(params, dataset, config, experiment).to(device)
    common.seed_training_stream(seed)
    optimizer, scheduler = trainer._optimizer_and_scheduler(model, params, config.max_epochs)
    loader = trainer.make_loader(tx, ty, config.batch_size, True, seed)
    train_eval = trainer.make_loader(tx, ty, config.batch_size, False, seed)
    valid_eval = trainer.make_loader(vx, vy, config.batch_size, False, seed)
    best, best_epoch, stale, curves = (float("inf"), 1, 0, [])
    best_checkpoint = directory / "epoch_selection_model.pth"
    for epoch in range(1, config.max_epochs + 1):
        optimization_mse = common.train_epoch(model, loader, optimizer, device)
        if scheduler is not None:
            scheduler.step()
        train_metrics = evaluation_metrics(model, train_eval, ty, device)
        validation_metrics = evaluation_metrics(model, valid_eval, vy, device)
        train_mse, validation_mse = (train_metrics["MSE"], validation_metrics["MSE"])
        if validation_mse < best - 1e-10:
            best, best_epoch, stale = (validation_mse, epoch, 0)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "params": params,
                    "epoch": epoch,
                    "ablation_experiment": experiment,
                    "train_seed": seed,
                    "repeat": assignment["repeat"],
                    "fold": assignment["fold"],
                    "role": "epoch_selection_only; trained without validation or outer-test samples",
                },
                best_checkpoint,
            )
        else:
            stale += 1
        curves.append(
            {
                "epoch": epoch,
                "train_optimization_MSE": optimization_mse,
                "train_MSE": train_mse,
                "validation_MSE": validation_mse,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_validation_MSE": best,
                "best_epoch": best_epoch,
                "n_train": len(ti),
                "n_validation": len(vi),
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "MSE"},
                **{f"validation_{k}": v for k, v in validation_metrics.items() if k != "MSE"},
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  {experiment}: epoch {epoch}/{config.max_epochs}; train MSE={train_mse:.5f}; validation MSE={validation_mse:.5f}; best epoch={best_epoch}",
                flush=True,
            )
        if stale >= config.patience:
            break
    pd.DataFrame(curves).to_csv(directory / "validation_learning_curve.csv", index=False)
    model.load_state_dict(
        torch.load(best_checkpoint, map_location=device, weights_only=False)["state_dict"],
        strict=True,
    )
    predictions = []
    for split, data, targets, indices in (("train", tx, ty, ti), ("validation", vx, vy, vi)):
        loader_eval = trainer.make_loader(data, targets, config.batch_size, False, seed)
        predicted = trainer.predict(model, loader_eval, device)
        for local, index in enumerate(indices):
            predictions.append(
                {
                    "split": split,
                    "sample_index": index,
                    "sample_id": dataset.global_ids[index],
                    "acquisition_id": dataset.acquisition_ids[index],
                    "measured_TSS": float(targets[local]),
                    "predicted_TSS": float(predicted[local]),
                }
            )
    pd.DataFrame(predictions).to_csv(directory / "epoch_selection_predictions.csv", index=False)
    result = {
        "params": params,
        "best_epoch": best_epoch,
        "best_validation_MSE": best,
        "epochs_observed": len(curves),
        "reached_max_epochs": len(curves) == config.max_epochs,
        "best_epoch_at_limit": best_epoch == config.max_epochs,
        "selection_seed": seed,
        "hyperparameter_trials": 0,
        "selection_rule": "minimum grouped-holdout validation MSE; no outer-test feedback",
        "interpretation": "conditional fixed-recipe ablation, not independently tuned ablation optima",
    }
    save_json(directory / "epoch_selection.json", result)
    if best_epoch == config.max_epochs:
        print(
            f"WARNING: {experiment} selected the epoch limit ({best_epoch}); inspect the validation curve before interpreting the ablation.",
            flush=True,
        )
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def completed(directory, config):
    marker = directory / "fold_complete.json"
    if not marker.is_file():
        return False
    item = json.loads(marker.read_text())
    if item.get("status") != "complete":
        return False
    if (
        item.get("seeds") != config.train_seeds
        or item.get("protocol") != "fixed_recipe_group_holdout"
    ):
        raise ValueError("Completed task uses another protocol or seed list")
    required = [
        "validation_assignment.json",
        "reused_hyperparameters.json",
        "epoch_selection.json",
        "validation_learning_curve.csv",
        "epoch_selection_predictions.csv",
        "epoch_selection_model.pth",
        "epoch_selection_data.pt",
        "outer_data.pt",
        "predictions_by_seed.csv",
        "final_training_curves.csv",
        "method_attention.csv",
        "wavelength_attention.csv",
    ]
    for seed in config.train_seeds:
        required.extend([f"model_seed_{seed}.pth", f"inference_seed_{seed}.npz"])
    if any(not (directory / p).is_file() for p in required):
        raise ValueError(f"Missing files in a completed task: {directory}")
    return True


def outer_worker(config, fold, assignment, source, result_dir):
    torch.set_num_threads(config.cpu_threads)
    dataset = load_raw_dataset(Path(result_dir) / "dataset_snapshot.pt", config.target)
    device = torch.device(config.device)
    params = source["params"]
    prefix = f"R{fold['repeat'] + 1:02d}-F{fold['fold'] + 1:02d}"
    for experiment in config.experiments:
        directory = task_directory(result_dir, experiment, fold)
        if completed(directory, config):
            print(f"[{prefix}] {experiment}: already complete", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / "validation_assignment.json", assignment)
        save_json(directory / "reused_hyperparameters.json", source)
        print(
            f"[{prefix}] {experiment}: reuse fold parameters; select epochs on the grouped holdout",
            flush=True,
        )
        selected = select_epoch(dataset, config, experiment, assignment, params, directory, device)
        epochs = selected["best_epoch"]
        ti, vi = (fold["train_indices"], fold["test_indices"])
        tx, vx = common.normalized_inputs(dataset, ti, vi)
        ty, vy = (dataset.y[ti], dataset.y[vi])
        torch.save(
            {
                "train_indices": ti,
                "test_indices": vi,
                "train_X": tx,
                "test_X": vx,
                "train_y": ty,
                "test_y": vy,
            },
            directory / "outer_data.pt",
        )
        predictions, history, method_rows, wavelength_rows = ([], [], [], [])
        for seed in config.train_seeds:
            trainer.set_seed(seed)
            model = common.new_model(params, dataset, config, experiment).to(device)
            common.seed_training_stream(seed)
            check_ablation_structure(model, experiment)
            optimizer, scheduler = trainer._optimizer_and_scheduler(model, params, epochs)
            loader = trainer.make_loader(tx, ty, config.batch_size, True, seed)
            evaluation = trainer.make_loader(tx, ty, config.batch_size, False, seed)
            print(
                f"[{prefix}] {experiment}: final seed={seed}, validated epochs={epochs}", flush=True
            )
            for epoch in range(1, epochs + 1):
                loss = common.train_epoch(model, loader, optimizer, device)
                if scheduler is not None:
                    scheduler.step()
                metrics = evaluation_metrics(model, evaluation, ty, device)
                history.append(
                    {
                        "train_seed": seed,
                        "epoch": epoch,
                        "optimization_MSE": loss,
                        **{f"train_{k}": v for k, v in metrics.items()},
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "params": params,
                    "epochs": epochs,
                    "ablation_experiment": experiment,
                    "gate_modes": list(EXPERIMENTS[experiment]),
                    "input_metadata": {"input_shape": list(dataset.X.shape[1:]),
                        "processing_methods": dataset.processing_methods, "wavelengths": dataset.wavelengths,
                        "normalization": NORMALIZATION},
                    "train_seed": seed,
                    "repeat": fold["repeat"],
                    "fold": fold["fold"],
                    "hyperparameter_source": source,
                    "protocol": "fixed_recipe_group_holdout",
                },
                directory / f"model_seed_{seed}.pth",
            )
            arrays_out = {}
            for split, data, target, indices in (("train", tx, ty, ti), ("test", vx, vy, vi)):
                arrays = common.collect_inference(
                    model, data, target, indices, dataset, config, device
                )
                arrays_out.update({f"{split}_{key}": value for key, value in arrays.items()})
                for key, names, rows in (
                    ("raw_channel_weights", dataset.processing_methods, method_rows),
                    ("wavelength_weights", dataset.wavelengths, wavelength_rows),
                ):
                    weights = arrays[key].reshape(len(data), -1).astype(float)
                    for j, name in enumerate(names):
                        rows.append(
                            {
                                "repeat": fold["repeat"],
                                "fold": fold["fold"],
                                "train_seed": seed,
                                "split": split,
                                "index": j,
                                "name": name,
                                "mean": weights[:, j].mean(),
                                "sample_sd": weights[:, j].std(),
                                "min": weights[:, j].min(),
                                "max": weights[:, j].max(),
                            }
                        )
                if split == "test":
                    for local, index in enumerate(indices):
                        predictions.append(
                            {
                                "repeat": fold["repeat"],
                                "fold": fold["fold"],
                                "train_seed": seed,
                                "sample_index": index,
                                "sample_id": dataset.global_ids[index],
                                "acquisition_id": dataset.acquisition_ids[index],
                                "measured_TSS": float(target[local]),
                                "predicted_TSS": float(arrays["pred"].reshape(-1)[local]),
                            }
                        )
            np.savez_compressed(directory / f"inference_seed_{seed}.npz", **arrays_out)
            del model, optimizer, scheduler, arrays_out
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for name, rows in (
            ("predictions_by_seed", predictions),
            ("final_training_curves", history),
            ("method_attention", method_rows),
            ("wavelength_attention", wavelength_rows),
        ):
            pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False)
        save_json(
            directory / "fold_complete.json",
            {
                "status": "complete",
                "protocol": "fixed_recipe_group_holdout",
                "repeat": fold["repeat"],
                "fold": fold["fold"],
                "seeds": config.train_seeds,
                "experiment": experiment,
            },
        )
    return prefix


def summarize(config, manifest, dataset, result_dir):
    summaries, comparison = ({}, [])
    for experiment in config.experiments:
        directory = result_dir / experiment
        tasks = [task_directory(result_dir, experiment, fold) for fold in manifest["folds"]]
        if not all((completed(p, config) for p in tasks)):
            raise ValueError(f"Cannot summarize unfinished experiment: {experiment}")
        predictions = pd.concat(
            [pd.read_csv(p / "predictions_by_seed.csv") for p in tasks], ignore_index=True
        )
        expected = len(dataset.X) * len(config.train_seeds) * config.outer_repeats
        if (
            len(predictions) != expected
            or predictions.duplicated(["repeat", "train_seed", "sample_index"]).any()
        ):
            raise ValueError("Invalid/duplicate OOF records")
        for _, part in predictions.groupby(["repeat", "train_seed"]):
            if set(part.sample_index) != set(range(len(dataset.X))):
                raise ValueError("Incomplete once-per-repeat OOF coverage")
        predictions.to_csv(directory / "oof_predictions_by_seed.csv", index=False)
        ensemble = predictions.groupby(
            ["repeat", "fold", "sample_index", "sample_id", "acquisition_id"], as_index=False
        ).agg(
            measured_TSS=("measured_TSS", "first"), ensemble_predicted_TSS=("predicted_TSS", "mean")
        )
        ensemble.to_csv(directory / "oof_predictions_seed_ensemble.csv", index=False)
        metrics = pd.DataFrame(
            [
                {
                    "repeat": repeat,
                    **trainer.compute_metrics(g.measured_TSS, g.ensemble_predicted_TSS),
                }
                for repeat, g in ensemble.groupby("repeat")
            ]
        )
        seed_metrics = pd.DataFrame(
            [
                {
                    "repeat": repeat,
                    "train_seed": seed,
                    **trainer.compute_metrics(g.measured_TSS, g.predicted_TSS),
                }
                for (repeat, seed), g in predictions.groupby(["repeat", "train_seed"])
            ]
        )
        metrics.to_csv(directory / "metrics_by_repeat_ensemble.csv", index=False)
        seed_metrics.to_csv(directory / "metrics_by_repeat_and_seed.csv", index=False)
        seed_metrics.groupby("train_seed")[common.METRICS].agg(["mean", "std"]).to_csv(
            directory / "metrics_by_seed_mean_sd.csv"
        )
        summary = {
            "protocol": "fixed_recipe_group_holdout",
            "hyperparameter_search_trials": 0,
            "n_outer_repeats": config.outer_repeats,
            "seed_count": len(config.train_seeds),
            "seeds": config.train_seeds,
            "official_ensemble": {
                m: {"mean": metrics[m].mean(), "sd_across_repeats": metrics[m].std(ddof=1)}
                for m in common.METRICS
            },
            "individual_seed_performance": {
                m: {
                    "mean": seed_metrics[m].mean(),
                    "mean_within_repeat_seed_sd": seed_metrics.groupby("repeat")[m]
                    .std(ddof=1)
                    .mean(),
                }
                for m in common.METRICS
            },
            "cluster_bootstrap_95_ci": common.bootstrap_ci(
                ensemble, config.bootstrap_iterations, config.bootstrap_seed_base + 99991
            ),
            "interpretation": "conditional on IDASNet-selected hyperparameters, not per-ablation best achievable performance",
            "uncertainty_note": "Repeat SD requires at least two repetitions. Bootstrap intervals are conditional on fitted models.",
        }
        summaries[experiment] = summary
        save_json(directory / "summary_metrics.json", summary)
        selections, all_curves, source_rows = ([], [], [])
        for fold, task in zip(manifest["folds"], tasks):
            choice = json.loads((task / "epoch_selection.json").read_text())
            source = json.loads((task / "reused_hyperparameters.json").read_text())
            selections.append(
                {
                    "repeat": fold["repeat"],
                    "fold": fold["fold"],
                    **{k: v for k, v in choice.items() if k != "params"},
                }
            )
            frame = pd.read_csv(task / "validation_learning_curve.csv")
            frame.insert(0, "fold", fold["fold"])
            frame.insert(0, "repeat", fold["repeat"])
            all_curves.append(frame)
            source_rows.append(
                {
                    "repeat": fold["repeat"],
                    "fold": fold["fold"],
                    "selected_epochs": choice["best_epoch"],
                    "source_file": source["source_file"],
                    **source["params"],
                }
            )
        pd.DataFrame(selections).to_csv(
            directory / "epoch_selection_by_outer_fold.csv", index=False
        )
        pd.DataFrame(source_rows).to_csv(
            directory / "reused_hyperparameters_by_outer_fold.csv", index=False
        )
        pd.concat(all_curves, ignore_index=True).to_csv(
            directory / "validation_learning_curves_all_folds.csv", index=False
        )
        for kind in ("method", "wavelength"):
            frame = pd.concat(
                [pd.read_csv(p / f"{kind}_attention.csv") for p in tasks], ignore_index=True
            )
            frame.to_csv(directory / f"{kind}_attention_by_model.csv", index=False)
            if kind == "method":
                common.save_method_rank_data(frame, directory)
            means = frame.groupby(["split", "repeat", "fold", "index", "name"], as_index=False)[
                "mean"
            ].mean()
            means.to_csv(directory / f"{kind}_attention_by_fold.csv", index=False)
            means.groupby(["split", "index", "name"])["mean"].agg(
                ["mean", "std"]
            ).reset_index().to_csv(directory / f"{kind}_attention_summary.csv", index=False)
        save_json(
            directory / "attention_aggregation_metadata.json",
            {
                "within_model": "sample mean",
                "within_fold": "equal seed mean",
                "summary": "equal outer-fold mean",
                "train_test_separate": True,
                "fold_SD_is_not_independent_sample_CI": True,
                "attention_is_not_causal_importance": True,
            },
        )
        common.diagnostic_tables(ensemble, directory)
        comparison.append(
            {
                "experiment": experiment,
                "outer_repeats": config.outer_repeats,
                "seed_count": len(config.train_seeds),
                **{
                    f"{m}_{stat}": summary["official_ensemble"][m][stat]
                    for m in common.METRICS
                    for stat in ("mean", "sd_across_repeats")
                },
            }
        )
    table = pd.DataFrame(comparison)
    table.to_csv(result_dir / "architecture_comparison.csv", index=False)
    save_json(result_dir / "summary_all_experiments.json", summaries)
    print(table.to_string(index=False), flush=True)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw_pt", "fold_manifest", "reference_result", "result_root"):
        parser.add_argument("--" + flag.replace("_", "-"), type=Path, default=defaults[flag])
    parser.add_argument("--target", default="TSS")
    parser.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENTS), default=defaults["experiments"]
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=defaults["train_seeds"])
    for flag in (
        "outer_repeats",
        "bootstrap_seed_base",
        "validation_seed",
        "epoch_selection_seed",
        "max_epochs",
        "patience",
        "batch_size",
        "parallel_outer_folds",
        "cpu_threads",
        "bootstrap_iterations",
    ):
        parser.add_argument("--" + flag.replace("_", "-"), type=int, default=defaults[flag])
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults["validation_fraction"]
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--summarize", type=Path)
    return parser.parse_args()


def main():
    defaults = DEFAULTS
    args = parse_args(defaults)
    torch.set_num_threads(args.cpu_threads)
    if args.summarize:
        root = args.summarize.resolve()
        saved = load_run_config(root / "run_config.json")
        config = FixedConfig(**saved)
        dataset = load_raw_dataset(root / "dataset_snapshot.pt", config.target)
        manifest = json.loads((root / "outer_folds.json").read_text())
        summarize(config, manifest, dataset, root)
        print(f"Summarized saved predictions without training/search: {root}")
        return
    for key in (
        "outer_repeats",
        "max_epochs",
        "patience",
        "batch_size",
        "parallel_outer_folds",
        "cpu_threads",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.bootstrap_iterations < 0 or not 0 < args.validation_fraction < 1:
        raise ValueError("Invalid bootstrap count/validation fraction")
    if (
        not args.train_seeds
        or not args.experiments
        or len(args.train_seeds) != len(set(args.train_seeds))
        or (len(args.experiments) != len(set(args.experiments)))
    ):
        raise ValueError("Seed/experiment lists must be nonempty and have no duplicates")
    if any((seed < 0 or seed >= 2**32 for seed in args.train_seeds)):
        raise ValueError("Training seeds must be in [0, 2**32)")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    values = vars(args)
    config = FixedConfig(
        **{
            key: values[key]
            for key in FixedConfig.__dataclass_fields__
            if key not in ("device", "raw_pt", "fold_manifest", "reference_result")
        },
        device=device,
        raw_pt=str(args.raw_pt.resolve()),
        fold_manifest=str(args.fold_manifest.resolve()),
        reference_result=str(args.reference_result.resolve()),
    )
    dataset = load_raw_dataset(Path(config.raw_pt), config.target)
    if (
        len(dataset.processing_methods) != dataset.X.shape[1]
        or len(dataset.wavelengths) != dataset.X.shape[2]
    ):
        raise ValueError("Missing method/wavelength metadata")
    manifest = common.validate_manifest(dataset, config.fold_manifest, config.outer_repeats)
    source_parameters = validate_parameter_source(dataset, config, manifest)
    assignments = [make_validation_split(dataset, fold, config.validation_fraction,
        config.validation_seed, config.epoch_selection_seed) for fold in manifest["folds"]]
    if args.resume:
        root = args.resume.resolve()
        previous = load_run_config(root / "run_config.json")
        current = asdict(config)
        for key in ("parallel_outer_folds", "cpu_threads"):
            previous.pop(key, None)
            current.pop(key, None)
        saved_dataset = load_raw_dataset(root / "dataset_snapshot.pt", config.target)
        if (previous != current or not common.same_dataset(saved_dataset, dataset)
            or json.loads((root / "outer_folds.json").read_text()) != manifest
            or json.loads((root / "validation_folds.json").read_text()) != assignments
            or json.loads((root / "reused_hyperparameters.json").read_text()) != source_parameters):
            raise ValueError("Resume data, partitions or settings differ from the saved run")
    else:
        root = trainer.next_result_directory(args.result_root.resolve())
        trainer.save_dataset_snapshot(Path(config.raw_pt), root)
        save_json(root / "run_config.json", asdict(config))
        save_json(root / "outer_folds.json", manifest)
        save_json(root / "validation_folds.json", assignments)
        save_json(root / "reused_hyperparameters.json", source_parameters)
        save_json(root / "dataset_metadata.json", {
            "input_shape": list(dataset.X.shape), "sample_ids": dataset.global_ids,
            "acquisition_ids": dataset.acquisition_ids, "dates": dataset.dates,
            "processing_methods": dataset.processing_methods, "wavelengths_nm": dataset.wavelengths})
    print(
        f"Attention ablations: {len(config.experiments)} settings; "
        f"{len(manifest['folds'])} outer tasks; seeds={config.train_seeds}\n"
        "Fixed fold-specific parameters; grouped validation selects training epochs\n"
        f"Result: {root}",
        flush=True,
    )
    jobs = list(zip(manifest["folds"], assignments, source_parameters))
    if config.parallel_outer_folds == 1:
        for fold, assignment, source in jobs:
            outer_worker(config, fold, assignment, source, str(root))
    else:
        with ProcessPoolExecutor(
            max_workers=config.parallel_outer_folds, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(outer_worker, config, fold, assignment, source, str(root))
                for fold, assignment, source in jobs
            ]
            for future in as_completed(futures):
                print(f"Completed outer task {future.result()}", flush=True)
    summarize(config, manifest, dataset, root)
    save_json(
        root / "run_complete.json",
        {
            "status": "complete",
            "protocol": "fixed_recipe_group_holdout",
            "experiments": config.experiments,
            "outer_tasks": len(jobs),
            "hyperparameter_search_trials": 0,
        },
    )
    print(f"Fixed-recipe architecture CV completed: {root}", flush=True)


if __name__ == "__main__":
    main()
