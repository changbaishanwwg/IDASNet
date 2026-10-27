"""Study combinations of preprocessing-method counts and wavelength counts.

Inputs are selected from fixed global attention rankings, then models are
refitted on the saved acquisition-grouped folds. Full-spectrum normalization
precedes cropping. Rankings use folds from the same dataset, so this study
does not provide independent validation of feature selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from data.folds import load_raw_dataset
from data import NORMALIZATION
from idasnet.protocol import TRAINING, load_run_config
from experiments import ablation as fixed

common = fixed.common
trainer = fixed.trainer
save_json = common.save_json
PROTOCOL = "fixed_global_input_combinations"
WARNING = (
    "Rankings pool the outer folds of the same dataset. "
    "This analysis does not independently validate feature selection."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Experiment settings
DEFAULT_SOURCE_RESULT = ROOT / "results/Ablation/1"
DEFAULT_RANKING_RESULT = ROOT / "results/IDASNet/1"
DEFAULT_RESULT_ROOT = ROOT / "results/JointInputs"
DEFAULT_METHOD_TOPKS = [1, 2, 3]
DEFAULT_WAVELENGTH_TOPKS = [5, 10, 20, 30, 40, 50]
DEFAULT_TRAIN_SEEDS = list(TRAINING["train_seeds"])
DEFAULT_OUTER_REPEATS = TRAINING["outer_repeats"]
DEFAULT_MAX_EPOCHS = TRAINING["max_epochs"]
DEFAULT_PATIENCE = TRAINING["patience"]
DEFAULT_BATCH_SIZE = TRAINING["batch_size"]
DEFAULT_PARALLEL_OUTER_FOLDS = 4
DEFAULT_CPU_THREADS = 1
DEFAULT_BOOTSTRAP_ITERATIONS = TRAINING["bootstrap_iterations"]
DEFAULTS = dict(
    source_result=DEFAULT_SOURCE_RESULT,
    ranking_result=DEFAULT_RANKING_RESULT,
    result_root=DEFAULT_RESULT_ROOT,
    method_topks=DEFAULT_METHOD_TOPKS,
    wavelength_topks=DEFAULT_WAVELENGTH_TOPKS,
    train_seeds=DEFAULT_TRAIN_SEEDS,
    outer_repeats=DEFAULT_OUTER_REPEATS,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    batch_size=DEFAULT_BATCH_SIZE,
    parallel_outer_folds=DEFAULT_PARALLEL_OUTER_FOLDS,
    cpu_threads=DEFAULT_CPU_THREADS,
    bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
)


@dataclass(frozen=True)
class SelectionConfig:
    source_result: str
    ranking_result: str
    method_topks: list[int]
    wavelength_topks: list[int]
    train_seeds: list[int]
    outer_repeats: int
    max_epochs: int
    patience: int
    batch_size: int
    parallel_outer_folds: int
    cpu_threads: int
    device: str
    bootstrap_iterations: int
    target: str = "TSS"
    bootstrap_seed_base: int = 2026


def fold_relative(fold):
    return Path(f"repeat_{fold['repeat']+1:02d}") / f"fold_{fold['fold']+1:02d}"


def task_path(root, name, fold):
    return Path(root) / name / fold_relative(fold)


def build_candidates(config, dataset):
    """Build the Cartesian product of requested method and wavelength counts."""
    channels, bands = dataset.X.shape[1:]
    waves = np.asarray(dataset.wavelengths, dtype=float)
    if len(waves) != bands or not np.isfinite(waves).all() or not np.all(np.diff(waves) > 0):
        raise ValueError("Source wavelengths must be finite, unique, strictly increasing")
    if (
        len(dataset.processing_methods) != channels
        or len(set(dataset.processing_methods)) != channels
    ):
        raise ValueError("Missing/duplicate method metadata")
    for counts, minimum, maximum in (
        (config.method_topks, 1, channels),
        (config.wavelength_topks, 4, bands),
    ):
        if not counts or len(counts) != len(set(counts)):
            raise ValueError("Empty or duplicate top-k counts")
        if any(not isinstance(k, int) or not minimum <= k <= maximum for k in counts):
            raise ValueError(f"Invalid top-k count; expected {minimum}..{maximum}")
    result = []
    for m in config.method_topks:
        for w in config.wavelength_topks:
            if (m, w) == (channels, bands):
                raise ValueError(
                    "Complete-input combinations are not run; request reduced joint subsets only"
                )
            result.append(
                {"name": f"joint_top_{m}_{w}", "family": "joint", "method_k": m, "wavelength_k": w}
            )
    return result


def stable_rank(scores):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Non-finite/empty attention scores")
    # Break ties by input index.
    return np.lexsort((np.arange(len(scores)), -scores)).tolist()


def selection_plan(dataset, candidate, scores):
    method_order = stable_rank(scores["method"])
    wavelength_order = stable_rank(scores["wavelength"])
    if len(method_order) != dataset.X.shape[1] or len(wavelength_order) != dataset.X.shape[2]:
        raise ValueError("Attention dimensions differ from full input")
    methods = sorted(method_order[: candidate["method_k"]])
    bands = sorted(
        wavelength_order[: candidate["wavelength_k"]], key=lambda i: dataset.wavelengths[i]
    )
    return {
        **candidate,
        "experiment_type": "exploratory",
        "protocol": PROTOCOL,
        "ranking_scope": "pooled outer folds and training seeds",
        "independent_feature_selection_evaluation": False,
        "method_indices": methods,
        "wavelength_indices": bands,
        "selected_methods": [dataset.processing_methods[i] for i in methods],
        "selected_wavelengths_nm": [dataset.wavelengths[i] for i in bands],
        "method_importance_order": method_order,
        "wavelength_importance_order": wavelength_order,
        "original_input_shape": list(dataset.X.shape[1:]),
        "input_shape": [len(methods), len(bands)],
        "method_input_order": "dataset method order",
        "wavelength_input_order": "strictly increasing nm",
        "normalization_order": "normalize full spectra before cropping",
        "hardware_band_reduction_claim": False,
    }


def crop_inputs(x, plan):
    if x.ndim != 3 or list(x.shape[1:]) != plan["original_input_shape"]:
        raise ValueError("Crop expects the full normalized input shape")
    methods = torch.as_tensor(plan["method_indices"], dtype=torch.long, device=x.device)
    bands = torch.as_tensor(plan["wavelength_indices"], dtype=torch.long, device=x.device)
    return x.index_select(1, methods).index_select(2, bands).contiguous()


def new_model(params, plan):
    return trainer.build_model(params, *plan["input_shape"])


def training_budget(config):
    return {"max_epochs": int(config.max_epochs), "patience": int(config.patience)}


def execution_counts(config, candidates, jobs):
    if not candidates or any((item["family"] != "joint" for item in candidates)):
        raise ValueError("Expected method-count and wavelength-count combinations")
    validation = len(jobs) * len(candidates)
    final = validation * len(config.train_seeds)
    return {
        "joint_candidates": len(candidates),
        "new_validation_fits": validation,
        "new_final_fits": final,
        "new_training_fits": validation + final,
    }


def source_check(dataset, config):
    root = Path(config.source_result)
    source = load_run_config(root / "run_config.json")
    if source["train_seeds"] != config.train_seeds:
        raise ValueError("Training seeds differ from the reference ablation run")
    if read_json(root / "run_complete.json").get("status") != "complete":
        raise ValueError("Reference ablation run is incomplete")
    if source["target"] != config.target or source["batch_size"] != config.batch_size:
        raise ValueError("Reference target or batch size differs")
    manifest = common.validate_manifest(dataset, root / "outer_folds.json", config.outer_repeats)
    assignments = {(item["repeat"], item["fold"]): item for item in read_json(root / "validation_folds.json")}
    parameters = {(item["repeat"], item["fold"]): item for item in read_json(root / "reused_hyperparameters.json")}
    jobs = []
    for fold in manifest["folds"]:
        key = fold["repeat"], fold["fold"]
        assignment = assignments[key]
        expected = fixed.make_validation_split(dataset, fold, source["validation_fraction"],
            source["validation_seed"], source["epoch_selection_seed"])
        if assignment != expected:
            raise ValueError("Reference validation partition differs")
        relative = Path("no_attention") / fold_relative(fold)
        marker = read_json(root / relative / "fold_complete.json")
        if marker.get("status") != "complete" or marker["seeds"] != source["train_seeds"]:
            raise ValueError("Reference fold is incomplete")
        if read_json(root / relative / "validation_assignment.json") != assignment:
            raise ValueError("Fold validation assignment differs from the saved partition")
        selected = read_json(root / relative / "reused_hyperparameters.json")
        if selected != parameters[key]:
            raise ValueError("Fold hyperparameters differ from the reference run")
        for name in ("epoch_selection_data.pt", "outer_data.pt"):
            if not (root / relative / name).is_file():
                raise FileNotFoundError(root / relative / name)
        jobs.append({"fold": fold, "assignment": assignment, "source_params": selected})
    return manifest, jobs


def load_global_rankings(dataset, config):
    """Load training-sample attention pooled across the saved folds and seeds."""
    root = Path(config.ranking_result)
    source = load_run_config(root / "run_config.json")
    seeds = source["train_seeds"]
    if seeds != config.train_seeds or source["target"] != config.target:
        raise ValueError("Ranking seeds or target differ from the current run")
    if not common.same_dataset(load_raw_dataset(root / "dataset_snapshot.pt", config.target), dataset):
        raise ValueError("Ranking and refit datasets differ")
    reference = common.validate_manifest(dataset, Path(config.source_result) / "outer_folds.json", source["outer_repeats"])
    manifest = common.validate_manifest(dataset, root / "outer_folds.json", source["outer_repeats"])
    if manifest != reference:
        raise ValueError("Ranking and ablation partitions differ")
    folds = {(fold["repeat"], fold["fold"]) for fold in manifest["folds"]}
    scores = {}
    for kind, labels, index_col, name_col, summary_col in (
        ("method", dataset.processing_methods, "method_index", "method", "Method"),
        ("wavelength", dataset.wavelengths, "wavelength_index", "wavelength_nm", "Wavelength_nm")):
        records = pd.read_csv(root / f"training_{kind}_attention_by_fold_and_seed.csv")
        keys = ["repeat", "fold", "train_seed", index_col]
        if (len(records) != len(folds) * len(seeds) * len(labels)
            or records.duplicated(keys).any()
            or set(zip(records["repeat"], records["fold"])) != folds
            or set(records.train_seed) != set(seeds)
            or not records.source.eq("outer_training_samples_only").all()
            or not np.isfinite(records.importance).all()
            or not records.importance.between(0, 1).all()):
            raise ValueError(f"Incomplete training-attention records: {kind}")
        for _, part in records.groupby(["repeat", "fold", "train_seed"]):
            if set(part[index_col]) != set(range(len(labels))):
                raise ValueError("Attention feature indices differ from the dataset")
            for index, label in zip(part[index_col], part[name_col]):
                expected = labels[int(index)]
                if (str(label) != str(expected) if kind == "method" else not np.isclose(float(label), expected)):
                    raise ValueError("Attention labels differ from the dataset")
        fold_means = records.groupby(["repeat", "fold", index_col]).importance.mean()
        reconstructed = fold_means.groupby(level=index_col).mean().reindex(range(len(labels))).to_numpy()
        published = pd.read_csv(root / f"{kind}_importance_mean.csv")
        if len(published) != len(labels) or published[index_col].duplicated().any():
            raise ValueError("Invalid attention summary")
        published = published.set_index(index_col).reindex(range(len(labels)))
        if not np.all(published.N_Repeats == source["outer_repeats"]):
            raise ValueError("Attention summary does not cover every repetition")
        if (list(published[summary_col]) != list(labels) if kind == "method"
            else not np.allclose(published[summary_col], labels)):
            raise ValueError("Attention summary labels differ")
        values = published.Mean_Importance.to_numpy(dtype=float)
        if not np.allclose(values, reconstructed, rtol=1e-6, atol=1e-7):
            raise ValueError("Attention summary does not match the saved model weights")
        scores[kind] = values.tolist()
    return {"protocol": PROTOCOL, "ranking_result": str(root), "scores": scores,
        "all_source_train_seeds": seeds, "outer_fold_count": len(folds),
        "method_order": stable_rank(scores["method"]), "wavelength_order": stable_rank(scores["wavelength"]),
        "scope": "Fixed attention rankings pooled across outer folds; feature selection is not independently validated."}


def frozen_scores(root):
    ranking = read_json(Path(root) / "fixed_global_rankings.json")
    if ranking.get("protocol") != PROTOCOL:
        raise ValueError("Ranking configuration differs from the current experiment")
    scores = {kind: np.asarray(values, dtype=float) for kind, values in ranking["scores"].items()}
    if set(scores) != {"method", "wavelength"} or any(values.ndim != 1 or not len(values) or not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Invalid fixed attention scores")
    return ranking, scores


def save_rankings(path, dataset, scores, stage):
    rows = []
    for kind, names in (
        ("method", dataset.processing_methods),
        ("wavelength", dataset.wavelengths),
    ):
        order = stable_rank(scores[kind])
        rows.extend(
            {
                "stage": stage,
                "kind": kind,
                "rank": rank,
                "original_index": index,
                "name": names[index],
                "mean_weight": scores[kind][index],
            }
            for rank, index in enumerate(order, 1)
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def prepare_fold_data(dataset, config, job, root, device):
    fold, assignment = (job["fold"], job["assignment"])
    folder = Path(root) / "fold_data" / fold_relative(fold)
    folder.mkdir(parents=True, exist_ok=True)
    ti, vi = (assignment["train_indices"], assignment["validation_indices"])
    tx, vx = common.normalized_inputs(dataset, ti, vi)
    ot, oe = (fold["train_indices"], fold["test_indices"])
    ox, ex = common.normalized_inputs(dataset, ot, oe)
    source = Path(config.source_result) / "no_attention" / fold_relative(fold)
    old = torch.load(source / "epoch_selection_data.pt", map_location="cpu", weights_only=False)
    for key, value in (
        ("train_X", tx),
        ("validation_X", vx),
        ("train_y", dataset.y[ti]),
        ("validation_y", dataset.y[vi]),
    ):
        if not torch.equal(old[key], value):
            raise ValueError(f"Ranking teacher input/target mismatch: {key}")
    if old["train_indices"] != ti or old["validation_indices"] != vi:
        raise ValueError("Ranking teacher contains another training/validation split")
    old_outer = torch.load(source / "outer_data.pt", map_location="cpu", weights_only=False)
    if (
        old_outer["train_indices"] != ot
        or old_outer["test_indices"] != oe
        or (not torch.equal(old_outer["train_X"], ox))
        or (not torch.equal(old_outer["test_X"], ex))
        or (not torch.equal(old_outer["train_y"], dataset.y[ot]))
        or (not torch.equal(old_outer["test_y"], dataset.y[oe]))
    ):
        raise ValueError("Outer normalization/membership differs from the reference")
    data = {
        "selection_train_X": tx,
        "selection_validation_X": vx,
        "selection_train_y": dataset.y[ti],
        "selection_validation_y": dataset.y[vi],
        "outer_train_X": ox,
        "outer_test_X": ex,
        "outer_train_y": dataset.y[ot],
        "outer_test_y": dataset.y[oe],
        "assignment": assignment,
        "outer_fold": fold,
    }
    torch.save(data, folder / "normalized_full_inputs.pt")
    _, scores = frozen_scores(root)
    return data, scores, scores


def inference_metadata(dataset, plan):
    return replace(
        dataset,
        processing_methods=plan["selected_methods"],
        wavelengths=plan["selected_wavelengths_nm"],
    )


def fit_validation(dataset, config, job, plan, data, folder, device):
    """Choose epochs by grouped validation, conditional on globally informed inputs."""
    tx = crop_inputs(data["selection_train_X"], plan)
    vx = crop_inputs(data["selection_validation_X"], plan)
    ty, vy = (data["selection_train_y"], data["selection_validation_y"])
    seed = job["assignment"]["epoch_selection_seed"]
    params = job["source_params"]["params"]
    trainer.set_seed(seed)
    model = new_model(params, plan).to(device)
    common.seed_training_stream(seed)
    optimizer, scheduler = trainer._optimizer_and_scheduler(model, params, config.max_epochs)
    train_loader = trainer.make_loader(tx, ty, config.batch_size, True, seed)
    train_eval = trainer.make_loader(tx, ty, config.batch_size, False, seed)
    valid_eval = trainer.make_loader(vx, vy, config.batch_size, False, seed)
    best, epoch_best, stale, rows = (float("inf"), 1, 0, [])
    for epoch in range(1, config.max_epochs + 1):
        loss = common.train_epoch(model, train_loader, optimizer, device)
        if scheduler is not None:
            scheduler.step()
        tm = fixed.evaluation_metrics(model, train_eval, ty, device)
        vm = fixed.evaluation_metrics(model, valid_eval, vy, device)
        if vm["MSE"] < best - 1e-10:
            best, epoch_best, stale = (vm["MSE"], epoch, 0)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "params": params,
                    "epoch": epoch,
                    "selection_plan": plan,
                "input_metadata": {"input_shape": list(dataset.X.shape[1:]),
                    "processing_methods": dataset.processing_methods, "wavelengths": dataset.wavelengths,
                    "normalization": NORMALIZATION},
                    "role": "epoch_selection_with_globally_selected_inputs",
                    "train_seed": seed,
                    "protocol": PROTOCOL,
                    "experiment_type": "exploratory",
                    "warning": WARNING,
                    "training_budget": training_budget(config),
                    "batch_size": config.batch_size,
                    "independent_feature_selection_evaluation": False,
                    "fold_data_ref": str(
                        Path("fold_data") / fold_relative(job["fold"]) / "normalized_full_inputs.pt"
                    ),
                    "fold_data_stage": "selection_training_and_validation",
                    "repeat": job["fold"]["repeat"],
                    "fold": job["fold"]["fold"],
                },
                folder / "epoch_selection_model.pth",
            )
        else:
            stale += 1
        rows.append(
            {
                "epoch": epoch,
                "train_optimization_MSE": loss,
                **{f"train_{k}": v for k, v in tm.items()},
                **{f"validation_{k}": v for k, v in vm.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_validation_MSE": best,
                "best_epoch": epoch_best,
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  {plan['name']}: validation epoch={epoch}; MSE={vm['MSE']:.5f}; best={epoch_best}",
                flush=True,
            )
        if stale >= config.patience:
            break
    pd.DataFrame(rows).to_csv(folder / "validation_learning_curve.csv", index=False)
    checkpoint = torch.load(
        folder / "epoch_selection_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    predictions = []
    for split, x, y, indices in (
        ("train", tx, ty, job["assignment"]["train_indices"]),
        ("validation", vx, vy, job["assignment"]["validation_indices"]),
    ):
        predicted = trainer.predict(
            model, trainer.make_loader(x, y, config.batch_size, False, seed), device
        )
        predictions.extend(
            (
                {
                    "split": split,
                    "sample_index": i,
                    "sample_id": dataset.global_ids[i],
                    "acquisition_id": dataset.acquisition_ids[i],
                    "measured_TSS": float(y[j]),
                    "predicted_TSS": float(predicted[j]),
                }
                for j, i in enumerate(indices)
            )
        )
    pd.DataFrame(predictions).to_csv(folder / "epoch_selection_predictions.csv", index=False)
    selection = {
        "best_epoch": epoch_best,
        "best_validation_MSE": best,
        "epochs_observed": len(rows),
        "selection_seed": seed,
        "best_epoch_at_limit": epoch_best == config.max_epochs,
        "selection_rule": "minimum grouped-validation MSE, conditional on FIXED GLOBAL inputs",
        "protocol": PROTOCOL,
        "experiment_type": "exploratory",
        "warning": WARNING,
        "training_budget": training_budget(config),
        "batch_size": config.batch_size,
        "independent_feature_selection_evaluation": False,
        "hyperparameter_search_trials": 0,
    }
    save_json(folder / "epoch_selection.json", selection)
    if selection["best_epoch_at_limit"]:
        print(f"WARNING: {plan['name']} best epoch reached the configured limit", flush=True)
    del model, optimizer, scheduler
    return selection


def fit_final(dataset, config, job, plan, selection, data, folder, device):
    tx = crop_inputs(data["outer_train_X"], plan)
    vx = crop_inputs(data["outer_test_X"], plan)
    ty, vy = (data["outer_train_y"], data["outer_test_y"])
    params, epochs = (job["source_params"]["params"], int(selection["best_epoch"]))
    metadata = inference_metadata(dataset, plan)
    predictions, history, method_rows, wave_rows = ([], [], [], [])
    for seed in config.train_seeds:
        trainer.set_seed(seed)
        model = new_model(params, plan).to(device)
        common.seed_training_stream(seed)
        optimizer, scheduler = trainer._optimizer_and_scheduler(model, params, epochs)
        train_loader = trainer.make_loader(tx, ty, config.batch_size, True, seed)
        evaluate = trainer.make_loader(tx, ty, config.batch_size, False, seed)
        print(
            f"  {plan['name']}: final seed={seed}, epochs={epochs}, input={plan['input_shape']}",
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            loss = common.train_epoch(model, train_loader, optimizer, device)
            if scheduler is not None:
                scheduler.step()
            metrics = fixed.evaluation_metrics(model, evaluate, ty, device)
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
                "selection_plan": plan,
                "input_metadata": {"input_shape": list(dataset.X.shape[1:]),
                    "processing_methods": dataset.processing_methods, "wavelengths": dataset.wavelengths,
                    "normalization": NORMALIZATION},
                "train_seed": seed,
                "repeat": job["fold"]["repeat"],
                "fold": job["fold"]["fold"],
                "fold_data_stage": "outer_training_and_test",
                "protocol": PROTOCOL,
                "experiment_type": "exploratory",
                "warning": WARNING,
                "training_budget": training_budget(config),
                "batch_size": config.batch_size,
                "independent_feature_selection_evaluation": False,
                "fold_data_ref": str(
                    Path("fold_data") / fold_relative(job["fold"]) / "normalized_full_inputs.pt"
                ),
            },
            folder / f"model_seed_{seed}.pth",
        )
        arrays_out = {}
        for split, x, y, indices in (
            ("train", tx, ty, job["fold"]["train_indices"]),
            ("test", vx, vy, job["fold"]["test_indices"]),
        ):
            arrays = common.collect_inference(model, x, y, indices, metadata, config, device)
            arrays_out.update({f"{split}_{k}": v for k, v in arrays.items()})
            for kind, key, names, original, destination in (
                (
                    "method",
                    "raw_channel_weights",
                    plan["selected_methods"],
                    plan["method_indices"],
                    method_rows,
                ),
                (
                    "wavelength",
                    "wavelength_weights",
                    plan["selected_wavelengths_nm"],
                    plan["wavelength_indices"],
                    wave_rows,
                ),
            ):
                weights = arrays[key].reshape(len(x), -1).astype(float)
                for position, (name, index) in enumerate(zip(names, original)):
                    destination.append(
                        {
                            "repeat": job["fold"]["repeat"],
                            "fold": job["fold"]["fold"],
                            "train_seed": seed,
                            "split": split,
                            "index": index,
                            "input_index": position,
                            "name": name,
                            "mean": weights[:, position].mean(),
                            "sample_sd": weights[:, position].std(),
                            "min": weights[:, position].min(),
                            "max": weights[:, position].max(),
                        }
                    )
            if split == "test":
                predictions.extend(
                    (
                        {
                            "repeat": job["fold"]["repeat"],
                            "fold": job["fold"]["fold"],
                            "train_seed": seed,
                            "sample_index": index,
                            "sample_id": dataset.global_ids[index],
                            "acquisition_id": dataset.acquisition_ids[index],
                            "measured_TSS": float(y[j]),
                            "predicted_TSS": float(arrays["pred"].reshape(-1)[j]),
                        }
                        for j, index in enumerate(indices)
                    )
                )
        np.savez_compressed(folder / f"inference_seed_{seed}.npz", **arrays_out)
        del model, optimizer, scheduler, arrays_out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for name, rows in (
        ("predictions_by_seed", predictions),
        ("final_training_curves", history),
        ("method_attention", method_rows),
        ("wavelength_attention", wave_rows),
    ):
        pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)


def completed(folder, config):
    marker_path = folder / "fold_complete.json"
    if not marker_path.is_file():
        return False
    marker = read_json(marker_path)
    if marker["protocol"] != PROTOCOL or marker["seeds"] != config.train_seeds:
        raise ValueError("Completed task protocol/seeds do not match this run")
    if marker["training_budget"] != training_budget(config):
        raise ValueError("Completed task training budget does not match this run")
    if marker["role"] != "fresh_refit":
        raise ValueError("Expected a freshly fitted method/wavelength combination")
    files = [
        "predictions_by_seed.csv",
        "epoch_selection.json",
        "validation_learning_curve.csv",
        "epoch_selection_predictions.csv",
        "final_training_curves.csv",
        "method_attention.csv",
        "wavelength_attention.csv",
        "selection_plans.json",
        "reused_hyperparameters.json",
        "epoch_selection_model.pth",
    ]
    files.extend(
        (
            f"{prefix}_seed_{seed}.{suffix}"
            for seed in config.train_seeds
            for prefix, suffix in (("model", "pth"), ("inference", "npz"))
        )
    )
    if not all(((folder / name).is_file() for name in files)):
        raise ValueError(f"Files are missing from completed task {folder}")
    return marker["status"] == "complete"


def outer_worker(config, job, candidates, result_dir):
    torch.set_num_threads(config.cpu_threads)
    fold = job["fold"]
    execution_counts(config, candidates, [job])
    if all((completed(task_path(result_dir, c["name"], fold), config) for c in candidates)):
        print(f"{fold_relative(fold)}: all candidates already complete", flush=True)
        return str(fold_relative(fold))
    dataset = load_raw_dataset(Path(result_dir) / "dataset_snapshot.pt", config.target)
    device = torch.device(config.device)
    data, vscores, fscores = prepare_fold_data(dataset, config, job, result_dir, device)
    for candidate in candidates:
        folder = task_path(result_dir, candidate["name"], fold)
        if completed(folder, config):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        vp = selection_plan(dataset, candidate, vscores)
        fp = selection_plan(dataset, candidate, fscores)
        save_json(folder / "selection_plans.json", {"validation_stage": vp, "final_stage": fp})
        save_json(folder / "reused_hyperparameters.json", job["source_params"])
        print(
            f"[{fold_relative(fold)}] {candidate['name']}: compact inputs; no new Optuna search",
            flush=True,
        )
        selected = fit_validation(dataset, config, job, vp, data, folder, device)
        fit_final(dataset, config, job, fp, selected, data, folder, device)
        save_json(
            folder / "fold_complete.json",
            {
                "status": "complete",
                "protocol": PROTOCOL,
                "role": "fresh_refit",
                "repeat": fold["repeat"],
                "fold": fold["fold"],
                "seeds": config.train_seeds,
                "training_budget": training_budget(config),
                "experiment_type": "exploratory",
                "independent_feature_selection_evaluation": False,
            },
        )
    return str(fold_relative(fold))


def bootstrap_draws(ensemble, iterations, seed):
    from idasnet.statistics import cluster_bootstrap_draws
    draws = cluster_bootstrap_draws(ensemble, iterations, seed)
    return np.column_stack([draws[m] for m in common.METRICS])


def summarize_predictions(config, dataset, frame, folder):
    n = len(dataset.X)
    if (
        len(frame) != n * config.outer_repeats * len(config.train_seeds)
        or frame.duplicated(["repeat", "train_seed", "sample_index"]).any()
        or set(frame.train_seed) != set(config.train_seeds)
        or (not np.isfinite(frame[["measured_TSS", "predicted_TSS"]]).all().all())
    ):
        raise ValueError(f"Invalid OOF prediction records: {folder}")
    for _, part in frame.groupby(["repeat", "train_seed"]):
        if sorted(part.sample_index) != list(range(n)):
            raise ValueError("Every sample must have ONE OOF prediction per repeat/seed")
        indices = part.sample_index.to_numpy(dtype=int, copy=True)
        if not np.array_equal(part.sample_id.astype(str), dataset.global_ids[indices]):
            raise ValueError("OOF sample identity mismatch")
        if not np.allclose(part.measured_TSS, dataset.y[indices].numpy(), rtol=0, atol=1e-06):
            raise ValueError("OOF reference values differ from the dataset")
    ensemble = frame.groupby(
        ["repeat", "fold", "sample_index", "sample_id", "acquisition_id"], as_index=False
    ).agg(measured_TSS=("measured_TSS", "first"), ensemble_predicted_TSS=("predicted_TSS", "mean"))
    ensemble.to_csv(folder / "oof_predictions_seed_ensemble.csv", index=False)
    scores = pd.DataFrame(
        [
            {"repeat": repeat, **trainer.compute_metrics(b.measured_TSS, b.ensemble_predicted_TSS)}
            for repeat, b in ensemble.groupby("repeat")
        ]
    )
    seed_scores = pd.DataFrame(
        [
            {
                "repeat": repeat,
                "train_seed": seed,
                **trainer.compute_metrics(b.measured_TSS, b.predicted_TSS),
            }
            for (repeat, seed), b in frame.groupby(["repeat", "train_seed"])
        ]
    )
    scores.to_csv(folder / "metrics_by_repeat_ensemble.csv", index=False)
    seed_scores.to_csv(folder / "metrics_by_repeat_and_seed.csv", index=False)
    seed_scores.groupby("train_seed")[common.METRICS].agg(["mean", "std"]).to_csv(
        folder / "metrics_by_seed_mean_sd.csv"
    )
    draws = bootstrap_draws(
        ensemble, config.bootstrap_iterations, config.bootstrap_seed_base + 99991
    )
    summary = {
        "protocol": PROTOCOL,
        "experiment_type": "exploratory",
        "warning": WARNING,
        "training_budget": training_budget(config),
        "independent_feature_selection_evaluation": False,
        "outer_repeats": config.outer_repeats,
        "train_seeds": config.train_seeds,
        "exploratory_seed_ensemble": {
            metric: {"mean": scores[metric].mean(), "sd_across_repeats": scores[metric].std(ddof=1)}
            for metric in common.METRICS
        },
        "individual_seed_performance": {
            metric: {
                "mean": seed_scores[metric].mean(),
                "mean_within_repeat_seed_sd": seed_scores.groupby("repeat")[metric]
                .std(ddof=1)
                .mean(),
            }
            for metric in common.METRICS
        },
        "conditional_NH7_bootstrap_95_ci": (
            {
                metric: np.nanquantile(draws[:, j], [0.025, 0.975]).tolist()
                for j, metric in enumerate(common.METRICS)
            }
            if len(draws)
            else {}
        ),
        "uncertainty_note": "Whole-cluster draws shared across repeats; conditional on fitted models.",
        "interpretation": "Fixed global rankings and full-input hyperparameters; feature selection is not independently validated.",
    }
    save_json(folder / "summary_metrics.json", summary)
    common.diagnostic_tables(ensemble, folder)
    return (summary, ensemble, draws)


def save_combination_details(config, table, root):
    """Export selected inputs and their prediction metrics."""
    joint = table[table.family == "joint"].sort_values(["method_k", "wavelength_k"]).copy()
    joint["all_source_ranking_seed_count"] = len(
        read_json(root / "fixed_global_rankings.json")["all_source_train_seeds"]
    )
    plans = read_json(root / "fixed_joint_selection_plans.json")
    by_name = {plan["name"]: plan for plan in plans}
    joint["selected_methods"] = joint.experiment.map(
        lambda name: json.dumps(by_name[name]["selected_methods"])
    )
    joint["selected_wavelengths_nm"] = joint.experiment.map(
        lambda name: json.dumps(by_name[name]["selected_wavelengths_nm"])
    )
    joint.to_csv(root / "input_combinations_detail.csv", index=False)
    save_json(
        root / "input_combinations_metadata.json",
        {
            "protocol": PROTOCOL,
            "experiment_type": "exploratory",
            "warning": WARNING,
            "independent_feature_selection_evaluation": False,
            "table_title": "Exploratory prediction performance using fixed global attention-selected method/wavelength combinations",
            "cells": "Seed predictions averaged per sample, complete OOF metric per repeat, then repeat mean and descriptive SD",
            "refit_train_seeds": config.train_seeds,
            "outer_repeats": config.outer_repeats,
            "training_budget": training_budget(config),
            "joint_candidates": len(joint),
            "best_test_cell_is_not_an_unbiased_selected_model": True,
            "uncertainty_note": "Neither repeat SD nor conditional bootstrap intervals account for global selection bias.",
        },
    )


def summarize(config, manifest, dataset, candidates, root):
    summaries, ensembles, draws, comparison, selections = ({}, {}, {}, [], [])
    for candidate in candidates:
        name, folder = (candidate["name"], root / candidate["name"])
        tasks = [task_path(root, name, fold) for fold in manifest["folds"]]
        if not all((completed(task, config) for task in tasks)):
            raise ValueError(f"Cannot summarize incomplete candidate {name}")
        frame = pd.concat(
            [pd.read_csv(task / "predictions_by_seed.csv") for task in tasks], ignore_index=True
        )
        frame.to_csv(folder / "oof_predictions_by_seed.csv", index=False)
        curves, epochs = ([], [])
        for fold, task in zip(manifest["folds"], tasks):
            plans = read_json(task / "selection_plans.json")
            selection = read_json(task / "epoch_selection.json")
            epochs.append(
                {
                    "repeat": fold["repeat"],
                    "fold": fold["fold"],
                    "best_epoch": selection["best_epoch"],
                    "best_validation_MSE": selection["best_validation_MSE"],
                    "epochs_observed": selection.get("epochs_observed"),
                }
            )
            curve = pd.read_csv(task / "validation_learning_curve.csv")
            curve.insert(0, "fold", fold["fold"])
            curve.insert(0, "repeat", fold["repeat"])
            curves.append(curve)
            for stage, plan in plans.items():
                for kind, key, labels in (
                    ("method", "method_indices", dataset.processing_methods),
                    ("wavelength", "wavelength_indices", dataset.wavelengths),
                ):
                    selections.extend(
                        (
                            {
                                "experiment": name,
                                "stage": stage,
                                "repeat": fold["repeat"],
                                "fold": fold["fold"],
                                "kind": kind,
                                "original_index": index,
                                "name": labels[index],
                                "input_position": position,
                            }
                            for position, index in enumerate(plan[key])
                        )
                    )
        pd.DataFrame(epochs).to_csv(folder / "epoch_selection_by_outer_fold.csv", index=False)
        pd.concat(curves, ignore_index=True).to_csv(
            folder / "validation_learning_curves_all_folds.csv", index=False
        )
        for kind in ("method", "wavelength"):
            attention = pd.concat(
                [pd.read_csv(task / f"{kind}_attention.csv") for task in tasks], ignore_index=True
            )
            attention.to_csv(folder / f"{kind}_attention_by_model.csv", index=False)
            means = attention.groupby(["split", "repeat", "fold", "index", "name"], as_index=False)[
                "mean"
            ].mean()
            means.groupby(["split", "index", "name"])["mean"].agg(
                ["mean", "std", "count"]
            ).reset_index().to_csv(
                folder / f"{kind}_attention_summary_when_selected.csv", index=False
            )
        save_json(
            folder / "attention_aggregation_metadata.json",
            {
                "means_are_conditional_on_input_being_selected": True,
                "missing_inputs_are_not_zero_attention": True,
                "original_indices_preserved": True,
                "train_test_saved_separately": True,
            },
        )
        summaries[name], ensembles[name], draws[name] = summarize_predictions(
            config, dataset, frame, folder
        )
        comparison.append(
            {
                "experiment": name,
                "family": candidate["family"],
                "method_k": candidate["method_k"],
                "wavelength_k": candidate["wavelength_k"],
                "outer_repeats": config.outer_repeats,
                "seed_count": len(config.train_seeds),
                **{
                    f"{metric}_{stat}": summaries[name]["exploratory_seed_ensemble"][metric][stat]
                    for metric in common.METRICS
                    for stat in ("mean", "sd_across_repeats")
                },
            }
        )
    selected = pd.DataFrame(selections)
    selected.to_csv(root / "selected_inputs_by_fold.csv", index=False)
    frequencies = (
        selected.groupby(["experiment", "stage", "kind", "original_index", "name"])
        .size()
        .rename("selected_fold_count")
        .reset_index()
    )
    frequencies["fraction_of_outer_folds"] = frequencies.selected_fold_count / len(
        manifest["folds"]
    )
    frequencies.to_csv(root / "input_selection_frequencies.csv", index=False)
    table = pd.DataFrame(comparison)
    table["experiment_type"] = "exploratory"
    table["independent_feature_selection_evaluation"] = False
    table["max_epochs"] = config.max_epochs
    table["patience"] = config.patience
    table.to_csv(root / "input_selection_comparison.csv", index=False)
    save_combination_details(config, table, root)
    save_json(root / "summary_all_experiments.json", summaries)
    np.savez_compressed(root / "conditional_bootstrap_metric_draws.npz", **draws)
    print(table.to_string(index=False), flush=True)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, default=defaults["source_result"])
    parser.add_argument("--ranking-result", type=Path, default=defaults["ranking_result"])
    parser.add_argument("--result-root", type=Path, default=defaults["result_root"])
    for key in ("method_topks", "wavelength_topks", "train_seeds"):
        parser.add_argument(
            "--" + key.replace("_", "-"), type=int, nargs="+", default=defaults[key]
        )
    for key in (
        "outer_repeats",
        "max_epochs",
        "patience",
        "batch_size",
        "parallel_outer_folds",
        "cpu_threads",
        "bootstrap_iterations",
    ):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=defaults[key])
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
        config = SelectionConfig(**saved)
        dataset = load_raw_dataset(root / "dataset_snapshot.pt", config.target)
        summarize(
            config,
            read_json(root / "outer_folds.json"),
            dataset,
            read_json(root / "candidate_plan.json"),
            root,
        )
        print(f"Summarized saved predictions without training or source-result access: {root}")
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
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations cannot be negative")
    for key in ("train_seeds",):
        values = getattr(args, key)
        if (
            not values
            or len(values) != len(set(values))
            or any((not 0 <= n < 2**32 for n in values))
        ):
            raise ValueError(f"Invalid/duplicate {key}")
    source_root = args.source_result.resolve()
    source_config = load_run_config(source_root / "run_config.json")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable")
    excluded = ("source_result", "ranking_result", "device", "target", "bootstrap_seed_base")
    config = SelectionConfig(
        **{k: vars(args)[k] for k in SelectionConfig.__dataclass_fields__ if k not in excluded},
        source_result=str(source_root),
        ranking_result=str(args.ranking_result.resolve()),
        device=device,
        **{k: source_config[k] for k in ("target", "bootstrap_seed_base")},
    )
    dataset = load_raw_dataset(source_root / "dataset_snapshot.pt", config.target)
    candidates = build_candidates(config, dataset)
    if not candidates:
        raise ValueError("At least one reduced-input combination is required")
    manifest, jobs = source_check(dataset, config)
    counts = execution_counts(config, candidates, jobs)
    frozen = load_global_rankings(dataset, config)
    validation_folds = [job["assignment"] for job in jobs]
    parameters = [job["source_params"] for job in jobs]
    plans = [selection_plan(dataset, candidate, frozen["scores"]) for candidate in candidates]
    print(f"Input combinations: {len(candidates)}; outer folds: {len(jobs)}; seeds: {config.train_seeds}", flush=True)
    if args.resume:
        root = args.resume.resolve()
        old, current = load_run_config(root / "run_config.json"), asdict(config)
        for key in ("parallel_outer_folds", "cpu_threads"):
            old.pop(key, None)
            current.pop(key, None)
        if (old != current
            or read_json(root / "fixed_global_rankings.json") != frozen
            or read_json(root / "candidate_plan.json") != candidates
            or read_json(root / "outer_folds.json") != manifest
            or read_json(root / "validation_folds.json") != validation_folds
            or read_json(root / "reused_hyperparameters.json") != parameters
            or read_json(root / "fixed_joint_selection_plans.json") != plans
            or not common.same_dataset(load_raw_dataset(root / "dataset_snapshot.pt", config.target), dataset)):
            raise ValueError("Resume data, rankings or settings differ from the saved run")
    else:
        root = trainer.next_result_directory(args.result_root.resolve())
        shutil.copy2(source_root / "dataset_snapshot.pt", root / "dataset_snapshot.pt")
        shutil.copy2(source_root / "dataset_metadata.json", root / "dataset_metadata.json")
        save_json(root / "run_config.json", asdict(config))
        save_json(root / "outer_folds.json", manifest)
        save_json(root / "validation_folds.json", validation_folds)
        save_json(root / "candidate_plan.json", candidates)
        save_json(root / "fixed_global_rankings.json", frozen)
        save_json(root / "fixed_joint_selection_plans.json", plans)
        save_json(root / "reused_hyperparameters.json", parameters)
        save_rankings(root / "fixed_global_attention_rankings.csv", dataset, frozen["scores"], "fixed_global")
    print(f"Result: {root}", flush=True)
    for job in jobs:
        (root / "fold_data" / fold_relative(job["fold"])).mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            task_path(root, candidate["name"], job["fold"]).mkdir(parents=True, exist_ok=True)
    if config.parallel_outer_folds == 1:
        for job in jobs:
            outer_worker(config, job, candidates, str(root))
    else:
        with ProcessPoolExecutor(
            max_workers=config.parallel_outer_folds, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(outer_worker, config, job, candidates, str(root)) for job in jobs
            ]
            for future in as_completed(futures):
                print(f"Completed outer task: {future.result()}", flush=True)
    summarize(config, manifest, dataset, candidates, root)
    save_json(
        root / "run_complete.json",
        {
            "status": "complete",
            "protocol": PROTOCOL,
            "outer_tasks": len(jobs),
            "train_seeds": config.train_seeds,
            "candidates": [c["name"] for c in candidates],
            "experiment_type": "exploratory",
            "training_budget": training_budget(config),
            "execution_counts": counts,
            "independent_feature_selection_evaluation": False,
        },
    )
    print(f"Input-combination cross-validation completed: {root}", flush=True)


if __name__ == "__main__":
    main()
