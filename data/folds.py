"""Dataset validation and acquisition-grouped cross-validation splits.

All ROIs from one NH7 acquisition stay in the same fold. Input spectra are
normalized later, when the training scripts load each partition.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def load_outer_fold_manifest(path, dataset):
    """Load saved folds, validating exact sample identity and grouped coverage."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("n_samples") != len(dataset.X):
        raise ValueError("Reference manifest sample count differs")
    folds = []
    seen = set()
    for record in payload["folds"]:
        pair = (int(record["repeat"]), int(record["fold"]))
        if pair in seen:
            raise ValueError("Duplicate repeat/fold")
        seen.add(pair)
        train, test = (np.asarray(record[name], dtype=int) for name in ("train_indices", "test_indices"))
        for indices in (train, test):
            if not len(indices) or np.any(indices < 0) or np.any(indices >= len(dataset.X)) or len(set(indices)) != len(indices):
                raise ValueError("Invalid manifest sample indices")
        if set(train) & set(test) or set(train) | set(test) != set(range(len(dataset.X))):
            raise ValueError("Invalid manifest train/test coverage")
        for split, indices in (("train", train), ("test", test)):
            if dataset.global_ids[indices].tolist() != record[f"{split}_sample_ids"] or sorted(set(dataset.acquisition_ids[indices])) != sorted(record[f"{split}_groups"]):
                raise ValueError("Manifest sample/group identity differs from dataset")
        if set(dataset.acquisition_ids[train]) & set(dataset.acquisition_ids[test]):
            raise ValueError("Cluster leakage in saved outer partitions")
        folds.append(GroupFold(*pair, int(record["seed"]), train, test,
            tuple(record["train_groups"]), tuple(record["test_groups"])))
    for repeat in sorted({f.repeat for f in folds}):
        coverage = np.zeros(len(dataset.X), dtype=int)
        for fold in folds:
            if fold.repeat == repeat:
                coverage[fold.test_indices] += 1
        if not np.all(coverage == 1):
            raise ValueError("Expected one held-out prediction per berry and repeat")
    return folds


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

    methods = [str(v) for v in data.get("processing_methods", [])]
    wavelengths = [float(v) for v in data.get("wavelengths", [])]
    if len(methods) != X.shape[1] or len(set(methods)) != len(methods):
        raise ValueError("Missing, duplicate or misaligned preprocessing-channel metadata")
    if len(wavelengths) != X.shape[2] or len(wavelengths) < 2 or not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelength metadata must match a strictly increasing spectral axis")
    return RawDataset(
        X=X,
        y=y,
        global_ids=global_ids,
        acquisition_ids=acquisition_ids,
        dates=dates,
        processing_methods=[str(v) for v in data.get("processing_methods", [])],
        wavelengths=[float(v) for v in data.get("wavelengths", [])],
    )


def _random_balanced_group_assignment(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    """Randomly assign acquisitions while balancing sample counts.

    Ties are broken randomly to obtain different repeated partitions.
    Target values are not used.
    """
    unique, counts = np.unique(groups.astype(str), return_counts=True)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(unique) < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds the number of acquisitions={len(unique)}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled_groups = unique[order]
    shuffled_counts = counts[order]

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    # The first pass guarantees that every fold receives at least one group.
    for fold_index in range(n_splits):
        fold_groups[fold_index].append(str(shuffled_groups[fold_index]))
        fold_sizes[fold_index] += int(shuffled_counts[fold_index])

    for group, count in zip(shuffled_groups[n_splits:], shuffled_counts[n_splits:], strict=True):
        smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold_index = int(rng.choice(smallest))
        fold_groups[fold_index].append(str(group))
        fold_sizes[fold_index] += int(count)

    return fold_groups


def build_repeated_group_folds(
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 2026,
) -> list[GroupFold]:
    """Create repeated outer folds with complete, disjoint group coverage."""
    groups = np.asarray(groups, dtype=str)
    all_indices = np.arange(len(groups), dtype=np.int64)
    result: list[GroupFold] = []
    seen_signatures: set[tuple[tuple[str, ...], ...]] = set()

    repeat = 0
    seed_offset = 0
    max_attempts = max(100, n_repeats * 50)
    while repeat < n_repeats:
        if seed_offset >= max_attempts:
            raise RuntimeError("Could not construct enough distinct repeated group partitions")
        seed = base_seed + seed_offset
        seed_offset += 1
        fold_groups = _random_balanced_group_assignment(groups, n_splits, seed)
        signature = tuple(sorted(tuple(sorted(v)) for v in fold_groups))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        coverage = np.zeros(len(groups), dtype=np.int64)
        for fold_index, test_group_list in enumerate(fold_groups):
            test_mask = np.isin(groups, np.asarray(test_group_list, dtype=str))
            test_indices = all_indices[test_mask]
            train_indices = all_indices[~test_mask]
            train_groups = tuple(sorted(np.unique(groups[train_indices]).tolist()))
            test_groups = tuple(sorted(np.unique(groups[test_indices]).tolist()))
            if set(train_groups) & set(test_groups):
                raise RuntimeError("Acquisition leakage detected while building outer folds")
            coverage[test_indices] += 1
            result.append(
                GroupFold(
                    repeat=repeat,
                    fold=fold_index,
                    seed=seed,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_groups=train_groups,
                    test_groups=test_groups,
                )
            )
        if not np.all(coverage == 1):
            raise RuntimeError("Every sample must occur in exactly one test fold per repeat")
        repeat += 1

    return result


def iter_inner_group_folds(
    outer_train_indices: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic GroupKFold indices in full-dataset coordinates."""
    from sklearn.model_selection import GroupKFold

    outer_train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    outer_groups = np.asarray(groups, dtype=str)[outer_train_indices]
    unique_groups = np.unique(outer_groups)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        raise ValueError("At least two outer-training acquisitions are required")
    splitter = GroupKFold(n_splits=actual_splits)
    placeholder = np.zeros(len(outer_train_indices), dtype=np.float32)
    for local_train, local_valid in splitter.split(placeholder, groups=outer_groups):
        train_indices = outer_train_indices[local_train]
        valid_indices = outer_train_indices[local_valid]
        if set(groups[train_indices]) & set(groups[valid_indices]):
            raise RuntimeError("Acquisition leakage detected in inner GroupKFold")
        yield train_indices, valid_indices


def save_outer_fold_manifest(
    path: Path,
    folds: list[GroupFold],
    dataset: RawDataset,
) -> None:
    """Save sample IDs and acquisition assignments for every outer fold."""
    records = []
    for item in folds:
        records.append(
            {
                "repeat": item.repeat,
                "fold": item.fold,
                "seed": item.seed,
                "train_indices": item.train_indices.tolist(),
                "test_indices": item.test_indices.tolist(),
                "train_groups": list(item.train_groups),
                "test_groups": list(item.test_groups),
                "train_sample_ids": dataset.global_ids[item.train_indices].tolist(),
                "test_sample_ids": dataset.global_ids[item.test_indices].tolist(),
            }
        )
    payload = {
        "group_key": "metadata.acquisition_ids",
        "n_samples": len(dataset.X),
        "n_acquisitions": int(len(np.unique(dataset.acquisition_ids))),
        "n_repeats": int(len({f.repeat for f in folds})),
        "n_splits": int(len({f.fold for f in folds if f.repeat == 0})),
        "folds": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
