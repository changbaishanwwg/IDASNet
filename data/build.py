"""Build IDASNet data from dark-corrected NH7 cubes and white references."""

from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter


PREPROCESSING = {
    'sg_window': 11,
    'sg_polynomial': 3,
    'derivative_spacing_nm': 5,
    'sg_mode': 'interp',
    'pixel_snv_std_correction': 0,
    'epsilon': 1e-08,
}

HEIGHT, WIDTH = 1024, 1280
WAVELENGTHS = list(range(350, 1101, 5))
N_BANDS = len(WAVELENGTHS)
PROCESSING_METHODS = [
    "reflectance_SG",
    "reflectance_SGder",
    "reflectance_SG+SNV",
    "reflectance_SGder+SNV",
    "reflectance_SNV",
]
TARGET_COLUMNS = [
    "Density",
    "Total Acid",
    "pH",
    "TSS",
    "Tartaric acid",
    "Malic acid",
    "Alpha Amino",
    "Ammonia",
    "Gluconic acid",
    "VA",
]
INVALID_TARGET_VALUES = {-999.0, -9999.0, -888.0}

# Default paths and experiment settings
# Input dates are subdirectories of DEFAULT_INPUT_ROOT and are combined into one dataset.
DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent / "raw"
DEFAULT_DATES = ["20250908", "20261005", "20261006"]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_ROI_SCALE_FACTOR = 0.9
DEFAULT_WHITEBOARD_MASK_FRACTION = 0.2
DEFAULT_OUTLIER_Z_THRESHOLD = 3.0
DEFAULT_OUTLIER_REQUIRED_BAND_FRACTION = 0.95
DEFAULT_MIN_PIXELS = 10
DEFAULT_OVERWRITE = False


@dataclass
class QualityRecord:
    acquisition_date: str
    sample_id: str
    status: str
    reason: str
    source: str = ""
    detail: str = ""


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def natural_id_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_labelme_groups(json_path: Path) -> dict[str, list[np.ndarray]]:
    """Group polygons by ID. Repeated polygons in one image are one sample."""
    with json_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for shape in document.get("shapes", []):
        sample_id = normalize_id(shape.get("label", ""))
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if sample_id and shape.get("shape_type", "polygon") == "polygon" and len(points) >= 3:
            groups[sample_id].append(points)
    return dict(groups)


def load_labelme_polygons(json_path: Path) -> list[np.ndarray]:
    """Load every valid polygon, regardless of its label."""
    with json_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    polygons = []
    for shape in document.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if shape.get("shape_type", "polygon") == "polygon" and len(points) >= 3:
            polygons.append(points)
    return polygons


def scale_polygon(points: np.ndarray, factor: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    return np.rint(center + (points - center) * factor).astype(np.int32)


def build_roi_mask(polygons: Iterable[np.ndarray], scale_factor: float) -> np.ndarray:
    """Union every polygon belonging to the same sample into one ROI."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for polygon in polygons:
        scaled = scale_polygon(polygon, scale_factor)
        scaled[:, 0] = np.clip(scaled[:, 0], 0, WIDTH - 1)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, HEIGHT - 1)
        cv2.fillPoly(mask, [scaled], 1)
    return mask.astype(bool)


def shrink_mask_to_fraction(mask: np.ndarray, target_fraction: float) -> np.ndarray:
    """Morphologically erode a mask to approximately the requested area."""
    binary = mask.astype(np.uint8)
    original_area = int(binary.sum())
    if original_area == 0:
        return binary.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    low, high = 0, min(mask.shape) // 2
    best = binary.copy()
    best_error = abs(1.0 - target_fraction)
    while low <= high:
        iterations = (low + high) // 2
        eroded = cv2.erode(binary, kernel, iterations=iterations)
        fraction = float(eroded.sum()) / original_area
        error = abs(fraction - target_fraction)
        if eroded.any() and error < best_error:
            best, best_error = eroded.copy(), error
        if fraction > target_fraction:
            low = iterations + 1
        else:
            high = iterations - 1
    return best.astype(bool)


def extract_white_reference(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute W-D with the per-wavelength mean ± 2σ filter."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.full(N_BANDS, np.nan, dtype=np.float32)
    spectra = np.asarray(cube[ys, :, xs], dtype=np.float32)
    means = spectra.mean(axis=0)
    stds = spectra.std(axis=0)
    accepted = (spectra >= means - 2.0 * stds) & (spectra <= means + 2.0 * stds)
    filtered = np.where(accepted, spectra, np.nan)
    return np.nanmean(filtered, axis=0).astype(np.float32)


def robust_common_pixel_mask(
    spectra: np.ndarray, z_threshold: float, required_band_fraction: float
) -> np.ndarray:
    """Select whole pixels once and reuse the mask for every wavelength/method."""
    finite_rows = np.isfinite(spectra).all(axis=1)
    result = np.zeros(len(spectra), dtype=bool)
    finite_spectra = spectra[finite_rows]
    if len(finite_spectra) == 0:
        return result
    median = np.median(finite_spectra, axis=0)
    mad = np.median(np.abs(finite_spectra - median), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(finite_spectra, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    stable_bands = scale > 1e-12
    if stable_bands.any():
        z = np.abs(finite_spectra[:, stable_bands] - median[stable_bands])
        z /= scale[stable_bands]
        keep = (z <= z_threshold).mean(axis=1) >= required_band_fraction
    else:
        keep = np.ones(len(finite_spectra), dtype=bool)
    result[np.flatnonzero(finite_rows)] = keep
    return result


def snv_per_pixel(spectra: np.ndarray) -> np.ndarray:
    means = spectra.mean(axis=1, keepdims=True)
    stds = spectra.std(axis=1, keepdims=True, ddof=PREPROCESSING["pixel_snv_std_correction"])
    return (spectra - means) / (stds + PREPROCESSING["epsilon"])


def preprocess_roi_pixels(spectra: np.ndarray) -> dict[str, np.ndarray]:
    options = dict(window_length=PREPROCESSING["sg_window"],
                   polyorder=PREPROCESSING["sg_polynomial"],
                   axis=1, mode=PREPROCESSING["sg_mode"])
    sg = savgol_filter(spectra, **options).astype(np.float32)
    derivative = savgol_filter(spectra, deriv=1,
        delta=PREPROCESSING["derivative_spacing_nm"], **options).astype(np.float32)
    return {
        "reflectance_SG": sg,
        "reflectance_SGder": derivative,
        "reflectance_SG+SNV": snv_per_pixel(sg),
        "reflectance_SGder+SNV": snv_per_pixel(derivative),
        "reflectance_SNV": snv_per_pixel(spectra),
    }


def read_biochemistry(
    csv_path: Path, date: str
) -> tuple[dict[str, dict[str, object]], set[str], set[str], list[QualityRecord]]:
    frame = pd.read_csv(csv_path)
    # Retain samples based on ID and TSS; other assays are optional metadata.
    missing = [column for column in ["ID", "TSS"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {missing}")
    frame = frame.copy()
    frame["_id"] = frame["ID"].map(normalize_id)
    counts = Counter(frame["_id"])
    observed_ids = {sample_id for sample_id in frame["_id"] if sample_id}
    duplicates = {sample_id for sample_id, count in counts.items() if sample_id and count > 1}
    records = [
        QualityRecord(date, sample_id, "dropped", "duplicate_biochemistry_id", str(csv_path))
        for sample_id in sorted(duplicates, key=natural_id_key)
    ]
    valid: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        sample_id = row["_id"]
        if not sample_id or sample_id in duplicates:
            continue
        values: dict[str, object] = {}
        for column in TARGET_COLUMNS:
            if column not in frame.columns:
                values[column] = np.nan
                continue
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if (
                pd.isna(value)
                or float(value) in INVALID_TARGET_VALUES
                or not np.isfinite(float(value))
            ):
                values[column] = np.nan
            else:
                values[column] = float(value)
        if pd.isna(values["TSS"]):
            records.append(
                QualityRecord(date, sample_id, "dropped", "invalid_TSS", str(csv_path), "TSS")
            )
            continue
        values["Assay_Date"] = str(row.get("Date", "")).strip()
        valid[sample_id] = values
    return valid, observed_ids, duplicates, records


def discover_acquisitions(
    date_dir: Path,
) -> tuple[dict[str, tuple[Path, Path]], list[QualityRecord]]:
    date = date_dir.name
    nh7_files = {p.stem: p for p in (date_dir / "nh7").glob("*.nh7")}
    json_files = {p.stem: p for p in (date_dir / "json").glob("*.json")}
    whiteboard_files = {p.stem: p for p in (date_dir / "whiteboard").glob("*.json")}
    records: list[QualityRecord] = []
    all_stems = set(nh7_files) | set(json_files) | set(whiteboard_files)
    for stem in sorted(all_stems):
        missing = []
        if stem not in nh7_files:
            missing.append("nh7")
        if stem not in json_files:
            missing.append("roi_annotation")
        if stem not in whiteboard_files:
            missing.append("whiteboard_annotation")
        if missing:
            records.append(
                QualityRecord(
                    date, "*", "dropped", "missing_acquisition_file", stem, ";".join(missing)
                )
            )
    valid_stems = set(nh7_files) & set(json_files) & set(whiteboard_files)
    return {
        stem: (nh7_files[stem], whiteboard_files[stem]) for stem in sorted(valid_stems)
    }, records


def build_date_dataset(
    date_dir: Path,
    roi_scale_factor: float,
    whiteboard_mask_fraction: float,
    z_threshold: float,
    required_band_fraction: float,
    min_pixels: int,
) -> tuple[list[dict[str, object]], list[QualityRecord]]:
    date = date_dir.name
    biochem_path = date_dir / "brix" / "MHAM.csv"
    if not biochem_path.exists():
        raise FileNotFoundError(f"Biochemistry table not found: {biochem_path}")
    biochemistry, observed_biochem, duplicate_biochem, records = read_biochemistry(
        biochem_path, date
    )
    acquisitions, file_records = discover_acquisitions(date_dir)
    records.extend(file_records)

    annotations: dict[str, tuple[str, list[np.ndarray], Path]] = {}
    cross_file_duplicates: set[str] = set()
    for stem in acquisitions:
        json_path = date_dir / "json" / f"{stem}.json"
        for sample_id, polygons in load_labelme_groups(json_path).items():
            if sample_id in annotations and annotations[sample_id][0] != stem:
                cross_file_duplicates.add(sample_id)
            else:
                annotations[sample_id] = (stem, polygons, json_path)
    for sample_id in sorted(cross_file_duplicates, key=natural_id_key):
        records.append(
            QualityRecord(date, sample_id, "dropped", "duplicate_roi_id_across_acquisitions")
        )

    roi_ids = set(annotations) - cross_file_duplicates
    biochem_ids = set(biochemistry)
    for sample_id in sorted(roi_ids - observed_biochem, key=natural_id_key):
        records.append(QualityRecord(date, sample_id, "dropped", "missing_biochemistry_id"))
    for sample_id in sorted(biochem_ids - roi_ids, key=natural_id_key):
        records.append(QualityRecord(date, sample_id, "dropped", "missing_roi_id"))

    by_acquisition: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(roi_ids & biochem_ids, key=natural_id_key):
        by_acquisition[annotations[sample_id][0]].append(sample_id)

    rows: list[dict[str, object]] = []
    expected_bytes = HEIGHT * WIDTH * N_BANDS * np.dtype(np.uint16).itemsize
    for stem, sample_ids in sorted(by_acquisition.items()):
        nh7_path, whiteboard_path = acquisitions[stem]
        if nh7_path.stat().st_size != expected_bytes:
            records.extend(
                QualityRecord(date, sid, "dropped", "invalid_nh7_size", str(nh7_path))
                for sid in sample_ids
            )
            continue
        cube = np.memmap(nh7_path, dtype=np.uint16, mode="r", shape=(HEIGHT, N_BANDS, WIDTH))
        whiteboard_polygons = load_labelme_polygons(whiteboard_path)
        whiteboard_mask = build_roi_mask(whiteboard_polygons, 1.0)
        whiteboard_mask = shrink_mask_to_fraction(whiteboard_mask, whiteboard_mask_fraction)
        white_reference = extract_white_reference(cube, whiteboard_mask)
        if (
            not whiteboard_polygons
            or not np.isfinite(white_reference).all()
            or np.any(white_reference <= 1e-8)
        ):
            records.extend(
                QualityRecord(date, sid, "dropped", "invalid_white_reference", str(whiteboard_path))
                for sid in sample_ids
            )
            del cube
            continue
        for sample_id in sample_ids:
            _, polygons, json_path = annotations[sample_id]
            ys, xs = np.where(build_roi_mask(polygons, roi_scale_factor))
            if len(ys) < min_pixels:
                records.append(
                    QualityRecord(
                        date,
                        sample_id,
                        "dropped",
                        "insufficient_roi_pixels",
                        str(json_path),
                        str(len(ys)),
                    )
                )
                continue
            # NH7 stores DN-D and the whiteboard spectrum stores W-D.
            dn_minus_d = np.asarray(cube[ys, :, xs], dtype=np.float32)
            spectra = dn_minus_d / white_reference
            common_mask = robust_common_pixel_mask(spectra, z_threshold, required_band_fraction)
            retained = int(common_mask.sum())
            if retained < min_pixels:
                records.append(
                    QualityRecord(
                        date,
                        sample_id,
                        "dropped",
                        "insufficient_pixels_after_outlier_filter",
                        str(nh7_path),
                        f"{retained}/{len(spectra)}",
                    )
                )
                continue
            retained_spectra = spectra[common_mask]
            retained_dn_minus_d = dn_minus_d[common_mask]
            method_means = {
                method: values.mean(axis=0)
                for method, values in preprocess_roi_pixels(retained_spectra).items()
            }
            raw_dn_mean = retained_dn_minus_d.mean(axis=0)
            raw_reflectance_mean = retained_spectra.mean(axis=0)
            if any(
                v.shape != (N_BANDS,) or not np.isfinite(v).all()
                for v in (*method_means.values(), raw_dn_mean, raw_reflectance_mean)
            ):
                records.append(
                    QualityRecord(
                        date,
                        sample_id,
                        "dropped",
                        "incomplete_or_nonfinite_spectrum",
                        str(nh7_path),
                    )
                )
                continue
            row: dict[str, object] = {
                "Global_ID": f"{date}_{sample_id}",
                "Date": date,
                "Assay_Date": biochemistry[sample_id]["Assay_Date"],
                "ID": sample_id,
                "Acquisition_ID": f"{date}_{stem}",
                "Retained_Pixels": retained,
            }
            row.update({target: biochemistry[sample_id][target] for target in TARGET_COLUMNS})
            for wavelength, dn_value, reflectance_value, white_value in zip(
                WAVELENGTHS, raw_dn_mean, raw_reflectance_mean, white_reference, strict=True
            ):
                row[f"{wavelength}_dn_minus_d"] = float(dn_value)
                row[f"{wavelength}_reflectance_raw"] = float(reflectance_value)
                row[f"{wavelength}_white_dn_minus_d"] = float(white_value)
            for method in PROCESSING_METHODS:
                for wavelength, value in zip(WAVELENGTHS, method_means[method], strict=True):
                    row[f"{wavelength}_{method}"] = float(value)
            rows.append(row)
            records.append(
                QualityRecord(
                    date,
                    sample_id,
                    "kept",
                    "valid",
                    stem,
                    f"pixels={retained}/{len(spectra)};polygons={len(polygons)}",
                )
            )
        del cube
    return rows, records


def save_outputs(
    rows: list[dict[str, object]],
    records: list[QualityRecord],
    output_dir: Path,
    configuration: dict[str, object],
    overwrite: bool,
) -> None:
    if not rows:
        raise RuntimeError("No valid samples remain after quality control.")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "Indoor_Multichannel_Dataset_Cleaned.csv"
    pt_path = output_dir / "raw.pt"
    dn_path = output_dir / "ROI_DN_minus_D_Spectra.csv"
    reflectance_path = output_dir / "ROI_Raw_Reflectance_Spectra.csv"
    white_path = output_dir / "White_Reference_DN_minus_D_Spectra.csv"
    report_path = output_dir / "quality_report.csv"
    manifest_path = output_dir / "build_manifest.json"
    outputs = [csv_path, pt_path, dn_path, reflectance_path, white_path, report_path, manifest_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    frame = pd.DataFrame(rows).sort_values(["Date", "ID"]).reset_index(drop=True)
    wavelength_columns = [str(w) for w in WAVELENGTHS]
    roi_metadata = ["Global_ID", "Date", "Acquisition_ID", "ID", "Retained_Pixels"]
    for suffix, destination in (
        ("dn_minus_d", dn_path),
        ("reflectance_raw", reflectance_path),
    ):
        source_columns = [f"{w}_{suffix}" for w in WAVELENGTHS]
        spectral_frame = frame[[*roi_metadata, *source_columns]].copy()
        spectral_frame.columns = [*roi_metadata, *wavelength_columns]
        spectral_frame.to_csv(destination, index=False, encoding="utf-8-sig")
    white_columns = [f"{w}_white_dn_minus_d" for w in WAVELENGTHS]
    white_frame = (
        frame[["Acquisition_ID", "Date", *white_columns]].drop_duplicates("Acquisition_ID").copy()
    )
    white_frame.columns = ["Acquisition_ID", "Date", *wavelength_columns]
    white_frame.to_csv(white_path, index=False, encoding="utf-8-sig")
    feature_names = [f"{w}_{method}" for method in PROCESSING_METHODS for w in WAVELENGTHS]
    frame = frame[
        ["Global_ID", "Date", "Assay_Date", "Acquisition_ID", "ID", *TARGET_COLUMNS, *feature_names]
    ]
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    x = np.stack(
        [
            frame[[f"{w}_{method}" for w in WAVELENGTHS]].to_numpy(np.float32)
            for method in PROCESSING_METHODS
        ],
        axis=1,
    )
    if x.shape != (len(frame), len(PROCESSING_METHODS), N_BANDS) or not np.isfinite(x).all():
        raise RuntimeError(f"Final feature tensor failed validation: shape={x.shape}")
    dataset = {
        "X": torch.from_numpy(x),
        "y": {c: torch.tensor(frame[c].to_numpy(np.float32)) for c in TARGET_COLUMNS},
        "metadata": {
            "global_ids": frame["Global_ID"].to_numpy(),
            "dates": frame["Date"].to_numpy(),
            "assay_dates": frame["Assay_Date"].to_numpy(),
            "original_ids": frame["ID"].to_numpy(),
            "acquisition_ids": frame["Acquisition_ID"].to_numpy(),
            "target_names": TARGET_COLUMNS,
        },
        "processing_methods": PROCESSING_METHODS,
        "wavelengths": WAVELENGTHS,
        "target_names": TARGET_COLUMNS,
        "feature_names": feature_names,
        "normalization": {"method": "none; applied by SplitPT after splitting"},
        "radiometry": {
            "input_status": "dark_corrected_DN_minus_D",
            "reflectance_formula": "(DN-D)/(W-D)",
            "whiteboard_mask_fraction": configuration["whiteboard_mask_fraction"],
            "whiteboard_outlier_filter": "per-wavelength mean +/- 2 standard deviations",
        },
    }
    torch.save(dataset, pt_path)
    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    reason_counts = Counter(r.reason for r in records if r.status == "dropped")
    manifest = {
        "configuration": configuration,
        "outputs": {
            "csv": csv_path.name,
            "pytorch": pt_path.name,
            "roi_dn_minus_d": dn_path.name,
            "roi_raw_reflectance": reflectance_path.name,
            "white_reference_dn_minus_d": white_path.name,
            "quality_report": report_path.name,
        },
        "dataset": {
            "samples": len(frame),
            "shape": list(x.shape),
            "processing_methods": PROCESSING_METHODS,
            "wavelengths_nm": WAVELENGTHS,
            "targets": TARGET_COLUMNS,
        },
        "spectral_records": {
            "roi_pixel_rule": "mean over the same retained ROI pixels used for the model features",
            "nh7_signal": "DN-D; uncorrected DN is not recoverable from this file",
            "white_reference": "W-D after whiteboard masking and per-wavelength outlier filtering",
            "raw_reflectance_formula": "(DN-D)/(W-D)",
        },
        "quality_control": {
            "kept": sum(r.status == "kept" for r in records),
            "dropped_by_reason": dict(sorted(reason_counts.items())),
            "same_id_polygons_within_one_annotation": "unioned into one ROI",
            "invalid_target_rule": "drop only when TSS is invalid; other invalid targets are stored as NaN",
            "outlier_filter": "one robust whole-pixel mask shared by all bands and methods",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(frame)} samples with X shape {x.shape}")
    print(f"CSV: {csv_path}\nPyTorch: {pt_path}")
    print(f"Spectral records: {dn_path}, {reflectance_path}, {white_path}")
    print(f"Quality report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-scale-factor", type=float, default=DEFAULT_ROI_SCALE_FACTOR)
    parser.add_argument(
        "--whiteboard-mask-fraction", type=float, default=DEFAULT_WHITEBOARD_MASK_FRACTION
    )
    parser.add_argument("--outlier-z-threshold", type=float, default=DEFAULT_OUTLIER_Z_THRESHOLD)
    parser.add_argument(
        "--outlier-required-band-fraction",
        type=float,
        default=DEFAULT_OUTLIER_REQUIRED_BAND_FRACTION,
    )
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.roi_scale_factor <= 1:
        raise ValueError("--roi-scale-factor must be in (0, 1]")
    if not 0 < args.whiteboard_mask_fraction <= 1:
        raise ValueError("--whiteboard-mask-fraction must be in (0, 1]")
    if not 0 < args.outlier_required_band_fraction <= 1:
        raise ValueError("--outlier-required-band-fraction must be in (0, 1]")
    rows: list[dict[str, object]] = []
    records: list[QualityRecord] = []
    for date in args.dates:
        date_dir = args.input_root / date
        if not date_dir.is_dir():
            raise FileNotFoundError(f"Acquisition directory not found: {date_dir}")
        print(f"Processing acquisition date {date} ...")
        date_rows, date_records = build_date_dataset(
            date_dir,
            args.roi_scale_factor,
            args.whiteboard_mask_fraction,
            args.outlier_z_threshold,
            args.outlier_required_band_fraction,
            args.min_pixels,
        )
        rows.extend(date_rows)
        records.extend(date_records)
    global_ids = [str(row["Global_ID"]) for row in rows]
    duplicate_globals = {v for v, count in Counter(global_ids).items() if count > 1}
    if duplicate_globals:
        rows = [row for row in rows if row["Global_ID"] not in duplicate_globals]
        for global_id in sorted(duplicate_globals):
            date, sample_id = global_id.split("_", 1)
            records.append(QualityRecord(date, sample_id, "dropped", "duplicate_global_id"))
    configuration = {
        "input_root": str(args.input_root.resolve()),
        "dates": args.dates,
        "roi_scale_factor": args.roi_scale_factor,
        "whiteboard_mask_fraction": args.whiteboard_mask_fraction,
        "outlier_z_threshold": args.outlier_z_threshold,
        "outlier_required_band_fraction": args.outlier_required_band_fraction,
        "min_pixels": args.min_pixels,
    }
    save_outputs(rows, records, args.output_dir, configuration, args.overwrite)


if __name__ == "__main__":
    main()
