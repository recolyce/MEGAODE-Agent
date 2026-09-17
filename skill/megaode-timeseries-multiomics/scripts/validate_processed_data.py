#!/usr/bin/env python3
"""Validate BioMaster processed_data/ before MEGAODE starts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REQUIRED_FILES = (
    "sample_metadata.csv",
    "proteomics.csv",
    "metabolomics.csv",
    "protein_annotations.csv",
    "metabolite_annotations.csv",
    "dataset_manifest.yaml",
    "preprocessing_report.md",
)
REQUIRED_META = ("sample_id", "subject_id", "time", "time_unit", "condition", "batch")
VALID_STATUS = {
    "MODELING READY",
    "MODELING READY WITH WARNINGS",
    "NOT MODELING READY",
}


def _fail(errors: list[str], extra: dict | None = None) -> int:
    payload = {"ok": False, "errors": errors}
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1


def _read_manifest(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        return {"_raw": text}
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {"_raw": text}


def _read_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    first = str(frame.columns[0])
    if first != "sample_id":
        warnings.append(f"{path.name} first column is {first!r}; treating it as sample_id")
        frame = frame.rename(columns={frame.columns[0]: "sample_id"})
    matrix = frame.set_index("sample_id")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix, warnings


def _numeric_matrix(name: str, matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    converted = matrix.apply(pd.to_numeric, errors="coerce")
    coerced = int((converted.isna() & matrix.notna()).sum().sum())
    if coerced:
        warnings.append(f"{name}: coerced {coerced} non-numeric values to NA")
    inf_count = int(converted.isin([math.inf, -math.inf]).sum().sum()) if len(converted.columns) else 0
    if inf_count:
        raise ValueError(f"{name} contains ±Inf")
    if converted.isna().all().all():
        raise ValueError(f"{name} has no numeric values")
    return converted, warnings


def _report_status(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    return last if last in VALID_STATUS else None


def _dataset_name(root: Path, manifest: dict) -> str:
    for key in ("dataset_name", "name", "dataset"):
        value = manifest.get(key)
        if value:
            return str(value)
    return root.name if root.name != "processed_data" else (root.parent.name or "dataset")


def validate(root: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        return {"ok": False, "errors": [f"missing required files: {missing}"]}

    metadata = pd.read_csv(root / "sample_metadata.csv")
    absent = [col for col in REQUIRED_META if col not in metadata.columns]
    if absent:
        errors.append(f"sample_metadata.csv missing columns: {absent}")
        return {"ok": False, "errors": errors}
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    if metadata["sample_id"].duplicated().any():
        dup = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].astype(str).unique().tolist()
        errors.append(f"sample_id is not unique, e.g. {dup[:5]}")
    metadata["time"] = pd.to_numeric(metadata["time"], errors="coerce")
    if metadata["time"].isna().any():
        errors.append("sample_metadata.time must be numeric and sortable")

    try:
        proteomics, w1 = _read_matrix(root / "proteomics.csv")
        metabolomics, w2 = _read_matrix(root / "metabolomics.csv")
        warnings.extend(w1 + w2)
        proteomics, w3 = _numeric_matrix("proteomics.csv", proteomics)
        metabolomics, w4 = _numeric_matrix("metabolomics.csv", metabolomics)
        warnings.extend(w3 + w4)
    except ValueError as exc:
        errors.append(str(exc))
        return {"ok": False, "errors": errors}

    meta_ids = set(metadata["sample_id"])
    prot_ids = set(proteomics.index.astype(str))
    met_ids = set(metabolomics.index.astype(str))
    if not prot_ids.issubset(meta_ids):
        errors.append(f"proteomics sample_id not in metadata, e.g. {sorted(prot_ids - meta_ids)[:5]}")
    if not met_ids.issubset(meta_ids):
        errors.append(f"metabolomics sample_id not in metadata, e.g. {sorted(met_ids - meta_ids)[:5]}")
    paired = prot_ids & met_ids
    if not paired:
        errors.append("proteomics and metabolomics have no paired samples")
    if proteomics.columns.duplicated().any():
        errors.append("proteomics feature names are duplicated")
    if metabolomics.columns.duplicated().any():
        errors.append("metabolomics feature names are duplicated")

    if not metadata["time"].isna().all():
        by_subject = metadata.groupby(metadata["subject_id"].astype(str))["time"].nunique()
        n_times = int(metadata["time"].nunique())
        if int((by_subject >= 2).sum()) == 0 and n_times < 2:
            errors.append("no valid longitudinal structure: need a subject with ≥2 timepoints or ≥2 distinct times")

    report = (root / "preprocessing_report.md").read_text(encoding="utf-8")
    status = _report_status(report)
    if status is None:
        errors.append("preprocessing_report.md must end with MODELING READY, MODELING READY WITH WARNINGS, or NOT MODELING READY")

    manifest = _read_manifest(root / "dataset_manifest.yaml")
    expected = {
        "n_samples": int(len(metadata)),
        "n_proteins": int(proteomics.shape[1]),
        "n_metabolites": int(metabolomics.shape[1]),
        "n_paired_samples": int(len(paired)),
    }
    aliases = {
        "n_samples": ("n_samples", "sample_count", "samples"),
        "n_proteins": ("n_proteins", "n_protein_features", "protein_count", "proteins"),
        "n_metabolites": ("n_metabolites", "n_metabolite_features", "metabolite_count", "metabolites"),
        "n_paired_samples": ("n_paired_samples", "paired_samples", "n_paired"),
    }
    for key, names in aliases.items():
        present = next((manifest[name] for name in names if name in manifest and manifest[name] is not None), None)
        if present is None:
            warnings.append(f"dataset_manifest.yaml missing {key}")
            continue
        try:
            if int(present) != expected[key]:
                errors.append(f"dataset_manifest.yaml {key}={present} != actual {expected[key]}")
        except (TypeError, ValueError):
            errors.append(f"dataset_manifest.yaml {key} is not an integer")

    if status == "NOT MODELING READY":
        errors.append("preprocessing_report status is NOT MODELING READY")

    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "modeling_status": status}

    return {
        "ok": True,
        "modeling_status": status,
        "dataset_name": _dataset_name(root, manifest),
        "warnings": warnings,
        "n_samples": expected["n_samples"],
        "n_proteins": expected["n_proteins"],
        "n_metabolites": expected["n_metabolites"],
        "n_paired_samples": expected["n_paired_samples"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate processed_data/ for MEGAODE")
    parser.add_argument("processed_data", type=Path)
    args = parser.parse_args()
    root = args.processed_data
    if not root.exists():
        return _fail([f"processed_data directory does not exist: {root}"])
    result = validate(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
