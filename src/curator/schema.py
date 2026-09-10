"""Validate a BioMaster modeling-ready template."""

from __future__ import annotations

from pathlib import Path

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


class TemplateError(ValueError):
    pass


def find_template_dir(source: Path) -> Path:
    source = source.resolve()
    if (source / "sample_metadata.csv").exists():
        return source
    nested = source / "processed_data"
    if (nested / "sample_metadata.csv").exists():
        return nested
    raise TemplateError(f"No BioMaster template under {source}")


def validate_files(root: Path) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise TemplateError(f"Missing template files: {missing}")
    return list(REQUIRED_FILES)


def already_log_transformed(manifest_text: str) -> bool:
    text = manifest_text.lower()
    return "log2" in text or "log-transformed" in text or "log transformed" in text
