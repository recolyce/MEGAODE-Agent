"""Load a BioMaster template into memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.curator.schema import (
    TemplateError,
    already_log_transformed,
    find_template_dir,
    validate_files,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:  # noqa: BLE001 — BioMaster manifests may be YAML-ish comments
            pass
    return {"_raw": text}


def _wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns[0] != "sample_id":
        raise TemplateError(f"{path.name} must start with sample_id")
    matrix = frame.set_index("sample_id")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


@dataclass
class BioMasterDataset:
    source: Path
    name: str
    metadata: pd.DataFrame
    proteomics: pd.DataFrame
    metabolomics: pd.DataFrame
    protein_annotations: pd.DataFrame
    metabolite_annotations: pd.DataFrame
    manifest: dict[str, Any]
    report: str
    already_logged: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def tissues(self) -> list[str]:
        if "tissue" not in self.metadata.columns:
            return []
        return sorted(self.metadata["tissue"].astype(str).unique())


def load_biomaster(source: Path) -> BioMasterDataset:
    root = find_template_dir(Path(source))
    validate_files(root)
    metadata = pd.read_csv(root / "sample_metadata.csv")
    missing = [c for c in ("sample_id", "subject_id", "time", "time_unit", "condition", "batch") if c not in metadata.columns]
    if missing:
        raise TemplateError(f"sample_metadata.csv missing columns: {missing}")
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["time"] = pd.to_numeric(metadata["time"], errors="coerce")
    if metadata["time"].isna().any():
        raise TemplateError("sample_metadata.time must be numeric")
    proteomics = _wide(root / "proteomics.csv")
    metabolomics = _wide(root / "metabolomics.csv")
    protein_ids = set(proteomics.index.astype(str))
    metabolite_ids = set(metabolomics.index.astype(str))
    meta_ids = set(metadata["sample_id"])
    if protein_ids != metabolite_ids:
        raise TemplateError("proteomics and metabolomics sample_id sets differ")
    if not protein_ids.issubset(meta_ids):
        extra = sorted(protein_ids - meta_ids)[:5]
        raise TemplateError(f"matrix sample_id not in metadata, e.g. {extra}")
    metadata = metadata.loc[metadata["sample_id"].isin(protein_ids)].copy()
    metadata = metadata.sort_values(["sample_id"], kind="mergesort").reset_index(drop=True)
    proteomics = proteomics.loc[metadata["sample_id"]]
    metabolomics = metabolomics.loc[metadata["sample_id"]]
    protein_ann = pd.read_csv(root / "protein_annotations.csv")
    metabolite_ann = pd.read_csv(root / "metabolite_annotations.csv")
    id_col_p = "protein_id" if "protein_id" in protein_ann.columns else "feature_id"
    id_col_m = "metabolite_id" if "metabolite_id" in metabolite_ann.columns else "feature_id"
    protein_ann = protein_ann.rename(columns={id_col_p: "feature_id"})
    metabolite_ann = metabolite_ann.rename(columns={id_col_m: "feature_id"})
    protein_ann["feature_id"] = protein_ann["feature_id"].astype(str)
    metabolite_ann["feature_id"] = metabolite_ann["feature_id"].astype(str)
    report = (root / "preprocessing_report.md").read_text(encoding="utf-8")
    manifest_text = (root / "dataset_manifest.yaml").read_text(encoding="utf-8")
    manifest = _read_manifest(root / "dataset_manifest.yaml")
    warnings: list[str] = []
    if "MODELING READY WITH WARNINGS" in report:
        warnings.append("BioMaster status: MODELING READY WITH WARNINGS")
    elif "NOT MODELING READY" in report:
        warnings.append("BioMaster status: NOT MODELING READY")
    if not (root / "sample_qc_flags.csv").exists() and "sample_qc_flags" in report:
        warnings.append("sample_qc_flags.csv not in template; flagged samples were retained")
    name = root.name
    if isinstance(manifest.get("dataset"), str):
        name = str(manifest["dataset"]).split()[0]
    return BioMasterDataset(
        source=root,
        name=name,
        metadata=metadata,
        proteomics=proteomics,
        metabolomics=metabolomics,
        protein_annotations=protein_ann,
        metabolite_annotations=metabolite_ann,
        manifest=manifest,
        report=report,
        already_logged=already_log_transformed(manifest_text + "\n" + report),
        warnings=warnings,
    )
