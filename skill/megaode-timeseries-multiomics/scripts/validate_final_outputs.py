#!/usr/bin/env python3
"""Validate BioMaster final MEGAODE interpretation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_CLASSIFICATIONS = {
    "known_supported",
    "potential_discovery",
    "uncertain",
    "not_literature_reviewed",
}
REQUIRED_CSV_COLUMNS = (
    "dataset",
    "unit",
    "source_modality",
    "source_id",
    "source_name",
    "target_modality",
    "target_id",
    "target_name",
    "direction",
    "models_supporting",
    "raw_attribution_summary",
    "attribution_score",
    "model_consistency_score",
    "context_score",
    "direction_consistency_score",
    "stability_score",
    "experimental_priority_score",
    "experimental_priority_rank",
    "experimental_priority_percentile",
    "selected_for_literature_validation",
    "literature_evidence_level",
    "classification",
    "notes",
)
CAUSALITY_PATTERNS = (
    "attribution ≠ biological causality",
    "attribution != biological causality",
    "attribution ≠ causality",
    "attribution != causality",
    "不自动代表真实生物学因果",
)


def _na(value: object) -> bool:
    text = str(value).strip().upper()
    return text in {"", "NA", "NAN", "NONE", "NULL"}


def _find_outputs(final_dir: Path, dataset: str | None) -> tuple[Path | None, Path | None, Path | None]:
    csvs = sorted(final_dir.glob("*_cross_modal_pair_scores.csv"))
    lits = sorted(final_dir.glob("*_high_score_literature_and_discoveries.md"))
    reports = sorted(final_dir.glob("*_cross_modal_biological_report.md"))
    if dataset:
        csv = final_dir / f"{dataset}_cross_modal_pair_scores.csv"
        lit = final_dir / f"{dataset}_high_score_literature_and_discoveries.md"
        report = final_dir / f"{dataset}_cross_modal_biological_report.md"
        return (
            csv if csv.exists() else (csvs[0] if csvs else None),
            lit if lit.exists() else (lits[0] if lits else None),
            report if report.exists() else (reports[0] if reports else None),
        )
    return (
        csvs[0] if csvs else None,
        lits[0] if lits else None,
        reports[0] if reports else None,
    )


def validate(final_dir: Path, dataset: str | None = None) -> dict:
    errors: list[str] = []
    csv_path, lit_path, report_path = _find_outputs(final_dir, dataset)
    if csv_path is None:
        errors.append("missing <dataset>_cross_modal_pair_scores.csv")
    if lit_path is None:
        errors.append("missing <dataset>_high_score_literature_and_discoveries.md")
    if report_path is None:
        errors.append("missing <dataset>_cross_modal_biological_report.md")
    if errors:
        return {"ok": False, "errors": errors}

    scores = pd.read_csv(csv_path)
    if not len(scores):
        errors.append("cross_modal_pair_scores.csv is empty")
    missing_cols = [col for col in REQUIRED_CSV_COLUMNS if col not in scores.columns]
    if missing_cols:
        errors.append(f"pair scores missing columns: {missing_cols}")

    if "source_modality" in scores.columns and "target_modality" in scores.columns:
        same = scores.loc[scores["source_modality"].astype(str) == scores["target_modality"].astype(str)]
        if len(same):
            errors.append(f"same-modality pairs are not allowed: {len(same)}")

    if "experimental_priority_score" in scores.columns:
        numeric = pd.to_numeric(scores["experimental_priority_score"], errors="coerce")
        if numeric.isna().any():
            errors.append("experimental_priority_score must be numeric in 0–100")
        elif float(numeric.min()) < 0 or float(numeric.max()) > 100:
            errors.append("experimental_priority_score is outside 0–100")

    if "experimental_priority_rank" in scores.columns and len(scores):
        ranks = pd.to_numeric(scores["experimental_priority_rank"], errors="coerce")
        if ranks.isna().any():
            errors.append("experimental_priority_rank contains non-numeric values")
        else:
            expected = set(range(1, len(scores) + 1))
            actual = set(int(v) for v in ranks.tolist())
            if actual != expected:
                errors.append("experimental_priority_rank has duplicates or gaps")

    if "classification" in scores.columns:
        bad = sorted(set(scores["classification"].astype(str)) - REQUIRED_CLASSIFICATIONS)
        if bad:
            errors.append(f"illegal classification values: {bad}")

    if {"classification", "literature_evidence_level"} <= set(scores.columns):
        unread = scores.loc[scores["classification"].astype(str) == "not_literature_reviewed"]
        if len(unread):
            bad_level = unread.loc[~unread["literature_evidence_level"].map(_na)]
            if len(bad_level):
                errors.append("not_literature_reviewed pairs must have literature_evidence_level == NA")

    lit = lit_path.read_text(encoding="utf-8").strip()
    report = report_path.read_text(encoding="utf-8").strip()
    if not lit:
        errors.append("high_score_literature_and_discoveries.md is empty")
    if not report:
        errors.append("cross_modal_biological_report.md is empty")
    if report and not any(token in report for token in CAUSALITY_PATTERNS):
        errors.append("biological report must state that attribution ≠ causality")

    if errors:
        return {"ok": False, "errors": errors}
    return {
        "ok": True,
        "csv": str(csv_path),
        "literature_md": str(lit_path),
        "report_md": str(report_path),
        "n_pairs": int(len(scores)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final BioMaster outputs")
    parser.add_argument("final_dir", type=Path)
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()
    if not args.final_dir.exists():
        print(json.dumps({"ok": False, "errors": [f"final dir not found: {args.final_dir}"]}, ensure_ascii=False))
        return 1
    result = validate(args.final_dir, args.dataset or None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
