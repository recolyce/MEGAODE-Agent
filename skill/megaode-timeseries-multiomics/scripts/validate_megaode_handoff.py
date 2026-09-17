#!/usr/bin/env python3
"""Validate MEGAODE → BioMaster handoff/ before interpretation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_FILES = (
    "megaode_run_summary.json",
    "model_metrics.csv",
    "cross_modal_contributions.csv",
    "generated_models.csv",
    "protein_annotations.csv",
    "metabolite_annotations.csv",
    "handoff_manifest.yaml",
)


def validate(handoff: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (handoff / name).exists()]
    if missing:
        return {"ok": False, "errors": [f"missing handoff files: {missing}"]}

    summary = json.loads((handoff / "megaode_run_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(handoff / "model_metrics.csv")
    contrib = pd.read_csv(handoff / "cross_modal_contributions.csv")

    if not len(metrics):
        errors.append("model_metrics.csv is empty")
    required_metric = {"dataset", "unit", "direction", "model", "model_role", "selected_model"}
    absent = sorted(required_metric - set(metrics.columns))
    if absent:
        errors.append(f"model_metrics.csv missing columns: {absent}")
    invented = [col for col in ("p-value", "pvalue", "confidence_interval", "replicate", "fold_metric") if col in metrics.columns]
    if invented:
        warnings.append(f"unexpected inferential columns present: {invented}")

    if not len(contrib):
        errors.append("cross_modal_contributions.csv is empty")
    else:
        need = {"source_modality", "target_modality", "source_feature", "target_feature", "method", "score"}
        absent = sorted(need - set(contrib.columns))
        if absent:
            errors.append(f"cross_modal_contributions.csv missing columns: {absent}")
        else:
            same = contrib.loc[contrib["source_modality"].astype(str) == contrib["target_modality"].astype(str)]
            if len(same):
                errors.append(f"same-modality pairs leaked into handoff: {len(same)}")

    if not summary.get("ok"):
        errors.append("megaode_run_summary.json ok is not true")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    return {
        "ok": True,
        "dataset": summary.get("dataset"),
        "n_metric_rows": int(len(metrics)),
        "n_cross_modal_rows": int(len(contrib)),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate handoff/")
    parser.add_argument("handoff", type=Path)
    args = parser.parse_args()
    if not args.handoff.exists():
        print(json.dumps({"ok": False, "errors": [f"handoff not found: {args.handoff}"]}, ensure_ascii=False))
        return 1
    result = validate(args.handoff)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
