#!/usr/bin/env python3
"""Collect unit/direction MEGAODE artifacts into a BioMaster handoff/ folder."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _model_role(name: str, generated_names: set[str]) -> str:
    if name in generated_names:
        return "generated"
    return "library_learned"


def _dataset_from_processed(processed_data: Path | None) -> str:
    if processed_data is None or not processed_data.exists():
        return ""
    manifest_path = processed_data / "dataset_manifest.yaml"
    if manifest_path.exists() and yaml is not None:
        loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for key in ("dataset_name", "name", "dataset"):
                if loaded.get(key):
                    return str(loaded[key])
    return processed_data.name if processed_data.name != "processed_data" else processed_data.parent.name


def _find_units(artifacts: Path) -> list[Path]:
    roots: list[Path] = []
    for evaluation in artifacts.rglob("evaluation/evaluation.json"):
        roots.append(evaluation.parents[1])
    return sorted(set(roots))


def _generated_names(generated_dir: Path | None, extra: list[str]) -> set[str]:
    names = {name for name in extra if name}
    if generated_dir and generated_dir.exists():
        names.update(path.stem for path in generated_dir.glob("*.py") if not path.name.startswith("_"))
    return names


def collect(
    artifacts: Path,
    handoff: Path,
    processed_data: Path | None = None,
    generated_dir: Path | None = None,
) -> dict[str, Any]:
    handoff.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_from_processed(processed_data)
    metric_rows: list[dict[str, Any]] = []
    contribution_frames: list[pd.DataFrame] = []
    generated_rows: list[dict[str, Any]] = []
    unit_summaries: list[dict[str, Any]] = []
    discovered_generated: list[str] = []

    for bundle_root in _find_units(artifacts):
        rel = bundle_root.relative_to(artifacts)
        parts = list(rel.parts)
        unit_dataset = parts[0] if parts else dataset
        dataset = dataset or unit_dataset
        task = parts[1] if len(parts) > 1 else "last_interval"
        unit = parts[2] if len(parts) > 2 else bundle_root.name
        direction = parts[3] if len(parts) > 3 else ""

        evaluation = _read_json(bundle_root / "evaluation" / "evaluation.json")
        handoff_json = _read_json(bundle_root / "contribution" / "biomaster_handoff.json")
        prior = _read_json(bundle_root / "prior_summary.json")
        if not prior:
            prior_hits = list((artifacts / unit_dataset / "priors").glob("*/prior_summary.json")) if unit_dataset else []
            if prior_hits:
                prior = _read_json(prior_hits[0])
        selected = str(evaluation.get("selected_model") or "")
        generated_name = str((handoff_json.get("generated_model") if handoff_json else "") or "")
        for row in evaluation.get("rows") or []:
            name = str(row.get("model") or "")
            role = str(row.get("model_role") or "")
            if role == "generated" or (name and name not in {
                "mlp",
                "neural_ode",
                "feature_chunk_lstm",
                "cross_attn_fusion",
                "gated_fusion",
                "koopman_ae",
                "dual_lstm",
                "mmvae_forecast",
                "mogonet_fusion",
                "graph_omics_ode",
                "prior_fusion_mlp",
            }):
                discovered_generated.append(name)
            metric_rows.append(
                {
                    "dataset": dataset or unit_dataset,
                    "unit": unit,
                    "direction": direction or str(handoff_json.get("direction") or ""),
                    "task": task,
                    "model": name,
                    "model_role": role or _model_role(name, set()),
                    "test_median_pcc": row.get("test_median_pcc"),
                    "val_median_pcc": row.get("val_median_pcc"),
                    "optuna_val_median_pcc": row.get("optuna_val_median_pcc"),
                    "test_protein_median_pcc": row.get("test_protein_median_pcc"),
                    "test_metabolite_median_pcc": row.get("test_metabolite_median_pcc"),
                    "selected_model": selected,
                    "error": row.get("error") or "",
                }
            )
        pair_csv = bundle_root / "contribution" / "contribution_pairs.csv"
        if pair_csv.exists():
            table = pd.read_csv(pair_csv)
            if len(table):
                table["dataset"] = dataset or unit_dataset
                table["unit"] = unit
                table["direction"] = table["direction"] if "direction" in table.columns else (direction or "")
                contribution_frames.append(table)
        generated_rows.append(
            {
                "dataset": dataset or unit_dataset,
                "unit": unit,
                "direction": direction,
                "selected_model": selected,
                "generated_model": generated_name,
                "methods": ",".join(handoff_json.get("methods") or []),
                "interpretation_md": str(bundle_root / "interpretation.md") if (bundle_root / "interpretation.md").exists() else "",
                "report_md": str(bundle_root / "report.md") if (bundle_root / "report.md").exists() else "",
                "prior_path": prior.get("path") or "",
                "prior_sources": ",".join(prior.get("sources") or []),
                "prior_n_edges": prior.get("n_edges"),
            }
        )
        unit_summaries.append(
            {
                "dataset": dataset or unit_dataset,
                "unit": unit,
                "direction": direction,
                "task": task,
                "selected_model": selected,
                "evaluation": str(bundle_root / "evaluation" / "evaluation.json"),
                "contribution_pairs": str(pair_csv) if pair_csv.exists() else "",
                "method_scores": str(bundle_root / "contribution" / "method_scores.csv")
                if (bundle_root / "contribution" / "method_scores.csv").exists()
                else "",
                "biomaster_handoff": str(bundle_root / "contribution" / "biomaster_handoff.json")
                if (bundle_root / "contribution" / "biomaster_handoff.json").exists()
                else "",
                "interpretation_md": _read_text(bundle_root / "interpretation.md")[:2000],
                "report_md": _read_text(bundle_root / "report.md")[:2000],
                "prior": prior,
            }
        )

    generated_names = _generated_names(generated_dir, discovered_generated)
    metrics = pd.DataFrame(metric_rows)
    if len(metrics):
        metrics["model_role"] = [
            "generated" if str(row.model) in generated_names else (row.model_role or "library_learned")
            for row in metrics.itertuples()
        ]
        metrics.to_csv(handoff / "model_metrics.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "dataset",
                "unit",
                "direction",
                "model",
                "model_role",
                "test_median_pcc",
                "selected_model",
                "error",
            ]
        ).to_csv(handoff / "model_metrics.csv", index=False)

    if contribution_frames:
        all_pairs = pd.concat(contribution_frames, ignore_index=True)
        if "source_modality" in all_pairs.columns and "target_modality" in all_pairs.columns:
            cross = all_pairs.loc[all_pairs["source_modality"].astype(str) != all_pairs["target_modality"].astype(str)].copy()
        else:
            cross = all_pairs
        keep = [
            col
            for col in (
                "dataset",
                "unit",
                "direction",
                "model",
                "source_feature",
                "source_name",
                "source_modality",
                "target_feature",
                "target_name",
                "target_modality",
                "method",
                "score",
                "signed_score",
                "signed_gradient",
                "signed_coefficient",
                "prior",
            )
            if col in cross.columns
        ]
        cross[keep].to_csv(handoff / "cross_modal_contributions.csv", index=False)
        n_cross = int(len(cross))
    else:
        pd.DataFrame(
            columns=[
                "dataset",
                "unit",
                "direction",
                "model",
                "source_feature",
                "source_name",
                "source_modality",
                "target_feature",
                "target_name",
                "target_modality",
                "method",
                "score",
                "prior",
            ]
        ).to_csv(handoff / "cross_modal_contributions.csv", index=False)
        n_cross = 0

    pd.DataFrame(generated_rows).to_csv(handoff / "generated_models.csv", index=False)

    if processed_data is not None:
        for name in ("protein_annotations.csv", "metabolite_annotations.csv"):
            src = processed_data / name
            if src.exists():
                shutil.copy2(src, handoff / name)
    if not (handoff / "protein_annotations.csv").exists():
        pd.DataFrame().to_csv(handoff / "protein_annotations.csv", index=False)
    if not (handoff / "metabolite_annotations.csv").exists():
        pd.DataFrame().to_csv(handoff / "metabolite_annotations.csv", index=False)

    summary = {
        "dataset": dataset,
        "task": "last_interval",
        "direction": "multimodal",
        "n_units": len(unit_summaries),
        "n_metric_rows": int(len(metrics)),
        "n_cross_modal_rows": n_cross,
        "generated_models": sorted(generated_names),
        "units": unit_summaries,
        "ok": bool(unit_summaries),
    }
    (handoff / "megaode_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": dataset,
        "task": "last_interval",
        "files": [
            "megaode_run_summary.json",
            "model_metrics.csv",
            "cross_modal_contributions.csv",
            "generated_models.csv",
            "protein_annotations.csv",
            "metabolite_annotations.csv",
        ],
        "n_cross_modal_rows": n_cross,
        "n_metric_rows": int(len(metrics)),
        "note": "Handoff only. BioMaster computes Experimental Priority Score and literature grades later.",
    }
    dest = handoff / "handoff_manifest.yaml"
    if yaml is not None:
        dest.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MEGAODE artifacts into handoff/")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--processed-data", type=Path, default=None)
    parser.add_argument("--generated-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.artifacts.exists():
        print(json.dumps({"ok": False, "errors": [f"artifacts not found: {args.artifacts}"]}, ensure_ascii=False))
        return 1
    summary = collect(args.artifacts, args.handoff, args.processed_data, args.generated_dir)
    print(json.dumps({"ok": bool(summary.get("ok")), **summary}, ensure_ascii=False, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
