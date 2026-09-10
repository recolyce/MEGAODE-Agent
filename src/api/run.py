"""CLI for the LangGraph last_interval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.api.graph import run_pipeline
from src.models.registry import DEFAULT_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curator → prior → models via LangGraph")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--unit", default="")
    parser.add_argument("--direction", default="multimodal", help="multimodal joint forecast (both omics in and out)")
    parser.add_argument("--models", default="")
    parser.add_argument("--n-trials", type=int, default=15, help="Optuna trials per model on fold 1")
    parser.add_argument("--prior", default="auto")
    parser.add_argument("--prior-sources", default="", help="comma list: name_rule,kegg_reactome,string,pretrained")
    parser.add_argument("--organism", default="")
    parser.add_argument("--n-hv", type=int, default=512)
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument("--no-codegen", action="store_true")
    args = parser.parse_args()
    state = {
        "source": str(args.source),
        "artifacts": str(args.artifacts),
        "task": "last_interval",
        "models": [name.strip() for name in (args.models or ",".join(DEFAULT_MODELS)).split(",") if name.strip()],
        "n_trials": int(args.n_trials),
        "prior_backend": args.prior,
        "n_hv": args.n_hv,
        "run_attribution": not args.no_attribution,
        "run_codegen": not args.no_codegen,
    }
    if args.prior_sources:
        state["prior_sources"] = [part.strip() for part in args.prior_sources.split(",") if part.strip()]
    if args.unit:
        state["unit"] = args.unit
    state["direction"] = "multimodal"
    if args.organism:
        state["organism"] = args.organism
    final = run_pipeline(state)
    report = final.get("report") or json.dumps(final, ensure_ascii=False, default=str, indent=2)
    print(report)
    if final.get("report_path"):
        print(f"\nreport file: {final['report_path']}", flush=True)
    metrics = final.get("metrics") or []
    if metrics:
        print("\nmetrics:", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
