"""CLI: attribute selected models on an existing ModelingBundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.interpretation.contribution import compute_contributions


def main() -> None:
    parser = argparse.ArgumentParser(description="Write BioMaster attribution score tables")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--models", default="", help="default: selected_model from evaluation.json")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--prior", default="auto")
    parser.add_argument("--organism", default="human")
    parser.add_argument("--prior-sources", default="")
    args = parser.parse_args()
    names = [part.strip() for part in args.models.split(",") if part.strip()]
    sources = [part.strip() for part in args.prior_sources.split(",") if part.strip()] or None
    summary = compute_contributions(
        str(args.source),
        str(args.bundle),
        names,
        artifacts=str(args.artifacts),
        prior_backend=args.prior,
        organism=args.organism,
        prior_sources=sources,
        selected_only=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
