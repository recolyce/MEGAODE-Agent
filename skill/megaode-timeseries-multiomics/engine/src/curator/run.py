"""CLI: BioMaster template → last_interval ModelingBundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.curator.loader import load_biomaster
from src.curator.tasks import get_task


def run(
    source: Path,
    task: str,
    artifacts: Path,
    units: list[str] | None,
    n_hv: int,
    directions: list[str] | None,
) -> list[Path]:
    data = load_biomaster(source)
    build = get_task(task)
    if units:
        chosen = units
    elif data.tissues:
        chosen = data.tissues
    else:
        chosen = ["all"]
    written: list[Path] = []
    for unit in chosen:
        print(f"curate {data.name} {task} unit={unit}", flush=True)
        bundles = build(
            data,
            unit=unit,
            n_high_variance=n_hv,
            directions=tuple(directions) if directions else None,
        )
        for bundle in bundles:
            dest = artifacts / bundle.dataset / bundle.task / bundle.unit / bundle.direction
            bundle.save(dest)
            written.append(dest)
            print(
                f"  {bundle.direction} lag={bundle.lag_mode} "
                f"train={len(bundle.train.pairs)} val={len(bundle.val.pairs)} "
                f"test={len(bundle.test.pairs)} x={bundle.n_expr} y={len(bundle.y_features)}",
                flush=True,
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate a BioMaster template for modeling")
    parser.add_argument("--source", type=Path, default=Path("/personal/data/biomaster-processed/gm-aging"))
    parser.add_argument("--task", default="last_interval")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--unit", action="append", dest="units")
    parser.add_argument("--n-hv", type=int, default=512)
    parser.add_argument("--direction", action="append", dest="directions")
    args = parser.parse_args()
    run(args.source, args.task, args.artifacts, args.units, args.n_hv, args.directions)


if __name__ == "__main__":
    main()
