"""Smoke: last_interval multimodal bundle with mlp."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.curator.bundle import ModelingBundle
from src.curator.loader import load_biomaster
from src.curator.tasks import get_task
from src.eval.evaluator import evaluate_models
from src.prior.registry import PriorRegistry


def ensure_bundle(source: Path, artifacts: Path, unit: str, n_hv: int) -> ModelingBundle:
    data = load_biomaster(source)
    dest = artifacts / data.name / "last_interval" / unit / "multimodal"
    if (dest / "bundle.json").exists():
        return ModelingBundle.load(dest)
    bundles = get_task("last_interval")(data, unit=unit, n_high_variance=n_hv, directions=("multimodal",))
    bundle = bundles[0]
    bundle.save(dest)
    return bundle


def run(source: Path, artifacts: Path, unit: str, n_hv: int, n_trials: int) -> pd.DataFrame:
    bundle = ensure_bundle(source, artifacts, unit, n_hv)
    data = load_biomaster(source)
    prior = PriorRegistry().build(
        data,
        bundle.x_features,
        bundle.prior_relationship(),
        backend="name_rule",
        n_context=len(bundle.context_names),
        y_features=bundle.y_features,
    )
    bundle.attach_prior(prior)
    dest = artifacts / bundle.dataset / bundle.task / bundle.unit / bundle.direction / "evaluation"
    summary = evaluate_models(bundle, ["mlp"], dest, n_trials=n_trials)
    print(json_summary(summary), flush=True)
    return pd.DataFrame(summary.get("rows") or [])


def json_summary(summary: dict) -> str:
    import json

    return json.dumps(
        {
            "selected": summary.get("selected_model"),
            "excel": summary.get("excel"),
            "n_folds": summary.get("n_folds"),
            "cv": summary.get("cv"),
        },
        ensure_ascii=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test multimodal last_interval + Optuna")
    parser.add_argument("--source", type=Path, default=Path("/personal/data/biomaster-processed/ipop_exercise"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--unit", default="all")
    parser.add_argument("--n-hv", type=int, default=512)
    parser.add_argument("--n-trials", type=int, default=2)
    args = parser.parse_args()
    run(args.source, args.artifacts, args.unit, args.n_hv, args.n_trials)


if __name__ == "__main__":
    main()
