"""Smoke: gm-aging last_interval/liver with train_mean, last_value, ridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.curator.bundle import ModelingBundle
from src.curator.loader import load_biomaster
from src.curator.tasks import get_task
from src.models.registry import ModelRegistry
from src.prior.registry import PriorRegistry


def median_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        if np.std(a) < 1e-12:
            continue
        if np.std(b) < 1e-12:
            scores.append(0.0)
            continue
        scores.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.nanmedian(scores)) if scores else float("nan")


def ensure_bundle(source: Path, artifacts: Path, unit: str, direction: str, n_hv: int) -> ModelingBundle:
    dest = artifacts / Path(source).name / "last_interval" / unit / direction
    # dataset name may differ from folder name
    if (dest / "bundle.json").exists():
        return ModelingBundle.load(dest)
    data = load_biomaster(source)
    dest = artifacts / data.name / "last_interval" / unit / direction
    if (dest / "bundle.json").exists():
        return ModelingBundle.load(dest)
    bundles = get_task("last_interval")(
        data, unit=unit, n_high_variance=n_hv, directions=(direction,)
    )
    bundle = bundles[0]
    bundle.save(dest)
    return bundle


def run(source: Path, artifacts: Path, unit: str, direction: str, n_hv: int) -> pd.DataFrame:
    bundle = ensure_bundle(source, artifacts, unit, direction, n_hv)
    data = load_biomaster(source)
    rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
    prior = PriorRegistry().build(
        data, bundle.x_features, rel, backend="name_rule", n_context=len(bundle.context_names)
    )
    bundle.attach_prior(prior)
    PriorRegistry().save(prior, artifacts / bundle.dataset / "priors" / prior.name)
    registry = ModelRegistry()
    rows = []
    for name in ("train_mean", "last_value", "ridge"):
        model = registry.build(name)
        model.fit(bundle)
        pred = model.predict(bundle)
        pcc = median_pcc(bundle.test.Y, pred)
        rows.append(
            {
                "dataset": bundle.dataset,
                "task": bundle.task,
                "unit": bundle.unit,
                "direction": bundle.direction,
                "lag_mode": bundle.lag_mode,
                "model": name,
                "median_pcc": pcc,
                "n_test": int(bundle.test.X.shape[0]),
                **model.params(),
            }
        )
        print(f"  {name}: median PCC={pcc:.4f}", flush=True)
    table = pd.DataFrame(rows)
    out = artifacts / bundle.dataset / bundle.task / bundle.unit / "smoke_metrics.csv"
    table.to_csv(out, index=False)
    print("wrote", out, flush=True)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test last_interval models")
    parser.add_argument("--source", type=Path, default=Path("/personal/data/biomaster-processed/gm-aging"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--unit", default="liver")
    parser.add_argument("--direction", default="proteomics_to_metabolomics")
    parser.add_argument("--n-hv", type=int, default=512)
    args = parser.parse_args()
    run(args.source, args.artifacts, args.unit, args.direction, args.n_hv)


if __name__ == "__main__":
    main()
