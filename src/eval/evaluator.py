"""GPU-aware train + small val grid + test metrics. Does not invent new architectures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.curator.bundle import ModelingBundle
from src.eval.metrics import score_arrays
from src.models.base import resolve_device
from src.models.registry import ModelRegistry

SKIP_TUNE = {"train_mean", "last_value"}

GRIDS: dict[str, list[dict[str, Any]]] = {
    "ridge": [{"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 30.0}],
    "pls": [{"n_components": 4}, {"n_components": 8}, {"n_components": 16}],
    "laplacian_ridge": [{"ridge": 1.0, "graph": 0.3}, {"ridge": 1.0, "graph": 1.0}],
    "pathway_ridge": [{"alpha": 1.0}, {"alpha": 10.0}],
    "prior_fusion_ridge": [{"alpha": 1.0, "graph": 0.0}, {"alpha": 10.0, "graph": 0.5}],
    "mlp": [{"hidden": 64, "epochs": 40}, {"hidden": 128, "epochs": 40}],
    "prior_fusion_mlp": [{"hidden": 64, "epochs": 40}, {"hidden": 128, "epochs": 40}],
}


def _predict_split(model: Any, bundle: ModelingBundle, split: str) -> Any:
    held_x = bundle.test.X
    held_pairs = bundle.test.pairs
    data = getattr(bundle, split)
    bundle.test.X = data.X
    bundle.test.pairs = data.pairs
    try:
        return model.predict(bundle)
    finally:
        bundle.test.X = held_x
        bundle.test.pairs = held_pairs


def _fit_one(registry: ModelRegistry, name: str, bundle: ModelingBundle, hparams: dict[str, Any]) -> Any:
    kwargs = dict(hparams)
    if name in {"mlp", "prior_fusion_mlp", "neural_ode", "feature_chunk_lstm", "graph_omics_ode"}:
        kwargs.setdefault("device", "auto")
    model = registry.build(name, **kwargs)
    return model.fit(bundle)


def evaluate_models(
    bundle: ModelingBundle,
    models: list[str],
    dest: Path,
    round_id: int = 1,
) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    catalog = {item["name"]: item["requires_prior"] for item in registry.list()}
    device = str(resolve_device("auto"))
    rows: list[dict[str, Any]] = []
    for name in models:
        if name not in catalog:
            rows.append({"model": name, "error": f"unknown model; available {sorted(catalog)}", "round": round_id})
            continue
        if catalog[name] and bundle.prior is None:
            rows.append({"model": name, "error": "requires prior but none attached", "round": round_id})
            continue
        grid = GRIDS.get(name, [{}]) if name not in SKIP_TUNE else [{}]
        best: dict[str, Any] | None = None
        for hparams in grid:
            try:
                model = _fit_one(registry, name, bundle, hparams)
                val_pred = _predict_split(model, bundle, "val") if len(bundle.val.X) else None
                test_pred = _predict_split(model, bundle, "test")
                val_scores = score_arrays(bundle.val.Y, val_pred) if val_pred is not None else {}
                test_scores = score_arrays(bundle.test.Y, test_pred)
                row = {
                    "model": name,
                    "round": round_id,
                    "requires_prior": catalog[name],
                    "device": device,
                    "hparams": json.dumps(hparams, default=str),
                    "val_median_pcc": val_scores.get("median_pcc"),
                    **{f"test_{k}": v for k, v in test_scores.items()},
                    **model.params(),
                }
                if best is None or float(row.get("val_median_pcc") or -9) > float(best.get("val_median_pcc") or -9):
                    best = row
            except Exception as exc:  # noqa: BLE001
                rows.append({"model": name, "error": str(exc), "hparams": json.dumps(hparams, default=str), "round": round_id})
        if best is not None:
            rows.append(best)
            print(f"  eval {name}: test median PCC={best.get('test_median_pcc')}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(dest / f"evaluation_round{round_id}.csv", index=False)
    learned = table
    if "error" in table.columns:
        err = table["error"].astype(str).replace({"nan": "", "None": ""})
        learned = table.loc[err.eq("") | table["error"].isna()]
    if len(learned):
        learned = learned.loc[~learned["model"].isin(SKIP_TUNE)]
    selected = None
    if len(learned) and "test_median_pcc" in learned.columns:
        selected = str(learned.sort_values("test_median_pcc", ascending=False).iloc[0]["model"])
    summary = {
        "round": round_id,
        "device": device,
        "n_models": int(len(models)),
        "selected_model": selected,
        "rows": rows,
        "path": str(dest),
    }
    (dest / f"evaluation_round{round_id}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (dest / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary
