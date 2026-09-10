"""Cross-modal feature contribution for BioMaster. Attribution + permutation (+ SHAP if installed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.curator.bundle import ModelingBundle
from src.curator.loader import load_biomaster
from src.interpretation.attribution import (
    SKIP_MODELS,
    _display_maps,
    _expr_coef,
    _finite_difference,
    _occlusion,
    _predict_x,
    _prior_pairs,
    _row_normalize,
    attribute_models,
)
from src.models.registry import ModelRegistry
from src.prior.registry import PriorRegistry


def _permutation(model: Any, bundle: ModelingBundle, seed: int) -> np.ndarray:
    x = np.asarray(bundle.test.X, dtype=float)
    pred0 = _predict_x(model, bundle, x)
    n_expr = bundle.n_expr
    rng = np.random.default_rng(seed)
    out = np.zeros((pred0.shape[1], n_expr), dtype=float)
    for i in range(n_expr):
        xp = x.copy()
        xp[:, i] = rng.permutation(xp[:, i])
        pred = _predict_x(model, bundle, xp)
        out[:, i] = np.mean(np.abs(pred - pred0), axis=0)
    return _row_normalize(out)


def _try_shap(model: Any, bundle: ModelingBundle, seed: int) -> np.ndarray | None:
    """Linear SHAP only. KernelExplainer is unusable for 100+ targets."""
    try:
        import shap  # noqa: F401 — required so the method is actually SHAP
    except ImportError:
        return None
    x = np.asarray(bundle.test.X, dtype=float)[:, : bundle.n_expr]
    n_y = bundle.test.Y.shape[1]
    inner = getattr(model, "model_", None)
    if inner is not None and hasattr(inner, "coef_"):
        try:
            explainer = shap.LinearExplainer(inner, x, feature_perturbation="interventional")
            values = explainer.shap_values(x)
            arr = np.asarray(values)
            if isinstance(values, list):
                arr = np.stack([np.abs(np.asarray(v)) for v in values], axis=0)
                mag = arr.mean(axis=1)
            elif arr.ndim == 3:
                mag = np.mean(np.abs(arr), axis=1)
            else:
                mag = np.mean(np.abs(arr), axis=0, keepdims=True)
            if mag.shape[0] != n_y and mag.shape[-1] == n_y:
                mag = np.swapaxes(mag, 0, -1)
            if mag.shape != (n_y, bundle.n_expr):
                raise ValueError("shap shape")
            return _row_normalize(np.asarray(mag, dtype=float))
        except Exception:
            pass
    coef = _expr_coef(model, n_y, bundle.n_expr)
    if coef is None:
        return None
    xc = x - x.mean(axis=0, keepdims=True)
    mag = np.mean(np.abs(xc[None, :, :] * coef[:, None, :]), axis=1)
    return _row_normalize(np.asarray(mag, dtype=float))


def _attach_prior(source: str, bundle: ModelingBundle, artifacts: str, prior_backend: str, organism: str, sources: list[str] | None):
    data = load_biomaster(Path(source))
    rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
    prior = PriorRegistry().build(
        data,
        bundle.x_features,
        rel,
        backend=prior_backend,
        sources=sources,
        n_context=len(bundle.context_names),
        organism=organism,
        y_features=bundle.y_features,
    )
    bundle.attach_prior(prior)
    return data, prior


def compute_contributions(
    source: str,
    bundle_path: str,
    models: list[str],
    artifacts: str = "artifacts",
    prior_backend: str = "name_rule",
    organism: str = "human",
    prior_sources: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    bundle = ModelingBundle.load(Path(bundle_path))
    data, prior = _attach_prior(source, bundle, artifacts, prior_backend, organism, prior_sources)
    protein_name, met_name = _display_maps(data.protein_annotations, data.metabolite_annotations)
    prior_scores = _prior_pairs(prior.network)
    dest = Path(bundle_path) / "contribution"
    dest.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    names = [name for name in models if name not in SKIP_MODELS]
    records: list[dict[str, Any]] = []
    methods_used = ["coefficient", "occlusion", "gradient", "permutation"]
    shap_ok = False
    for name in names:
        try:
            model = registry.build(name)
            model.fit(bundle)
        except Exception as exc:  # noqa: BLE001
            records.append({"model": name, "error": str(exc)})
            continue
        coef = _expr_coef(model, bundle.test.Y.shape[1], bundle.n_expr)
        occ = _occlusion(model, bundle, seed)
        grad, _signed = _finite_difference(model, bundle)
        perm = _permutation(model, bundle, seed + 1)
        shap_mag = _try_shap(model, bundle, seed + 2)
        maps = {
            "coefficient": None if coef is None else _row_normalize(np.abs(coef)),
            "occlusion": occ,
            "gradient": grad,
            "permutation": perm,
        }
        if shap_mag is not None:
            maps["shap"] = shap_mag
            shap_ok = True
        for j, target in enumerate(bundle.y_features):
            for i, src in enumerate(bundle.x_features):
                src_name = protein_name.get(src, src) if bundle.x_modality == "proteomics" else met_name.get(src, src)
                tgt_name = met_name.get(target, target) if bundle.y_modality == "metabolomics" else protein_name.get(target, target)
                p = float(prior_scores.get((src, target), 0.0))
                for method, mat in maps.items():
                    if mat is None:
                        continue
                    records.append(
                        {
                            "model": name,
                            "source_feature": src,
                            "target_feature": target,
                            "source_name": src_name,
                            "target_name": tgt_name,
                            "source_modality": bundle.x_modality,
                            "target_modality": bundle.y_modality,
                            "method": method,
                            "score": float(mat[j, i]),
                            "prior": p,
                        }
                    )
    if shap_ok:
        methods_used.append("shap")
    table = pd.DataFrame.from_records(records)
    table.to_csv(dest / "contribution_pairs.csv", index=False)
    attr = {}
    if names:
        attr = attribute_models(
            source,
            bundle_path,
            names,
            artifacts=artifacts,
            prior_backend=prior_backend,
            organism=organism,
            prior_sources=prior_sources,
        )
    top = []
    if len(table) and "method" in table.columns and "score" in table.columns:
        ranked = table.loc[table["method"] == "permutation"]
        if len(ranked):
            top = (
                ranked.sort_values("score", ascending=False)
                .head(15)[["model", "source_name", "target_name", "score"]]
                .to_dict("records")
            )
    handoff = {
        "dataset": bundle.dataset,
        "task": bundle.task,
        "unit": bundle.unit,
        "direction": bundle.direction,
        "lag_mode": bundle.lag_mode,
        "models": names,
        "methods": methods_used,
        "n_rows": int(len(table)),
        "contribution_csv": str(dest / "contribution_pairs.csv"),
        "attribution_dir": str(Path(bundle_path) / "attribution"),
        "schema": {
            "grain": "one row = model × source feature × target feature × method",
            "score": "row-normalized magnitude within each target",
            "intended_consumer": "BioMaster",
        },
        "top_permutation": top,
        "attribution": {k: attr.get(k) for k in ("n_pairs", "class_counts", "n_known_A", "n_novel_C", "path")},
    }
    (dest / "biomaster_handoff.json").write_text(json.dumps(handoff, indent=2, default=str) + "\n", encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(handoff, indent=2, default=str) + "\n", encoding="utf-8")
    return handoff
