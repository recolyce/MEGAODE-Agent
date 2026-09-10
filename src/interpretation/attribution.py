"""Multi-method protein–metabolite pair attribution on a ModelingBundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.curator.bundle import ModelingBundle
from src.curator.loader import load_biomaster
from src.models.registry import ModelRegistry
from src.prior.registry import PriorRegistry

SKIP_MODELS = {"train_mean", "last_value"}
N_BOOT = 20
TOP_K = 64
TOP_PER_TARGET = 50
W_SC = {"A": 0.25, "S": 0.20, "D": 0.10, "R": 0.15, "T": 0.10, "P": 0.10, "L": 0.10}
W_DE = {"A": 0.35, "S": 0.25, "D": 0.15, "R": 0.15, "T": 0.10}


def _predict_x(model: Any, bundle: ModelingBundle, x: np.ndarray) -> np.ndarray:
    held = bundle.test.X
    bundle.test.X = x
    try:
        return np.asarray(model.predict(bundle), dtype=float)
    finally:
        bundle.test.X = held


def _row_normalize(mag: np.ndarray) -> np.ndarray:
    denom = mag.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return mag / denom


def _expr_coef(model: Any, n_y: int, n_expr: int) -> np.ndarray | None:
    coef = None
    inner = getattr(model, "model_", None)
    if inner is not None and hasattr(inner, "coef_"):
        coef = np.asarray(inner.coef_, dtype=float)
    elif getattr(model, "coef_", None) is not None:
        coef = np.asarray(model.coef_, dtype=float)
    if coef is None:
        return None
    if coef.shape == (n_expr + 3, n_y) or (coef.ndim == 2 and coef.shape[0] != n_y and coef.shape[1] == n_y):
        coef = coef.T
    if coef.ndim != 2 or coef.shape[0] != n_y:
        return None
    return coef[:, :n_expr]


def _occlusion(model: Any, bundle: ModelingBundle, seed: int) -> np.ndarray:
    x = np.asarray(bundle.test.X, dtype=float)
    pred0 = _predict_x(model, bundle, x)
    n_expr = bundle.n_expr
    n_y = pred0.shape[1]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_y, n_expr), dtype=float)
    for i in range(n_expr):
        xp = x.copy()
        xp[:, i] = rng.permutation(xp[:, i])
        pred = _predict_x(model, bundle, xp)
        out[:, i] = np.mean(np.abs(pred - pred0), axis=0)
    return _row_normalize(out)


def _finite_difference(model: Any, bundle: ModelingBundle) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(bundle.test.X, dtype=float)
    n_expr = bundle.n_expr
    pred0 = _predict_x(model, bundle, x)
    n_y = pred0.shape[1]
    mag = np.zeros((n_y, n_expr), dtype=float)
    signed = np.zeros((n_y, n_expr), dtype=float)
    for i in range(n_expr):
        std = float(np.std(x[:, i]))
        eps = 0.01 if std < 1e-8 else 0.01 * std
        up = x.copy()
        down = x.copy()
        up[:, i] += eps
        down[:, i] -= eps
        grad = (_predict_x(model, bundle, up) - _predict_x(model, bundle, down)) / (2.0 * eps)
        signed[:, i] = np.mean(grad, axis=0)
        mag[:, i] = np.mean(np.abs(grad), axis=0)
    return _row_normalize(mag), signed


def _perturbation_r(model: Any, bundle: ModelingBundle, signed: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(bundle.test.X, dtype=float)
    pred0 = _predict_x(model, bundle, x)
    rng = np.random.default_rng(seed)
    n_y, n_expr = signed.shape
    hits = np.zeros_like(signed)
    valid = np.zeros_like(signed)
    for i in range(n_expr):
        xp = x.copy()
        xp[:, i] = rng.permutation(xp[:, i])
        dx = x[:, i] - xp[:, i]
        usable = np.abs(dx) > 1e-12
        if not usable.any():
            continue
        dy = pred0 - _predict_x(model, bundle, xp)
        expected = np.sign(dx)[:, None] * np.sign(signed[:, i])[None, :]
        observed = np.sign(dy)
        agree = (expected == observed) & usable[:, None] & (np.sign(signed[:, i])[None, :] != 0)
        hits[:, i] = agree.sum(axis=0)
        valid[:, i] = usable.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = hits / np.where(valid > 0, valid, np.nan)
    return np.nan_to_num(out, nan=0.0)


def _bootstrap_linear(
    model: Any,
    bundle: ModelingBundle,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    coef0 = _expr_coef(model, bundle.train.Y.shape[1], bundle.n_expr)
    if coef0 is None:
        return None
    alpha = float(getattr(model, "alpha", 10.0))
    n_y, n_expr = coef0.shape
    rng = np.random.default_rng(seed)
    x = np.asarray(bundle.train.X, dtype=float)
    y = np.asarray(bundle.train.Y, dtype=float)
    boot = np.empty((n_boot, n_y, n_expr), dtype=float)
    from sklearn.linear_model import Ridge

    for b in range(n_boot):
        idx = rng.integers(0, x.shape[0], size=x.shape[0])
        fitted = Ridge(alpha=alpha).fit(x[idx], y[idx])
        coef = np.asarray(fitted.coef_, dtype=float)
        if coef.shape != (n_y, x.shape[1]):
            coef = coef.T
        boot[b] = coef[:, :n_expr]
    ranks = np.abs(boot).argsort(axis=2).argsort(axis=2)
    in_top = ranks >= (n_expr - min(TOP_K, n_expr))
    stability = in_top.mean(axis=0)
    main_sign = np.sign(coef0)
    directional = (np.sign(boot) == main_sign[None, :, :]).mean(axis=0)
    directional = np.where(main_sign == 0, 0.0, directional)
    return stability, directional


def _prior_pairs(network: pd.DataFrame) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    protein_paths: dict[str, set[str]] = {}
    met_paths: dict[str, set[str]] = {}
    for row in network.itertuples(index=False):
        rel = str(row.relationship)
        src, tgt = str(row.source_node), str(row.target_node)
        if rel == "enzyme_to_metabolite_reaction":
            scores[(src, tgt)] = max(scores.get((src, tgt), 0.0), 1.0)
            scores[(tgt, src)] = max(scores.get((tgt, src), 0.0), 1.0)
        elif rel == "protein_to_pathway":
            protein_paths.setdefault(src, set()).add(tgt)
        elif rel == "metabolite_to_pathway":
            met_paths.setdefault(src, set()).add(tgt)
        elif rel == "pathway_comembership":
            scores[(src, tgt)] = max(scores.get((src, tgt), 0.0), 0.5)
            scores[(tgt, src)] = max(scores.get((tgt, src), 0.0), 0.5)
    for protein, p_paths in protein_paths.items():
        for metabolite, m_paths in met_paths.items():
            if p_paths & m_paths:
                scores[(protein, metabolite)] = max(scores.get((protein, metabolite), 0.0), 0.5)
                scores[(metabolite, protein)] = max(scores.get((metabolite, protein), 0.0), 0.5)
    return scores


def _display_maps(protein_ann: pd.DataFrame, met_ann: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    def pick(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, str]:
        names = {}
        for rec in frame.to_dict("records"):
            feat = str(rec.get("feature_id") or "")
            label = ""
            for col in columns:
                value = rec.get(col)
                if value is not None and str(value) not in {"", "nan", "None"}:
                    label = str(value).split(";")[0]
                    break
            names[feat] = label or feat
        return names

    return (
        pick(protein_ann, ("gene_symbol", "gene_symbols", "genes_all")),
        pick(met_ann, ("standard_english_name", "feature_name", "metabolite_identification", "metabolite_name")),
    )


def _classify(rank: int, s: float, d: float, r: float, p: float) -> str:
    unstable = rank <= 5 and (s < 0.25 or d < 0.5 or r < 0.4)
    if unstable:
        return "D"
    if rank <= 10 and p >= 0.5 and s >= 0.30 and d >= 0.50 and r >= 0.40:
        return "A"
    if rank <= 3 and p == 0.0 and s >= 0.40 and d >= 0.65 and r >= 0.55:
        return "C"
    if rank <= 10 and 0.0 < p < 1.0 and s >= 0.25:
        return "B"
    return ""


def _attribute_one(
    model: Any,
    bundle: ModelingBundle,
    prior_scores: dict[tuple[str, str], float],
    seed: int,
) -> dict[str, np.ndarray]:
    n_expr = bundle.n_expr
    n_y = bundle.train.Y.shape[1]
    coef = _expr_coef(model, n_y, n_expr)
    occlusion = _occlusion(model, bundle, seed + 3)
    grad_mag, grad_signed = _finite_difference(model, bundle)
    if coef is not None:
        attr = _row_normalize(np.abs(coef))
        signed = coef
        methods = ["coefficient", "occlusion", "finite_difference"]
    else:
        attr = occlusion
        signed = grad_signed
        methods = ["occlusion", "finite_difference"]
    boot = _bootstrap_linear(model, bundle, N_BOOT, seed)
    if boot is None:
        stability = np.clip(1.0 - np.abs(occlusion - grad_mag), 0.0, 1.0)
        directional = (np.sign(grad_signed) != 0).astype(float)
    else:
        stability, directional = boot
    robustness = _perturbation_r(model, bundle, signed, seed + 17)
    return {
        "A": attr,
        "occlusion": occlusion,
        "gradient": grad_mag,
        "S": stability,
        "D": directional,
        "R": robustness,
        "signed": signed,
        "_methods": methods,  # type: ignore[dict-item]
    }


def _rows_for_model(
    maps: dict[str, np.ndarray],
    bundle: ModelingBundle,
    model_name: str,
    prior_scores: dict[tuple[str, str], float],
    protein_name: dict[str, str],
    met_name: dict[str, str],
    methods: list[str],
) -> pd.DataFrame:
    n_y = len(bundle.y_features)
    n_expr = bundle.n_expr
    attr = maps["A"]
    records = []
    for j, target in enumerate(bundle.y_features):
        for i, source in enumerate(bundle.x_features):
            p = float(prior_scores.get((source, target), 0.0))
            src_mod, tgt_mod = bundle.x_modality, bundle.y_modality
            records.append(
                {
                    "model": model_name,
                    "direction": bundle.direction,
                    "source_feature": source,
                    "target_feature": target,
                    "source_modality": src_mod,
                    "target_modality": tgt_mod,
                    "source_name": protein_name.get(source, source) if src_mod == "proteomics" else met_name.get(source, source),
                    "target_name": met_name.get(target, target) if tgt_mod == "metabolomics" else protein_name.get(target, target),
                    "coefficient": float(attr[j, i]),
                    "occlusion": float(maps["occlusion"][j, i]),
                    "gradient": float(maps["gradient"][j, i]),
                    "prior": p,
                    "methods": ",".join(methods),
                }
            )
    return pd.DataFrame.from_records(records)


def attribute_models(
    source: str,
    bundle_path: str,
    models: list[str],
    artifacts: str = "artifacts",
    prior_backend: str = "name_rule",
    organism: str = "human",
    seed: int = 42,
    prior_sources: list[str] | None = None,
) -> dict[str, Any]:
    bundle = ModelingBundle.load(Path(bundle_path))
    data = load_biomaster(Path(source))
    rel = bundle.prior_relationship()
    prior = PriorRegistry().build(
        data,
        bundle.x_features,
        rel,
        backend=prior_backend,
        sources=prior_sources,
        n_context=len(bundle.context_names),
        organism=organism,
        y_features=bundle.y_features,
    )
    bundle.attach_prior(prior)
    prior_scores = _prior_pairs(prior.network)
    protein_name, met_name = _display_maps(data.protein_annotations, data.metabolite_annotations)
    registry = ModelRegistry()
    dest = Path(bundle_path) / "attribution"
    dest.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    used: list[str] = []
    for name in models:
        if name in SKIP_MODELS:
            skipped.append(f"{name}: not a cross-modal map")
            continue
        try:
            model = registry.build(name)
            model.fit(bundle)
            maps = _attribute_one(model, bundle, prior_scores, seed)
            methods = list(maps.pop("_methods"))  # type: ignore[arg-type]
            table = _rows_for_model(maps, bundle, name, prior_scores, protein_name, met_name, methods)
            table.to_csv(dest / f"{name}_pair_scores.csv", index=False)
            frames.append(table)
            used.append(name)
            print(f"  attributed {name}: {len(table)} pairs", flush=True)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{name}: {exc}")
    if not frames:
        return {"n_pairs": 0, "models": [], "skipped": skipped, "path": str(dest)}
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(dest / "method_scores.csv", index=False)
    summary = {
        "models": used,
        "skipped": skipped,
        "n_pairs": int(len(combined)),
        "methods": ["coefficient", "occlusion", "gradient"],
        "path": str(dest),
        "note": "Raw attribution scores only. BioMaster computes confidence / novelty later.",
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary
