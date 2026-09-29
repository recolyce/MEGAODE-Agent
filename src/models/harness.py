"""Modeling-stage checks. Cross-modal flow is required for later attribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.curator.bundle import ModelingBundle
from src.interpretation.attribution import _predict_x


class CrossModalFlowError(ValueError):
    """Protein and metabolite blocks do not influence each other's outputs."""


def _mean_abs(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a))) if a.size else 0.0


def _shift_block(x: np.ndarray, cols: slice, rng: np.random.Generator) -> np.ndarray:
    out = np.array(x, dtype=float, copy=True)
    block = np.array(out[:, cols], dtype=float, copy=True)
    if block.shape[0] > 1:
        for j in range(block.shape[1]):
            block[:, j] = rng.permutation(block[:, j])
    out[:, cols] = block + 1.0
    return out


def check_cross_modal_flow(
    model: Any,
    bundle: ModelingBundle,
    *,
    min_rel: float = 1e-5,
    seed: int = 0,
) -> dict[str, Any]:
    """Perturb one omics block; the other omics' outputs must move.

    Per-node maps (Linear(1) encoder + independent field + per-node head) fail:
    protein outputs ignore metabolites, so cross-modal gradients are identically 0.
    """
    n_p = int(getattr(bundle, "n_protein", 0) or 0)
    n_m = int(getattr(bundle, "n_metabolite", 0) or 0)
    if n_p <= 0 or n_m <= 0:
        return {"ok": True, "skipped": True, "reason": "not a joint multimodal bundle"}
    x = np.asarray(bundle.test.X, dtype=float)
    if x.shape[0] < 1 or x.shape[1] < n_p + n_m:
        raise CrossModalFlowError("test X is too small for a cross-modal flow check")
    pred0 = np.asarray(_predict_x(model, bundle, x), dtype=float)
    if pred0.shape[1] < n_p + n_m:
        raise CrossModalFlowError(f"predict width {pred0.shape[1]} < n_protein+n_metabolite {n_p + n_m}")
    rng = np.random.default_rng(seed)
    pred_flip_p = np.asarray(_predict_x(model, bundle, _shift_block(x, slice(0, n_p), rng)), dtype=float)
    pred_flip_m = np.asarray(_predict_x(model, bundle, _shift_block(x, slice(n_p, n_p + n_m), rng)), dtype=float)
    met_scale = _mean_abs(pred0[:, n_p : n_p + n_m]) + 1e-6
    prot_scale = _mean_abs(pred0[:, :n_p]) + 1e-6
    prot_to_met = _mean_abs(pred_flip_p[:, n_p : n_p + n_m] - pred0[:, n_p : n_p + n_m]) / met_scale
    met_to_prot = _mean_abs(pred_flip_m[:, :n_p] - pred0[:, :n_p]) / prot_scale
    report = {
        "ok": prot_to_met >= min_rel and met_to_prot >= min_rel,
        "prot_to_met": prot_to_met,
        "met_to_prot": met_to_prot,
        "min_rel": min_rel,
    }
    if report["ok"]:
        return report
    raise CrossModalFlowError(
        "cross-modal flow required: protein outputs must depend on metabolite inputs and "
        f"vice versa (prot→met {prot_to_met:.2e}, met→prot {met_to_prot:.2e}, min {min_rel:.0e}). "
        "Do not use a per-node Linear(1) field plus a per-node head; mix the two omics "
        "(GCN / flatten head / dual-stream fuse) so attribution has nonzero cross-modal gradients."
    )
