from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV

from src.curator.bundle import ModelingBundle
from src.models.base import split_context
from src.prior.features import pathway_design


class GraphSmoothEnet:
    name = "graph_smooth_enet"
    requires_prior = True

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.model_: RidgeCV | None = None
        self.smooth_: np.ndarray | None = None
        self.weight_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "GraphSmoothEnet":
        if bundle.prior is None:
            raise RuntimeError("needs bundle.prior")

        expr, ctx = split_context(bundle.train.X, bundle.n_expr)
        n_expr = expr.shape[1]

        # Build feature weights from protein embeddings
        weight = np.ones(n_expr, dtype=float)
        feats = getattr(bundle.prior, "features", None)
        emb = getattr(feats, "protein_emb", None) if feats is not None else None
        if emb is not None:
            emb = np.asarray(emb, dtype=float)
            if emb.shape[0] == n_expr:
                nrm = np.linalg.norm(emb, axis=1)
                nrm = nrm / (float(np.max(nrm)) + 1e-8)
                weight = 0.5 + 0.5 * nrm

        # Build graph smoothing filter from adjacency (expr-only slice)
        smooth = np.eye(n_expr)
        adj = getattr(feats, "adjacency", None) if feats is not None else None
        if adj is not None:
            A = np.asarray(adj, dtype=float)
            A = A[:n_expr, :n_expr]
            if A.shape[0] == n_expr:
                deg = A.sum(axis=1)
                deg = np.where(deg > 0, deg, 1.0)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
                A_norm = D_inv_sqrt @ A @ D_inv_sqrt
                smooth = (1 - self.alpha) * np.eye(n_expr) + self.alpha * A_norm

        self.smooth_ = smooth
        self.weight_ = weight

        expr_smooth = expr @ smooth.T
        membership = getattr(bundle.prior, "membership", {}) or {}
        path = pathway_design(expr_smooth * weight.reshape(1, -1), list(bundle.x_features), membership)

        x = np.hstack([expr_smooth * weight.reshape(1, -1), path, ctx])
        self.model_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=3)
        self.model_.fit(x, bundle.train.Y)
        return self

    def _design(self, x: np.ndarray, bundle: ModelingBundle) -> np.ndarray:
        expr, ctx = split_context(x, bundle.n_expr)
        expr_smooth = expr @ self.smooth_.T
        membership = getattr(bundle.prior, "membership", {}) or {}
        path = pathway_design(expr_smooth * self.weight_.reshape(1, -1), list(bundle.x_features), membership)
        return np.hstack([expr_smooth * self.weight_.reshape(1, -1), path, ctx])

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(bundle.test.X, bundle)), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": float(self.alpha)}
