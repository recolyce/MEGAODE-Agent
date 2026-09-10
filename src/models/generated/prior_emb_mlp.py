from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.curator.bundle import ModelingBundle
from src.models.base import split_context
from src.prior.features import pathway_design


class PriorEmbMLP:
    name = "prior_emb_mlp"
    requires_prior = True

    def __init__(self, hidden_size: int = 16, alpha: float = 0.1, random_state: int = 42) -> None:
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.random_state = random_state
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(hidden_size,),
            alpha=alpha,
            random_state=random_state,
            max_iter=1000,
        )
        self.membership: dict[str, list[str]] = {}
        self.weight_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "PriorEmbMLP":
        if bundle.prior is None:
            raise RuntimeError("PriorEmbMLP requires bundle.prior")
        self.membership = getattr(bundle.prior, "membership", {}) or {}
        expr, ctx = split_context(bundle.train.X, bundle.n_expr)
        path = pathway_design(expr, list(bundle.x_features), self.membership)

        weight = np.ones((1, expr.shape[1]), dtype=float)
        feats = getattr(bundle.prior, "features", None)
        emb = getattr(feats, "protein_emb", None) if feats is not None else None
        if emb is not None:
            emb = np.asarray(emb, dtype=float)
            if emb.shape[0] == expr.shape[1]:
                nrm = np.linalg.norm(emb, axis=1)
                nrm = nrm / (float(np.max(nrm)) + 1e-8)
                weight = (0.5 + 0.5 * nrm).reshape(1, -1)
        self.weight_ = weight

        x = np.hstack([expr * self.weight_, path, ctx])
        self.model_.fit(x, bundle.train.Y)
        return self

    def _design(self, x: np.ndarray, bundle: ModelingBundle) -> np.ndarray:
        expr, ctx = split_context(x, bundle.n_expr)
        path = pathway_design(expr, list(bundle.x_features), self.membership)
        return np.hstack([expr * self.weight_, path, ctx])

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(bundle.test.X, bundle)), dtype=float)

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden_size": int(self.hidden_size),
            "alpha": float(self.alpha),
            "random_state": int(self.random_state),
        }