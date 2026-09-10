"""Deterministic prior-gated ridge written when the coding LLM is unavailable."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from src.curator.bundle import ModelingBundle
from src.models.base import split_context
from src.prior.features import pathway_design


class PriorGatedRidge:
    name = "prior_gated_ridge"
    requires_prior = True

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.model_ = Ridge(alpha=alpha)
        self.membership: dict[str, list[str]] = {}

    def fit(self, bundle: ModelingBundle) -> "PriorGatedRidge":
        if bundle.prior is None:
            raise RuntimeError("prior_gated_ridge needs bundle.prior")
        self.membership = getattr(bundle.prior, "membership", {}) or {}
        expr, ctx = split_context(bundle.train.X, bundle.n_expr)
        path = pathway_design(expr, list(bundle.x_features), self.membership)
        gate = np.ones((1, expr.shape[1]), dtype=float)
        features = getattr(bundle.prior, "features", None)
        if features is not None and getattr(features, "group_index", None) is not None:
            gi = np.asarray(features.group_index)
            if gi.shape[0] == expr.shape[1]:
                gate[0, gi < 0] = 0.25
        x = np.hstack([expr * gate, path, ctx])
        self.model_.fit(x, bundle.train.Y)
        self.gate_ = gate
        return self

    def _design(self, x: np.ndarray, bundle: ModelingBundle) -> np.ndarray:
        expr, ctx = split_context(x, bundle.n_expr)
        path = pathway_design(expr, list(bundle.x_features), self.membership)
        return np.hstack([expr * self.gate_, path, ctx])

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(bundle.test.X, bundle)), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": float(self.alpha), "gated": True}
