"""Pathway / random-group / Laplacian Ridge. Need bundle.prior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.curator.bundle import ModelingBundle
from src.models.base import split_context
from src.prior.graph import pathway_score_matrix


class PathwayRidgeModel:
    name = "pathway_ridge"
    requires_prior = True

    def __init__(self, alpha: float = 1.0, min_members: int = 2) -> None:
        self.alpha = alpha
        self.min_members = min_members
        self.model_ = Ridge(alpha=alpha)
        self.x_columns_: list[str] = []
        self.feature_names_: list[str] = []
        self.n_expr_ = 0
        self.membership: dict[str, list[str]] = {}

    def fit(self, bundle: ModelingBundle) -> "PathwayRidgeModel":
        if bundle.prior is None or not bundle.prior.membership:
            raise RuntimeError("pathway_ridge needs bundle.prior.membership")
        self.membership = bundle.prior.membership
        self.n_expr_ = bundle.n_expr
        self.feature_names_ = list(bundle.x_features)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        scores = pathway_score_matrix(
            pd.DataFrame(expr, columns=self.feature_names_),
            self.membership,
            min_members=self.min_members,
        )
        self.x_columns_ = list(scores.columns)
        self.model_.fit(np.hstack([scores.to_numpy(dtype=float), ctx]), bundle.train.Y)
        return self

    def _design(self, x: np.ndarray) -> np.ndarray:
        expr, ctx = split_context(x, self.n_expr_)
        scores = pathway_score_matrix(
            pd.DataFrame(expr, columns=self.feature_names_),
            self.membership,
            min_members=self.min_members,
        ).reindex(columns=self.x_columns_, fill_value=0.0)
        return np.hstack([scores.to_numpy(dtype=float), ctx])

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(bundle.test.X)), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": self.alpha, "n_pathway_features": len(self.x_columns_)}


class RandomGroupRidgeModel:
    name = "random_group_ridge"
    requires_prior = True

    def __init__(self, n_groups: int | None = None, alpha: float = 1.0, seed: int = 42) -> None:
        self.n_groups = n_groups
        self.alpha = alpha
        self.seed = seed
        self.model_ = Ridge(alpha=alpha)
        self.assignments_: np.ndarray | None = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "RandomGroupRidgeModel":
        self.n_expr_ = bundle.n_expr
        expr, _ctx = split_context(bundle.train.X, self.n_expr_)
        n_features = expr.shape[1]
        n_groups = self.n_groups
        if n_groups is None:
            n_groups = max(len(getattr(bundle.prior, "membership", {}) or {}), 1)
        n_groups = max(1, min(int(n_groups), n_features))
        rng = np.random.default_rng(self.seed)
        self.assignments_ = rng.integers(0, n_groups, size=n_features)
        self.model_.fit(self._scores(bundle.train.X), bundle.train.Y)
        return self

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if self.assignments_ is None:
            raise RuntimeError("RandomGroupRidgeModel has not been fit")
        expr, ctx = split_context(x, self.n_expr_)
        n_groups = int(self.assignments_.max()) + 1
        out = np.zeros((expr.shape[0], n_groups), dtype=float)
        for group in range(n_groups):
            mask = self.assignments_ == group
            if mask.any():
                out[:, group] = expr[:, mask].mean(axis=1)
        return np.hstack([out, ctx])

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(self._scores(bundle.test.X)), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": self.alpha, "seed": self.seed}


class LaplacianRidgeModel:
    name = "laplacian_ridge"
    requires_prior = True

    def __init__(self, ridge: float = 1.0, graph: float = 1.0) -> None:
        self.ridge = ridge
        self.graph = graph
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "LaplacianRidgeModel":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("laplacian_ridge needs bundle.prior.laplacian")
        x = np.asarray(bundle.train.X, dtype=float)
        y = np.asarray(bundle.train.Y, dtype=float)
        laplacian = np.asarray(bundle.prior.laplacian, dtype=float)
        x_mean = x.mean(axis=0, keepdims=True)
        y_mean = y.mean(axis=0, keepdims=True)
        xc, yc = x - x_mean, y - y_mean
        n_features = xc.shape[1]
        if laplacian.shape != (n_features, n_features):
            raise ValueError(f"Laplacian {laplacian.shape} != n_features={n_features}")
        lhs = xc.T @ xc + self.ridge * np.eye(n_features) + self.graph * laplacian
        coef = np.linalg.solve(lhs, xc.T @ yc)
        self.coef_ = coef
        self.x_mean_ = x_mean
        self.intercept_ = y_mean - x_mean @ coef
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("LaplacianRidgeModel has not been fit")
        return np.asarray(bundle.test.X, dtype=float) @ self.coef_ + self.intercept_

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "ridge": self.ridge, "graph": self.graph}
