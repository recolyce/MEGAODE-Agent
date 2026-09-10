"""Ridge and PLS on ModelingBundle arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

from src.curator.bundle import ModelingBundle


class RidgeModel:
    name = "ridge"
    requires_prior = False

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.model_ = Ridge(alpha=alpha)

    def fit(self, bundle: ModelingBundle) -> "RidgeModel":
        self.model_.fit(bundle.train.X, bundle.train.Y)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(bundle.test.X), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": float(self.alpha)}


class PLSModel:
    name = "pls"
    requires_prior = False

    def __init__(self, n_components: int = 8) -> None:
        self.n_components = n_components
        self.model_: PLSRegression | None = None

    def fit(self, bundle: ModelingBundle) -> "PLSModel":
        x, y = bundle.train.X, bundle.train.Y
        n_comp = max(1, min(self.n_components, x.shape[0] - 1, x.shape[1], y.shape[1]))
        self.n_components = n_comp
        self.model_ = PLSRegression(n_components=n_comp, scale=False)
        self.model_.fit(x, y)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("PLSModel has not been fit")
        return np.asarray(self.model_.predict(bundle.test.X), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "n_components": int(self.n_components)}
