from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from src.curator.bundle import ModelingBundle


class MetabLaplacianRidge:
    name = "metab_laplacian_ridge"
    requires_prior = True

    def __init__(self, alpha: float = 1.0, lambda_metab: float = 1.0, gamma: float = 0.1) -> None:
        self.alpha = alpha
        self.lambda_metab = lambda_metab
        self.gamma = gamma
        self.W_: np.ndarray | None = None
        self.smoother_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "MetabLaplacianRidge":
        if bundle.prior is None:
            raise RuntimeError("needs bundle.prior")
        feats = getattr(bundle.prior, "features", None)
        if feats is None:
            raise RuntimeError("needs prior.features")
        metab_emb = getattr(feats, "metabolite_emb", None)
        if metab_emb is None:
            raise RuntimeError("needs metabolite_emb")
        metab_emb = np.asarray(metab_emb, dtype=float)
        X = np.asarray(bundle.train.X, dtype=float)
        Y = np.asarray(bundle.train.Y, dtype=float)

        # Build metabolite similarity Laplacian
        K = rbf_kernel(metab_emb, gamma=self.gamma)
        K = (K + K.T) / 2
        D_diag = K.sum(axis=1)
        L = np.diag(D_diag) - K
        n_y = Y.shape[1]

        # Ridge regression
        XtX = X.T @ X
        ridge = XtX + self.alpha * np.eye(X.shape[1])
        self.W_ = np.linalg.solve(ridge, X.T @ Y)

        # Precompute Laplacian smoother: (I + lambda * L)^{-1}
        I = np.eye(n_y)
        self.smoother_ = np.linalg.solve(I + self.lambda_metab * L, I)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        X = np.asarray(bundle.test.X, dtype=float)
        Y_raw = X @ self.W_
        Y_smooth = Y_raw @ self.smoother_.T
        return np.asarray(Y_smooth, dtype=float)

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "alpha": float(self.alpha),
            "lambda_metab": float(self.lambda_metab),
            "gamma": float(self.gamma),
        }
