from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from src.curator.bundle import ModelingBundle


class PriorEmbeddingRidge:
    name = "prior_embedding_ridge"
    requires_prior = True

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self._prior_weight: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "PriorEmbeddingRidge":
        if bundle.prior is None or bundle.prior.features is None:
            raise RuntimeError("prior with features required")
        protein_emb = getattr(bundle.prior.features, "protein_emb", None)
        metabolite_emb = getattr(bundle.prior.features, "metabolite_emb", None)
        if protein_emb is None or metabolite_emb is None:
            raise RuntimeError("protein_emb and metabolite_emb required")
        protein_emb = np.asarray(protein_emb, dtype=float)
        metabolite_emb = np.asarray(metabolite_emb, dtype=float)
        if protein_emb.shape[0] != bundle.n_expr:
            raise RuntimeError("protein_emb rows != n_expr")

        prot_norm = protein_emb / (np.linalg.norm(protein_emb, axis=1, keepdims=True) + 1e-8)
        met_norm = metabolite_emb / (np.linalg.norm(metabolite_emb, axis=1, keepdims=True) + 1e-8)
        prior_weight = met_norm @ prot_norm.T  # (n_y, n_expr)
        self._prior_weight = prior_weight

        X = np.asarray(bundle.train.X, dtype=float)
        Y = np.asarray(bundle.train.Y, dtype=float)
        n_features = X.shape[1]
        sqrt_alpha = np.sqrt(self.alpha)

        # identity only on the expression columns, zero on context columns
        eye_pad = np.zeros((bundle.n_expr, n_features), dtype=float)
        eye_pad[:, :bundle.n_expr] = np.eye(bundle.n_expr)

        X_aug = np.vstack([X, sqrt_alpha * eye_pad])
        Y_aug = np.vstack([Y, sqrt_alpha * prior_weight.T])

        model = Ridge(alpha=0.0, solver="auto")
        model.fit(X_aug, Y_aug)
        self.coef_ = model.coef_
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        X = np.asarray(bundle.test.X, dtype=float)
        return X @ self.coef_.T

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": float(self.alpha)}
