import numpy as np
from sklearn.linear_model import Ridge
from src.models.base import BaseModel

class PriorMaskedRidge(BaseModel):
    name = "prior_masked_ridge"
    requires_prior = True

    def __init__(self, hparams=None):
        self.hparams = hparams or {}
        self.alpha = self.hparams.get("alpha", 1.0)
        self.coefs_ = []
        self.mask = None

    def fit(self, bundle):
        pair_prior = bundle.prior.features.pair_prior
        if pair_prior is None:
            raise ValueError("pair_prior is required for PriorMaskedRidge")
        self.mask = np.asarray(pair_prior, dtype=float)
        X, Y = bundle.train.X, bundle.train.Y
        n_expr = bundle.n_expr
        X_expr = X[:, :n_expr]
        n_y = Y.shape[1]
        self.coefs_ = []
        for j in range(n_y):
            col_mask = self.mask[:, j] > 0
            if col_mask.sum() == 0:
                intercept = float(Y[:, j].mean())
                self.coefs_.append((np.array([], dtype=int), np.array([]), intercept))
                continue
            cols = np.where(col_mask)[0]
            X_j = X_expr[:, cols]
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X_j, Y[:, j])
            self.coefs_.append((cols, model.coef_, model.intercept_))

    def predict(self, bundle):
        X_test = bundle.test.X
        n_expr = bundle.n_expr
        X_expr = X_test[:, :n_expr]
        n_y = len(self.coefs_)
        Y_pred = np.zeros((X_test.shape[0], n_y), dtype=float)
        for j, (cols, coef, intercept) in enumerate(self.coefs_):
            if len(cols) == 0:
                Y_pred[:, j] = intercept
            else:
                X_j = X_expr[:, cols]
                Y_pred[:, j] = X_j @ coef + intercept
        return Y_pred

    def params(self):
        return {"alpha": self.alpha}