import numpy as np
import sklearn
from src.models.base import split_context
from src.prior.features import pathway_design

class PriorGateRidgeFixed:
    name = "prior_gate_ridge_fixed"
    requires_prior = True

    def __init__(self, alpha=1.0, lam_prior=0.1, **kwargs):
        self.alpha = self._robust_float(alpha, 1.0)
        self.lam_prior = self._robust_float(lam_prior, 0.1)
        # Capture any extra kwargs to avoid failure
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.coef_ = None

    def _robust_float(self, value, default):
        if value is None:
            return default
        if isinstance(value, dict):
            # Extract the first scalar value from dict
            first_val = list(value.values())[0]
            return float(first_val)
        return float(value)

    def fit(self, bundle):
        X = bundle.train.X.values.astype(np.float64)
        Y = bundle.train.Y.values.astype(np.float64)
        n_expr = bundle.n_expr
        prior = bundle.prior
        # Get Laplacian matrix (dense or sparse)
        L_matrix = prior.features.laplacian
        if L_matrix is None:
            L_matrix = np.eye(X.shape[1])
        else:
            try:
                L_matrix = L_matrix.toarray()
            except AttributeError:
                pass
            L_matrix = np.asarray(L_matrix, dtype=np.float64)
        # Compute XtX and XtY
        XtX = X.T @ X
        XtY = X.T @ Y
        p = X.shape[1]
        reg = self.alpha * np.eye(p) + self.lam_prior * L_matrix
        # Solve for coefficients
        try:
            self.coef_ = np.linalg.solve(XtX + reg, XtY)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.coef_ = np.linalg.pinv(XtX + reg) @ XtY
        return self

    def predict(self, bundle):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet. Call fit first.")
        X_test = bundle.test.X.values.astype(np.float64)
        return X_test @ self.coef_

    def params(self):
        return {"alpha": self.alpha, "lam_prior": self.lam_prior}
