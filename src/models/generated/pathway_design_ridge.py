import numpy as np
from sklearn.linear_model import Ridge
from src.models.base import split_context
from src.prior.features import pathway_design

class PathwayDesignRidge:
    name = "pathway_design_ridge"
    requires_prior = True

    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = float(alpha) if alpha is not None else 1.0
        # absorb extra kwargs silently
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.coef_ = None
        self.intercept_ = None
        self.ridge_ = None

    def fit(self, bundle):
        X = bundle.train.X
        Y = bundle.train.Y.values.astype(np.float64)
        n_expr = bundle.n_expr
        membership = bundle.prior.membership
        # split expression and context (if any)
        X_expr, X_context = split_context(X, n_expr)
        # compute pathway design matrix (n_samples, n_pathways)
        Z = pathway_design(X_expr, membership)
        # train ridge regression
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.ridge_.fit(Z, Y)
        self.coef_ = self.ridge_.coef_
        self.intercept_ = self.ridge_.intercept_
        return self

    def predict(self, bundle):
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit first.")
        X_test = bundle.test.X
        n_expr = bundle.n_expr
        membership = bundle.prior.membership
        X_expr, X_context = split_context(X_test, n_expr)
        Z_test = pathway_design(X_expr, membership)
        return self.ridge_.predict(Z_test)

    def params(self):
        return {"alpha": self.alpha}
