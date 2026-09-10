import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from src.models.base import split_context, resolve_device

class PriorGBM:
    name = "prior_gbm"
    requires_prior = True

    def fit(self, bundle):
        self.bundle = bundle
        prior = bundle.prior.features
        self.group_index = prior.group_index[:bundle.n_expr]
        self.protein_emb = prior.protein_emb
        self.pair_prior = prior.pair_prior
        if hasattr(self.pair_prior, 'toarray'):
            self.pair_prior = self.pair_prior.toarray()

        X = bundle.train.X.values if hasattr(bundle.train.X, 'values') else bundle.train.X
        Y = bundle.train.Y.values if hasattr(bundle.train.Y, 'values') else bundle.train.Y
        n_samples = X.shape[0]
        n_expr = bundle.n_expr
        expr_X = X[:, :n_expr]

        # Group features
        group_ids = self.group_index
        group_mean = np.zeros((n_samples, n_expr))
        group_std = np.zeros((n_samples, n_expr))
        for i in range(n_expr):
            g = group_ids[i]
            if g >= 0:
                mask = group_ids == g
                group_mean[:, i] = np.mean(expr_X[:, mask], axis=1)
                group_std[:, i] = np.std(expr_X[:, mask], axis=1)
            else:
                group_mean[:, i] = np.mean(expr_X, axis=1)
                group_std[:, i] = np.std(expr_X, axis=1)

        # Linear projections
        proj_protein = expr_X @ self.protein_emb
        proj_pair = expr_X @ self.pair_prior

        X_aug = np.hstack([expr_X, group_mean, group_std, proj_protein, proj_pair])

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_aug)

        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            multi_output=True
        )
        self.model.fit(X_scaled, Y)
        return self

    def predict(self, bundle):
        X_test = bundle.test.X.values if hasattr(bundle.test.X, 'values') else bundle.test.X
        n_test = X_test.shape[0]
        n_expr = bundle.n_expr
        expr_X = X_test[:, :n_expr]

        group_ids = self.group_index
        group_mean = np.zeros((n_test, n_expr))
        group_std = np.zeros((n_test, n_expr))
        for i in range(n_expr):
            g = group_ids[i]
            if g >= 0:
                mask = group_ids == g
                group_mean[:, i] = np.mean(expr_X[:, mask], axis=1)
                group_std[:, i] = np.std(expr_X[:, mask], axis=1)
            else:
                group_mean[:, i] = np.mean(expr_X, axis=1)
                group_std[:, i] = np.std(expr_X, axis=1)

        proj_protein = expr_X @ self.protein_emb
        proj_pair = expr_X @ self.pair_prior

        X_aug = np.hstack([expr_X, group_mean, group_std, proj_protein, proj_pair])
        X_scaled = self.scaler.transform(X_aug)
        return self.model.predict(X_scaled)

    def params(self):
        return {
            "feature_engineering": "group_mean_std + protein_emb_proj + pair_prior_proj",
            "model": "HistGradientBoostingRegressor",
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_depth": 4,
            "scaling": "StandardScaler"
        }
