import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.base import split_context, resolve_device

class PathwayForest:
    name = "pathway_forest"
    requires_prior = True

    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        self.n_expr = None
        self.n_groups = None
        self.n_metab = None
        self.protein_feat_ids = None
        self.metab_feat_ids = None
        self.pathway_groups = None
        self.pair_prior = None

    def _extract_features(self, X):
        """Augment X with pathway group means and metabolite scores from prior."""
        # X is DataFrame with protein feature_id columns
        if isinstance(X, pd.DataFrame):
            prot_ids = X.columns.tolist()
        else:
            prot_ids = self.protein_feat_ids
            X = pd.DataFrame(X, columns=prot_ids)
        n_samples = X.shape[0]

        # 1) Pathway group means (protein side)
        group_feats = np.zeros((n_samples, self.n_groups), dtype=np.float64)
        for idx, pathway in enumerate(self.pathway_groups):
            members = [pid for pid in self.pathway_groups[pathway] if pid in prot_ids]
            if len(members) > 0:
                group_feats[:, idx] = X[members].mean(axis=1).values
        # 2) Metabolite scores from pair_prior (protein expression -> metabolite)
        # pair_prior is (n_expr, n_y) – dense or sparse
        X_prot = X[prot_ids].values  # (n, n_expr)
        if hasattr(self.pair_prior, 'toarray'):
            pair_dense = self.pair_prior.toarray()
        else:
            pair_dense = np.asarray(self.pair_prior)
        metab_scores = X_prot @ pair_dense  # (n, n_y)

        # Combine: original + group + metabolite scores
        X_aug = np.hstack([X_prot, group_feats, metab_scores])
        return X_aug

    def fit(self, bundle):
        self.n_expr = bundle.n_expr
        prior = bundle.prior
        # Get feature IDs
        X_train = bundle.train.X
        Y_train = bundle.train.Y
        if isinstance(X_train, pd.DataFrame):
            self.protein_feat_ids = X_train.columns.tolist()
        else:
            self.protein_feat_ids = [f'prot_{i}' for i in range(X_train.shape[1])]
        if isinstance(Y_train, pd.DataFrame):
            self.metab_feat_ids = Y_train.columns.tolist()
        else:
            self.metab_feat_ids = [f'metab_{i}' for i in range(Y_train.shape[1])]

        # Build pathway groups from prior.membership (only protein members)
        membership = prior.membership
        self.pathway_groups = {}
        for pathway, members in membership.items():
            prot_members = [m for m in members if m in self.protein_feat_ids]
            if prot_members:
                self.pathway_groups[pathway] = prot_members
        self.n_groups = len(self.pathway_groups)

        # Get pair_prior (protein->metabolite edges)
        self.pair_prior = prior.features.pair_prior
        if self.pair_prior is not None:
            # Ensure shape matches (n_expr, n_y)
            if self.pair_prior.shape[0] != self.n_expr or self.pair_prior.shape[1] != len(self.metab_feat_ids):
                # Attempt to reindex (assuming feature_id order matches)
                # In practice, pair_prior rows/cols correspond to feature_ids in order?
                # If not, we try to match using prior info; for simplicity assume same order.
                pass
        self.n_metab = len(self.metab_feat_ids)

        # Augment training features
        X_train_aug = self._extract_features(bundle.train.X)
        Y_train_vals = bundle.train.Y.values if isinstance(bundle.train.Y, pd.DataFrame) else bundle.train.Y

        # Train RandomForest
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_aug, Y_train_vals)

    def predict(self, bundle):
        X_test = bundle.test.X
        X_test_aug = self._extract_features(X_test)
        pred = self.model.predict(X_test_aug)
        if isinstance(pred, np.ndarray) and pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }
