import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from src.models.base import split_context

class PathwayBoosted:
    name = "pathway_boosted"
    requires_prior = True

    def __init__(self, n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.n_expr = None
        self.n_metab = None
        self.pathway_groups = []
        self.pair_prior_matrix = None
        self.protein_ids = None
        self.metab_ids = None

    def _build_pathway_features(self, X, all_protein_ids):
        """Compute pathway mean expression and metabolite scores from pair_prior."""
        n = X.shape[0]
        # Pathway means
        pathway_feats = np.zeros((n, len(self.pathway_groups)), dtype=np.float64)
        for i, group in enumerate(self.pathway_groups):
            cols = [pid for pid in group if pid in all_protein_ids]
            if cols and len(cols) > 0:
                pathway_feats[:, i] = pd.DataFrame(X, columns=all_protein_ids)[cols].mean(axis=1).values
        # Metabolite scores from pair_prior (if available)
        if self.pair_prior_matrix is not None:
            X_prot = pd.DataFrame(X, columns=all_protein_ids).values
            if hasattr(self.pair_prior_matrix, 'toarray'):
                pair_dense = self.pair_prior_matrix.toarray()
            else:
                pair_dense = self.pair_prior_matrix
            metab_scores = X_prot @ pair_dense  # (n, n_metab)
        else:
            metab_scores = np.zeros((n, self.n_metab), dtype=np.float64)
        # Combine: original (already included separately), pathway means, metabolite scores
        return np.hstack([pathway_feats, metab_scores])

    def fit(self, bundle):
        self.n_expr = bundle.n_expr
        prior = bundle.prior
        X_train = bundle.train.X
        Y_train = bundle.train.Y
        
        # Determine feature IDs
        if isinstance(X_train, pd.DataFrame):
            self.protein_ids = X_train.columns.tolist()
        else:
            self.protein_ids = [f'prot_{i}' for i in range(X_train.shape[1])]
        if isinstance(Y_train, pd.DataFrame):
            self.metab_ids = Y_train.columns.tolist()
        else:
            self.metab_ids = [f'metab_{i}' for i in range(Y_train.shape[1])]
        self.n_metab = len(self.metab_ids)

        # Build pathway groups from prior.membership (only protein members)
        membership = prior.membership
        if membership:
            for pathway, members in membership.items():
                prot_members = [m for m in members if m in self.protein_ids]
                if prot_members:
                    self.pathway_groups.append(prot_members)
        # Build pair_prior matrix from prior.features.pair_prior
        pair_prior = prior.features.pair_prior
        if pair_prior is not None:
            if isinstance(pair_prior, dict):
                # Convert dict (protein->list of metabolite indices) to dense matrix
                n_prot = len(self.protein_ids)
                n_met = len(self.metab_ids)
                mat = np.zeros((n_prot, n_met), dtype=np.float64)
                for prot_id, metab_list in pair_prior.items():
                    if prot_id in self.protein_ids:
                        prot_idx = self.protein_ids.index(prot_id)
                        for m_id in metab_list:
                            if m_id in self.metab_ids:
                                met_idx = self.metab_ids.index(m_id)
                                mat[prot_idx, met_idx] = 1.0
                self.pair_prior_matrix = mat
            else:
                # Assume it's array-like; ensure shape matches
                if pair_prior.shape[0] == self.n_expr and pair_prior.shape[1] == self.n_metab:
                    self.pair_prior_matrix = pair_prior
                else:
                    self.pair_prior_matrix = None
        else:
            self.pair_prior_matrix = None

        # Build features for training
        X_base = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        extra_features = self._build_pathway_features(X_base, self.protein_ids)
        X_combined = np.hstack([X_base, extra_features])

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_combined)

        # Train multi-output random forest
        base_estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model = MultiOutputRegressor(base_estimator, n_jobs=-1)
        self.model.fit(X_scaled, Y_train)

    def predict(self, bundle):
        X_test = bundle.test.X
        X_base = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        extra_features = self._build_pathway_features(X_base, self.protein_ids)
        X_combined = np.hstack([X_base, extra_features])
        X_scaled = self.scaler.transform(X_combined)
        return self.model.predict(X_scaled)

    def params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }
