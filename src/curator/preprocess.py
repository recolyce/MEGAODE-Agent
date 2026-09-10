"""Train-fold-only imputation, scaling, and high-variance selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPreprocessor:
    already_logged: bool
    group_col: str | None = "tissue"
    n_high_variance: int = 512
    kept_features: list[str] = field(default_factory=list)
    medians: dict[str, dict[str, float]] = field(default_factory=dict)
    global_medians: dict[str, float] = field(default_factory=dict)
    scaler_mean: list[float] = field(default_factory=list)
    scaler_scale: list[float] = field(default_factory=list)

    def fit(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        train_ids: list[str],
    ) -> "FoldPreprocessor":
        work = matrix.loc[train_ids].copy()
        if not self.already_logged:
            values = work.to_numpy(dtype=float, copy=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.where(np.isfinite(values) & (values > 0), np.log2(values), np.nan)
            work = pd.DataFrame(logged, index=work.index, columns=work.columns)
        meta = metadata.set_index("sample_id")
        groups = (
            meta.loc[work.index, self.group_col].astype(str)
            if self.group_col and self.group_col in meta.columns
            else pd.Series("all", index=work.index)
        )
        self.medians = {}
        for group, idx in groups.groupby(groups).groups.items():
            block = work.loc[idx]
            self.medians[str(group)] = {
                col: float(val) for col, val in block.median(axis=0, skipna=True).items() if pd.notna(val)
            }
        self.global_medians = {
            col: float(val) for col, val in work.median(axis=0, skipna=True).items() if pd.notna(val)
        }
        imputed = self._impute(work, groups)
        variances = imputed.var(axis=0, skipna=True).fillna(0.0)
        ordered = variances.sort_values(ascending=False).index.tolist()
        n_keep = min(self.n_high_variance, len(ordered))
        self.kept_features = ordered[:n_keep]
        selected = imputed.loc[:, self.kept_features].fillna(0.0)
        scaler = StandardScaler()
        scaler.fit(selected.to_numpy(dtype=float))
        self.scaler_mean = scaler.mean_.astype(float).tolist()
        self.scaler_scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(float).tolist()
        return self

    def transform(self, matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        work = matrix.copy()
        if not self.already_logged:
            values = work.to_numpy(dtype=float, copy=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.where(np.isfinite(values) & (values > 0), np.log2(values), np.nan)
            work = pd.DataFrame(logged, index=work.index, columns=work.columns)
        meta = metadata.set_index("sample_id")
        groups = (
            meta.loc[work.index, self.group_col].astype(str)
            if self.group_col and self.group_col in meta.columns
            else pd.Series("all", index=work.index)
        )
        imputed = self._impute(work, groups)
        selected = imputed.reindex(columns=self.kept_features).fillna(0.0)
        mean = np.asarray(self.scaler_mean, dtype=float)
        scale = np.asarray(self.scaler_scale, dtype=float)
        scaled = (selected.to_numpy(dtype=float) - mean) / scale
        return pd.DataFrame(scaled, index=selected.index, columns=self.kept_features)

    def _impute(self, matrix: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
        out = matrix.copy()
        for group, idx in groups.groupby(groups).groups.items():
            med = self.medians.get(str(group), self.global_medians)
            block = out.loc[idx]
            fill = pd.Series({col: med.get(col, self.global_medians.get(col, 0.0)) for col in block.columns})
            out.loc[idx] = block.fillna(fill)
        return out

    def to_params(self) -> dict[str, Any]:
        return asdict(self)
