"""Train-mean and last-value baselines on a ModelingBundle."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.curator.bundle import ModelingBundle


class TrainMeanModel:
    name = "train_mean"
    requires_prior = False

    def __init__(self) -> None:
        self.means_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "TrainMeanModel":
        self.means_ = np.nanmean(bundle.train.Y, axis=0)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("TrainMeanModel has not been fit")
        n = bundle.test.X.shape[0]
        return np.broadcast_to(self.means_, (n, self.means_.shape[0])).copy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name}


class LastValueModel:
    """Y at persist_y_sample_id (same trajectory at x_time)."""

    name = "last_value"
    requires_prior = False

    def __init__(self) -> None:
        self.fallback_: np.ndarray | None = None
        self.definition = ""

    def fit(self, bundle: ModelingBundle) -> "LastValueModel":
        self.fallback_ = np.nanmean(bundle.train.Y, axis=0)
        self.definition = (
            "same-trajectory Y at x_time (true lag: same subject; pseudo: tissue|sex|replicate)"
        )
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.fallback_ is None:
            raise RuntimeError("LastValueModel has not been fit")
        lookup = bundle.y_lookup
        rows = []
        for sid in bundle.test.pairs["persist_y_sample_id"].astype(str):
            if sid in lookup.index:
                rows.append(lookup.loc[sid, bundle.y_features].to_numpy(dtype=float))
            else:
                rows.append(self.fallback_)
        return np.vstack(rows)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "definition": self.definition}
