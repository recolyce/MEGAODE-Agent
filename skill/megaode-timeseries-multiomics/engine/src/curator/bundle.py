"""ModelingBundle: the only object models consume."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SplitArrays:
    X: np.ndarray
    Y: np.ndarray
    pairs: pd.DataFrame


@dataclass
class ModelingBundle:
    dataset: str
    task: str
    unit: str
    direction: str
    lag_mode: str
    x_modality: str
    y_modality: str
    x_features: list[str]
    y_features: list[str]
    context_names: list[str]
    n_expr: int
    n_protein: int
    n_metabolite: int
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    pairs: pd.DataFrame
    y_lookup: pd.DataFrame
    metadata: pd.DataFrame
    protocol: dict[str, Any]
    preprocess_params: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    prior: Any | None = None

    def is_multimodal(self) -> bool:
        return self.direction == "multimodal" or (
            "proteomics" in str(self.x_modality) and "metabolomics" in str(self.x_modality)
        )

    def prior_relationship(self) -> str:
        if self.is_multimodal():
            return "joint"
        return "protein_to_pathway" if self.x_modality == "proteomics" else "metabolite_to_pathway"

    def attach_prior(self, prior: Any) -> "ModelingBundle":
        self.prior = prior
        return self

    def save(self, dest: Path) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self.pairs.to_csv(dest / "pairs.csv", index=False)
        self.y_lookup.to_csv(dest / "y_lookup.csv")
        self.metadata.to_csv(dest / "sample_metadata.csv", index=False)
        np.save(dest / "train_X.npy", self.train.X)
        np.save(dest / "train_Y.npy", self.train.Y)
        np.save(dest / "val_X.npy", self.val.X)
        np.save(dest / "val_Y.npy", self.val.Y)
        np.save(dest / "test_X.npy", self.test.X)
        np.save(dest / "test_Y.npy", self.test.Y)
        payload = {
            "dataset": self.dataset,
            "task": self.task,
            "unit": self.unit,
            "direction": self.direction,
            "lag_mode": self.lag_mode,
            "x_modality": self.x_modality,
            "y_modality": self.y_modality,
            "x_features": self.x_features,
            "y_features": self.y_features,
            "context_names": self.context_names,
            "n_expr": self.n_expr,
            "n_protein": self.n_protein,
            "n_metabolite": self.n_metabolite,
            "protocol": self.protocol,
            "preprocess_params": self.preprocess_params,
            "warnings": self.warnings,
        }
        (dest / "bundle.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return dest

    @classmethod
    def load(cls, dest: Path) -> "ModelingBundle":
        dest = Path(dest)
        payload = json.loads((dest / "bundle.json").read_text(encoding="utf-8"))
        pairs = pd.read_csv(dest / "pairs.csv")
        y_lookup = pd.read_csv(dest / "y_lookup.csv", index_col=0)
        metadata = pd.read_csv(dest / "sample_metadata.csv")
        splits = {}
        for name in ("train", "val", "test"):
            x = np.load(dest / f"{name}_X.npy")
            y = np.load(dest / f"{name}_Y.npy")
            sub = pairs.loc[pairs["split"] == name].reset_index(drop=True)
            splits[name] = SplitArrays(X=x, Y=y, pairs=sub)
        return cls(
            dataset=payload["dataset"],
            task=payload["task"],
            unit=payload["unit"],
            direction=payload["direction"],
            lag_mode=payload["lag_mode"],
            x_modality=payload["x_modality"],
            y_modality=payload["y_modality"],
            x_features=list(payload["x_features"]),
            y_features=list(payload["y_features"]),
            context_names=list(payload["context_names"]),
            n_expr=int(payload["n_expr"]),
            n_protein=int(payload.get("n_protein") or payload["n_expr"]),
            n_metabolite=int(payload.get("n_metabolite") or 0),
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            pairs=pairs,
            y_lookup=y_lookup,
            metadata=metadata,
            protocol=payload["protocol"],
            preprocess_params=payload["preprocess_params"],
            warnings=list(payload.get("warnings") or []),
        )
