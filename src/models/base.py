"""Shared model protocol. Every registered model fits and predicts a ModelingBundle."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from src.curator.bundle import ModelingBundle


class BaseModel(Protocol):
    name: str
    requires_prior: bool

    def fit(self, bundle: ModelingBundle) -> "BaseModel":
        ...

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        ...

    def params(self) -> dict[str, Any]:
        ...


def resolve_device(requested: str = "auto"):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_context(x: np.ndarray, n_expr: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.shape[1] < n_expr:
        raise ValueError("X has fewer columns than expression features")
    return x[:, :n_expr], x[:, n_expr:]
