"""Contract the coding agent must follow when writing a new model."""

from __future__ import annotations

# Generated model files: deny process/network, allow any other installed package.
FORBIDDEN = {
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "os",
    "sys",
    "requests",
    "urllib",
    "http",
    "ctypes",
    "multiprocessing",
    "importlib",
    "pickle",
}
ALLOWED_IMPORT_PREFIXES = ()  # kept for compatibility; AST now uses FORBIDDEN only

EXAMPLE = '''
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from src.curator.bundle import ModelingBundle
from src.models.base import split_context
from src.prior.features import pathway_design


class PathwayEmbRidge:
    name = "pathway_emb_ridge"
    requires_prior = True

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.model_ = Ridge(alpha=alpha)
        self.membership: dict[str, list[str]] = {}
        self.weight_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "PathwayEmbRidge":
        if bundle.prior is None:
            raise RuntimeError("needs bundle.prior")
        self.membership = getattr(bundle.prior, "membership", {}) or {}
        expr, ctx = split_context(bundle.train.X, bundle.n_expr)
        path = pathway_design(expr, list(bundle.x_features), self.membership)
        weight = np.ones((1, expr.shape[1]), dtype=float)
        feats = getattr(bundle.prior, "features", None)
        emb = getattr(feats, "protein_emb", None) if feats is not None else None
        if emb is not None:
            emb = np.asarray(emb, dtype=float)
            if emb.shape[0] == expr.shape[1]:
                nrm = np.linalg.norm(emb, axis=1)
                nrm = nrm / (float(np.max(nrm)) + 1e-8)
                weight = (0.5 + 0.5 * nrm).reshape(1, -1)
        self.weight_ = weight
        x = np.hstack([expr * weight, path, ctx])
        self.model_.fit(x, bundle.train.Y)
        return self

    def _design(self, x: np.ndarray, bundle: ModelingBundle) -> np.ndarray:
        expr, ctx = split_context(x, bundle.n_expr)
        path = pathway_design(expr, list(bundle.x_features), self.membership)
        return np.hstack([expr * self.weight_, path, ctx])

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(bundle.test.X, bundle)), dtype=float)

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": float(self.alpha)}
'''

SYSTEM = """You write ONE new last_interval model. You have tools. Each turn reply with ONE JSON object, no markdown.

Tools:
{"tool":"list_installed_packages","args":{"query":"optional substring"}}
{"tool":"install_python_packages","args":{"packages":["lightgbm"]}}
{"tool":"run_terminal","args":{"command":"PYTHONPATH=. python -c 'import lightgbm; print(lightgbm.__version__)'"}}
{"tool":"read_repo_file","args":{"path":"src/models/classic.py"}}
{"tool":"write_generated_model","args":{"model_name":"snake","class_name":"Pascal","code":"full module text"}}
{"tool":"smoke_generated_model","args":{"model_name":"snake"}}
{"done":true,"model_name":"snake","reason":"one sentence"}

Workflow: list_installed_packages → install anything extra you need → write_generated_model → smoke_generated_model → done.
run_terminal is only python/pip in this repo. Do not install from git URLs.

Model rules:
- New snake_case name, not ridge/pls/prior_gated_ridge/prior_fusion_ridge/laplacian_ridge/pathway_ridge.
- Class: name, requires_prior, fit(bundle), predict(bundle), params().
- predict uses only bundle.test.X. numpy arrays have no .values (use np.asarray).
- bundle.prior.membership / .laplacian / .features.protein_emb are available.
  from src.prior.features import pathway_design
  from src.models.base import split_context
- Generated code may import any INSTALLED scientific package. It must NOT import os, sys,
  subprocess, socket, pathlib, requests, urllib. Use tools to pip-install missing packages.
- Keep the model small. No downloads inside fit/predict.

Example pattern you may adapt (change the name):
""" + EXAMPLE
