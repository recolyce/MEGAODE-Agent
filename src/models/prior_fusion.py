"""Prior-injected baselines: pathway scores (+ optional Laplacian) into Ridge/MLP."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.prior.features import pathway_design


def _prior_parts(bundle: ModelingBundle) -> tuple[dict[str, list[str]], list[str]]:
    prior = bundle.prior
    if prior is None:
        raise RuntimeError("prior_fusion models need bundle.prior")
    membership = getattr(prior, "membership", None) or {}
    return membership, list(bundle.x_features)


def _design(x: np.ndarray, bundle: ModelingBundle, membership: dict[str, list[str]]) -> np.ndarray:
    expr, ctx = split_context(x, bundle.n_expr)
    path = pathway_design(expr, list(bundle.x_features), membership)
    return np.hstack([expr, path, ctx])


class PriorFusionRidgeModel:
    name = "prior_fusion_ridge"
    requires_prior = True

    def __init__(self, alpha: float = 10.0, graph: float = 0.0) -> None:
        self.alpha = alpha
        self.graph = graph
        self.model_ = Ridge(alpha=alpha)
        self.membership: dict[str, list[str]] = {}
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "PriorFusionRidgeModel":
        self.membership, _ = _prior_parts(bundle)
        x = _design(bundle.train.X, bundle, self.membership)
        y = np.asarray(bundle.train.Y, dtype=float)
        if self.graph > 0 and getattr(bundle.prior, "laplacian", None) is not None:
            lap = np.asarray(bundle.prior.laplacian, dtype=float)
            # Regularize only the expression block; pathway+context stay ridge-only.
            n_expr = bundle.n_expr
            n_feat = x.shape[1]
            extra = np.zeros((n_feat, n_feat), dtype=float)
            block = lap[:n_expr, :n_expr] if lap.shape[0] >= n_expr else lap
            extra[: block.shape[0], : block.shape[1]] = self.graph * block
            x_mean = x.mean(axis=0, keepdims=True)
            y_mean = y.mean(axis=0, keepdims=True)
            xc, yc = x - x_mean, y - y_mean
            lhs = xc.T @ xc + self.alpha * np.eye(n_feat) + extra
            self.coef_ = np.linalg.solve(lhs, xc.T @ yc)
            self.intercept_ = y_mean - x_mean @ self.coef_
            self.model_ = None
        else:
            self.model_.fit(x, y)
            self.coef_ = np.asarray(self.model_.coef_, dtype=float).T
            self.intercept_ = np.asarray(self.model_.intercept_, dtype=float)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        x = _design(bundle.test.X, bundle, self.membership)
        if self.model_ is not None:
            return np.asarray(self.model_.predict(x), dtype=float)
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("PriorFusionRidgeModel has not been fit")
        return x @ self.coef_ + self.intercept_

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": float(self.alpha), "graph": float(self.graph)}


class PriorFusionMLPModel:
    name = "prior_fusion_mlp"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.membership: dict[str, list[str]] = {}
        self.model_ = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "PriorFusionMLPModel":
        import torch

        from src.models.neural import _MLPNet, _fit_torch

        self.membership, _ = _prior_parts(bundle)
        torch.manual_seed(self.seed)
        x = _design(bundle.train.X, bundle, self.membership)
        y = bundle.train.Y
        model = _MLPNet(x.shape[1], y.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            xv = torch.as_tensor(_design(bundle.val.X, bundle, self.membership), dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("PriorFusionMLPModel has not been fit")
        x = _design(bundle.test.X, bundle, self.membership)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        return np.asarray(pred.detach().cpu().numpy(), dtype=float)

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "best_epoch": int(self.best_epoch_),
            "device": str(self.device),
        }
