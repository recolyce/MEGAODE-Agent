"""Adapted baselines: unimodal-temporal or multimodal-nontemporal papers → last_interval.

Each class is a small adapter, not a paper reproduction. Time lives in context
columns [t0, t1, sex_F, ...]. Dual-stream models split X via split_modalities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_modalities
from src.models.neural import _fit_torch


def _parts(x: np.ndarray, bundle: ModelingBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_modalities(x, bundle.n_protein, bundle.n_metabolite)


def _tensors(arrays: tuple[np.ndarray, ...], device) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(arr, dtype=torch.float32, device=device) for arr in arrays)


def _rk4(func: nn.Module, z: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, steps: int = 3) -> torch.Tensor:
    dt = (t1 - t0).unsqueeze(1) / float(steps)
    t = t0
    for _ in range(steps):
        k1 = func(t, z)
        k2 = func(t + 0.5 * dt.squeeze(1), z + 0.5 * dt * k1)
        k3 = func(t + 0.5 * dt.squeeze(1), z + 0.5 * dt * k2)
        k4 = func(t + dt.squeeze(1), z + dt * k3)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt.squeeze(1)
    return z


class _TimeFilmNet(nn.Module):
    """FiLM (Perez 2018): time/context modulates a joint expression encoder."""

    def __init__(self, n_expr: int, n_ctx: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_expr, hidden), nn.GELU())
        self.film = nn.Linear(n_ctx + 1, hidden * 2)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, expr: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        dt = (ctx[:, 1] - ctx[:, 0]).unsqueeze(1)
        gamma, beta = self.film(torch.cat([ctx, dt], dim=1)).chunk(2, dim=1)
        h = self.enc(expr)
        return self.head(torch.tanh(gamma) * h + beta)


class TimeFilmModel:
    name = "time_film"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 50,
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
        self.model_: _TimeFilmNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "TimeFilmModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        expr = np.concatenate([p, m], axis=1)
        model = _TimeFilmNet(expr.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = _tensors((expr, ctx), self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            vp, vm, vc = _parts(bundle.val.X, bundle)
            xv = _tensors((np.concatenate([vp, vm], axis=1), vc), self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TimeFilmModel has not been fit")
        p, m, ctx = _parts(bundle.test.X, bundle)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors((np.concatenate([p, m], axis=1), ctx), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "source": "FiLM Perez 2018 time-conditioned MLP adapter",
        }


class _TimeGRUNet(nn.Module):
    """One GRU-D / T-LSTM step: decay hidden by dt, then a single GRU update (Che 2018; Baytas 2017)."""

    def __init__(self, n_expr: int, n_ctx: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.to_h = nn.Linear(n_expr, hidden)
        self.decay = nn.Linear(1, 1)
        self.wz = nn.Linear(n_expr, hidden)
        self.uz = nn.Linear(hidden, hidden, bias=False)
        self.wr = nn.Linear(n_expr, hidden)
        self.ur = nn.Linear(hidden, hidden, bias=False)
        self.wh = nn.Linear(n_expr, hidden)
        self.uh = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden + n_ctx, n_out)

    def forward(self, expr: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        dt = (ctx[:, 1] - ctx[:, 0]).unsqueeze(1).clamp(min=0.0)
        h0 = torch.tanh(self.to_h(expr))
        gamma = torch.exp(-torch.relu(self.decay(dt)))
        h = gamma * h0
        z = torch.sigmoid(self.wz(expr) + self.uz(h))
        r = torch.sigmoid(self.wr(expr) + self.ur(h))
        h_tilde = torch.tanh(self.wh(expr) + self.uh(r * h))
        h1 = (1.0 - z) * h + z * h_tilde
        return self.head(torch.cat([h1, ctx], dim=1))


class TimeGRUModel:
    name = "time_gru"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
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
        self.model_: _TimeGRUNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "TimeGRUModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        expr = np.concatenate([p, m], axis=1)
        model = _TimeGRUNet(expr.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = _tensors((expr, ctx), self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            vp, vm, vc = _parts(bundle.val.X, bundle)
            xv = _tensors((np.concatenate([vp, vm], axis=1), vc), self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TimeGRUModel has not been fit")
        p, m, ctx = _parts(bundle.test.X, bundle)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors((np.concatenate([p, m], axis=1), ctx), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "source": "GRU-D / T-LSTM one-step dt-decay adapter",
        }


class _DeepCCANet(nn.Module):
    """Deep CCA towers (Andrew 2013) + forecast head. Alignment is MSE between view latents."""

    def __init__(self, n_p: int, n_m: int, n_ctx: int, n_out: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.p = nn.Sequential(nn.Linear(n_p, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.m = nn.Sequential(nn.Linear(n_m, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.head = nn.Linear(latent * 2 + n_ctx, n_out)

    def encode(self, p: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.p(p), self.m(m)

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        zp, zm = self.encode(p, m)
        return self.head(torch.cat([zp, zm, ctx], dim=1))


class DeepCCAFusionModel:
    name = "deep_cca_fusion"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 64,
        latent: int = 16,
        align: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.latent = latent
        self.align = align
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _DeepCCANet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "DeepCCAFusionModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _DeepCCANet(p.shape[1], m.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden, self.latent).to(
            self.device
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pt, mt, ct = _tensors((p, m, ctx), self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        has_val = len(bundle.val.X) > 0
        if has_val:
            pv, mv, cv = _tensors(_parts(bundle.val.X, bundle), self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        best_state = None
        best_val = float("inf")
        best_epoch = self.epochs
        model.train()
        for epoch in range(1, self.epochs + 1):
            opt.zero_grad(set_to_none=True)
            pred = model(pt, mt, ct)
            zp, zm = model.encode(pt, mt)
            rec = torch.mean((pred - yt) ** 2)
            (rec + self.align * torch.mean((zp - zm) ** 2)).backward()
            opt.step()
            if has_val:
                model.eval()
                with torch.no_grad():
                    val = float(torch.mean((model(pv, mv, cv) - yv) ** 2).item())
                model.train()
                if val < best_val:
                    best_val = val
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.best_epoch_ = best_epoch
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DeepCCAFusionModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "latent": self.latent,
            "align": self.align,
            "epochs": self.epochs,
            "source": "Deep CCA Andrew 2013 dual-view + forecast adapter",
        }


class _ODEFunc(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(z)


class _DualODENet(nn.Module):
    """Per-modality Neural ODE (Chen 2018) then fuse — unimodal temporal applied twice."""

    def __init__(self, n_p: int, n_m: int, n_ctx: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.enc_p = nn.Sequential(nn.Linear(n_p, hidden), nn.Tanh())
        self.enc_m = nn.Sequential(nn.Linear(n_m, hidden), nn.Tanh())
        self.func_p = _ODEFunc(hidden)
        self.func_m = _ODEFunc(hidden)
        self.head = nn.Linear(hidden * 2 + n_ctx, n_out)

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        t0, t1 = ctx[:, 0], ctx[:, 1]
        zp = _rk4(self.func_p, self.enc_p(p), t0, t1)
        zm = _rk4(self.func_m, self.enc_m(m), t0, t1)
        return self.head(torch.cat([zp, zm, ctx], dim=1))


class DualNeuralODEModel:
    name = "dual_neural_ode"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 1e-3,
        epochs: int = 40,
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
        self.model_: _DualODENet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "DualNeuralODEModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _DualODENet(p.shape[1], m.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = _tensors((p, m, ctx), self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            xv = _tensors(_parts(bundle.val.X, bundle), self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DualNeuralODEModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "source": "dual-stream Neural ODE (Chen 2018 per modality)",
        }
