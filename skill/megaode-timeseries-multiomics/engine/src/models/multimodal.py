"""Multimodal last-interval models: dual-stream fusion plus concat Neural ODE adapters."""

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


class _CrossAttnNet(nn.Module):
    """CrossAttOmics / TMO-Net style bidirectional cross-attention (Beaude 2025; Sun 2024)."""

    def __init__(self, n_p: int, n_m: int, n_ctx: int, n_out: int, hidden: int, n_heads: int) -> None:
        super().__init__()
        heads = max(1, int(n_heads))
        while hidden % heads and heads > 1:
            heads -= 1
        self.p_enc = nn.Sequential(nn.Linear(n_p, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.m_enc = nn.Sequential(nn.Linear(n_m, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.p_from_m = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.m_from_p = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + n_ctx, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        hp = self.p_enc(p).unsqueeze(1)
        hm = self.m_enc(m).unsqueeze(1)
        p2, _ = self.p_from_m(hp, hm, hm)
        m2, _ = self.m_from_p(hm, hp, hp)
        return self.head(torch.cat([p2.squeeze(1), m2.squeeze(1), ctx], dim=1))


class CrossAttnFusionModel:
    name = "cross_attn_fusion"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 64,
        n_heads: int = 4,
        lr: float = 1e-3,
        epochs: int = 40,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.n_heads = n_heads
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _CrossAttnNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "CrossAttnFusionModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _CrossAttnNet(p.shape[1], m.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden, self.n_heads).to(
            self.device
        )
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
            raise RuntimeError("CrossAttnFusionModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "epochs": self.epochs,
            "source": "CrossAttOmics / TMO-Net cross-attention adapter",
        }


class _GatedNet(nn.Module):
    """PULSE / MoACG style gated late fusion of two omics encoders."""

    def __init__(self, n_p: int, n_m: int, n_ctx: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.p_enc = nn.Sequential(nn.Linear(n_p, hidden), nn.GELU())
        self.m_enc = nn.Sequential(nn.Linear(n_m, hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.head = nn.Linear(hidden + n_ctx, n_out)

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        hp = self.p_enc(p)
        hm = self.m_enc(m)
        g = self.gate(torch.cat([hp, hm], dim=1))
        return self.head(torch.cat([g * hp + (1.0 - g) * hm, ctx], dim=1))


class GatedFusionModel:
    name = "gated_fusion"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 64,
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
        self.model_: _GatedNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "GatedFusionModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _GatedNet(p.shape[1], m.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden).to(self.device)
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
            raise RuntimeError("GatedFusionModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs, "source": "gated multimodal fusion"}


class _KoopmanNet(nn.Module):
    """EKATP-style encoder → linear Koopman step → decoder (Lusch 2018; Liu 2021)."""

    def __init__(self, n_in: int, n_out: int, latent: int, hidden: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, latent))
        self.K = nn.Linear(latent, latent, bias=False)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.K(self.enc(x)))


class KoopmanAEModel:
    name = "koopman_ae"
    requires_prior = False

    def __init__(
        self,
        latent: int = 32,
        hidden: int = 64,
        lr: float = 1e-3,
        epochs: int = 40,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.latent = latent
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _KoopmanNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "KoopmanAEModel":
        torch.manual_seed(self.seed)
        x, y = bundle.train.X, bundle.train.Y
        model = _KoopmanNet(x.shape[1], y.shape[1], self.latent, self.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            xv = torch.as_tensor(bundle.val.X, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("KoopmanAEModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            xt = torch.as_tensor(bundle.test.X, dtype=torch.float32, device=self.device)
            return self.model_(xt).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "latent": self.latent,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "source": "EKATP Koopman autoencoder adapter",
        }


class _DualLSTMNet(nn.Module):
    def __init__(self, hidden: int, n_ctx: int, n_out: int) -> None:
        super().__init__()
        self.p_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.m_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden * 2 + n_ctx, n_out)

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        _o, (hp, _) = self.p_lstm(p.unsqueeze(-1))
        _o, (hm, _) = self.m_lstm(m.unsqueeze(-1))
        return self.head(torch.cat([hp[-1], hm[-1], ctx], dim=1))


class DualLSTMModel:
    name = "dual_lstm"
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
        self.model_: _DualLSTMNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "DualLSTMModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _DualLSTMNet(self.hidden, ctx.shape[1], bundle.train.Y.shape[1]).to(self.device)
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
            raise RuntimeError("DualLSTMModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs, "source": "dual-stream LSTM"}


class _MMVAENet(nn.Module):
    """TMO-Net style dual VAE latents fused to forecast the next joint state."""

    def __init__(self, n_p: int, n_m: int, n_ctx: int, n_out: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.p_mu = nn.Sequential(nn.Linear(n_p, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.p_lv = nn.Linear(n_p, latent)
        self.m_mu = nn.Sequential(nn.Linear(n_m, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.m_lv = nn.Linear(n_m, latent)
        self.head = nn.Linear(latent * 2 + n_ctx, n_out)

    def encode(self, p: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.p_mu(p), self.p_lv(p), self.m_mu(m), self.m_lv(m)

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        p_mu, p_lv, m_mu, m_lv = self.encode(p, m)
        if self.training:
            p_z = p_mu + torch.exp(0.5 * p_lv) * torch.randn_like(p_mu)
            m_z = m_mu + torch.exp(0.5 * m_lv) * torch.randn_like(m_mu)
        else:
            p_z, m_z = p_mu, m_mu
        return self.head(torch.cat([p_z, m_z, ctx], dim=1))


class MMVAEForecastModel:
    name = "mmvae_forecast"
    requires_prior = False

    def __init__(
        self,
        hidden: int = 64,
        latent: int = 16,
        beta: float = 1e-3,
        lr: float = 1e-3,
        epochs: int = 40,
        weight_decay: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.latent = latent
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _MMVAENet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "MMVAEForecastModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _MMVAENet(p.shape[1], m.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden, self.latent).to(
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
            p_mu, p_lv, m_mu, m_lv = model.encode(pt, mt)
            rec = torch.mean((pred - yt) ** 2)
            kl = 0.5 * (
                torch.mean(p_mu**2 + torch.exp(p_lv) - 1.0 - p_lv) + torch.mean(m_mu**2 + torch.exp(m_lv) - 1.0 - m_lv)
            )
            (rec + self.beta * kl).backward()
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
            raise RuntimeError("MMVAEForecastModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "latent": self.latent,
            "beta": self.beta,
            "epochs": self.epochs,
            "source": "TMO-Net dual-VAE forecast adapter",
        }


class _MogonetNet(nn.Module):
    """MOGONET-inspired dual towers + compact view-correlation bilinear fusion (Wang 2021)."""

    def __init__(self, n_p: int, n_m: int, n_ctx: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.p = nn.Sequential(nn.Linear(n_p, hidden), nn.GELU())
        self.m = nn.Sequential(nn.Linear(n_m, hidden), nn.GELU())
        self.bilinear = nn.Bilinear(hidden, hidden, hidden)
        self.head = nn.Linear(hidden * 3 + n_ctx, n_out)

    def forward(self, p: torch.Tensor, m: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        hp = self.p(p)
        hm = self.m(m)
        return self.head(torch.cat([hp, hm, self.bilinear(hp, hm), ctx], dim=1))


class MogonetFusionModel:
    name = "mogonet_fusion"
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
        self.model_: _MogonetNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "MogonetFusionModel":
        torch.manual_seed(self.seed)
        p, m, ctx = _parts(bundle.train.X, bundle)
        model = _MogonetNet(p.shape[1], m.shape[1], ctx.shape[1], bundle.train.Y.shape[1], self.hidden).to(self.device)
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
            raise RuntimeError("MogonetFusionModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(*_tensors(_parts(bundle.test.X, bundle), self.device)).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs, "source": "MOGONET dual-tower adapter"}
