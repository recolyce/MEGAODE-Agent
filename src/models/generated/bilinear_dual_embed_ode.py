"""BilinearDualEmbedODE: dual protein/metabolite embedding-conditioned vector field + RK4 integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context


class BilinearDualEmbedODE:
    name = "bilinear_dual_embed_ode"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 1e-3,
        epochs: int = 20,
        rk_steps: int = 4,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.rk_steps = rk_steps
        self.seed = seed
        self.device = resolve_device("auto")
        self.model_ = None
        self.protein_emb_ = None
        self.metabolite_emb_ = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "BilinearDualEmbedODE":
        if bundle.prior is None or bundle.prior.features is None:
            raise RuntimeError("bilinear_dual_embed_ode needs prior.features")
        self.n_expr_ = bundle.n_expr
        prot_emb = np.asarray(
            bundle.prior.features.protein_emb[: self.n_expr_, :], dtype=float
        )
        metab_emb = np.asarray(
            bundle.prior.features.metabolite_emb[: self.n_expr_, :], dtype=float
        )
        self.protein_emb_ = prot_emb
        self.metabolite_emb_ = metab_emb

        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]
        n_ctx = ctx.shape[1]
        emb_dim = prot_emb.shape[1]
        latent_dim = 2 * emb_dim

        model = nn.Module()
        model.vec_field = nn.Sequential(
            nn.Linear(latent_dim + n_ctx, self.hidden),
            nn.Softplus(),
            nn.Linear(self.hidden, self.hidden),
            nn.Softplus(),
            nn.Linear(self.hidden, latent_dim),
        )
        model.head = nn.Sequential(
            nn.Linear(latent_dim + n_ctx, self.hidden),
            nn.Softplus(),
            nn.Linear(self.hidden, n_out),
        )
        model = model.to(self.device)

        pe = torch.as_tensor(prot_emb, dtype=torch.float32, device=self.device)
        me = torch.as_tensor(metab_emb, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)

        h_init = torch.cat([xt @ pe, xt @ me], dim=1)
        dt = (ct[:, 1] - ct[:, 0]).view(-1, 1)
        dt_step = dt / float(self.rk_steps)

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            h = h_init
            for _step in range(self.rk_steps):
                k1 = model.vec_field(torch.cat([h, ct], dim=1))
                k2 = model.vec_field(torch.cat([h + 0.5 * dt_step * k1, ct], dim=1))
                k3 = model.vec_field(torch.cat([h + 0.5 * dt_step * k2, ct], dim=1))
                k4 = model.vec_field(torch.cat([h + dt_step * k3, ct], dim=1))
                h = h + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pred = model.head(torch.cat([h, ct], dim=1))
            torch.mean((pred - yt) ** 2).backward()
            opt.step()

        self.model_ = model
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("BilinearDualEmbedODE has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        model = self.model_
        model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
            ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
            pe = torch.as_tensor(self.protein_emb_, dtype=torch.float32, device=self.device)
            me = torch.as_tensor(self.metabolite_emb_, dtype=torch.float32, device=self.device)
            h = torch.cat([xt @ pe, xt @ me], dim=1)
            dt = (ct[:, 1] - ct[:, 0]).view(-1, 1)
            dt_step = dt / float(self.rk_steps)
            for _step in range(self.rk_steps):
                k1 = model.vec_field(torch.cat([h, ct], dim=1))
                k2 = model.vec_field(torch.cat([h + 0.5 * dt_step * k1, ct], dim=1))
                k3 = model.vec_field(torch.cat([h + 0.5 * dt_step * k2, ct], dim=1))
                k4 = model.vec_field(torch.cat([h + dt_step * k3, ct], dim=1))
                h = h + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pred = model.head(torch.cat([h, ct], dim=1))
        return pred.cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "rk_steps": self.rk_steps,
        }
