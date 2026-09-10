"""EmbedODE: embedding-conditioned ODE with RK2 integration over last_interval dt."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context


class _EmbedODENet(nn.Module):
    """ODE net whose vector field is conditioned on protein embeddings."""

    def __init__(
        self,
        n_proteins: int,
        n_metabolites: int,
        hidden: int,
        protein_emb: torch.Tensor,
        metabolite_emb: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_proteins = n_proteins
        self.hidden = hidden
        self.register_buffer("protein_emb", protein_emb)
        self.register_buffer("metabolite_emb", metabolite_emb)

        emb_dim = protein_emb.shape[1]
        self.proj_expr = nn.Linear(1, hidden)
        self.proj_emb = nn.Linear(emb_dim, hidden)

        self.func_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )

        self.head = nn.Linear(n_proteins * hidden + 2, n_metabolites)

    def forward(self, expr: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        batch = expr.size(0)
        expr_enc = self.proj_expr(expr.unsqueeze(-1))
        emb_enc = self.proj_emb(self.protein_emb).unsqueeze(0)
        h = expr_enc + emb_enc

        dt = ((ctx[:, 1] - ctx[:, 0]) / 2.0).view(-1, 1, 1)
        for _ in range(2):
            k1 = self.func_net(h)
            k2 = self.func_net(h + 0.5 * dt * k1)
            h = h + dt * k2

        h_flat = h.reshape(batch, -1)
        return self.head(torch.cat([h_flat, ctx], dim=1))


class EmbedODE:
    name = "embed_ode"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 16,
        lr: float = 1e-3,
        epochs: int = 20,
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
        self.model_: _EmbedODENet | None = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "EmbedODE":
        if bundle.prior is None:
            raise RuntimeError("embed_ode needs bundle.prior")
        if bundle.prior.protein_emb is None or bundle.prior.metabolite_emb is None:
            raise RuntimeError("embed_ode needs prior.protein_emb and prior.metabolite_emb")

        self.n_expr_ = bundle.n_expr
        protein_emb = torch.as_tensor(
            np.asarray(bundle.prior.protein_emb, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        metabolite_emb = torch.as_tensor(
            np.asarray(bundle.prior.metabolite_emb, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]

        model = _EmbedODENet(
            n_proteins=self.n_expr_,
            n_metabolites=n_out,
            hidden=self.hidden,
            protein_emb=protein_emb,
            metabolite_emb=metabolite_emb,
        ).to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)

        best_state = None
        best_val = float("inf")
        model.train()
        for _epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = model(xt, ct)
            torch.mean((pred - yt) ** 2).backward()
            opt.step()

            if len(bundle.val.X):
                model.eval()
                with torch.no_grad():
                    ve, vc = split_context(bundle.val.X, self.n_expr_)
                    val = float(
                        torch.mean(
                            (
                                model(
                                    torch.as_tensor(ve, dtype=torch.float32, device=self.device),
                                    torch.as_tensor(vc, dtype=torch.float32, device=self.device),
                                )
                                - torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
                            )
                            ** 2
                        ).item()
                    )
                model.train()
                if val < best_val:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("EmbedODE has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(
                torch.as_tensor(expr, dtype=torch.float32, device=self.device),
                torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
            ).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
        }
