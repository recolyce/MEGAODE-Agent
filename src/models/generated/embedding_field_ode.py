"""EmbeddingFieldODE: pretrained-embedding-conditioned vector field + RK2 integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context


class _EmbeddingFieldODENet(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_out: int,
        hidden: int,
        n_context: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, hidden)
        self.field = nn.Sequential(
            nn.Linear(hidden + emb_dim, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(n_nodes * hidden + n_context, n_out)

    def forward(
        self,
        expr: torch.Tensor,
        ctx: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(expr.unsqueeze(-1))
        t0, t1 = ctx[:, 0], ctx[:, 1]
        dt = ((t1 - t0) / 2.0).view(-1, 1, 1)
        emb_b = emb.unsqueeze(0).expand(h.size(0), -1, -1)
        for _step in range(2):
            k1 = self.field(torch.cat([h, emb_b], dim=-1))
            k2 = self.field(torch.cat([h + 0.5 * dt * k1, emb_b], dim=-1))
            h = h + dt * k2
        return self.head(torch.cat([h.reshape(h.size(0), -1), ctx], dim=1))


class EmbeddingFieldODE:
    name = "embedding_field_ode"
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
        self.model_: _EmbeddingFieldODENet | None = None
        self.emb_: torch.Tensor | None = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "EmbeddingFieldODE":
        if bundle.prior is None:
            raise RuntimeError("embedding_field_ode needs bundle.prior")
        features = bundle.prior.features
        if features is None:
            raise RuntimeError("embedding_field_ode needs prior.features")
        prot_emb = np.asarray(features.protein_emb, dtype=np.float32)
        metab_emb = np.asarray(features.metabolite_emb, dtype=np.float32)
        self.n_expr_ = bundle.n_expr
        prot_emb = prot_emb[: self.n_expr_, :]
        metab_emb = metab_emb[: self.n_expr_, :]
        emb = np.concatenate([prot_emb, metab_emb], axis=1)
        self.emb_ = torch.as_tensor(emb, dtype=torch.float32, device=self.device)

        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]
        n_ctx = ctx.shape[1]
        model = _EmbeddingFieldODENet(
            self.n_expr_, n_out, self.hidden, n_ctx, emb.shape[1]
        ).to(self.device)
        opt = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)

        best_state = None
        best_val = float("inf")
        model.train()
        for _epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = model(xt, ct, self.emb_)
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
                                    self.emb_,
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
        if self.model_ is None or self.emb_ is None:
            raise RuntimeError("EmbeddingFieldODE has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            return (
                self.model_(
                    torch.as_tensor(expr, dtype=torch.float32, device=self.device),
                    torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                    self.emb_,
                )
                .cpu()
                .numpy()
            )

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
        }
