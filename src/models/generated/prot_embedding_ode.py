"""ProtEmbeddingODE: protein-embedding-conditioned graph ODE for metabolomics prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.models.graph_ode import normalized_adjacency


class _ProtEmbedODEFunc(nn.Module):
    """ODE right-hand side: graph convolution + protein-embedding bias."""

    def __init__(self, hidden: int, prot_emb_dim: int) -> None:
        super().__init__()
        self.gcn = nn.Linear(hidden, hidden)
        self.prot_cond = nn.Linear(prot_emb_dim, hidden)
        self.act = nn.Softplus()

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, prot_emb: torch.Tensor
    ) -> torch.Tensor:
        gcn_out = torch.matmul(adj, self.gcn(h))
        prot_bias = self.prot_cond(prot_emb).unsqueeze(0)
        return self.act(gcn_out + prot_bias)


class _ProtEmbedODENet(nn.Module):
    """Full network: encode -> RK4 integrate with protein embeddings -> head."""

    def __init__(
        self,
        n_nodes: int,
        n_out: int,
        hidden: int,
        n_context: int,
        prot_emb_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, hidden)
        self.func = _ProtEmbedODEFunc(hidden, prot_emb_dim)
        self.head = nn.Linear(n_nodes * hidden + n_context, n_out)

    def forward(
        self,
        expr: torch.Tensor,
        ctx: torch.Tensor,
        adj: torch.Tensor,
        prot_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(expr.unsqueeze(-1))
        t0, t1 = ctx[:, 0], ctx[:, 1]
        steps = 2
        dt = ((t1 - t0) / float(steps)).view(-1, 1, 1)
        for _ in range(steps):
            k1 = self.func(h, adj, prot_emb)
            k2 = self.func(h + 0.5 * dt * k1, adj, prot_emb)
            k3 = self.func(h + 0.5 * dt * k2, adj, prot_emb)
            k4 = self.func(h + dt * k3, adj, prot_emb)
            h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.head(torch.cat([h.reshape(h.size(0), -1), ctx], dim=1))


class ProtEmbeddingODE:
    """Prior-injected ODE model using protein embeddings to condition graph dynamics.

    Encodes protein expression into latent states, evolves them via a
    graph-informed ODE whose right-hand side is augmented with learned
    projections of pre-trained protein embeddings, then predicts
    metabolite abundances.
    """

    name = "prot_embedding_ode"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 12,
        lr: float = 1e-3,
        epochs: int = 20,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _ProtEmbedODENet | None = None
        self.adj_: np.ndarray | None = None
        self.prot_emb_: np.ndarray | None = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "ProtEmbeddingODE":
        if bundle.prior is None:
            raise RuntimeError("prot_embedding_ode needs bundle.prior")
        if bundle.prior.laplacian is None:
            raise RuntimeError("prot_embedding_ode needs bundle.prior.laplacian")
        if bundle.prior.protein_emb is None:
            raise RuntimeError("prot_embedding_ode needs bundle.prior.protein_emb")

        self.n_expr_ = bundle.n_expr
        lap = np.asarray(
            bundle.prior.laplacian[: self.n_expr_, : self.n_expr_], dtype=float
        )
        self.adj_ = normalized_adjacency(lap)
        self.prot_emb_ = np.asarray(
            bundle.prior.protein_emb[: self.n_expr_, :], dtype=float
        )

        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]
        n_ctx = ctx.shape[1]
        prot_emb_dim = self.prot_emb_.shape[1]

        model = _ProtEmbedODENet(
            self.n_expr_, n_out, self.hidden, n_ctx, prot_emb_dim
        ).to(self.device)
        adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
        prot_emb = torch.as_tensor(
            self.prot_emb_, dtype=torch.float32, device=self.device
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(
            bundle.train.Y, dtype=torch.float32, device=self.device
        )

        best_state = None
        best_val = float("inf")
        model.train()
        for _epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = model(xt, ct, adj, prot_emb)
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
                                    torch.as_tensor(
                                        ve, dtype=torch.float32, device=self.device
                                    ),
                                    torch.as_tensor(
                                        vc, dtype=torch.float32, device=self.device
                                    ),
                                    adj,
                                    prot_emb,
                                )
                                - torch.as_tensor(
                                    bundle.val.Y,
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            ** 2
                        ).item()
                    )
                model.train()
                if val < best_val:
                    best_val = val
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None or self.adj_ is None or self.prot_emb_ is None:
            raise RuntimeError("ProtEmbeddingODE has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            return (
                self.model_(
                    torch.as_tensor(expr, dtype=torch.float32, device=self.device),
                    torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                    torch.as_tensor(
                        self.adj_, dtype=torch.float32, device=self.device
                    ),
                    torch.as_tensor(
                        self.prot_emb_, dtype=torch.float32, device=self.device
                    ),
                )
                .cpu()
                .numpy()
            )

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
        }
