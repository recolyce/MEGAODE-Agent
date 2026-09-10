"""EmbeddingGraphODE: protein-embedding gated graph ODE for proteomics→metabolomics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.models.graph_ode import normalized_adjacency


class _EmbedODEFunc(nn.Module):
    """Graph ODE right-hand side with embedding-conditioned sigmoid gating."""

    def __init__(self, hidden: int, emb_dim: int) -> None:
        super().__init__()
        self.gcn = nn.Linear(hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(emb_dim, hidden), nn.Sigmoid())
        self.act = nn.Softplus()

    def forward(self, h: torch.Tensor, adj: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        g = self.gate(emb)  # [n_nodes, hidden]
        out = torch.matmul(adj, self.gcn(h))
        return self.act(out * g.unsqueeze(0))


class EmbeddingGraphODE:
    name = "embedding_graph_ode"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 16,
        lr: float = 1e-3,
        epochs: int = 20,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.device = resolve_device("auto")
        self.model_ = None
        self.adj_ = None
        self.prot_emb_ = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "EmbeddingGraphODE":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("embedding_graph_ode needs prior.laplacian")
        self.n_expr_ = bundle.n_expr
        n_out = bundle.train.Y.shape[1]

        lap = np.asarray(
            bundle.prior.laplacian[: self.n_expr_, : self.n_expr_], dtype=float
        )
        self.adj_ = normalized_adjacency(lap)

        if bundle.prior.features is None or bundle.prior.features.protein_emb is None:
            raise RuntimeError("embedding_graph_ode needs prior.features.protein_emb")
        prot_raw = np.asarray(bundle.prior.features.protein_emb, dtype=float)
        self.prot_emb_ = prot_raw[: self.n_expr_, :]
        emb_dim = self.prot_emb_.shape[1]

        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_ctx = ctx.shape[1]

        torch.manual_seed(self.seed)
        model = nn.Module()
        model.encoder = nn.Linear(1, self.hidden)
        model.func = _EmbedODEFunc(self.hidden, emb_dim)
        model.head = nn.Linear(self.n_expr_ * self.hidden + n_ctx, n_out)
        model = model.to(self.device)

        adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
        emb = torch.as_tensor(self.prot_emb_, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            h = model.encoder(xt.unsqueeze(-1))
            t0, t1 = ct[:, 0], ct[:, 1]
            steps = 2
            dt = ((t1 - t0) / float(steps)).view(-1, 1, 1)
            for _s in range(steps):
                k1 = model.func(h, adj, emb)
                k2 = model.func(h + 0.5 * dt * k1, adj, emb)
                k3 = model.func(h + 0.5 * dt * k2, adj, emb)
                k4 = model.func(h + dt * k3, adj, emb)
                h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            pred = model.head(torch.cat([h.reshape(h.size(0), -1), ct], dim=1))
            torch.mean((pred - yt) ** 2).backward()
            opt.step()

        self.model_ = model
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("EmbeddingGraphODE has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
            ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
            adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
            emb = torch.as_tensor(self.prot_emb_, dtype=torch.float32, device=self.device)
            h = self.model_.encoder(xt.unsqueeze(-1))
            t0, t1 = ct[:, 0], ct[:, 1]
            steps = 2
            dt = ((t1 - t0) / float(steps)).view(-1, 1, 1)
            for _s in range(steps):
                k1 = self.model_.func(h, adj, emb)
                k2 = self.model_.func(h + 0.5 * dt * k1, adj, emb)
                k3 = self.model_.func(h + 0.5 * dt * k2, adj, emb)
                k4 = self.model_.func(h + dt * k3, adj, emb)
                h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            pred = self.model_.head(torch.cat([h.reshape(h.size(0), -1), ct], dim=1))
        return pred.cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs}
