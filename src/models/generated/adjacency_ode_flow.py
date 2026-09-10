"""AdjacencyODEFlow: graph-conditioned ODE over adjacency, RK4 over last_interval, head to Y."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.models.graph_ode import normalized_adjacency


class _AdjFlowFunc(nn.Module):
    """Vector field: adjacency message passing + residual."""

    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.msg = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(adj, self.msg(h))
        gate = torch.sigmoid(self.gate(torch.cat([h, aggregated], dim=-1)))
        return self.dropout(gate * aggregated + (1 - gate) * h)


class _AdjODENet(nn.Module):
    def __init__(
        self, n_nodes: int, n_out: int, hidden: int, n_context: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LayerNorm(hidden),
            nn.Softplus(),
        )
        self.func = _AdjFlowFunc(hidden, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(n_nodes * hidden + n_context),
            nn.Linear(n_nodes * hidden + n_context, n_out),
        )

    def forward(self, expr: torch.Tensor, ctx: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(expr.unsqueeze(-1))
        t0, t1 = ctx[:, 0], ctx[:, 1]
        dt = (t1 - t0).view(-1, 1, 1)
        # RK4 single step over last_interval
        k1 = self.func(h, adj)
        k2 = self.func(h + 0.5 * dt * k1, adj)
        k3 = self.func(h + 0.5 * dt * k2, adj)
        k4 = self.func(h + dt * k3, adj)
        h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        flat = h.reshape(h.size(0), -1)
        return self.head(torch.cat([flat, ctx], dim=-1))


class AdjacencyODEFlow:
    name = "adjacency_ode_flow"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 16,
        lr: float = 1e-3,
        epochs: int = 20,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _AdjODENet | None = None
        self.adj_: np.ndarray | None = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "AdjacencyODEFlow":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("adjacency_ode_flow needs bundle.prior.laplacian")
        self.n_expr_ = bundle.n_expr
        lap = bundle.prior.laplacian[: self.n_expr_, : self.n_expr_]
        self.adj_ = normalized_adjacency(lap)
        # If prior.features.adjacency exists on the same node set, blend it
        if bundle.prior.features is not None and bundle.prior.features.adjacency is not None:
            raw_adj = np.asarray(
                bundle.prior.features.adjacency[: self.n_expr_, : self.n_expr_], dtype=float
            )
            if raw_adj.shape == self.adj_.shape:
                self.adj_ = 0.3 * self.adj_ + 0.7 * raw_adj
                rowsum = self.adj_.sum(axis=1, keepdims=True)
                rowsum[rowsum == 0] = 1.0
                self.adj_ = self.adj_ / rowsum

        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]
        n_ctx = ctx.shape[1]
        model = _AdjODENet(self.n_expr_, n_out, self.hidden, n_ctx, self.dropout).to(self.device)
        adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        best_state = None
        best_val = float("inf")
        model.train()
        for _epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = model(xt, ct, adj)
            loss = torch.mean((pred - yt) ** 2)
            loss.backward()
            opt.step()
            if len(bundle.val.X):
                model.eval()
                with torch.no_grad():
                    ve, vc = split_context(bundle.val.X, self.n_expr_)
                    val_pred = model(
                        torch.as_tensor(ve, dtype=torch.float32, device=self.device),
                        torch.as_tensor(vc, dtype=torch.float32, device=self.device),
                        adj,
                    )
                    val_loss = float(
                        torch.mean(
                            (val_pred - torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)) ** 2
                        ).item()
                    )
                model.train()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None or self.adj_ is None:
            raise RuntimeError("AdjacencyODEFlow has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(
                torch.as_tensor(expr, dtype=torch.float32, device=self.device),
                torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device),
            ).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "lr": self.lr,
        }
