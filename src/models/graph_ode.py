"""GraphOmicsODE adapter: pathway graph on X nodes, ODE in time, linear head to Y."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context


def normalized_adjacency(laplacian: np.ndarray) -> np.ndarray:
    degree = np.diag(laplacian)
    adjacency = np.diag(degree) - laplacian
    np.fill_diagonal(adjacency, 0.0)
    deg = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    np.divide(1.0, np.sqrt(deg, where=deg > 0, out=np.ones_like(deg)), out=inv_sqrt, where=deg > 0)
    return adjacency * (inv_sqrt[:, None] * inv_sqrt[None, :])


class _GCN(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adj, self.linear(h))


class _GODEFunc(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gcn = _GCN(hidden, hidden)
        self.act = nn.Softplus()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.act(self.gcn(h, adj))


class _GraphOmicsODENet(nn.Module):
    def __init__(self, n_nodes: int, n_out: int, hidden: int, n_context: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, hidden)
        self.func = _GODEFunc(hidden)
        self.head = nn.Linear(n_nodes * hidden + n_context, n_out)

    def forward(self, expr: torch.Tensor, ctx: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(expr.unsqueeze(-1))
        t0, t1 = ctx[:, 0], ctx[:, 1]
        steps = 2
        dt = ((t1 - t0) / float(steps)).view(-1, 1, 1)
        for _ in range(steps):
            k1 = self.func(h, adj)
            k2 = self.func(h + 0.5 * dt * k1, adj)
            k3 = self.func(h + 0.5 * dt * k2, adj)
            k4 = self.func(h + dt * k3, adj)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.head(torch.cat([h.reshape(h.size(0), -1), ctx], dim=1))


class GraphOmicsODEModel:
    name = "graph_omics_ode"
    requires_prior = True

    def __init__(self, hidden: int = 16, lr: float = 1e-3, epochs: int = 25, weight_decay: float = 1e-4, seed: int = 42, device: str = "auto") -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _GraphOmicsODENet | None = None
        self.adj_: np.ndarray | None = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "GraphOmicsODEModel":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("graph_omics_ode needs bundle.prior.laplacian")
        self.n_expr_ = bundle.n_expr
        lap = bundle.prior.laplacian[: self.n_expr_, : self.n_expr_]
        self.adj_ = normalized_adjacency(lap)
        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        model = _GraphOmicsODENet(self.n_expr_, bundle.train.Y.shape[1], self.hidden, ctx.shape[1]).to(self.device)
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
                                    adj,
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
        if self.model_ is None or self.adj_ is None:
            raise RuntimeError("GraphOmicsODEModel has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(
                torch.as_tensor(expr, dtype=torch.float32, device=self.device),
                torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device),
            ).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs}
