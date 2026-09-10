"""EmbedOdeGcn: protein-embedding conditioned GNN-ODE with RK2 integration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.models.graph_ode import normalized_adjacency


class _EmbedODENet(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_out: int,
        hidden: int,
        n_context: int,
        prot_emb: np.ndarray,
    ) -> None:
        super().__init__()
        emb_dim = prot_emb.shape[1]
        self.prot_emb = nn.Parameter(
            torch.as_tensor(prot_emb, dtype=torch.float32), requires_grad=False
        )
        self.expr_enc = nn.Linear(1, hidden)
        self.emb_proj = nn.Linear(emb_dim, hidden)
        self.gcn1 = nn.Linear(hidden, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(n_nodes * hidden + n_context, n_out)

    def forward(
        self,
        expr: torch.Tensor,
        ctx: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        h_expr = self.expr_enc(expr.unsqueeze(-1))
        h_emb = self.emb_proj(self.prot_emb).unsqueeze(0).expand(expr.size(0), -1, -1)
        h = h_expr + h_emb

        t0, t1 = ctx[:, 0], ctx[:, 1]
        steps = 2
        dt = ((t1 - t0) / float(steps)).view(-1, 1, 1)
        for _ in range(steps):
            k1 = torch.nn.functional.softplus(
                torch.matmul(adj, self.gcn1(h)) + self.gcn2(h)
            )
            k2 = torch.nn.functional.softplus(
                torch.matmul(adj, self.gcn1(h + 0.5 * dt * k1))
                + self.gcn2(h + 0.5 * dt * k1)
            )
            h = h + dt * k2

        return self.head(torch.cat([h.reshape(h.size(0), -1), ctx], dim=1))


class EmbedOdeGcn:
    name = "embed_ode_gcn"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 32,
        lr: float = 1e-3,
        epochs: int = 18,
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
        self.adj_: np.ndarray | None = None
        self.n_expr_ = 0
        self.prot_emb_: np.ndarray | None = None

    def fit(self, bundle: ModelingBundle) -> "EmbedOdeGcn":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("embed_ode_gcn needs bundle.prior.laplacian")
        self.n_expr_ = bundle.n_expr
        lap = np.asarray(
            bundle.prior.laplacian[: self.n_expr_, : self.n_expr_], dtype=float
        )
        self.adj_ = normalized_adjacency(lap)

        if bundle.prior.protein_emb is not None:
            self.prot_emb_ = np.asarray(
                bundle.prior.protein_emb[: self.n_expr_, :], dtype=float
            )
        else:
            self.prot_emb_ = np.random.randn(self.n_expr_, 64).astype(float)

        torch.manual_seed(self.seed)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]
        model = _EmbedODENet(
            n_nodes=self.n_expr_,
            n_out=n_out,
            hidden=self.hidden,
            n_context=ctx.shape[1],
            prot_emb=self.prot_emb_,
        ).to(self.device)

        adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
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
                                    torch.as_tensor(
                                        ve, dtype=torch.float32, device=self.device
                                    ),
                                    torch.as_tensor(
                                        vc, dtype=torch.float32, device=self.device
                                    ),
                                    adj,
                                )
                                - torch.as_tensor(
                                    bundle.val.Y, dtype=torch.float32, device=self.device
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
        if self.model_ is None or self.adj_ is None:
            raise RuntimeError("EmbedOdeGcn has not been fit")
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        self.model_.eval()
        with torch.no_grad():
            return (
                self.model_(
                    torch.as_tensor(expr, dtype=torch.float32, device=self.device),
                    torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                    torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device),
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
