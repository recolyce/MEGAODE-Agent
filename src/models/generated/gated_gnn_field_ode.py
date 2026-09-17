"""Gated GNN-field ODE: static GNN context from prior embeddings conditions an RK4 vector field with time-modulated prior gate and residual MLP."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.models.graph_ode import normalized_adjacency


class _GNNContextEncoder(nn.Module):
    """Propagate pretrained node embeddings over the prior adjacency to produce per‑node GNN context vectors."""

    def __init__(self, n_nodes: int, emb_dim: int, hidden_ctx: int, n_layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden_ctx)
        self.gcn_layers = nn.ModuleList([nn.Linear(hidden_ctx, hidden_ctx) for _ in range(n_layers)])
        self.act = nn.Softplus()

    def forward(self, node_emb: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj(node_emb))
        for gcn in self.gcn_layers:
            h = self.act(torch.matmul(adj, gcn(h)))
        return h


class _GatedODEFunc(nn.Module):
    """ODE right‑hand side: deep MLP over [state | gated GNN context | pooled omics contexts | time] with a residual skip."""

    def __init__(self, hidden: int, hidden_ctx: int) -> None:
        super().__init__()
        in_dim = hidden + hidden_ctx + 2 * hidden_ctx + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.Softplus(),
            nn.Linear(hidden * 2, hidden),
        )
        self.res = nn.Linear(hidden, hidden)
        self.time_gate = nn.Sequential(nn.Linear(1, hidden_ctx), nn.Sigmoid())

    def forward(
        self,
        h: torch.Tensor,
        gnn_ctx: torch.Tensor,
        prot_ctx: torch.Tensor,
        metab_ctx: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = h.shape
        gate = self.time_gate(t)  # [B, hidden_ctx]
        gnn_exp = gnn_ctx.unsqueeze(0).expand(B, -1, -1)  # [B, N, hidden_ctx]
        gated_gnn = gnn_exp * gate.unsqueeze(1)           # time‑modulated prior gating
        prot_exp = prot_ctx.unsqueeze(1).expand(-1, N, -1)
        metab_exp = metab_ctx.unsqueeze(1).expand(-1, N, -1)
        t_exp = t.unsqueeze(1).expand(-1, N, -1)
        feat = torch.cat([h, gated_gnn, prot_exp, metab_exp, t_exp], dim=-1)
        return self.net(feat) + self.res(h)


class GatedGNNFieldODE:
    name = "gated_gnn_field_ode"
    requires_prior = True

    def __init__(
        self,
        hidden: int = 12,
        hidden_ctx: int = 16,
        lr: float = 1e-3,
        epochs: int = 20,
        num_steps: int = 4,
        device: str = "auto",
    ) -> None:
        self.hidden = hidden
        self.hidden_ctx = hidden_ctx
        self.lr = lr
        self.epochs = epochs
        self.num_steps = num_steps
        self.device = resolve_device(device)
        self.model_: tuple | None = None
        self.adj_: np.ndarray | None = None
        self.n_expr_ = 0
        self.n_protein_ = 0
        self.n_metabolite_ = 0
        self.gnn_ctx_: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def fit(self, bundle: ModelingBundle) -> "GatedGNNFieldODE":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("gated_gnn_field_ode needs bundle.prior.laplacian")

        self.n_expr_ = bundle.n_expr
        self.n_protein_ = bundle.n_protein
        self.n_metabolite_ = bundle.n_metabolite

        # ---- normalised adjacency --------------------------------------------------
        lap = np.asarray(bundle.prior.laplacian[: self.n_expr_, : self.n_expr_], dtype=float)
        self.adj_ = normalized_adjacency(lap)
        adj_t = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)

        # ---- build joint node embeddings from prior features -----------------------
        prot_emb = np.asarray(bundle.prior.features.protein_emb[: self.n_expr_, :], dtype=float)
        metab_emb = np.asarray(bundle.prior.features.metabolite_emb[: self.n_expr_, :], dtype=float)
        emb_dim = prot_emb.shape[1]
        node_emb = np.zeros((self.n_expr_, emb_dim), dtype=float)
        node_emb[: self.n_protein_, :] = prot_emb[: self.n_protein_, :]
        node_emb[self.n_protein_ :, :] = metab_emb[self.n_protein_ :, :]

        # ---- pre‑compute static per‑node GNN context ------------------------------
        gnn_encoder = _GNNContextEncoder(self.n_expr_, emb_dim, self.hidden_ctx).to(self.device)
        node_emb_t = torch.as_tensor(node_emb, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            gnn_out = gnn_encoder(node_emb_t, adj_t)
        self.gnn_ctx_ = gnn_out.detach()  # [N, hidden_ctx] frozen

        # ---- pooled omics‑type context vectors ------------------------------------
        prot_pool = self.gnn_ctx_[: self.n_protein_].mean(dim=0)      # [hidden_ctx]
        metab_pool = self.gnn_ctx_[self.n_protein_ :].mean(dim=0)     # [hidden_ctx]

        # ---- data -----------------------------------------------------------------
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)

        # ---- trainable components -------------------------------------------------
        encoder = nn.Linear(1, self.hidden).to(self.device)
        ode_func = _GatedODEFunc(self.hidden, self.hidden_ctx).to(self.device)
        head = nn.Linear(self.hidden, 1).to(self.device)

        all_params = list(encoder.parameters()) + list(ode_func.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(all_params, lr=self.lr)

        encoder.train()
        ode_func.train()
        head.train()

        t0 = ct[:, 0]
        t1 = ct[:, 1]
        dt = (t1 - t0) / float(self.num_steps)

        for _epoch in range(self.epochs):
            opt.zero_grad(set_to_none=True)

            h = encoder(xt.unsqueeze(-1))  # [B, N, hidden]

            for step in range(self.num_steps):
                t_cur = t0 + dt * float(step)
                t_mid = t_cur + 0.5 * dt
                t_nxt = t_cur + dt

                B = h.size(0)
                prot_b = prot_pool.unsqueeze(0).expand(B, -1)
                metab_b = metab_pool.unsqueeze(0).expand(B, -1)

                # RK4
                k1 = ode_func(h, self.gnn_ctx_, prot_b, metab_b, t_cur.unsqueeze(-1))
                k2 = ode_func(h + 0.5 * dt.view(-1, 1, 1) * k1, self.gnn_ctx_, prot_b, metab_b, t_mid.unsqueeze(-1))
                k3 = ode_func(h + 0.5 * dt.view(-1, 1, 1) * k2, self.gnn_ctx_, prot_b, metab_b, t_mid.unsqueeze(-1))
                k4 = ode_func(h + dt.view(-1, 1, 1) * k3, self.gnn_ctx_, prot_b, metab_b, t_nxt.unsqueeze(-1))
                h = h + (dt.view(-1, 1, 1) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            pred = head(h).squeeze(-1)
            loss = torch.mean((pred - yt) ** 2)
            loss.backward()
            opt.step()

        self.model_ = (encoder, ode_func, head)
        return self

    # ------------------------------------------------------------------
    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None or self.adj_ is None or self.gnn_ctx_ is None:
            raise RuntimeError("GatedGNNFieldODE has not been fit")

        encoder, ode_func, head = self.model_
        encoder.eval()
        ode_func.eval()
        head.eval()

        expr, ctx = split_context(bundle.test.X, self.n_expr_)

        prot_pool = self.gnn_ctx_[: self.n_protein_].mean(dim=0)
        metab_pool = self.gnn_ctx_[self.n_protein_ :].mean(dim=0)

        with torch.no_grad():
            xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
            ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)

            h = encoder(xt.unsqueeze(-1))

            t0 = ct[:, 0]
            t1 = ct[:, 1]
            dt = (t1 - t0) / float(self.num_steps)

            for step in range(self.num_steps):
                t_cur = t0 + dt * float(step)
                t_mid = t_cur + 0.5 * dt
                t_nxt = t_cur + dt

                B = h.size(0)
                prot_b = prot_pool.unsqueeze(0).expand(B, -1)
                metab_b = metab_pool.unsqueeze(0).expand(B, -1)

                k1 = ode_func(h, self.gnn_ctx_, prot_b, metab_b, t_cur.unsqueeze(-1))
                k2 = ode_func(h + 0.5 * dt.view(-1, 1, 1) * k1, self.gnn_ctx_, prot_b, metab_b, t_mid.unsqueeze(-1))
                k3 = ode_func(h + 0.5 * dt.view(-1, 1, 1) * k2, self.gnn_ctx_, prot_b, metab_b, t_mid.unsqueeze(-1))
                k4 = ode_func(h + dt.view(-1, 1, 1) * k3, self.gnn_ctx_, prot_b, metab_b, t_nxt.unsqueeze(-1))
                h = h + (dt.view(-1, 1, 1) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            pred = head(h).squeeze(-1)
        return pred.cpu().numpy()

    # ------------------------------------------------------------------
    def params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "hidden": self.hidden,
            "hidden_ctx": self.hidden_ctx,
            "epochs": self.epochs,
            "num_steps": self.num_steps,
        }
