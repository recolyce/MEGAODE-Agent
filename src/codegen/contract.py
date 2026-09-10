"""Contract the coding agent must follow when writing a new model."""

from __future__ import annotations

# Generated model files: deny process/network, allow any other installed package.
FORBIDDEN = {
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "os",
    "sys",
    "requests",
    "urllib",
    "http",
    "ctypes",
    "multiprocessing",
    "importlib",
    "pickle",
}
ALLOWED_IMPORT_PREFIXES = ()  # kept for compatibility; AST now uses FORBIDDEN only

EXAMPLE = '''
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device, split_context
from src.models.graph_ode import normalized_adjacency


class PriorGraphODE:
    name = "prior_graph_ode"
    requires_prior = True

    def __init__(self, hidden: int = 12, lr: float = 1e-3, epochs: int = 20) -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.device = resolve_device("auto")
        self.model_ = None
        self.adj_ = None
        self.n_expr_ = 0

    def fit(self, bundle: ModelingBundle) -> "PriorGraphODE":
        if bundle.prior is None or bundle.prior.laplacian is None:
            raise RuntimeError("needs prior.laplacian")
        self.n_expr_ = bundle.n_expr
        lap = np.asarray(bundle.prior.laplacian[: self.n_expr_, : self.n_expr_], dtype=float)
        self.adj_ = normalized_adjacency(lap)
        expr, ctx = split_context(bundle.train.X, self.n_expr_)
        n_out = bundle.train.Y.shape[1]
        n_ctx = ctx.shape[1]
        model = nn.Module()
        model.enc = nn.Linear(1, self.hidden)
        model.gcn = nn.Linear(self.hidden, self.hidden)
        model.head = nn.Linear(self.n_expr_ * self.hidden + n_ctx, n_out)
        model = model.to(self.device)
        adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            h = model.enc(xt.unsqueeze(-1))
            dt = ((ct[:, 1] - ct[:, 0]) / 2.0).view(-1, 1, 1)
            for _step in range(2):
                k1 = torch.nn.functional.softplus(torch.matmul(adj, model.gcn(h)))
                k2 = torch.nn.functional.softplus(torch.matmul(adj, model.gcn(h + dt * k1)))
                h = h + 0.5 * dt * (k1 + k2)
            pred = model.head(torch.cat([h.reshape(h.size(0), -1), ct], dim=1))
            torch.mean((pred - yt) ** 2).backward()
            opt.step()
        self.model_ = model
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        expr, ctx = split_context(bundle.test.X, self.n_expr_)
        model = self.model_
        model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(expr, dtype=torch.float32, device=self.device)
            ct = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)
            adj = torch.as_tensor(self.adj_, dtype=torch.float32, device=self.device)
            h = model.enc(xt.unsqueeze(-1))
            dt = ((ct[:, 1] - ct[:, 0]) / 2.0).view(-1, 1, 1)
            for _step in range(2):
                k1 = torch.nn.functional.softplus(torch.matmul(adj, model.gcn(h)))
                k2 = torch.nn.functional.softplus(torch.matmul(adj, model.gcn(h + dt * k1)))
                h = h + 0.5 * dt * (k1 + k2)
            pred = model.head(torch.cat([h.reshape(h.size(0), -1), ct], dim=1))
        return pred.cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs}
'''

SYSTEM = """You write ONE new last_interval model. You have tools. Each turn reply with ONE JSON object, no markdown.

Tools:
{"tool":"list_installed_packages","args":{"query":"optional substring"}}
{"tool":"install_python_packages","args":{"packages":["lightgbm"]}}
{"tool":"run_terminal","args":{"command":"PYTHONPATH=. python -c 'import torch; print(torch.__version__)'"}}
{"tool":"read_repo_file","args":{"path":"src/models/graph_ode.py"}}
{"tool":"write_generated_model","args":{"model_name":"snake","class_name":"Pascal","code":"full module text"}}
{"tool":"smoke_generated_model","args":{"model_name":"snake"}}
{"done":true,"model_name":"snake","reason":"one sentence"}

Workflow: list_installed_packages → read src/models/graph_ode.py → write_generated_model → smoke_generated_model → done.
run_terminal is only python/pip in this repo. Do not install from git URLs.

Model rules:
- The new model MUST be a prior-injected dynamical ODE (same family as graph_omics_ode):
  protein+metabolite graph or embedding-conditioned vector field, integrate hidden state over last_interval dt (RK2/RK4), then a head to both-omics Y.
- Task is multimodal: X = [proteins | metabolites | context], Y = [proteins | metabolites] at the next time.
- Use bundle.prior.laplacian / features.adjacency / protein_emb / metabolite_emb. Read graph_ode.normalized_adjacency.
- New snake_case name. Do NOT write ridge, elastic-net, GBM, output Laplacian smoother, or copy graph_omics_ode's class name.
- Class: name, requires_prior=True, fit(bundle), predict(bundle), params().
- predict uses only bundle.test.X. numpy arrays have no .values (use np.asarray).
- Generated code may import any INSTALLED scientific package. It must NOT import os, sys,
  subprocess, socket, pathlib, requests, urllib. Use tools to pip-install missing packages.
- Keep it small (few epochs). No downloads inside fit/predict.

Example pattern you may adapt (change the name):
""" + EXAMPLE
