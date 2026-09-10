"""MLP, feature-chunk LSTM, and latent Neural ODE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.curator.bundle import ModelingBundle
from src.models.base import resolve_device


class _MLPNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_torch(model, opt, xt, yt, xv, yv, epochs: int):
    best_state = None
    best_val = float("inf")
    best_epoch = epochs
    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(xt) if not isinstance(xt, tuple) else model(*xt)
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        opt.step()
        if xv is not None:
            model.eval()
            with torch.no_grad():
                vp = model(xv) if not isinstance(xv, tuple) else model(*xv)
                val = float(torch.mean((vp - yv) ** 2).item())
            model.train()
            if val < best_val:
                best_val = val
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


class MLPModel:
    name = "mlp"
    requires_prior = False

    def __init__(self, hidden: int = 128, lr: float = 1e-3, epochs: int = 80, weight_decay: float = 1e-4, seed: int = 42, device: str = "auto") -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _MLPNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "MLPModel":
        torch.manual_seed(self.seed)
        x, y = bundle.train.X, bundle.train.Y
        model = _MLPNet(x.shape[1], y.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            xv = torch.as_tensor(bundle.val.X, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLPModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            xt = torch.as_tensor(bundle.test.X, dtype=torch.float32, device=self.device)
            return self.model_(xt).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs, "best_epoch": self.best_epoch_}


class _ChunkLSTMNet(nn.Module):
    def __init__(self, chunk_size: int, hidden: int, n_out: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(chunk_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, chunks: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(chunks, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _out, (h_n, _c) = self.lstm(packed)
        return self.head(h_n[-1])


def _chunk(x: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    n, p = x.shape
    n_chunks = int(np.ceil(p / chunk_size))
    padded = np.zeros((n, n_chunks * chunk_size), dtype=np.float32)
    padded[:, :p] = x
    return padded.reshape(n, n_chunks, chunk_size), np.full(n, n_chunks, dtype=np.int64)


class FeatureChunkLSTMModel:
    name = "feature_chunk_lstm"
    requires_prior = False

    def __init__(self, chunk_size: int = 64, hidden: int = 32, lr: float = 1e-3, epochs: int = 80, weight_decay: float = 1e-4, seed: int = 42, device: str = "auto") -> None:
        self.chunk_size = chunk_size
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _ChunkLSTMNet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "FeatureChunkLSTMModel":
        torch.manual_seed(self.seed)
        chunks, lengths = _chunk(bundle.train.X, self.chunk_size)
        model = _ChunkLSTMNet(self.chunk_size, self.hidden, bundle.train.Y.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = (
            torch.as_tensor(chunks, dtype=torch.float32, device=self.device),
            torch.as_tensor(lengths, dtype=torch.int64),
        )
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            c, l = _chunk(bundle.val.X, self.chunk_size)
            xv = (torch.as_tensor(c, dtype=torch.float32, device=self.device), torch.as_tensor(l, dtype=torch.int64))
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("FeatureChunkLSTMModel has not been fit")
        chunks, lengths = _chunk(bundle.test.X, self.chunk_size)
        self.model_.eval()
        with torch.no_grad():
            xt = torch.as_tensor(chunks, dtype=torch.float32, device=self.device)
            lt = torch.as_tensor(lengths, dtype=torch.int64)
            return self.model_(xt, lt).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "role": "feature_encoder_not_temporal", "chunk_size": self.chunk_size}


class _ODEFunc(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t
        return self.net(z)


class _LatentODENet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, n_context: int) -> None:
        super().__init__()
        self.n_expr = n_in - n_context
        self.n_context = n_context
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.func = _ODEFunc(hidden)
        self.decoder = nn.Linear(hidden + n_context, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = x[:, self.n_expr :]
        t0 = x[:, self.n_expr]
        t1 = x[:, self.n_expr + 1]
        z0 = self.encoder(x)
        steps = 3
        dt = (t1 - t0).unsqueeze(1) / float(steps)
        z = z0
        t = t0
        for _ in range(steps):
            k1 = self.func(t, z)
            k2 = self.func(t + 0.5 * dt.squeeze(1), z + 0.5 * dt * k1)
            k3 = self.func(t + 0.5 * dt.squeeze(1), z + 0.5 * dt * k2)
            k4 = self.func(t + dt.squeeze(1), z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt.squeeze(1)
        return self.decoder(torch.cat([z, ctx], dim=1))


class NeuralODEModel:
    name = "neural_ode"
    requires_prior = False

    def __init__(self, hidden: int = 64, lr: float = 1e-3, epochs: int = 40, weight_decay: float = 1e-4, seed: int = 42, device: str = "auto") -> None:
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = resolve_device(device)
        self.model_: _LatentODENet | None = None
        self.best_epoch_ = epochs

    def fit(self, bundle: ModelingBundle) -> "NeuralODEModel":
        torch.manual_seed(self.seed)
        n_ctx = len(bundle.context_names)
        model = _LatentODENet(bundle.train.X.shape[1], bundle.train.Y.shape[1], self.hidden, n_ctx).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt = torch.as_tensor(bundle.train.X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(bundle.train.Y, dtype=torch.float32, device=self.device)
        xv = yv = None
        if len(bundle.val.X):
            xv = torch.as_tensor(bundle.val.X, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(bundle.val.Y, dtype=torch.float32, device=self.device)
        self.model_, self.best_epoch_ = _fit_torch(model, opt, xt, yt, xv, yv, self.epochs)
        return self

    def predict(self, bundle: ModelingBundle) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("NeuralODEModel has not been fit")
        self.model_.eval()
        with torch.no_grad():
            xt = torch.as_tensor(bundle.test.X, dtype=torch.float32, device=self.device)
            return self.model_(xt).cpu().numpy()

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "hidden": self.hidden, "epochs": self.epochs, "source": "Chen et al. 2018 latent adapter"}
