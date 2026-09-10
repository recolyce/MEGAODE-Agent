"""Turn a PriorAttachment into arrays a model can concatenate or regularize with."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.prior.graph import membership_sets, pad_laplacian, shared_pathway_laplacian


SOFT_PAIR = {"enzyme_to_metabolite_reaction", "pathway_comembership"}


@dataclass
class PriorFeatures:
    """Fixed-shape extras aligned to bundle.x_features / y_features / context."""

    sources: list[str]
    laplacian: np.ndarray | None
    adjacency: np.ndarray | None
    group_index: np.ndarray
    group_names: list[str]
    protein_emb: np.ndarray | None
    metabolite_emb: np.ndarray | None
    pair_prior: dict[tuple[str, str], float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def params(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "laplacian": None if self.laplacian is None else list(self.laplacian.shape),
            "adjacency": None if self.adjacency is None else list(self.adjacency.shape),
            "n_groups": int(len(self.group_names)),
            "protein_emb": None if self.protein_emb is None else list(self.protein_emb.shape),
            "metabolite_emb": None if self.metabolite_emb is None else list(self.metabolite_emb.shape),
            "n_pair_prior": int(len(self.pair_prior)),
            "notes": self.notes,
        }


def _group_index(features: list[str], membership: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    names = sorted(membership)
    index = {name: i for i, name in enumerate(names)}
    out = np.full(len(features), -1, dtype=int)
    feat_pos = {name: i for i, name in enumerate(features)}
    for group, members in membership.items():
        gid = index[group]
        for member in members:
            pos = feat_pos.get(member)
            if pos is not None and out[pos] < 0:
                out[pos] = gid
    return out, names


def _pair_prior(network: pd.DataFrame) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    if network.empty:
        return scores
    for row in network.itertuples(index=False):
        rel = str(row.relationship)
        key = (str(row.source_node), str(row.target_node))
        if rel == "enzyme_to_metabolite_reaction":
            scores[key] = max(scores.get(key, 0.0), 1.0)
        elif rel in SOFT_PAIR or rel == "protein_to_pathway":
            scores[key] = max(scores.get(key, 0.0), 0.5)
    return scores


def _align_emb(table: dict[str, np.ndarray], features: list[str]) -> np.ndarray | None:
    if not table or not features:
        return None
    dim = max(int(np.asarray(vec).size) for vec in table.values())
    out = np.zeros((len(features), dim), dtype=float)
    n_hit = 0
    for i, name in enumerate(features):
        vec = table.get(name)
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float).ravel()
        out[i, : min(dim, arr.size)] = arr[:dim]
        n_hit += 1
    return out if n_hit else None


def materialize_features(
    network: pd.DataFrame,
    features: list[str],
    y_features: list[str],
    relationship: str,
    n_context: int,
    pretrained: Any | None,
    sources: list[str],
    extra_adjacency: np.ndarray | None = None,
) -> PriorFeatures:
    notes: list[str] = []
    membership = membership_sets(network, features, relationship) if len(network) else {}
    laplacian = None
    adjacency = None
    if len(network) and features:
        laplacian = pad_laplacian(shared_pathway_laplacian(network, features), n_context)
        n = len(features)
        adjacency = np.zeros((n + n_context, n + n_context), dtype=float)
        adjacency[:n, :n] = np.diag(np.diag(laplacian[:n, :n])) - laplacian[:n, :n]
        if extra_adjacency is not None and extra_adjacency.shape[0] == n:
            adjacency[:n, :n] = np.maximum(adjacency[:n, :n], extra_adjacency)
            lap_extra = np.diag(extra_adjacency.sum(axis=1)) - extra_adjacency
            laplacian[:n, :n] = laplacian[:n, :n] + lap_extra
            notes.append("STRING/proxy adjacency added to Laplacian")
    group_index, group_names = _group_index(features, membership)
    protein_emb = metabolite_emb = None
    if pretrained is not None:
        protein_emb = _align_emb(getattr(pretrained, "protein", {}) or {}, features)
        metabolite_emb = _align_emb(getattr(pretrained, "metabolite", {}) or {}, y_features)
        notes.extend(list(getattr(pretrained, "notes", None) or []))
    return PriorFeatures(
        sources=list(sources),
        laplacian=laplacian,
        adjacency=adjacency,
        group_index=group_index,
        group_names=group_names,
        protein_emb=protein_emb,
        metabolite_emb=metabolite_emb,
        pair_prior=_pair_prior(network),
        notes=notes,
    )


def pathway_design(expr: np.ndarray, features: list[str], membership: dict[str, list[str]], min_members: int = 2) -> np.ndarray:
    """Sample × pathway mean scores; empty membership → zeros (n, 1)."""
    from src.prior.graph import pathway_score_matrix

    if not membership:
        return np.zeros((expr.shape[0], 1), dtype=float)
    frame = pd.DataFrame(expr, columns=features)
    try:
        scores = pathway_score_matrix(frame, membership, min_members=min_members)
    except RuntimeError:
        return np.zeros((expr.shape[0], 1), dtype=float)
    return scores.to_numpy(dtype=float)
