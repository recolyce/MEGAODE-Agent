"""Membership and Laplacian helpers from a protein–metabolite–pathway network."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERSHIP_REL = {"protein_to_pathway", "metabolite_to_pathway"}
SOFT_REL = {"pathway_comembership", "enzyme_to_metabolite_reaction"}


def load_network(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def membership_sets(
    network: pd.DataFrame,
    features: list[str],
    relationship: str,
) -> dict[str, list[str]]:
    allowed = set(features)
    groups: dict[str, list[str]] = defaultdict(list)
    sub = network.loc[network["relationship"] == relationship]
    for row in sub.itertuples(index=False):
        if row.source_node in allowed:
            groups[str(row.target_node)].append(str(row.source_node))
    return {key: sorted(set(values)) for key, values in groups.items() if values}


def pathway_score_matrix(
    frame: pd.DataFrame,
    membership: dict[str, list[str]],
    min_members: int = 2,
    include_unmapped: bool = True,
) -> pd.DataFrame:
    mapped: set[str] = set()
    columns: dict[str, pd.Series] = {}
    for pathway, members in sorted(membership.items()):
        present = [name for name in members if name in frame.columns]
        if len(present) < min_members:
            continue
        mapped.update(present)
        columns[pathway] = frame.loc[:, present].mean(axis=1, skipna=True)
    if include_unmapped:
        leftover = [name for name in frame.columns if name not in mapped]
        if leftover:
            columns["unmapped"] = frame.loc[:, leftover].mean(axis=1, skipna=True)
    if not columns:
        raise RuntimeError("No pathway groups overlapped the selected features")
    return pd.DataFrame(columns, index=frame.index)


def shared_pathway_laplacian(network: pd.DataFrame, features: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(features)}
    n = len(features)
    adjacency = np.zeros((n, n), dtype=float)
    pathway_members: dict[str, list[str]] = defaultdict(list)
    sub = network.loc[network["relationship"].isin(MEMBERSHIP_REL)]
    for row in sub.itertuples(index=False):
        if row.source_node in index:
            pathway_members[str(row.target_node)].append(str(row.source_node))
    for members in pathway_members.values():
        present = [name for name in set(members) if name in index]
        if len(present) < 2:
            continue
        for i, left in enumerate(present):
            for right in present[i + 1 :]:
                a, b = index[left], index[right]
                adjacency[a, b] = 1.0
                adjacency[b, a] = 1.0
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def pad_laplacian(laplacian: np.ndarray, n_extra: int) -> np.ndarray:
    if n_extra <= 0:
        return laplacian
    n = laplacian.shape[0]
    out = np.zeros((n + n_extra, n + n_extra), dtype=float)
    out[:n, :n] = laplacian
    return out
