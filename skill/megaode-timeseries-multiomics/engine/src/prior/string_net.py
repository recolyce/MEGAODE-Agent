"""STRING v12 edges from TMO-agent-prior (or a name-rule proxy if files are missing)."""

from __future__ import annotations

import gzip
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.prior.name_rule import _gene, _protein_modules
from src.prior.paths import string_dir

REF = "Szklarczyk et al. STRING v12; local files under TMO_PRIOR_ROOT/string"
TAXON = {"human": "9606", "mouse": "10090"}
MIN_SCORE = 400


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _wanted_keys(protein_ann: pd.DataFrame) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = defaultdict(list)
    for rec in protein_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        values = {
            _gene(pd.Series(rec)),
            str(rec.get("string_preferred_name") or "").upper(),
            str(rec.get("gene_symbol") or "").upper(),
            str(rec.get("uniprot_swissprot") or rec.get("uniprot_accession") or "").upper(),
            feat.upper(),
            feat.split(".")[-1].upper() if "." in feat else "",
        }
        for value in values:
            value = value.strip()
            if value and value != "NAN":
                keys[value].append(feat)
    return keys


def _map_string_ids(aliases_path: Path, protein_ann: pd.DataFrame) -> dict[str, set[str]]:
    wanted = _wanted_keys(protein_ann)
    ensp_to_features: dict[str, set[str]] = defaultdict(set)
    if not aliases_path.exists():
        return {}
    with _open(aliases_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sid, alias = parts[0], parts[1].upper()
            if alias in wanted:
                ensp_to_features[sid].update(wanted[alias])
    return ensp_to_features


def _links_path(root: Path, taxon: str) -> Path | None:
    candidates = [
        root / f"{taxon}.protein.links.full.v12.0.txt.gz",
        root / f"{taxon}.protein.links.detailed.v12.0.txt.gz",
        root / f"{taxon}.protein.links.v12.0.txt.gz",
        root / "protein.links.tsv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _score_index(header: str) -> tuple[int, int, int]:
    cols = header.strip().split()
    lower = [c.lower() for c in cols]
    p1 = lower.index("protein1") if "protein1" in lower else 0
    p2 = lower.index("protein2") if "protein2" in lower else 1
    score = lower.index("combined_score") if "combined_score" in lower else len(cols) - 1
    return p1, p2, score


def build_string_network(
    protein_ann: pd.DataFrame,
    features: list[str],
    cache: Path | None = None,
    organism: str = "human",
    min_score: int = MIN_SCORE,
) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame, list[str]]:
    allowed = set(features)
    feat_pos = {name: i for i, name in enumerate(features)}
    n = len(features)
    adjacency = np.zeros((n, n), dtype=float)
    edges: list[dict[str, str]] = []
    warnings: list[str] = []
    root = Path(cache) if cache is not None else string_dir()
    taxon = TAXON.get(organism, "9606")
    aliases = root / f"{taxon}.protein.aliases.v12.0.txt"
    links_path = _links_path(root, taxon)
    used_official = False
    if links_path is not None:
        ensp = _map_string_ids(aliases, protein_ann)
        if not ensp:
            warnings.append(f"STRING aliases at {aliases.name} did not match annotation gene/UniProt IDs")
        else:
            keep = set(ensp)
            with _open(links_path) as handle:
                header = handle.readline()
                i1, i2, iscore = _score_index(header)
                for line in handle:
                    parts = line.split()
                    if len(parts) <= max(i1, i2, iscore):
                        continue
                    a, b = parts[i1], parts[i2]
                    if a not in keep or b not in keep or a == b:
                        continue
                    score = float(parts[iscore])
                    if score < min_score:
                        continue
                    weight = score / 1000.0 if score > 1 else score
                    for src in ensp[a]:
                        for tgt in ensp[b]:
                            if src not in allowed or tgt not in allowed or src == tgt:
                                continue
                            i, j = feat_pos[src], feat_pos[tgt]
                            adjacency[i, j] = max(adjacency[i, j], weight)
                            adjacency[j, i] = adjacency[i, j]
                            edges.append(
                                {
                                    "source_node": src,
                                    "target_node": tgt,
                                    "relationship": "protein_protein_interaction",
                                    "database": "STRING",
                                    "evidence": f"{a}-{b}:{int(score)}",
                                    "confidence": "high" if weight >= 0.7 else "medium",
                                    "reference": REF,
                                    "via": taxon,
                                }
                            )
            used_official = bool(edges)
            if not edges:
                warnings.append(f"STRING {taxon} links overlapped IDs but no pair passed score>={min_score}")
    if not used_official:
        warnings.append(
            "Official STRING unused or empty; protein–protein edges are name-rule module co-membership (string_proxy)"
        )
        groups: dict[str, list[str]] = defaultdict(list)
        for rec in protein_ann.to_dict("records"):
            feat = str(rec["feature_id"])
            if feat not in allowed:
                continue
            for module in _protein_modules(_gene(pd.Series(rec))):
                groups[module].append(feat)
        for module, members in groups.items():
            uniq = sorted(set(members))
            if len(uniq) < 2:
                continue
            for i, src in enumerate(uniq):
                for tgt in uniq[i + 1 :]:
                    a, b = feat_pos[src], feat_pos[tgt]
                    adjacency[a, b] = max(adjacency[a, b], 0.5)
                    adjacency[b, a] = adjacency[a, b]
                    edges.append(
                        {
                            "source_node": src,
                            "target_node": tgt,
                            "relationship": "protein_protein_interaction",
                            "database": "string_proxy",
                            "evidence": module,
                            "confidence": "medium",
                            "reference": REF,
                            "via": module,
                        }
                    )
    network = pd.DataFrame(edges)
    if not network.empty:
        network = network.drop_duplicates(["source_node", "target_node", "relationship"])
    provenance = pd.DataFrame(
        [
            {
                "source": "STRING" if used_official else "string_proxy",
                "date_accessed": date.today().isoformat(),
                "url_or_file": str(root),
                "version": "v12" if used_official else "proxy",
                "license": "STRING CC BY 4.0",
                "notes": "; ".join(warnings) or f"STRING {taxon} edges aligned to bundle proteins.",
            }
        ]
    )
    return network, (adjacency if n else None), provenance, warnings
