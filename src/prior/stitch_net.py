"""STITCH protein–chemical edges from TMO-agent-prior."""

from __future__ import annotations

import gzip
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from src.prior.paths import stitch_dir
from src.prior.string_net import TAXON, _map_string_ids
from src.prior.paths import string_dir

REF = "Szklarczyk et al. STITCH v5; local files under TMO_PRIOR_ROOT/stitch"
MIN_SCORE = 400


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _cid_keys(cid: object) -> set[str]:
    raw = str(cid or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return set()
    n = int(digits)
    return {
        f"CIDm{n}",
        f"CIDs{n}",
        f"CIDm{n:08d}",
        f"CIDs{n:08d}",
        f"CIDm{n:09d}",
        f"CIDs{n:09d}",
    }


def _chemicals(metabolite_ann: pd.DataFrame) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for rec in metabolite_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        for key in _cid_keys(rec.get("pubchem_cid")):
            index[key].append(feat)
    return index


def _links_path(root: Path, taxon: str) -> Path | None:
    for name in (
        f"{taxon}.protein_chemical.links.detailed.v5.0.tsv.gz",
        f"{taxon}.protein_chemical.links.detailed.v5.0",
    ):
        path = root / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def build_stitch_network(
    protein_ann: pd.DataFrame,
    metabolite_ann: pd.DataFrame,
    x_features: list[str],
    y_features: list[str],
    organism: str = "human",
    min_score: int = MIN_SCORE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    taxon = TAXON.get(organism, "9606")
    links = _links_path(stitch_dir(), taxon)
    if links is None:
        return pd.DataFrame(), pd.DataFrame(), [f"No STITCH protein–chemical file for {taxon}"]
    aliases = string_dir() / f"{taxon}.protein.aliases.v12.0.txt"
    ensp = _map_string_ids(aliases, protein_ann)
    chems = _chemicals(metabolite_ann)
    if not ensp:
        warnings.append("STITCH: no protein IDs matched STRING aliases")
    if not chems:
        warnings.append("STITCH: no pubchem_cid in metabolite annotations")
    edges: list[dict[str, str]] = []
    if ensp and chems:
        keep_p, keep_c = set(ensp), set(chems)
        with _open(links) as handle:
            header = handle.readline().strip().split()
            lower = [c.lower() for c in header]
            ic = lower.index("chemical") if "chemical" in lower else 0
            ip = lower.index("protein") if "protein" in lower else 1
            iscore = lower.index("combined_score") if "combined_score" in lower else len(header) - 1
            for line in handle:
                parts = line.split()
                if len(parts) <= max(ic, ip, iscore):
                    continue
                chem, prot = parts[ic], parts[ip]
                if chem not in keep_c or prot not in keep_p:
                    continue
                score = float(parts[iscore])
                if score < min_score:
                    continue
                for src in ensp[prot]:
                    if src not in x_features and src not in y_features:
                        continue
                    for tgt in chems[chem]:
                        if tgt not in x_features and tgt not in y_features:
                            continue
                        edges.append(
                            {
                                "source_node": src,
                                "target_node": tgt,
                                "relationship": "enzyme_to_metabolite_reaction",
                                "database": "STITCH",
                                "evidence": f"{prot}-{chem}:{int(score)}",
                                "confidence": "high" if score >= 700 else "medium",
                                "reference": REF,
                                "via": taxon,
                            }
                        )
    if not edges:
        warnings.append(f"STITCH {taxon} produced no protein–metabolite edges at score>={min_score}")
    network = pd.DataFrame(edges)
    if not network.empty:
        network = network.drop_duplicates(["source_node", "target_node", "relationship"])
    provenance = pd.DataFrame(
        [
            {
                "source": "STITCH",
                "date_accessed": date.today().isoformat(),
                "url_or_file": str(links),
                "version": "v5.0",
                "license": "STITCH / STRING CC",
                "notes": "; ".join(warnings) or f"{len(network)} STITCH edges",
            }
        ]
    )
    return network, provenance, warnings
