"""KEGG / Reactome protein–metabolite pathway edges from BioMaster annotations.

Downloads a few REST tables into a local cache. Metabolite HMDB→KEGG xrefs are
optional; if the cache has no compound map, only protein pathway edges are written.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ORG_CODE = {"mouse": "mmu", "human": "hsa"}


def _kegg_files(org_code: str) -> dict[str, str]:
    return {
        f"conv_{org_code}_uniprot.txt": f"https://rest.kegg.jp/conv/{org_code}/uniprot",
        f"link_pathway_{org_code}.txt": f"https://rest.kegg.jp/link/pathway/{org_code}",
        f"list_pathway_{org_code}.txt": f"https://rest.kegg.jp/list/pathway/{org_code}",
        "link_pathway_compound.txt": "https://rest.kegg.jp/link/pathway/compound",
    }


KEGG_FILES = _kegg_files("mmu")
REACTOME_UNIPROT = "https://reactome.org/download/current/UniProt2Reactome.txt"
KEGG_REF = "Kanehisa et al. KEGG REST https://rest.kegg.jp"
REACTOME_REF = "Milacic et al. Nucleic Acids Res. 2024; Reactome download current"


def _fetch(url: str, dest: Path, timeout: int = 25) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = Request(url, headers={"User-Agent": "tmo-data-ready/0.1"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            dest.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001 — cache miss is non-fatal
        dest.write_text("", encoding="utf-8")
        print(f"skip {url}: {exc}", flush=True)
    return dest


def _two_col(path: Path) -> list[tuple[str, str]]:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def _tokens(value: object) -> list[str]:
    text = str(value or "")
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip() and part.lower() != "nan"]


def _pad_hmdb(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return "HMDB" + digits.zfill(7) if digits else ""


def prepare_cache(cache: Path, extra: Path | None = None, organism: str = "mouse") -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    files = _kegg_files(ORG_CODE.get(organism, "mmu"))
    if extra and extra.exists():
        for name in files:
            src = extra / name
            dest = cache / name
            if src.exists() and not dest.exists():
                dest.write_bytes(src.read_bytes())
        for name in ("UniProt2Reactome.txt", "metabolite_identifier_map.csv"):
            src = extra / name
            dest = cache / name
            if src.exists() and not dest.exists():
                dest.write_bytes(src.read_bytes())
    for name, url in files.items():
        _fetch(url, cache / name)
    if not (cache / "UniProt2Reactome.txt").exists():
        _fetch(REACTOME_UNIPROT, cache / "UniProt2Reactome.txt")
    return cache


def build_kegg_reactome_network(
    protein_ann: pd.DataFrame,
    metabolite_ann: pd.DataFrame,
    cache: Path,
    organism: str = "mouse",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    org = ORG_CODE.get(organism, "mmu")
    prefix = f"{org}:"
    uniprot_to_org: dict[str, set[str]] = defaultdict(set)
    for left, right in _two_col(cache / f"conv_{org}_uniprot.txt"):
        if left.startswith(prefix):
            left, right = right, left
        uniprot_to_org[left.replace("up:", "").replace("uniprot:", "")].add(right)
    org_to_path: dict[str, set[str]] = defaultdict(set)
    for left, right in _two_col(cache / f"link_pathway_{org}.txt"):
        gene, pathway = (left, right) if left.startswith(prefix) else (right, left)
        org_to_path[gene].add(pathway)
    path_names = {left: right for left, right in _two_col(cache / f"list_pathway_{org}.txt")}

    uniprot_to_reactome: dict[str, set[str]] = defaultdict(set)
    reactome_path = cache / "UniProt2Reactome.txt"
    if reactome_path.exists() and reactome_path.stat().st_size:
        wanted = "mus musculus" if organism == "mouse" else "homo sapiens"
        with reactome_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                accession, reactome_id, _url, _name, _evidence, species = parts[:6]
                if wanted not in species.lower():
                    continue
                uniprot_to_reactome[accession].add(reactome_id)

    id_cols = [
        c
        for c in ("uniprot_accession", "uniprot_ids", "uniprot_ids_all", "uniprot_swissprot")
        if c in protein_ann.columns
    ]
    edges: list[dict[str, str]] = []
    protein_paths: dict[str, set[str]] = defaultdict(set)
    for rec in protein_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        accessions: list[str] = []
        for col in id_cols:
            accessions.extend(_tokens(rec.get(col)))
        for acc in accessions:
            for gene in uniprot_to_org.get(acc, ()):
                for pathway in org_to_path.get(gene, ()):
                    protein_paths[feat].add(pathway)
                    edges.append(
                        {
                            "source_node": feat,
                            "target_node": pathway,
                            "relationship": "protein_to_pathway",
                            "database": "KEGG",
                            "evidence": f"{acc}->{gene}",
                            "confidence": "high",
                            "reference": KEGG_REF,
                            "via": path_names.get(pathway, pathway),
                        }
                    )
            for pathway in uniprot_to_reactome.get(acc, ()):
                protein_paths[feat].add(pathway)
                edges.append(
                    {
                        "source_node": feat,
                        "target_node": pathway,
                        "relationship": "protein_to_pathway",
                        "database": "Reactome",
                        "evidence": acc,
                        "confidence": "high",
                        "reference": REACTOME_REF,
                        "via": pathway,
                    }
                )

    kegg_to_path: dict[str, set[str]] = defaultdict(set)
    for left, right in _two_col(cache / "link_pathway_compound.txt"):
        compound, pathway = (left, right) if "path:" in right or right.startswith("map") else (right, left)
        kegg_to_path[compound.replace("cpd:", "")].add(pathway)

    hmdb_to_kegg: dict[str, set[str]] = defaultdict(set)
    map_path = cache / "metabolite_identifier_map.csv"
    if map_path.exists() and map_path.stat().st_size:
        mapped = pd.read_csv(map_path)
        for rec in mapped.to_dict("records"):
            hmdb = _pad_hmdb(rec.get("hmdb_id") or rec.get("database_identifier"))
            kegg = str(rec.get("kegg_id") or rec.get("kegg") or "").replace("cpd:", "")
            if hmdb and kegg:
                hmdb_to_kegg[hmdb].add(kegg)

    metabolite_paths: dict[str, set[str]] = defaultdict(set)
    for rec in metabolite_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        hmdb = _pad_hmdb(rec.get("hmdb_id") or rec.get("database_identifier"))
        kegg_ids = set(hmdb_to_kegg.get(hmdb, ()))
        raw_kegg = str(rec.get("kegg_id") or "").replace("cpd:", "").strip()
        if raw_kegg and raw_kegg.lower() != "nan":
            kegg_ids.add(raw_kegg)
        for kegg in kegg_ids:
            for pathway in kegg_to_path.get(kegg, ()):
                metabolite_paths[feat].add(pathway)
                edges.append(
                    {
                        "source_node": feat,
                        "target_node": pathway,
                        "relationship": "metabolite_to_pathway",
                        "database": "KEGG",
                        "evidence": f"{hmdb}->{kegg}" if hmdb else kegg,
                        "confidence": "medium",
                        "reference": KEGG_REF,
                        "via": pathway,
                    }
                )

    network = pd.DataFrame(edges)
    if not network.empty:
        network = network.drop_duplicates(["source_node", "target_node", "relationship"])
    provenance = pd.DataFrame(
        [
            {
                "source": "KEGG+Reactome",
                "date_accessed": date.today().isoformat(),
                "url_or_file": str(cache),
                "version": "rest-current",
                "license": "KEGG academic / Reactome CC",
                "organism": organism,
                "notes": "Missing edges are incomplete mapping, not impossibility.",
            }
        ]
    )
    return network, provenance
