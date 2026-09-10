"""PriorRegistry: name-rule, KEGG/Reactome, STRING/proxy, graph matrices, embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.curator.loader import BioMasterDataset
from src.prior.features import PriorFeatures, materialize_features
from src.prior.graph import membership_sets
from src.prior.kegg_reactome import build_kegg_reactome_network, prepare_cache
from src.prior.name_rule import build_name_rule_network
from src.prior.paths import kegg_dir, prior_root, string_dir
from src.prior.pretrained import PretrainedEmbeddings, build_pretrained
from src.prior.stitch_net import build_stitch_network
from src.prior.string_net import build_string_network

BACKENDS = ("name_rule", "kegg_reactome", "string", "stitch", "pretrained", "auto")
SOURCE_CHOICES = ("name_rule", "kegg_reactome", "string", "stitch", "pretrained")


@dataclass
class PriorAttachment:
    name: str
    network: pd.DataFrame
    provenance: pd.DataFrame
    membership: dict[str, list[str]]
    laplacian: np.ndarray | None
    pretrained: PretrainedEmbeddings
    features: PriorFeatures | None = None
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sources": self.sources,
            "n_edges": int(len(self.network)),
            "relationships": self.network["relationship"].value_counts().to_dict() if len(self.network) else {},
            "n_groups": len(self.membership),
            "pretrained": self.pretrained.params(),
            "features": None if self.features is None else self.features.params(),
            "warnings": self.warnings,
        }


def annotation_prior_hints(data: BioMasterDataset) -> dict[str, Any]:
    protein = data.protein_annotations
    met = data.metabolite_annotations
    has_uniprot = any(c in protein.columns for c in ("uniprot_accession", "uniprot_ids", "uniprot_swissprot"))
    has_kegg_met = "kegg_id" in met.columns and met["kegg_id"].notna().any()
    has_hmdb = any(c in met.columns for c in ("hmdb_id", "database_identifier"))
    has_string = "string_preferred_name" in protein.columns
    has_smiles = "smiles" in met.columns
    suggested = ["name_rule", "pretrained"]
    if has_uniprot or has_kegg_met or has_hmdb:
        suggested.append("kegg_reactome")
    if has_string or True:
        suggested.append("string")
    if "pubchem_cid" in met.columns and met["pubchem_cid"].notna().any():
        suggested.append("stitch")
    return {
        "protein_columns": list(protein.columns),
        "metabolite_columns": list(met.columns),
        "has_gene_symbol": "gene_symbol" in protein.columns,
        "has_uniprot": has_uniprot,
        "has_string_name": has_string,
        "has_kegg_metabolite": bool(has_kegg_met),
        "has_hmdb": bool(has_hmdb),
        "has_smiles": has_smiles,
        "available_sources": list(SOURCE_CHOICES),
        "suggested_sources": suggested,
        "notes": [
            f"Knowledge root: {prior_root()}",
            "STRING/STITCH read TMO_PRIOR_ROOT (official files or proxy).",
            "pretrained: ESM-2 35M when UniProt sequences resolve; SMILES via PubChem then Uni-Mol2 84M (Morgan fallback).",
        ],
    }


class PriorRegistry:
    def __init__(self, cache_dir: Path | None = None, extra_cache: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else prior_root()
        self.extra_cache = Path(extra_cache) if extra_cache is not None else None

    def available(self) -> list[str]:
        return list(SOURCE_CHOICES)

    def build(
        self,
        data: BioMasterDataset,
        features: list[str],
        relationship: str,
        backend: str = "auto",
        n_context: int = 3,
        organism: str = "mouse",
        sources: list[str] | None = None,
        y_features: list[str] | None = None,
    ) -> PriorAttachment:
        warnings: list[str] = []
        wanted = [str(s) for s in (sources or [backend]) if str(s)]
        if "auto" in wanted or not wanted:
            hints = annotation_prior_hints(data)
            wanted = list(hints["suggested_sources"])
        wanted = [s for s in wanted if s in SOURCE_CHOICES or s == "auto"]
        if not wanted:
            wanted = ["name_rule", "pretrained"]

        frames: list[pd.DataFrame] = []
        provenances: list[pd.DataFrame] = []
        extra_adj = None

        if "name_rule" in wanted:
            network, provenance = build_name_rule_network(data.protein_annotations, data.metabolite_annotations)
            frames.append(network)
            provenances.append(provenance)

        if "kegg_reactome" in wanted:
            cache = prepare_cache(kegg_dir(), self.extra_cache, organism=organism)
            kegg_net, kegg_prov = build_kegg_reactome_network(
                data.protein_annotations, data.metabolite_annotations, cache, organism=organism
            )
            if kegg_net.empty:
                warnings.append("KEGG/Reactome produced no edges; name_rule still used if requested")
            else:
                frames.append(kegg_net)
                provenances.append(kegg_prov)

        protein_ids = set()
        if "feature_id" in data.protein_annotations.columns:
            protein_ids = set(data.protein_annotations["feature_id"].astype(str))
        else:
            protein_ids = set(map(str, data.proteomics.columns))
        protein_only = [name for name in features if name in protein_ids]

        if "string" in wanted:
            string_net, extra_adj, string_prov, string_warn = build_string_network(
                data.protein_annotations,
                protein_only or features,
                cache=string_dir(),
                organism=organism,
            )
            warnings.extend(string_warn)
            if not string_net.empty:
                frames.append(string_net)
                provenances.append(string_prov)

        if "stitch" in wanted:
            stitch_net, stitch_prov, stitch_warn = build_stitch_network(
                data.protein_annotations,
                data.metabolite_annotations,
                features,
                y_features or [],
                organism=organism,
            )
            warnings.extend(stitch_warn)
            if not stitch_net.empty:
                frames.append(stitch_net)
                provenances.append(stitch_prov)

        if not frames:
            network, provenance = build_name_rule_network(data.protein_annotations, data.metabolite_annotations)
            frames.append(network)
            provenances.append(provenance)
            wanted = list(dict.fromkeys(["name_rule", *wanted]))
            warnings.append("no requested network produced edges; fell back to name_rule")

        network = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not network.empty:
            network = network.drop_duplicates(["source_node", "target_node", "relationship"])
        provenance = pd.concat(provenances, ignore_index=True) if provenances else pd.DataFrame()
        if relationship in {"joint", "multimodal", "both"}:
            membership = {}
            for rel in ("protein_to_pathway", "metabolite_to_pathway"):
                part = membership_sets(network, features, rel) if len(network) else {}
                for key, values in part.items():
                    membership[key] = sorted(set(membership.get(key, []) + values))
        else:
            membership = membership_sets(network, features, relationship) if len(network) else {}
        pretrained = build_pretrained(
            data.protein_annotations,
            data.metabolite_annotations,
            features=features,
            y_features=y_features or [],
        )
        feat = materialize_features(
            network,
            features,
            y_features or [],
            relationship,
            n_context,
            pretrained,
            wanted,
            extra_adjacency=extra_adj,
        )
        name = "+".join(wanted)
        return PriorAttachment(
            name=name,
            network=network,
            provenance=provenance,
            membership=membership,
            laplacian=feat.laplacian,
            pretrained=pretrained,
            features=feat,
            sources=wanted,
            warnings=warnings,
        )

    def save(self, prior: PriorAttachment, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        prior.network.to_csv(dest / "biological_network.csv", index=False)
        prior.provenance.to_csv(dest / "network_provenance.csv", index=False)
        if prior.features is not None:
            if prior.features.laplacian is not None:
                np.save(dest / "laplacian.npy", prior.features.laplacian)
            if prior.features.adjacency is not None:
                np.save(dest / "adjacency.npy", prior.features.adjacency)
            if prior.features.protein_emb is not None:
                np.save(dest / "protein_emb.npy", prior.features.protein_emb)
            if prior.features.metabolite_emb is not None:
                np.save(dest / "metabolite_emb.npy", prior.features.metabolite_emb)
        (dest / "prior_summary.json").write_text(
            __import__("json").dumps(prior.params(), indent=2) + "\n", encoding="utf-8"
        )
        return dest
