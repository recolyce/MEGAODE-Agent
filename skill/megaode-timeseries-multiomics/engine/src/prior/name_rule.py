"""Name-rule modules from gene symbols and metabolite names. Organism-agnostic fallback."""

from __future__ import annotations

from datetime import date

import pandas as pd

REF = "Name-rule prior from gene symbols and metabolite names only"

PROTEIN_MODULES: dict[str, tuple[str, ...]] = {
    "lipid_lipoprotein": (
        "APOA1", "APOA2", "APOA4", "APOB", "APOC1", "APOC2", "APOC3", "APOC4",
        "APOD", "APOE", "APOF", "APOH", "APOL1", "APOM", "APMAP", "LCAT",
        "CETP", "PLTP", "LPA", "PON1", "PON3", "GPLD1",
    ),
    "complement": (
        "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C1RL", "C2", "C3", "C5", "C6",
        "C7", "C8A", "C8B", "C8G", "C9", "CFB", "CFD", "CFH", "CFI", "CFP",
        "CFHR1", "CFHR2", "CFHR4", "CFHR5", "C4B", "C4BPA", "C4BPB", "MASP1",
        "MASP2", "MBL2", "FCN2", "FCN3",
    ),
    "coagulation": (
        "F2", "F5", "F7", "F9", "F10", "F11", "F12", "F13A1", "F13B", "FGA",
        "FGB", "FGG", "PLG", "SERPINC1", "SERPIND1", "SERPINF2", "PROC",
        "PROS1", "PROZ", "KLKB1", "KNG1", "THBS1", "GP1BA", "GP5",
    ),
    "acute_phase": (
        "HP", "HPR", "ORM1", "ORM2", "SAA1", "SAA2", "SAA4", "LBP", "CD14",
        "CRP", "AHSG", "ITIH1", "ITIH2", "ITIH3", "ITIH4", "LRG1", "SERPINA1",
        "SERPINA3",
    ),
    "carrier": ("ALB", "TTR", "TF", "TFRC", "HPX", "GC", "AFM", "AMBP", "RBP4", "AZGP1"),
    "redox_detox": ("GPX3", "PRDX2", "QSOX1", "SELENOP", "CP", "PON1", "PON3", "BCHE"),
    "igf_growth": ("IGF2", "IGFBP3", "IGFALS", "IGF1", "PZP"),
}

METABOLITE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "lipid_lipoprotein",
        (
            "fa", "fatty", "oleic", "palmit", "linole", "arachid", "docosahexa",
            "eicosapenta", "cholesterol", "lysoc", "sphing", "carnitine",
            "triglycer", "cholate", "bile", "tauro", "glycochol",
        ),
    ),
    ("energy_tca", ("pyruv", "lact", "citric", "citrate", "malic", "ketoglutar", "succin", "glucose")),
    (
        "amino_acid",
        (
            "serine", "valine", "threonine", "cysteine", "alanine", "proline",
            "lysine", "methionine", "phenylalanine", "arginine", "tyrosine",
            "tryptophan", "histidine", "isoleucine", "leucine", "asparagine",
            "aspartic", "glutamine", "glutamic", "glycine", "ornithine", "taurine",
        ),
    ),
    ("sulfur_redox", ("glutathione", "cysteine", "cystine", "taurine")),
    ("tryptophan", ("tryptophan", "kynuren", "indole", "indoxyl", "xanthine")),
    ("purine", ("uric", "xanthine", "hypoxanthine", "allantoin", "inosine")),
    ("steroid", ("cortisol", "dhea", "androsterone", "pregnen", "testosterone")),
    ("heme_vitamin", ("bilirubin", "retinol", "pyridox", "pantothenic", "niacinamide")),
    ("bile_acid", ("cholate", "chenodeoxy", "bile", "tauro", "glycochol", "deoxychol")),
    ("phospholipid", ("lysoc", "lysope", "lysopi", "sphing")),
]

ENZYME_METABOLITE: list[tuple[str, tuple[str, ...]]] = [
    ("LCAT", ("cholesterol", "lysoc", "choline")),
    ("PON1", ("lactone", "lipid")),
    ("GPX3", ("glutathione",)),
    ("ALB", ("fatty", "oleic", "palmit", "bilirubin")),
    ("APOA1", ("cholesterol", "fatty")),
    ("APOB", ("cholesterol", "fatty")),
    ("APOE", ("cholesterol", "fatty")),
]


def _gene(row: pd.Series) -> str:
    raw = str(row.get("gene_symbol") or row.get("gene_symbols") or row.get("genes_all") or "")
    return raw.split(";")[0].split(".")[0].strip().upper()


def _met_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(col) or "")
        for col in (
            "feature_id",
            "metabolite_identification",
            "database_identifier",
            "metabolite_name",
            "feature_name",
            "standard_english_name",
            "molecule_class",
        )
    ).lower()


def _protein_modules(gene: str) -> list[str]:
    modules = []
    if gene.startswith(("IGH", "IGK", "IGL")) or gene in {"IGKC", "JCHAIN", "PIGR"}:
        modules.append("immunoglobulin")
    for module, members in PROTEIN_MODULES.items():
        if gene in members:
            modules.append(module)
    return sorted(set(modules))


def _metabolite_modules(text: str) -> list[str]:
    return sorted({module for module, keys in METABOLITE_RULES if any(key in text for key in keys)})


def build_name_rule_network(protein_ann: pd.DataFrame, metabolite_ann: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges: list[dict[str, str]] = []
    protein_mod: dict[str, list[str]] = {}
    metabolite_mod: dict[str, list[str]] = {}
    gene_to_features: dict[str, list[str]] = {}

    for rec in protein_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        gene = _gene(pd.Series(rec))
        gene_to_features.setdefault(gene, []).append(feat)
        mods = _protein_modules(gene)
        protein_mod[feat] = mods
        for module in mods:
            edges.append(
                {
                    "source_node": feat,
                    "target_node": module,
                    "relationship": "protein_to_pathway",
                    "database": "name_rules",
                    "evidence": f"gene {gene}",
                    "confidence": "medium",
                    "reference": REF,
                    "via": module,
                }
            )

    for rec in metabolite_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        text = _met_text(pd.Series(rec))
        mods = _metabolite_modules(text)
        metabolite_mod[feat] = mods
        for module in mods:
            edges.append(
                {
                    "source_node": feat,
                    "target_node": module,
                    "relationship": "metabolite_to_pathway",
                    "database": "name_rules",
                    "evidence": text[:80],
                    "confidence": "medium",
                    "reference": REF,
                    "via": module,
                }
            )

    for gene, keys in ENZYME_METABOLITE:
        for p_feat in gene_to_features.get(gene, []):
            for rec in metabolite_ann.to_dict("records"):
                text = _met_text(pd.Series(rec))
                if any(key in text for key in keys):
                    edges.append(
                        {
                            "source_node": p_feat,
                            "target_node": str(rec["feature_id"]),
                            "relationship": "enzyme_to_metabolite_reaction",
                            "database": "name_rules",
                            "evidence": f"{gene} ~ {keys}",
                            "confidence": "medium",
                            "reference": REF,
                            "via": gene,
                        }
                    )

    network = pd.DataFrame(edges)
    if not network.empty:
        network = network.drop_duplicates(["source_node", "target_node", "relationship"])
    provenance = pd.DataFrame(
        [
            {
                "source": "name_rules",
                "date_accessed": date.today().isoformat(),
                "url_or_file": "src/prior/name_rule.py",
                "version": "1.0",
                "license": "internal rule set",
                "notes": "Missing edges are incomplete rules, not impossibility.",
            }
        ]
    )
    return network, provenance
