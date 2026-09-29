"""Unimodal, non-temporal models kept as modules to borrow — never registered to run.

These papers do not match last_interval (joint next-time forecast). Their layers
(pathway masks, hierarchical VNNs, foundation encoders, alignment heads) are
what later adapters or codegen may copy. ``coerce_eval_models`` never sees this list.
"""

from __future__ import annotations

from typing import Any

# run=False is the contract: inspect may surface these; the evaluator must not.
REFERENCE_MODULES: list[dict[str, Any]] = [
    {
        "name": "p-net",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Reactome-aligned sparse layers (gene → pathway → process). Useful mask for prior_fusion / codegen, not a forecast head.",
        "paper": "Elmarakeby et al., Nature 2021",
        "url": "https://doi.org/10.1038/s41586-021-03922-4",
    },
    {
        "name": "pasnet",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Gene-layer to pathway-layer sparse DNN (Hao 2018). Same mask idea as P-NET, flatter hierarchy.",
        "paper": "Hao et al., BMC Bioinformatics 2018",
        "url": "https://doi.org/10.1186/s12859-018-2500-z",
    },
    {
        "name": "dcell",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Visible Neural Network: GO/CliXO subsystems as hidden units (Ma 2018). Hierarchy for interpretability, not time.",
        "paper": "Ma et al., Nat Methods 2018",
        "url": "https://doi.org/10.1038/s41592-018-0009-z",
    },
    {
        "name": "drugcell",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "DCell + drug branch (Kuenzi 2020). Dual-input VNN pattern if a treatment condition appears.",
        "paper": "Kuenzi et al., Science 2020",
        "url": "https://doi.org/10.1126/science.aaz2353",
    },
    {
        "name": "scvi",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "VAE encoder/decoder + NB likelihood (Lopez 2018). Latent design only; not a next-visit map.",
        "paper": "Lopez et al., Nat Methods 2018",
        "url": "https://doi.org/10.1038/s41592-018-0229-2",
    },
    {
        "name": "totalvi",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Joint RNA+protein VAE (Gayoso 2021). Two likelihoods, one latent — contrast with our dual-tower forecast VAEs.",
        "paper": "Gayoso et al., Nat Methods 2021",
        "url": "https://doi.org/10.1038/s41592-020-01050-x",
    },
    {
        "name": "geneformer",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Rank-value tokenizer + transformer (Theodoris 2023). Frozen token embeddings as a prior, not a sequence over visits.",
        "paper": "Theodoris et al., Nature 2023",
        "url": "https://doi.org/10.1038/s41586-023-06139-9",
    },
    {
        "name": "scgpt",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Gene-token foundation model (Cui 2024). Same: pretrained embeddings, not last_interval dynamics.",
        "paper": "Cui et al., Nat Methods 2024",
        "url": "https://doi.org/10.1038/s41592-024-02201-0",
    },
    {
        "name": "harmony",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Soft k-means batch correction in PCA space (Korsunsky 2019). Do not fit across test intervals.",
        "paper": "Korsunsky et al., Nat Methods 2019",
        "url": "https://doi.org/10.1038/s41592-019-0619-0",
    },
    {
        "name": "seurat-wnn",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Weighted nearest-neighbor multimodal graph (Hao 2021). Sample graph, not feature dynamics.",
        "paper": "Hao et al., Cell 2021",
        "url": "https://doi.org/10.1016/j.cell.2021.04.048",
    },
    {
        "name": "glue",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Graph-linked unified embedding across unpaired omics (Cao 2022). Alignment module only.",
        "paper": "Cao & Gao, Nat Biotechnol 2022",
        "url": "https://doi.org/10.1038/s41587-022-01284-4",
    },
    {
        "name": "scvelo",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Spliced/unspliced RNA velocity (Bergen 2020). Needs two RNA layers; not proteomics/metabolomics visits.",
        "paper": "Bergen et al., Nat Biotechnol 2020",
        "url": "https://doi.org/10.1038/s41587-020-0591-3",
    },
    {
        "name": "deepinsight",
        "kind": "unimodal_nontemporal",
        "run": False,
        "borrow": "Features → 2-D image then CNN (Sharma 2019). Wrong inductive bias for last_interval vectors.",
        "paper": "Sharma et al., Sci Rep 2019",
        "url": "https://doi.org/10.1038/s41598-019-47765-6",
    },
]


def list_reference_modules() -> list[dict[str, Any]]:
    return [dict(item, role="reference_only") for item in REFERENCE_MODULES]
