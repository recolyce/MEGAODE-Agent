"""External prior-knowledge root. Not stored in the code repo."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PRIOR_ROOT = Path("/personal/workspace/TMO-agent-prior")


def prior_root() -> Path:
    return Path(os.environ.get("TMO_PRIOR_ROOT", DEFAULT_PRIOR_ROOT))


def kegg_dir() -> Path:
    return prior_root() / "kegg"


def string_dir() -> Path:
    return prior_root() / "string"


def stitch_dir() -> Path:
    return prior_root() / "stitch"


def esm_weights_dir() -> Path:
    return prior_root() / "esm" / "weights"


def unimol_dir() -> Path:
    return prior_root() / "unimol"


def unimol_weights_dir() -> Path:
    dest = prior_root() / "unimol_weights"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def ensure_unimol2_84m_weight() -> Path:
    """Place Uni-Mol2 84M where unimol_tools looks: {UNIMOL_WEIGHT_DIR}/modelzoo/84M/checkpoint.pt."""
    dest = unimol_weights_dir() / "modelzoo" / "84M" / "checkpoint.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    candidates = [
        unimol_dir() / "unimol2" / "checkpoints" / "84M" / "checkpoint.pt",
        Path("/root/workspace/Uni-Mol/unimol2/checkpoints/84M/checkpoint.pt"),
    ]
    for src in candidates:
        if src.exists() and src.stat().st_size > 0:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(src.resolve())
            return dest
    return dest


def sequence_dir() -> Path:
    return prior_root() / "sequences" / "uniprot"


def smiles_dir() -> Path:
    return prior_root() / "smiles" / "pubchem"


def embedding_dir(kind: str) -> Path:
    return prior_root() / "embeddings" / kind
