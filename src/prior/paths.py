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


def sequence_dir() -> Path:
    return prior_root() / "sequences" / "uniprot"


def smiles_dir() -> Path:
    return prior_root() / "smiles" / "pubchem"


def embedding_dir(kind: str) -> Path:
    return prior_root() / "embeddings" / kind
