"""Protein ESM and metabolite SMILES embeddings, with hashed-name fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.prior.paths import (
    embedding_dir,
    ensure_unimol2_84m_weight,
    esm_weights_dir,
    sequence_dir,
    smiles_dir,
    unimol_weights_dir,
)


@dataclass
class PretrainedEmbeddings:
    protein: dict[str, np.ndarray] = field(default_factory=dict)
    metabolite: dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "none"
    notes: list[str] = field(default_factory=list)

    def params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_protein": len(self.protein),
            "n_metabolite": len(self.metabolite),
            "notes": self.notes,
        }


def _hash_vec(text: str, dim: int = 32) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    token = str(text or "").upper()
    if not token or token == "NAN":
        return vec
    for i in range(len(token)):
        gram = token[i : i + 3]
        vec[hash(gram) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def _http_text(url: str, timeout: int = 20) -> str:
    req = Request(url, headers={"User-Agent": "tmo-prior/0.2"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _fasta_seq(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
    return "".join(lines)


def uniprot_sequence(accession: str) -> str | None:
    acc = str(accession or "").strip()
    if not acc or acc.lower() == "nan":
        return None
    dest = sequence_dir()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{acc}.fasta"
    if path.exists() and path.stat().st_size > 0:
        return _fasta_seq(path.read_text(encoding="utf-8"))
    try:
        text = _http_text(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta")
    except Exception:
        return None
    if not text.startswith(">"):
        return None
    path.write_text(text, encoding="utf-8")
    return _fasta_seq(text)


def pubchem_smiles(cid: object) -> str | None:
    digits = "".join(ch for ch in str(cid or "") if ch.isdigit())
    if not digits:
        return None
    dest = smiles_dir()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{digits}.smi"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8").strip()
    try:
        text = _http_text(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{digits}/property/CanonicalSMILES/TXT"
        ).strip()
    except Exception:
        return None
    if not text or text.lower().startswith("<"):
        return None
    path.write_text(text + "\n", encoding="utf-8")
    return text


def _esm_embed(sequences: dict[str, str], notes: list[str]) -> dict[str, np.ndarray]:
    weight = esm_weights_dir() / "esm2_t12_35M_UR50D.pt"
    if not weight.exists():
        notes.append(f"ESM weight missing: {weight}")
        return {}
    try:
        import esm
        import torch
    except ImportError:
        notes.append("fair-esm is not installed")
        return {}
    cache = embedding_dir("esm")
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    need: dict[str, str] = {}
    for feat, seq in sequences.items():
        npy = cache / f"{feat.replace('/', '_')}.npy"
        if npy.exists():
            out[feat] = np.load(npy)
        elif seq:
            need[feat] = seq[:1022]
    if not need:
        if out:
            notes.append(f"ESM-2 35M embeddings from cache ({len(out)} proteins)")
        return out
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weight))
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    items = list(need.items())
    for start in range(0, len(items), 8):
        chunk = items[start : start + 8]
        _, _, tokens = batch_converter([(feat, seq) for feat, seq in chunk])
        with torch.no_grad():
            result = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        reps = result["representations"][model.num_layers]
        for i, (feat, seq) in enumerate(chunk):
            vec = reps[i, 1 : len(seq) + 1].mean(0).cpu().numpy().astype(float)
            np.save(cache / f"{feat.replace('/', '_')}.npy", vec)
            out[feat] = vec
    notes.append(f"ESM-2 t12_35M mean-pooled {len(need)} sequences; weight={weight}")
    return out


def _cls_reprs(raw: object) -> list[np.ndarray]:
    if isinstance(raw, dict):
        vecs = raw.get("cls_repr")
        if vecs is None:
            raise ValueError(f"Uni-Mol get_repr missing cls_repr; keys={list(raw)}")
        return [np.asarray(vec, dtype=float).ravel() for vec in vecs]
    arr = np.asarray(raw)
    if arr.ndim == 2:
        return [arr[i].astype(float) for i in range(arr.shape[0])]
    raise ValueError(f"Unexpected Uni-Mol repr type: {type(raw)}")


def _morgan_fingerprints(smiles_map: dict[str, str], notes: list[str]) -> dict[str, np.ndarray]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            return {}
        out: dict[str, np.ndarray] = {}
        for feat, smi in smiles_map.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
            out[feat] = np.asarray(fp, dtype=float)
        if out:
            notes.append("Morgan r=2 fingerprints (rdkit) because Uni-Mol runtime was unavailable")
        return out
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    out = {}
    for feat, smi in smiles_map.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[feat] = np.asarray(gen.GetFingerprint(mol), dtype=float)
    if out:
        notes.append("Morgan r=2 fingerprints (rdkit) because Uni-Mol runtime was unavailable")
    return out


def _unimol_or_morgan(smiles_map: dict[str, str], notes: list[str]) -> dict[str, np.ndarray]:
    cache = embedding_dir("unimol")
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    need: dict[str, str] = {}
    for feat, smi in smiles_map.items():
        npy = cache / f"{feat.replace('/', '_')}.npy"
        if npy.exists() and npy.stat().st_size > 0:
            out[feat] = np.load(npy)
        else:
            need[feat] = smi
    if not need:
        if out:
            notes.append(f"Uni-Mol2 84M embeddings from cache ({len(out)} metabolites)")
        return out

    os.environ["UNIMOL_WEIGHT_DIR"] = str(unimol_weights_dir())
    weight = ensure_unimol2_84m_weight()
    try:
        from unimol_tools import UniMolRepr
    except ImportError:
        notes.append("unimol_tools not installed; metabolite vectors are Morgan or hashed names")
        UniMolRepr = None  # type: ignore[assignment]
    if UniMolRepr is not None:
        if not weight.exists():
            notes.append(f"Uni-Mol2 84M weight missing: {weight}")
        else:
            try:
                model = UniMolRepr(
                    data_type="molecule",
                    remove_hs=False,
                    model_name="unimolv2",
                    model_size="84m",
                    batch_size=16,
                )
                feats = list(need)
                vecs = _cls_reprs(model.get_repr([need[feat] for feat in feats]))
                if len(vecs) != len(feats):
                    raise RuntimeError(f"Uni-Mol returned {len(vecs)} vectors for {len(feats)} SMILES")
                for feat, vec in zip(feats, vecs):
                    np.save(cache / f"{feat.replace('/', '_')}.npy", vec)
                    out[feat] = vec
                notes.append(f"Uni-Mol2 84M cls_repr for {len(need)} metabolites; weight={weight}")
                return out
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Uni-Mol inference failed: {exc}")
    out.update(_morgan_fingerprints(need, notes))
    return out


def build_pretrained(
    protein_ann: pd.DataFrame,
    metabolite_ann: pd.DataFrame,
    protein_table: dict[str, np.ndarray] | None = None,
    features: list[str] | None = None,
    y_features: list[str] | None = None,
) -> PretrainedEmbeddings:
    notes: list[str] = []
    protein = dict(protein_table or {})
    metabolite: dict[str, np.ndarray] = {}
    wanted_p = set(features or protein_ann["feature_id"].astype(str))
    sequences: dict[str, str] = {}
    for rec in protein_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        if feat not in wanted_p or feat in protein:
            continue
        acc = str(rec.get("uniprot_swissprot") or rec.get("uniprot_accession") or "").split(";")[0].strip()
        seq = uniprot_sequence(acc) if acc and acc.lower() != "nan" else None
        if seq:
            sequences[feat] = seq
    esm_vecs = _esm_embed(sequences, notes)
    protein.update(esm_vecs)
    for rec in protein_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        if feat not in wanted_p or feat in protein:
            continue
        protein[feat] = _hash_vec(str(rec.get("gene_symbol") or rec.get("string_preferred_name") or feat))
    wanted_m = set(y_features or metabolite_ann["feature_id"].astype(str))
    smiles_map: dict[str, str] = {}
    for rec in metabolite_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        if feat not in wanted_m:
            continue
        smi = rec.get("smiles")
        if not smi or str(smi) in {"nan", "None"}:
            smi = pubchem_smiles(rec.get("pubchem_cid"))
        if smi:
            smiles_map[feat] = str(smi)
    metabolite.update(_unimol_or_morgan(smiles_map, notes))
    hashed = 0
    for rec in metabolite_ann.to_dict("records"):
        feat = str(rec["feature_id"])
        if feat not in wanted_m or feat in metabolite:
            continue
        metabolite[feat] = _hash_vec(
            str(rec.get("standard_english_name") or rec.get("feature_name") or rec.get("metabolite_name") or feat)
        )
        hashed += 1
    if hashed:
        notes.append(f"{hashed} metabolites used hashed names (no SMILES / Uni-Mol)")
    method = "hashed_names"
    if esm_vecs:
        method = "esm2_t12_35M"
    if any("Uni-Mol2" in n for n in notes):
        method = f"{method}+unimol2_84m" if esm_vecs else "unimol2_84m"
    elif any("Morgan" in n or "Uni-Mol" in n for n in notes):
        method = f"{method}+smiles" if esm_vecs else "smiles"
    if protein_table:
        method = "external+" + method
    return PretrainedEmbeddings(protein=protein, metabolite=metabolite, method=method, notes=notes)
