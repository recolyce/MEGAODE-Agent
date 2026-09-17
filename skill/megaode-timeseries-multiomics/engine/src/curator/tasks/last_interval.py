"""Train on times before the last; test the last interval. Auto true/pseudo lag."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.curator.bundle import ModelingBundle, SplitArrays
from src.curator.loader import BioMasterDataset
from src.curator.preprocess import FoldPreprocessor
from src.curator.tasks.registry import register_task

DIRECTIONS = (
    ("multimodal", "proteomics+metabolomics", "proteomics+metabolomics"),
)
JOINT_DIRECTION = "multimodal"


def detect_lag_mode(metadata: pd.DataFrame) -> str:
    counts = metadata.groupby("subject_id")["time"].nunique()
    return "true" if int(counts.max()) >= 2 else "pseudo"


def trajectory_id(row: pd.Series, mode: str) -> str:
    if mode == "true":
        return str(row["subject_id"])
    tissue = str(row["tissue"]) if "tissue" in row.index and pd.notna(row["tissue"]) else "all"
    sex = str(row["sex"]) if "sex" in row.index and pd.notna(row["sex"]) else "NA"
    rep = str(row["replicate"]) if "replicate" in row.index and pd.notna(row["replicate"]) else str(row["sample_id"])
    return f"{tissue}|{sex}|{rep}"


def _unit_frames(data: BioMasterDataset, unit: str) -> pd.DataFrame:
    meta = data.metadata.copy()
    if unit != "pooled" and "tissue" in meta.columns:
        meta = meta.loc[meta["tissue"].astype(str) == unit].copy()
    if meta.empty:
        raise ValueError(f"No samples for unit={unit}")
    return meta.reset_index(drop=True)


def build_pairs(metadata: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = metadata.copy()
    frame["trajectory_id"] = frame.apply(lambda row: trajectory_id(row, mode), axis=1)
    times = sorted(float(t) for t in frame["time"].dropna().unique())
    if len(times) < 2:
        raise ValueError("last_interval needs at least two time points")
    t_last = times[-1]
    train_times = times[:-1]
    intervals = list(zip(times[:-1], times[1:]))
    train_intervals = [(a, b) for a, b in intervals if b != t_last]
    test_interval = intervals[-1]
    val_interval = train_intervals[-1] if train_intervals else test_interval

    by_traj: dict[str, dict[float, str]] = {}
    extra: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        tid = trajectory_id(pd.Series(row._asdict()), mode)
        by_traj.setdefault(tid, {})[float(row.time)] = str(row.sample_id)
        extra[tid] = {
            "subject_id": str(row.subject_id),
            "tissue": str(getattr(row, "tissue", "all")),
            "sex": str(getattr(row, "sex", "NA")),
        }

    records: list[dict[str, Any]] = []

    def add_pair(tid: str, x_time: float, y_time: float, kind: str, split: str) -> None:
        samples = by_traj[tid]
        if x_time not in samples or y_time not in samples:
            return
        persist = samples.get(x_time)
        records.append(
            {
                "trajectory_id": tid,
                "subject_id": extra[tid]["subject_id"],
                "tissue": extra[tid]["tissue"],
                "sex": extra[tid]["sex"],
                "x_sample_id": samples[x_time],
                "y_sample_id": samples[y_time],
                "persist_y_sample_id": persist,
                "x_time": x_time,
                "y_time": y_time,
                "kind": kind,
                "split": split,
            }
        )

    for tid, samples in by_traj.items():
        for x_time, y_time in train_intervals:
            add_pair(tid, x_time, y_time, "aligned_lag_train", "train")
        add_pair(tid, val_interval[0], val_interval[1], "aligned_lag_val", "val")
        add_pair(tid, test_interval[0], test_interval[1], "aligned_lag_test", "test")

    pairs = pd.DataFrame.from_records(records)
    protocol = {
        "task": "last_interval",
        "lag_mode": mode,
        "times": times,
        "train_times": train_times,
        "t_last": t_last,
        "train_intervals": train_intervals,
        "val_interval": list(val_interval),
        "test_interval": list(test_interval),
        "n_train": int((pairs["split"] == "train").sum()),
        "n_val": int((pairs["split"] == "val").sum()),
        "n_test": int((pairs["split"] == "test").sum()),
        "note": (
            "True lag uses subject_id trajectories. "
            "Pseudo lag uses tissue|sex|replicate. "
            "Y at t_last never enters preprocessor fitting. "
            "Both omics are inputs and both are predicted at the next time. "
            "Evaluator tunes on grouped fold-1 only (no 5-fold CV)."
        ),
    }
    return pairs, protocol


def _clock_scale(times: list[float], configured: float | str) -> float:
    if configured != "auto":
        return float(configured)
    return float(max(times)) if times else 1.0


def _context(
    pairs: pd.DataFrame, scale: float, pooled: bool, tissues: list[str]
) -> tuple[np.ndarray, list[str]]:
    sex = pairs["sex"].astype(str).str.upper().isin(("F", "FEMALE")).astype(float).to_numpy()
    t0 = pairs["x_time"].to_numpy(dtype=float) / scale
    t1 = pairs["y_time"].to_numpy(dtype=float) / scale
    cols = [t0, t1, sex]
    names = ["t0", "t1", "sex_F"]
    if pooled and tissues:
        for tissue in tissues:
            cols.append((pairs["tissue"].astype(str) == tissue).astype(float).to_numpy())
            names.append(f"tissue_{tissue}")
    return np.column_stack(cols), names


def _split_arrays(
    x_mat: pd.DataFrame,
    y_mat: pd.DataFrame,
    pairs: pd.DataFrame,
    scale: float,
    pooled: bool,
    tissues: list[str],
) -> SplitArrays:
    x = x_mat.loc[pairs["x_sample_id"].tolist()].to_numpy(dtype=float)
    y = y_mat.loc[pairs["y_sample_id"].tolist()].to_numpy(dtype=float)
    ctx, _names = _context(pairs, scale, pooled, tissues)
    return SplitArrays(X=np.hstack([x, ctx]), Y=y, pairs=pairs.reset_index(drop=True))


def _split_joint(
    protein: pd.DataFrame,
    metabolite: pd.DataFrame,
    pairs: pd.DataFrame,
    scale: float,
    pooled: bool,
    tissues: list[str],
) -> SplitArrays:
    xp = protein.loc[pairs["x_sample_id"].tolist()].to_numpy(dtype=float)
    xm = metabolite.loc[pairs["x_sample_id"].tolist()].to_numpy(dtype=float)
    yp = protein.loc[pairs["y_sample_id"].tolist()].to_numpy(dtype=float)
    ym = metabolite.loc[pairs["y_sample_id"].tolist()].to_numpy(dtype=float)
    ctx, _names = _context(pairs, scale, pooled, tissues)
    return SplitArrays(
        X=np.hstack([xp, xm, ctx]),
        Y=np.hstack([yp, ym]),
        pairs=pairs.reset_index(drop=True),
    )


@register_task("last_interval")
def curate_last_interval(
    data: BioMasterDataset,
    *,
    unit: str = "liver",
    n_high_variance: int = 512,
    clock_scale: float | str = "auto",
    directions: tuple[str, ...] | None = None,
) -> list[ModelingBundle]:
    mode = detect_lag_mode(data.metadata)
    unit_meta = _unit_frames(data, unit)
    pairs, protocol = build_pairs(unit_meta, mode)
    protocol["unit"] = unit
    protocol["dataset"] = data.name
    train_ids = sorted(
        set(pairs.loc[pairs["split"] == "train", "x_sample_id"])
        | set(pairs.loc[pairs["split"] == "train", "y_sample_id"])
    )
    times = protocol["times"]
    scale = _clock_scale(times, clock_scale)
    pooled = unit == "pooled"
    tissues = data.tissues if pooled else []
    wanted = {JOINT_DIRECTION}
    if directions:
        raw = {str(item) for item in directions if str(item)}
        if raw & {"both", "all", JOINT_DIRECTION, "proteomics_to_metabolomics", "metabolomics_to_proteomics"}:
            wanted = {JOINT_DIRECTION}
        else:
            wanted = raw

    protein_pp = FoldPreprocessor(
        already_logged=data.already_logged,
        group_col="tissue",
        n_high_variance=n_high_variance,
    ).fit(data.proteomics, data.metadata, train_ids)
    metabolite_pp = FoldPreprocessor(
        already_logged=data.already_logged,
        group_col=None,
        n_high_variance=n_high_variance,
    ).fit(data.metabolomics, data.metadata, train_ids)
    protein_t = protein_pp.transform(data.proteomics.loc[unit_meta["sample_id"]], unit_meta)
    metabolite_t = metabolite_pp.transform(data.metabolomics.loc[unit_meta["sample_id"]], unit_meta)

    bundles: list[ModelingBundle] = []
    if JOINT_DIRECTION not in wanted:
        return bundles
    _ctx_demo, context_names = _context(pairs.head(1), scale, pooled, tissues)
    del _ctx_demo
    train_p = pairs.loc[pairs["split"] == "train"].reset_index(drop=True)
    val_p = pairs.loc[pairs["split"] == "val"].reset_index(drop=True)
    test_p = pairs.loc[pairs["split"] == "test"].reset_index(drop=True)
    train = _split_joint(protein_t, metabolite_t, train_p, scale, pooled, tissues)
    val = _split_joint(protein_t, metabolite_t, val_p, scale, pooled, tissues)
    test = _split_joint(protein_t, metabolite_t, test_p, scale, pooled, tissues)
    y_lookup = pd.concat([protein_t, metabolite_t], axis=1)
    y_lookup.index.name = "sample_id"
    x_features = list(protein_t.columns) + list(metabolite_t.columns)
    y_features = list(x_features)
    n_protein = int(protein_t.shape[1])
    n_metabolite = int(metabolite_t.shape[1])
    protocol = {
        **protocol,
        "x_layout": "proteins_then_metabolites_then_context",
        "y_layout": "proteins_then_metabolites",
        "n_protein": n_protein,
        "n_metabolite": n_metabolite,
    }
    bundles.append(
        ModelingBundle(
            dataset=data.name,
            task="last_interval",
            unit=unit,
            direction=JOINT_DIRECTION,
            lag_mode=mode,
            x_modality="proteomics+metabolomics",
            y_modality="proteomics+metabolomics",
            x_features=x_features,
            y_features=y_features,
            context_names=context_names,
            n_expr=n_protein + n_metabolite,
            n_protein=n_protein,
            n_metabolite=n_metabolite,
            train=train,
            val=val,
            test=test,
            pairs=pairs.copy(),
            y_lookup=y_lookup,
            metadata=unit_meta,
            protocol=protocol,
            preprocess_params={
                "proteomics": protein_pp.to_params(),
                "metabolomics": metabolite_pp.to_params(),
                "clock_scale": scale,
                "train_sample_ids": train_ids,
            },
            warnings=list(data.warnings),
        )
    )
    return bundles
