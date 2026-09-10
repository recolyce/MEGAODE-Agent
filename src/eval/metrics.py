"""Test/val metrics for last_interval pairs. Each row is a pair, not a unique sample."""

from __future__ import annotations

import numpy as np


def _col_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> list[float]:
    scores = []
    for j in range(y_true.shape[1]):
        a = np.asarray(y_true[:, j], dtype=float)
        b = np.asarray(y_pred[:, j], dtype=float)
        if np.std(a) < 1e-12:
            continue
        if np.std(b) < 1e-12:
            scores.append(0.0)
            continue
        if method == "pearson":
            scores.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            scores.append(float(np.corrcoef(ra, rb)[0, 1]))
    return scores


def median_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = _col_corr(y_true, y_pred, "pearson")
    return float(np.nanmedian(vals)) if vals else float("nan")


def score_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson = _col_corr(y_true, y_pred, "pearson")
    spearman = _col_corr(y_true, y_pred, "spearman")
    err = y_pred - y_true
    return {
        "median_pcc": float(np.nanmedian(pearson)) if pearson else float("nan"),
        "mean_pcc": float(np.nanmean(pearson)) if pearson else float("nan"),
        "median_spearman": float(np.nanmedian(spearman)) if spearman else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "n_targets_scored": int(len(pearson)),
    }
