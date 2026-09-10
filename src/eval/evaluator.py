"""Fold-1 grouped holdout + Optuna search, then last-interval test metrics."""

from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.curator.bundle import ModelingBundle, SplitArrays
from src.eval.excel_report import write_metrics_excel
from src.eval.metrics import score_arrays
from src.models.base import resolve_device
from src.models.registry import ModelRegistry

N_FOLDS = 5
DEFAULT_TRIALS = 15
TORCH_MODELS = {
    "mlp",
    "prior_fusion_mlp",
    "neural_ode",
    "feature_chunk_lstm",
    "graph_omics_ode",
    "cross_attn_fusion",
    "gated_fusion",
    "koopman_ae",
    "dual_lstm",
    "mmvae_forecast",
    "mogonet_fusion",
}


def _predict_split(model: Any, bundle: ModelingBundle, split: str) -> Any:
    held_x = bundle.test.X
    held_pairs = bundle.test.pairs
    data = getattr(bundle, split)
    bundle.test.X = data.X
    bundle.test.pairs = data.pairs
    try:
        return model.predict(bundle)
    finally:
        bundle.test.X = held_x
        bundle.test.pairs = held_pairs


def development_split(bundle: ModelingBundle) -> SplitArrays:
    if len(bundle.val.X) == 0:
        return bundle.train
    return SplitArrays(
        X=np.vstack([bundle.train.X, bundle.val.X]),
        Y=np.vstack([bundle.train.Y, bundle.val.Y]),
        pairs=pd.concat([bundle.train.pairs, bundle.val.pairs], ignore_index=True),
    )


def apply_development_train(bundle: ModelingBundle) -> ModelingBundle:
    """Fit contribution models on the fold-1 train split stored on the bundle."""
    return bundle


@contextmanager
def _swap_train_val(bundle: ModelingBundle, train: SplitArrays, val: SplitArrays) -> Iterator[None]:
    held_train, held_val = bundle.train, bundle.val
    bundle.train = train
    bundle.val = val
    try:
        yield
    finally:
        bundle.train = held_train
        bundle.val = held_val


def _subset(split: SplitArrays, index: np.ndarray) -> SplitArrays:
    return SplitArrays(
        X=np.asarray(split.X)[index],
        Y=np.asarray(split.Y)[index],
        pairs=split.pairs.iloc[index].reset_index(drop=True),
    )


def _group_labels(pairs: pd.DataFrame) -> pd.Series:
    if "subject_id" in pairs.columns and pairs["subject_id"].nunique() >= 2:
        return pairs["subject_id"].astype(str)
    if "trajectory_id" in pairs.columns and pairs["trajectory_id"].nunique() >= 2:
        return pairs["trajectory_id"].astype(str)
    return pd.Series([str(i) for i in range(len(pairs))], index=pairs.index)


def cv_folds(pairs: pd.DataFrame, n_splits: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = _group_labels(pairs)
    n_groups = int(groups.nunique())
    splits = min(int(n_splits), n_groups)
    if splits < 2 or len(pairs) < 2:
        return []
    dummy = np.zeros((len(pairs), 1))
    return list(GroupKFold(n_splits=splits).split(dummy, groups=groups.to_numpy()))


def fold1_splits(bundle: ModelingBundle) -> tuple[SplitArrays, SplitArrays]:
    """Grouped 5-fold on train+val, keep only fold 1 (sklearn index 0)."""
    dev = development_split(bundle)
    folds = cv_folds(dev.pairs, N_FOLDS)
    if not folds:
        return bundle.train, bundle.val if len(bundle.val.X) else bundle.train
    train_idx, val_idx = folds[0]
    return _subset(dev, train_idx), _subset(dev, val_idx)


def _score_kwargs(bundle: ModelingBundle) -> dict[str, int]:
    return {"n_protein": int(bundle.n_protein or 0), "n_metabolite": int(bundle.n_metabolite or 0)}


def _fit_one(registry: ModelRegistry, name: str, bundle: ModelingBundle, hparams: dict[str, Any]) -> Any:
    kwargs = dict(hparams)
    if name in TORCH_MODELS:
        kwargs.setdefault("device", "auto")
    model = registry.build(name, **kwargs)
    return model.fit(bundle)


def _suggest(trial: Any, name: str, factory: Any) -> dict[str, Any]:
    if name == "mlp":
        return {
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "epochs": trial.suggest_categorical("epochs", [30, 50, 80]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
    if name == "prior_fusion_mlp":
        return {
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "epochs": trial.suggest_categorical("epochs", [30, 50, 80]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "neural_ode":
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "epochs": trial.suggest_categorical("epochs", [20, 40, 60]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "feature_chunk_lstm":
        return {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "chunk_size": trial.suggest_categorical("chunk_size", [32, 64]),
            "epochs": trial.suggest_categorical("epochs", [30, 50]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "graph_omics_ode":
        return {
            "hidden": trial.suggest_categorical("hidden", [8, 16, 32]),
            "epochs": trial.suggest_categorical("epochs", [15, 25, 40]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "cross_attn_fusion":
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "epochs": trial.suggest_categorical("epochs", [30, 50]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "gated_fusion":
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "epochs": trial.suggest_categorical("epochs", [30, 50, 80]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "koopman_ae":
        return {
            "latent": trial.suggest_categorical("latent", [16, 32, 64]),
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "epochs": trial.suggest_categorical("epochs", [30, 50]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "dual_lstm":
        return {
            "hidden": trial.suggest_categorical("hidden", [16, 32]),
            "epochs": trial.suggest_categorical("epochs", [20, 40]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "mmvae_forecast":
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64]),
            "latent": trial.suggest_categorical("latent", [8, 16, 32]),
            "beta": trial.suggest_float("beta", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [30, 50]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    if name == "mogonet_fusion":
        return {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "epochs": trial.suggest_categorical("epochs", [30, 50]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        }
    params: dict[str, Any] = {}
    try:
        names = set(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        return params
    if "hidden" in names:
        params["hidden"] = trial.suggest_categorical("hidden", [8, 16, 32, 64])
    if "epochs" in names:
        params["epochs"] = trial.suggest_categorical("epochs", [15, 25, 40])
    if "lr" in names:
        params["lr"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    return params


def _tune(
    registry: ModelRegistry,
    name: str,
    bundle: ModelingBundle,
    n_trials: int,
) -> tuple[dict[str, Any], float]:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from src.models.registry import _catalog

    factory = _catalog()[name][0]

    def objective(trial: optuna.Trial) -> float:
        hparams = _suggest(trial, name, factory)
        try:
            model = _fit_one(registry, name, bundle, hparams)
            if not len(bundle.val.X):
                return float("nan")
            pred = np.asarray(_predict_split(model, bundle, "val"), dtype=float)
            return float(score_arrays(bundle.val.Y, pred, **_score_kwargs(bundle)).get("median_pcc") or 0.0)
        except Exception as exc:  # noqa: BLE001
            trial.set_user_attr("error", str(exc))
            return -1.0

    if n_trials <= 1:
        return {}, float("nan")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), study_name=f"{name}_fold1")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = dict(study.best_trial.params) if study.best_trial is not None else {}
    best_value = float(study.best_value) if study.best_trial is not None else float("nan")
    return best, best_value


def evaluate_models(
    bundle: ModelingBundle,
    models: list[str],
    dest: Path,
    round_id: int = 1,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    catalog = {item["name"]: item["requires_prior"] for item in registry.list()}
    device = str(resolve_device("auto"))
    fold_train, fold_val = fold1_splits(bundle)
    rows: list[dict[str, Any]] = []
    for name in models:
        if name not in catalog:
            rows.append({"model": name, "error": f"unknown model; available {sorted(catalog)}", "round": round_id})
            continue
        if catalog[name] and bundle.prior is None:
            rows.append({"model": name, "error": "requires prior but none attached", "round": round_id})
            continue
        try:
            with _swap_train_val(bundle, fold_train, fold_val):
                hparams, val_optuna = _tune(registry, name, bundle, n_trials)
                model = _fit_one(registry, name, bundle, hparams)
                val_pred = _predict_split(model, bundle, "val") if len(bundle.val.X) else None
                test_pred = _predict_split(model, bundle, "test")
                params = model.params()
            val_scores = score_arrays(fold_val.Y, val_pred, **_score_kwargs(bundle)) if val_pred is not None else {}
            test_scores = score_arrays(bundle.test.Y, test_pred, **_score_kwargs(bundle))
            row = {
                "model": name,
                "round": round_id,
                "requires_prior": catalog[name],
                "device": device,
                "hparams": json.dumps(hparams, default=str),
                "n_folds": 1,
                "fold": 1,
                "n_trials": int(n_trials),
                "optuna_val_median_pcc": val_optuna,
                "val_median_pcc": val_scores.get("median_pcc", val_optuna),
                **{f"test_{k}": v for k, v in test_scores.items()},
                **params,
            }
            rows.append(row)
            print(
                f"  eval {name}: fold1 Optuna trials={n_trials} val median PCC={row.get('val_median_pcc')} "
                f"test median PCC={row.get('test_median_pcc')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": name, "error": str(exc), "round": round_id, "n_folds": 1, "fold": 1})
            print(f"  eval {name}: error {exc}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(dest / f"evaluation_round{round_id}.csv", index=False)
    excel_path = dest / "metrics_by_method.xlsx"
    try:
        write_metrics_excel(rows, excel_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  excel write failed: {exc}", flush=True)
        excel_path = dest / "metrics_by_method.xlsx"
    learned = table
    if "error" in table.columns:
        err = table["error"].astype(str).replace({"nan": "", "None": ""})
        learned = table.loc[err.eq("") | table["error"].isna()]
    selected = None
    if len(learned) and "test_median_pcc" in learned.columns:
        selected = str(learned.sort_values("test_median_pcc", ascending=False).iloc[0]["model"])
    summary = {
        "round": round_id,
        "device": device,
        "n_models": int(len(models)),
        "n_folds": 1,
        "fold": 1,
        "cv": "grouped_fold1_optuna",
        "n_trials": int(n_trials),
        "selected_model": selected,
        "rows": rows,
        "path": str(dest),
        "excel": str(excel_path),
    }
    (dest / f"evaluation_round{round_id}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (dest / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary
