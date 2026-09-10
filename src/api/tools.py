"""LangChain tools wrapping curator, prior, and model-library steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.curator.bundle import ModelingBundle
from src.curator.loader import load_biomaster
from src.curator.run import run as curate_run
from src.curator.tasks.last_interval import detect_lag_mode
from src.eval.evaluator import evaluate_models
from src.eval.metrics import median_pcc
from src.interpretation.attribution import attribute_models
from src.interpretation.contribution import compute_contributions
from src.models.registry import MODELS, ModelRegistry
from src.prior.registry import PriorRegistry, annotation_prior_hints


def inspect_dataset(source: str) -> dict[str, Any]:
    data = load_biomaster(Path(source))
    meta = data.metadata
    lag_mode = detect_lag_mode(meta)
    times = sorted(float(t) for t in meta["time"].dropna().unique())
    recommended_unit = "liver" if "liver" in data.tissues else (data.tissues[0] if data.tissues else "all")
    organism = "mouse" if "mouse" in str(data.manifest).lower() or "mmu" in data.report.lower() else "human"
    if "human" in data.report.lower() or "ipop" in data.name.lower():
        organism = "human"
    return {
        "name": data.name,
        "source": str(data.source),
        "n_samples": int(len(meta)),
        "n_subjects": int(meta["subject_id"].nunique()),
        "times": times,
        "time_unit": str(meta["time_unit"].iloc[0]) if "time_unit" in meta.columns else "",
        "tissues": data.tissues,
        "n_proteins": int(data.proteomics.shape[1]),
        "n_metabolites": int(data.metabolomics.shape[1]),
        "already_logged": bool(data.already_logged),
        "lag_mode": lag_mode,
        "recommended_unit": recommended_unit,
        "recommended_organism": organism,
        "warnings": list(data.warnings),
        "available_models": [{"name": name, "requires_prior": need} for name, (_f, need) in MODELS.items()],
        "available_priors": PriorRegistry().available(),
        "prior_hints": annotation_prior_hints(data),
    }


def curate_dataset(
    source: str,
    artifacts: str = "artifacts",
    task: str = "last_interval",
    unit: str | None = None,
    direction: str | None = None,
    n_hv: int = 512,
) -> dict[str, Any]:
    units = [unit] if unit else None
    directions = [direction] if direction else None
    paths = curate_run(Path(source), task, Path(artifacts), units, n_hv, directions)
    summaries = []
    for path in paths:
        bundle = ModelingBundle.load(path)
        summaries.append(
            {
                "path": str(path),
                "dataset": bundle.dataset,
                "unit": bundle.unit,
                "direction": bundle.direction,
                "lag_mode": bundle.lag_mode,
                "n_train": int(len(bundle.train.pairs)),
                "n_val": int(len(bundle.val.pairs)),
                "n_test": int(len(bundle.test.pairs)),
                "n_expr": bundle.n_expr,
                "n_y": len(bundle.y_features),
            }
        )
    return {"bundle_paths": [str(p) for p in paths], "bundles": summaries}


def build_priors(
    source: str,
    bundle_path: str,
    artifacts: str = "artifacts",
    backend: str = "name_rule",
    organism: str = "human",
    sources: list[str] | None = None,
) -> dict[str, Any]:
    data = load_biomaster(Path(source))
    bundle = ModelingBundle.load(Path(bundle_path))
    rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
    registry = PriorRegistry()
    prior = registry.build(
        data,
        bundle.x_features,
        rel,
        backend=backend,
        sources=sources,
        n_context=len(bundle.context_names),
        organism=organism,
        y_features=bundle.y_features,
    )
    dest = Path(artifacts) / bundle.dataset / "priors" / prior.name
    registry.save(prior, dest)
    return {**prior.params(), "path": str(dest)}


def fit_models(
    source: str,
    bundle_path: str,
    models: list[str],
    artifacts: str = "artifacts",
    prior_backend: str = "name_rule",
    organism: str = "human",
) -> dict[str, Any]:
    bundle = ModelingBundle.load(Path(bundle_path))
    registry = ModelRegistry()
    catalog = {item["name"]: item["requires_prior"] for item in registry.list()}
    needs_prior = any(catalog.get(name, False) for name in models)
    if needs_prior or prior_backend:
        data = load_biomaster(Path(source))
        rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
        prior = PriorRegistry().build(
            data,
            bundle.x_features,
            rel,
            backend=prior_backend,
            n_context=len(bundle.context_names),
            organism=organism,
            y_features=bundle.y_features,
        )
        bundle.attach_prior(prior)
    rows = []
    for name in models:
        if name not in catalog:
            rows.append({"model": name, "error": f"unknown model; available {sorted(catalog)}"})
            continue
        if catalog[name] and bundle.prior is None:
            rows.append({"model": name, "error": "requires prior but none attached"})
            continue
        try:
            model = registry.build(name)
            model.fit(bundle)
            pred = model.predict(bundle)
            rows.append(
                {
                    "model": name,
                    "median_pcc": median_pcc(bundle.test.Y, pred),
                    "n_test": int(bundle.test.X.shape[0]),
                    "requires_prior": catalog[name],
                    **model.params(),
                }
            )
        except Exception as exc:  # noqa: BLE001 — keep other models running
            rows.append({"model": name, "error": str(exc)})
    dest = Path(bundle_path).parent / "pipeline_metrics.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(dest, index=False)
    return {"metrics": rows, "path": str(dest)}


def attribute_selected_models(
    source: str,
    bundle_path: str,
    models: list[str],
    artifacts: str = "artifacts",
    prior_backend: str = "name_rule",
    organism: str = "human",
    prior_sources: list[str] | None = None,
) -> dict[str, Any]:
    return attribute_models(
        source,
        bundle_path,
        models,
        artifacts=artifacts,
        prior_backend=prior_backend,
        organism=organism,
        prior_sources=prior_sources,
    )


def evaluate_selected_models(
    source: str,
    bundle_path: str,
    models: list[str],
    artifacts: str = "artifacts",
    prior_backend: str = "auto",
    organism: str = "human",
    prior_sources: list[str] | None = None,
    round_id: int = 1,
) -> dict[str, Any]:
    bundle = ModelingBundle.load(Path(bundle_path))
    data = load_biomaster(Path(source))
    rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
    prior = PriorRegistry().build(
        data,
        bundle.x_features,
        rel,
        backend=prior_backend,
        sources=prior_sources,
        n_context=len(bundle.context_names),
        organism=organism,
        y_features=bundle.y_features,
    )
    bundle.attach_prior(prior)
    return evaluate_models(bundle, models, Path(bundle_path) / "evaluation", round_id=round_id)


def contribute_selected_models(
    source: str,
    bundle_path: str,
    models: list[str],
    artifacts: str = "artifacts",
    prior_backend: str = "auto",
    organism: str = "human",
    prior_sources: list[str] | None = None,
) -> dict[str, Any]:
    return compute_contributions(
        source,
        bundle_path,
        models,
        artifacts=artifacts,
        prior_backend=prior_backend,
        organism=organism,
        prior_sources=prior_sources,
    )


@tool
def inspect_biomaster_dataset(source: str) -> str:
    """Inspect a BioMaster template: subjects, times, tissues, lag mode, feature counts."""
    return json.dumps(inspect_dataset(source), ensure_ascii=False, default=str)


@tool
def curate_last_interval(
    source: str,
    artifacts: str = "artifacts",
    unit: str = "",
    direction: str = "",
    n_hv: int = 512,
) -> str:
    """Define last_interval pairs, fit train-fold preprocess, write ModelingBundles."""
    return json.dumps(
        curate_dataset(
            source,
            artifacts=artifacts,
            unit=unit or None,
            direction=direction or None,
            n_hv=n_hv,
        ),
        ensure_ascii=False,
        default=str,
    )


@tool
def build_prior_network(
    source: str,
    bundle_path: str,
    artifacts: str = "artifacts",
    backend: str = "name_rule",
    organism: str = "human",
    sources: str = "",
) -> str:
    """Materialize prior extra-inputs (name_rule, KEGG/Reactome, STRING/proxy, embeddings)."""
    wanted = [part.strip() for part in sources.split(",") if part.strip()] or None
    return json.dumps(
        build_priors(source, bundle_path, artifacts, backend, organism, wanted),
        ensure_ascii=False,
        default=str,
    )


@tool
def fit_registered_models(
    source: str,
    bundle_path: str,
    models: str,
    artifacts: str = "artifacts",
    prior_backend: str = "name_rule",
    organism: str = "human",
) -> str:
    """Fit comma-separated model names on a ModelingBundle and score median PCC on the test pairs."""
    names = [part.strip() for part in models.split(",") if part.strip()]
    return json.dumps(
        fit_models(source, bundle_path, names, artifacts, prior_backend, organism),
        ensure_ascii=False,
        default=str,
    )


@tool
def attribute_model_pairs(
    source: str,
    bundle_path: str,
    models: str,
    artifacts: str = "artifacts",
    prior_backend: str = "name_rule",
    organism: str = "human",
) -> str:
    """Multi-method attribution for every protein–metabolite pair of the selected models.

    Methods: coefficient (if linear), occlusion, finite-difference gradient,
    bootstrap stability, perturbation robustness, and prior support.
    Skips train_mean and last_value. Writes pair score tables under the bundle.
    """
    names = [part.strip() for part in models.split(",") if part.strip()]
    return json.dumps(
        attribute_selected_models(source, bundle_path, names, artifacts, prior_backend, organism),
        ensure_ascii=False,
        default=str,
    )


@tool
def evaluate_model_suite(
    source: str,
    bundle_path: str,
    models: str,
    artifacts: str = "artifacts",
    prior_backend: str = "auto",
    organism: str = "human",
    prior_sources: str = "",
) -> str:
    """Train selected models with a small val grid on GPU if available; write test metrics."""
    names = [part.strip() for part in models.split(",") if part.strip()]
    wanted = [part.strip() for part in prior_sources.split(",") if part.strip()] or None
    return json.dumps(
        evaluate_selected_models(source, bundle_path, names, artifacts, prior_backend, organism, wanted),
        ensure_ascii=False,
        default=str,
    )


@tool
def contribute_model_pairs(
    source: str,
    bundle_path: str,
    models: str,
    artifacts: str = "artifacts",
    prior_backend: str = "auto",
    organism: str = "human",
    prior_sources: str = "",
) -> str:
    """Cross-modal contribution (coefficient, occlusion, gradient, permutation, optional SHAP) for BioMaster."""
    names = [part.strip() for part in models.split(",") if part.strip()]
    wanted = [part.strip() for part in prior_sources.split(",") if part.strip()] or None
    return json.dumps(
        contribute_selected_models(source, bundle_path, names, artifacts, prior_backend, organism, wanted),
        ensure_ascii=False,
        default=str,
    )


@tool
def write_prior_injected_model(
    source: str,
    bundle_path: str,
    artifacts: str = "artifacts",
    prior_backend: str = "auto",
    organism: str = "human",
    prior_sources: str = "",
    instruction: str = "",
) -> str:
    """Ask the coding loop to write one new prior-injected model under src/models/generated/."""
    from src.codegen.agent import propose_model

    bundle = ModelingBundle.load(Path(bundle_path))
    wanted = [part.strip() for part in prior_sources.split(",") if part.strip()] or None
    data = load_biomaster(Path(source))
    rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
    prior = PriorRegistry().build(
        data,
        bundle.x_features,
        rel,
        backend=prior_backend,
        sources=wanted,
        n_context=len(bundle.context_names),
        organism=organism,
        y_features=bundle.y_features,
    )
    bundle.attach_prior(prior)
    return json.dumps(propose_model(bundle, instruction=instruction), ensure_ascii=False, default=str)


@tool
def list_python_packages(query: str = "") -> str:
    """List packages in the current Python environment (optional name substring)."""
    from src.codegen.env_tools import list_installed_packages

    return json.dumps(list_installed_packages(query), ensure_ascii=False)


@tool
def install_python_packages(packages: str) -> str:
    """Install comma-separated PyPI packages into the current environment with pip."""
    from src.codegen.env_tools import install_python_packages as _install

    names = [part.strip() for part in packages.split(",") if part.strip()]
    return json.dumps(_install(names), ensure_ascii=False)


@tool
def run_repo_terminal(command: str, timeout: int = 60) -> str:
    """Run a python or pip command in the repo. Other shells are blocked."""
    from src.codegen.env_tools import run_terminal

    return json.dumps(run_terminal(command, timeout=timeout), ensure_ascii=False, default=str)


PIPELINE_TOOLS = [
    inspect_biomaster_dataset,
    curate_last_interval,
    build_prior_network,
    fit_registered_models,
    evaluate_model_suite,
    write_prior_injected_model,
    list_python_packages,
    install_python_packages,
    run_repo_terminal,
    attribute_model_pairs,
    contribute_model_pairs,
]
