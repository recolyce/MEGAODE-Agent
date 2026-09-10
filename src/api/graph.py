"""LangGraph: inspect → plan → curate → prior → propose → evaluate → (rewrite once) → contribution → report."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.api.client import get_chat_model, has_llm_key, invoke_json
from src.api.tools import (
    build_priors,
    curate_dataset,
    inspect_dataset,
)
from src.codegen.agent import propose_model
from src.curator.bundle import ModelingBundle
from src.eval.evaluator import evaluate_models
from src.interpretation.contribution import compute_contributions
from src.models.registry import ModelRegistry
from src.prior.registry import PriorRegistry

DEFAULT_MODELS = ["train_mean", "last_value", "ridge", "pls", "laplacian_ridge", "prior_fusion_ridge"]
MAX_REWRITE = 1


class PipelineState(TypedDict, total=False):
    source: str
    artifacts: str
    task: str
    unit: str
    direction: str
    models: list[str]
    prior_backend: str
    prior_sources: list[str]
    organism: str
    n_hv: int
    inspection: dict[str, Any]
    plan_text: str
    bundle_paths: list[str]
    bundle_summaries: list[dict[str, Any]]
    prior_summary: dict[str, Any]
    generated: dict[str, Any]
    metrics: list[dict[str, Any]]
    eval_summary: dict[str, Any]
    eval_history: list[dict[str, Any]]
    selected_model: str
    want_rewrite: bool
    rewrite_count: int
    rewrite_instruction: str
    attribution: dict[str, Any]
    contribution: dict[str, Any]
    run_attribution: bool
    run_codegen: bool
    report: str
    report_path: str
    errors: list[str]


def _errors(state: PipelineState) -> list[str]:
    return list(state.get("errors") or [])


def _bundle_path(state: PipelineState) -> str | None:
    paths = state.get("bundle_paths") or []
    return paths[0] if paths else None


def inspect_node(state: PipelineState) -> dict[str, Any]:
    info = inspect_dataset(state["source"])
    updates: dict[str, Any] = {"inspection": info, "rewrite_count": int(state.get("rewrite_count") or 0)}
    if not state.get("unit"):
        updates["unit"] = info["recommended_unit"]
    if not state.get("organism"):
        updates["organism"] = info["recommended_organism"]
    if not state.get("direction"):
        updates["direction"] = "proteomics_to_metabolomics"
    if not state.get("models"):
        updates["models"] = list(DEFAULT_MODELS)
    if not state.get("prior_sources"):
        updates["prior_sources"] = list((info.get("prior_hints") or {}).get("suggested_sources") or ["name_rule", "pretrained"])
    if not state.get("prior_backend"):
        updates["prior_backend"] = "auto"
    return updates


def plan_node(state: PipelineState) -> dict[str, Any]:
    if not has_llm_key():
        raise RuntimeError("TMO_LLM_API_KEY required for plan; deterministic defaults are disabled")
    payload = {
        "inspection": {
            k: state.get("inspection", {}).get(k)
            for k in (
                "name",
                "lag_mode",
                "n_subjects",
                "n_samples",
                "times",
                "tissues",
                "n_proteins",
                "n_metabolites",
                "warnings",
                "prior_hints",
                "available_models",
            )
        },
        "current": {
            "unit": state.get("unit"),
            "direction": state.get("direction"),
            "models": state.get("models"),
            "prior_sources": state.get("prior_sources"),
            "organism": state.get("organism"),
        },
        "source_choices": ["name_rule", "kegg_reactome", "string", "stitch", "pretrained"],
    }
    try:
        parsed = invoke_json(
            "You plan a last_interval proteomics–metabolomics run and choose prior sources. "
            "Reply with JSON only: "
            '{"unit":str,"direction":"proteomics_to_metabolomics"|"metabolomics_to_proteomics",'
            '"organism":"human"|"mouse","prior_sources":[str],'
            '"baseline_models":[str],"reason":str}. '
            "prior_sources subset of name_rule,kegg_reactome,string,stitch,pretrained. "
            "baseline_models must already exist in available_models. "
            "No tissue → unit=all. Do not invent model architectures here.",
            json.dumps(payload, ensure_ascii=False, default=str),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"plan LLM failed: {exc}") from exc
    updates: dict[str, Any] = {"plan_text": json.dumps(parsed, ensure_ascii=False)}
    if parsed.get("unit"):
        updates["unit"] = str(parsed["unit"])
    if parsed.get("direction"):
        updates["direction"] = str(parsed["direction"])
    if parsed.get("organism"):
        updates["organism"] = str(parsed["organism"])
    if isinstance(parsed.get("prior_sources"), list) and parsed["prior_sources"]:
        updates["prior_sources"] = [str(s) for s in parsed["prior_sources"]]
    catalog = {item["name"] for item in ModelRegistry().list()}
    if isinstance(parsed.get("baseline_models"), list) and parsed["baseline_models"]:
        picked = [str(n) for n in parsed["baseline_models"] if str(n) in catalog]
        if picked:
            # keep honest baselines
            for must in ("train_mean", "last_value"):
                if must not in picked:
                    picked.insert(0, must)
            updates["models"] = picked
    return updates


def curate_node(state: PipelineState) -> dict[str, Any]:
    result = curate_dataset(
        state["source"],
        artifacts=state.get("artifacts") or "artifacts",
        task=state.get("task") or "last_interval",
        unit=state.get("unit"),
        direction=state.get("direction"),
        n_hv=int(state.get("n_hv") or 512),
    )
    return {
        "bundle_paths": result["bundle_paths"],
        "bundle_summaries": result["bundles"],
    }


def prior_node(state: PipelineState) -> dict[str, Any]:
    path = _bundle_path(state)
    if not path:
        return {"errors": _errors(state) + ["prior skipped: no bundle"]}
    summary = build_priors(
        state["source"],
        path,
        artifacts=state.get("artifacts") or "artifacts",
        backend=state.get("prior_backend") or "auto",
        organism=state.get("organism") or "human",
        sources=state.get("prior_sources"),
    )
    return {"prior_summary": summary}


def _attach(state: PipelineState) -> ModelingBundle:
    path = _bundle_path(state)
    if not path:
        raise RuntimeError("no bundle")
    bundle = ModelingBundle.load(path)
    from src.curator.loader import load_biomaster

    data = load_biomaster(state["source"])
    rel = "protein_to_pathway" if bundle.x_modality == "proteomics" else "metabolite_to_pathway"
    prior = PriorRegistry().build(
        data,
        bundle.x_features,
        rel,
        backend=state.get("prior_backend") or "auto",
        sources=state.get("prior_sources"),
        n_context=len(bundle.context_names),
        organism=state.get("organism") or "human",
        y_features=bundle.y_features,
    )
    bundle.attach_prior(prior)
    return bundle


def propose_node(state: PipelineState) -> dict[str, Any]:
    if state.get("run_codegen") is False:
        return {"generated": {"skipped": True}}
    try:
        bundle = _attach(state)
        result = propose_model(
            bundle,
            inspection=state.get("inspection"),
            prior_summary=state.get("prior_summary"),
            instruction=state.get("rewrite_instruction") or "",
            existing=list(state.get("models") or []),
        )
    except Exception as exc:  # noqa: BLE001
        return {"generated": {"error": str(exc)}, "errors": _errors(state) + [f"codegen: {exc}"]}
    models = list(state.get("models") or DEFAULT_MODELS)
    name = result.get("model")
    if name and name not in models:
        models.append(str(name))
    print(
        f"  proposed {name} llm={result.get('used_llm')} fallback={result.get('fallback')} repaired={result.get('repaired')} reason={result.get('reason')}",
        flush=True,
    )
    return {"generated": result, "models": models}


def evaluate_node(state: PipelineState) -> dict[str, Any]:
    path = _bundle_path(state)
    if not path:
        return {"errors": _errors(state) + ["evaluate skipped: no bundle"]}
    bundle = _attach(state)
    summary = evaluate_models(
        bundle,
        list(state.get("models") or DEFAULT_MODELS),
        Path(path) / "evaluation",
        round_id=int(state.get("rewrite_count") or 0) + 1,
    )
    history = list(state.get("eval_history") or [])
    history.append(summary)
    selected = summary.get("selected_model") or "ridge"
    want, instruction = _decide_rewrite(state, summary)
    return {
        "metrics": summary.get("rows") or [],
        "eval_summary": summary,
        "eval_history": history,
        "selected_model": selected,
        "want_rewrite": want,
        "rewrite_instruction": instruction,
    }


def _decide_rewrite(state: PipelineState, summary: dict[str, Any]) -> tuple[bool, str]:
    if int(state.get("rewrite_count") or 0) >= MAX_REWRITE:
        return False, ""
    if state.get("run_codegen") is False:
        return False, ""
    rows = [r for r in (summary.get("rows") or []) if not r.get("error")]
    by_name = {r["model"]: r for r in rows}
    last_value = float((by_name.get("last_value") or {}).get("test_median_pcc") or 0.0)
    best = summary.get("selected_model")
    best_pcc = float((by_name.get(best) or {}).get("test_median_pcc") or -1.0)
    generated = (state.get("generated") or {}).get("model")
    instruction = (
        f"Best learned model {best} test median PCC={best_pcc:.4f} vs last_value {last_value:.4f}. "
        "Rewrite a different prior-injected architecture (try Laplacian + pathway scores or embedding kernel). "
        f"Do not reuse {generated}."
    )
    if not has_llm_key():
        raise RuntimeError("TMO_LLM_API_KEY required to decide rewrite; threshold fallback is disabled")
    parsed = invoke_json(
        "You decide whether to allow ONE more model-structure rewrite. "
        "Reply JSON only: {\"rewrite\":bool,\"instruction\":str,\"reason\":str}. "
        "Rewrite only if a learned model fails to beat last_value by a meaningful margin "
        "or the generated model errored. Never request a second rewrite after this one.",
        json.dumps(
            {
                "metrics": rows,
                "selected": best,
                "generated": state.get("generated"),
                "lag_mode": (state.get("inspection") or {}).get("lag_mode"),
                "already_rewritten": int(state.get("rewrite_count") or 0),
            },
            ensure_ascii=False,
            default=str,
        ),
    )
    want = bool(parsed.get("rewrite"))
    return want, str(parsed.get("instruction") or instruction)


def rewrite_node(state: PipelineState) -> dict[str, Any]:
    updates = propose_node(state)
    updates["rewrite_count"] = int(state.get("rewrite_count") or 0) + 1
    updates["want_rewrite"] = False
    return updates


def contribution_node(state: PipelineState) -> dict[str, Any]:
    if state.get("run_attribution") is False:
        return {"contribution": {"skipped": True}, "attribution": {"skipped": True}}
    path = _bundle_path(state)
    if not path:
        return {"errors": _errors(state) + ["contribution skipped: no bundle"]}
    selected = state.get("selected_model") or "ridge"
    names = []
    for name in (selected, "ridge", ((state.get("generated") or {}).get("model"))):
        if name and name not in names and name not in {"train_mean", "last_value"}:
            names.append(str(name))
    if not names:
        return {"contribution": {"skipped": True, "reason": "no attributable models"}}
    summary = compute_contributions(
        state["source"],
        path,
        names,
        artifacts=state.get("artifacts") or "artifacts",
        prior_backend=state.get("prior_backend") or "auto",
        organism=state.get("organism") or "human",
        prior_sources=state.get("prior_sources"),
    )
    return {"contribution": summary, "attribution": summary.get("attribution") or {}}


def report_node(state: PipelineState) -> dict[str, Any]:
    facts = {
        "inspection": {
            k: state.get("inspection", {}).get(k)
            for k in ("name", "lag_mode", "n_subjects", "n_samples", "times", "warnings")
        },
        "plan": state.get("plan_text"),
        "prior": state.get("prior_summary"),
        "generated": state.get("generated"),
        "metrics": state.get("metrics"),
        "selected_model": state.get("selected_model"),
        "rewrite_count": state.get("rewrite_count"),
        "contribution": {
            k: (state.get("contribution") or {}).get(k)
            for k in ("models", "methods", "n_rows", "contribution_csv", "attribution_dir", "top_permutation")
        },
        "errors": state.get("errors"),
    }
    if not has_llm_key():
        raise RuntimeError("TMO_LLM_API_KEY required for report; JSON dump fallback is disabled")
    llm = get_chat_model()
    reply = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Write a short Chinese research note (plain text, no markdown heading larger than ##). "
                    "Cover lag mode, pair counts, priors used, whether a new model was written, "
                    "test median PCC vs last_value, whether a rewrite happened, and where BioMaster "
                    "should read contribution_pairs.csv. Do not invent numbers."
                ),
            },
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False, default=str)},
        ]
    )
    text = str(reply.content)
    path = _bundle_path(state)
    if path:
        dest = Path(path) / "report.md"
        dest.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"  wrote report {dest}", flush=True)
    return {"report": text, "report_path": str(Path(path) / "report.md") if path else ""}


def _after_evaluate(state: PipelineState) -> Literal["rewrite_model", "contribution"]:
    if state.get("want_rewrite") and int(state.get("rewrite_count") or 0) < MAX_REWRITE:
        return "rewrite_model"
    return "contribution"


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("inspect", inspect_node)
    graph.add_node("plan", plan_node)
    graph.add_node("curate", curate_node)
    graph.add_node("prior", prior_node)
    graph.add_node("propose", propose_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rewrite_model", rewrite_node)
    graph.add_node("contribution", contribution_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", "plan")
    graph.add_edge("plan", "curate")
    graph.add_edge("curate", "prior")
    graph.add_edge("prior", "propose")
    graph.add_edge("propose", "evaluate")
    graph.add_conditional_edges("evaluate", _after_evaluate, {"rewrite_model": "rewrite_model", "contribution": "contribution"})
    graph.add_edge("rewrite_model", "evaluate")
    graph.add_edge("contribution", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_pipeline(state: PipelineState) -> PipelineState:
    return build_pipeline().invoke(state)
