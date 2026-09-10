"""LangGraph: inspect → plan → curate → prior → propose → evaluate → (rewrite ≤3) → contribution → report."""

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
from src.interpretation.explain import write_interpretation_markdown
from src.models.registry import DEFAULT_MODELS, coerce_eval_models
from src.prior.registry import PriorRegistry

MAX_REWRITE = 3
JOINT_DIRECTION = "multimodal"


class PipelineState(TypedDict, total=False):
    source: str
    artifacts: str
    task: str
    unit: str
    direction: str
    bundle_index: int
    direction_runs: list[dict[str, Any]]
    models: list[str]
    prior_backend: str
    prior_sources: list[str]
    organism: str
    n_hv: int
    n_trials: int
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
    index = int(state.get("bundle_index") or 0)
    if 0 <= index < len(paths):
        return paths[index]
    return None


def _curate_directions(state: PipelineState) -> list[str] | None:
    return [JOINT_DIRECTION]


def inspect_node(state: PipelineState) -> dict[str, Any]:
    info = inspect_dataset(state["source"])
    updates: dict[str, Any] = {"inspection": info, "rewrite_count": int(state.get("rewrite_count") or 0)}
    if not state.get("unit"):
        updates["unit"] = info["recommended_unit"]
    if not state.get("organism"):
        updates["organism"] = info["recommended_organism"]
    if not state.get("direction"):
        updates["direction"] = JOINT_DIRECTION
    if not state.get("models"):
        updates["models"] = coerce_eval_models(list(DEFAULT_MODELS))
    if state.get("n_trials") is None:
        updates["n_trials"] = 15
    if state.get("bundle_index") is None:
        updates["bundle_index"] = 0
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
            "direction": JOINT_DIRECTION,
            "models": state.get("models"),
            "prior_sources": state.get("prior_sources"),
            "organism": state.get("organism"),
        },
        "source_choices": ["name_rule", "kegg_reactome", "string", "stitch", "pretrained"],
        "model_policy": {
            "required_baselines": [],
            "library_learned": list(DEFAULT_MODELS),
            "rule": "One multimodal direction: both omics in, both omics out at the next time. ML models only. Do not add leftover generated names. Do not invent architectures.",
        },
    }
    try:
        parsed = invoke_json(
            "You plan a last_interval multimodal proteomics+metabolomics forecast. "
            "Reply with JSON only: "
            '{"unit":str,"organism":"human"|"mouse","prior_sources":[str],'
            '"learned_models":[str],"reason":str}. '
            "There is only one direction: multimodal (both omics input, both omics next-time output). "
            "prior_sources subset of name_rule,kegg_reactome,string,stitch,pretrained. "
            "learned_models: use the library ML models already listed (mlp, neural_ode, "
            "feature_chunk_lstm, cross_attn_fusion, gated_fusion, koopman_ae, dual_lstm, "
            "mmvae_forecast, mogonet_fusion, graph_omics_ode, prior_fusion_mlp). "
            "Do not invent names or leftover generated files. No non-ML baselines. "
            "No tissue → unit=all.",
            json.dumps(payload, ensure_ascii=False, default=str),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"plan LLM failed: {exc}") from exc
    updates: dict[str, Any] = {"plan_text": json.dumps(parsed, ensure_ascii=False)}
    if parsed.get("unit"):
        updates["unit"] = str(parsed["unit"])
    updates["direction"] = JOINT_DIRECTION
    if parsed.get("organism"):
        updates["organism"] = str(parsed["organism"])
    if isinstance(parsed.get("prior_sources"), list) and parsed["prior_sources"]:
        updates["prior_sources"] = [str(s) for s in parsed["prior_sources"]]
    picked = parsed.get("learned_models") or parsed.get("baseline_models") or state.get("models") or DEFAULT_MODELS
    if isinstance(picked, list) and picked:
        updates["models"] = coerce_eval_models([str(n) for n in picked])
    else:
        updates["models"] = coerce_eval_models(list(DEFAULT_MODELS))
    return updates


def curate_node(state: PipelineState) -> dict[str, Any]:
    result = curate_dataset(
        state["source"],
        artifacts=state.get("artifacts") or "artifacts",
        task=state.get("task") or "last_interval",
        unit=state.get("unit"),
        directions=_curate_directions(state),
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
    rel = bundle.prior_relationship()
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
    name = result.get("model")
    models = coerce_eval_models(list(state.get("models") or DEFAULT_MODELS), extra=[str(name)] if name else None)
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
    generated = (state.get("generated") or {}).get("model")
    models = coerce_eval_models(list(state.get("models") or DEFAULT_MODELS), extra=[str(generated)] if generated else None)
    summary = evaluate_models(
        bundle,
        models,
        Path(path) / "evaluation",
        round_id=int(state.get("rewrite_count") or 0) + 1,
        n_trials=int(state.get("n_trials") or 15),
    )
    history = list(state.get("eval_history") or [])
    history.append(summary)
    selected = summary.get("selected_model") or (models[0] if models else "mlp")
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
    best = summary.get("selected_model")
    best_pcc = float((by_name.get(best) or {}).get("test_median_pcc") or -1.0)
    generated = (state.get("generated") or {}).get("model")
    already = int(state.get("rewrite_count") or 0)
    remaining = MAX_REWRITE - already
    instruction = (
        f"Best learned model {best} test median PCC={best_pcc:.4f} on multimodal next-time forecast. "
        "Write a different prior-injected dynamical ODE that takes both omics and predicts both. "
        "RK4 over last_interval dt. Do not write ridge / elastic-net / output smoother. "
        f"Do not reuse {generated}."
    )
    if not has_llm_key():
        raise RuntimeError("TMO_LLM_API_KEY required to decide rewrite; threshold fallback is disabled")
    parsed = invoke_json(
        f"You decide whether to request another model-structure rewrite. "
        f"Budget: at most {MAX_REWRITE} rewrites; already_rewritten is given; remaining={remaining}. "
        "Reply JSON only: {\"rewrite\":bool,\"instruction\":str,\"reason\":str}. "
        "Rewrite if the generated model errored or test median PCC is clearly weaker than the library ML models. "
        "Instruction must ask for a prior-injected multimodal dynamical ODE, not another linear ridge. "
        "Set rewrite=false when remaining is 0 or a library model is already clearly strongest.",
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
    selected = state.get("selected_model") or "mlp"
    names = []
    for name in (selected, ((state.get("generated") or {}).get("model"))):
        if name and name not in names:
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
    note = write_interpretation_markdown(
        path,
        contribution=summary,
        attribution=summary.get("attribution") or {},
        metrics=state.get("metrics") or [],
        selected=state.get("selected_model") or "",
        prior=state.get("prior_summary") or {},
    )
    summary = {**summary, "interpretation_md": note}
    return {"contribution": summary, "attribution": summary.get("attribution") or {}}


def report_node(state: PipelineState) -> dict[str, Any]:
    path = _bundle_path(state)
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
        "direction": Path(path).name if path else "",
        "contribution": {
            k: (state.get("contribution") or {}).get(k)
            for k in (
                "models",
                "methods",
                "n_rows",
                "contribution_csv",
                "method_scores_csv",
                "interpretation_md",
            )
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
                    "Cover the multimodal task (both omics in, both next-time omics out), lag mode, "
                    "priors used, whether a new model was written, test median PCC among ML models, "
                    "rewrite_count, and tell BioMaster to read "
                    "contribution_pairs.csv / method_scores.csv plus interpretation.md. "
                    "Do not mention scientific confidence, novelty, or pair classifications. "
                    "Do not invent numbers."
                ),
            },
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False, default=str)},
        ]
    )
    text = str(reply.content)
    dest_s = ""
    if path:
        dest = Path(path) / "report.md"
        dest.write_text(text.rstrip() + "\n", encoding="utf-8")
        dest_s = str(dest)
        print(f"  wrote report {dest}", flush=True)
    runs = list(state.get("direction_runs") or [])
    runs.append(
        {
            "direction": Path(path).name if path else "",
            "bundle": path,
            "report": text,
            "report_path": dest_s,
            "selected_model": state.get("selected_model"),
            "metrics": state.get("metrics"),
        }
    )
    return {"report": text, "report_path": dest_s, "direction_runs": runs}


def next_direction_node(state: PipelineState) -> dict[str, Any]:
    return {
        "bundle_index": int(state.get("bundle_index") or 0) + 1,
        "rewrite_count": 0,
        "want_rewrite": False,
        "rewrite_instruction": "",
        "generated": {},
        "metrics": [],
        "eval_summary": {},
        "eval_history": [],
        "selected_model": "",
        "contribution": {},
        "attribution": {},
    }


def _after_evaluate(state: PipelineState) -> Literal["rewrite_model", "contribution"]:
    if state.get("want_rewrite") and int(state.get("rewrite_count") or 0) < MAX_REWRITE:
        return "rewrite_model"
    return "contribution"


def _after_report(state: PipelineState) -> Literal["next_direction", "__end__"]:
    index = int(state.get("bundle_index") or 0)
    paths = state.get("bundle_paths") or []
    if index + 1 < len(paths):
        return "next_direction"
    return "__end__"


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
    graph.add_node("next_direction", next_direction_node)
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", "plan")
    graph.add_edge("plan", "curate")
    graph.add_edge("curate", "prior")
    graph.add_edge("prior", "propose")
    graph.add_edge("propose", "evaluate")
    graph.add_conditional_edges("evaluate", _after_evaluate, {"rewrite_model": "rewrite_model", "contribution": "contribution"})
    graph.add_edge("rewrite_model", "evaluate")
    graph.add_edge("contribution", "report")
    graph.add_conditional_edges("report", _after_report, {"next_direction": "next_direction", "__end__": END})
    graph.add_edge("next_direction", "prior")
    return graph.compile()


def run_pipeline(state: PipelineState) -> PipelineState:
    final = build_pipeline().invoke(state)
    runs = final.get("direction_runs") or []
    if len(runs) > 1:
        parts = []
        for run in runs:
            parts.append(f"## {run.get('direction')}\n\n{run.get('report') or ''}".rstrip())
        final["report"] = "\n\n".join(parts) + "\n"
        paths = [str(run.get("report_path") or "") for run in runs if run.get("report_path")]
        final["report_path"] = "; ".join(paths)
    return final
