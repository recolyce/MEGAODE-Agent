"""Tool-using coding loop: inspect env, pip install, write a prior-injected model."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from src.codegen.contract import FORBIDDEN, SYSTEM
from src.codegen.env_tools import dispatch
from src.curator.bundle import ModelingBundle
from src.models.registry import GENERATED_DIR, MODELS, ModelRegistry, register

SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
BLOCKED_NAMES = set(MODELS) | {"prior_gated_ridge"}
MAX_TURNS = 12


def _parse_action(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|text)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reply had no JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON was not an object")
    return parsed


def _check_ast(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "open", "__import__"}:
            raise ValueError(f"call not allowed: {node.func.id}")
        for raw in names:
            root = raw.split(".")[0]
            if root in FORBIDDEN:
                raise ValueError(f"import not allowed in generated models: {raw} (install/run via tools instead)")


def _load_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"src.models.generated.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(f"{path.name} has no class {class_name}")
    return cls


def _smoke(cls: type, bundle: ModelingBundle) -> dict[str, Any]:
    model = cls()
    model.fit(bundle)
    pred = model.predict(bundle)
    y = bundle.test.Y
    if pred.shape != y.shape:
        raise ValueError(f"predict shape {pred.shape} != Y {y.shape}")
    return {"ok": True, "shape": list(pred.shape)}


def write_generated(model_name: str, class_name: str, code: str, bundle: ModelingBundle) -> dict[str, Any]:
    if not SAFE_NAME.match(model_name):
        raise ValueError(f"bad model_name {model_name!r}")
    if model_name in BLOCKED_NAMES:
        raise ValueError(f"model_name {model_name!r} is reserved; pick a new name")
    _check_ast(code)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{model_name}.py"
    path.write_text(code, encoding="utf-8")
    cls = _load_class(path, class_name)
    if getattr(cls, "name", None) != model_name:
        cls.name = model_name
    smoke = _smoke(cls, bundle)
    register(model_name, cls, bool(getattr(cls, "requires_prior", True)))
    return {"ok": True, "model": model_name, "class_name": class_name, "path": str(path), "smoke": smoke}


def _catalog_text() -> str:
    return "\n".join(f"- {item['name']} prior={item['requires_prior']}" for item in ModelRegistry().list())


def _run_agent_tool(name: str, args: dict[str, Any], bundle: ModelingBundle, written: dict[str, Any] | None) -> dict[str, Any]:
    if name == "write_generated_model":
        return write_generated(str(args.get("model_name") or ""), str(args.get("class_name") or ""), str(args.get("code") or ""), bundle)
    if name == "smoke_generated_model":
        target = str(args.get("model_name") or (written or {}).get("model") or "")
        path = GENERATED_DIR / f"{target}.py"
        if not path.exists():
            return {"ok": False, "error": f"no generated file for {target}"}
        class_name = str((written or {}).get("class_name") or "")
        if not class_name:
            return {"ok": False, "error": "unknown class_name; write_generated_model first"}
        cls = _load_class(path, class_name)
        return {"ok": True, "model": target, "smoke": _smoke(cls, bundle)}
    return dispatch(name, args)


def propose_model(
    bundle: ModelingBundle,
    inspection: dict[str, Any] | None = None,
    prior_summary: dict[str, Any] | None = None,
    instruction: str = "",
    existing: list[str] | None = None,
) -> dict[str, Any]:
    from src.api.client import get_chat_model, has_llm_key

    if not has_llm_key():
        raise RuntimeError("TMO_LLM_API_KEY required to write a new model; deterministic fallback is disabled")
    payload = {
        "catalog": _catalog_text(),
        "n_expr": bundle.n_expr,
        "n_protein": getattr(bundle, "n_protein", bundle.n_expr),
        "n_metabolite": getattr(bundle, "n_metabolite", 0),
        "n_y": len(bundle.y_features),
        "n_train": int(len(bundle.train.pairs)),
        "lag_mode": bundle.lag_mode,
        "direction": bundle.direction,
        "x_modality": bundle.x_modality,
        "y_modality": bundle.y_modality,
        "prior": prior_summary,
        "inspection_priors": (inspection or {}).get("prior_hints"),
        "instruction": instruction
        or "Write and smoke one prior-injected dynamical ODE (graph/embedding vector field + RK over last_interval dt). Do not write another ridge.",
        "do_not_reuse": list(existing or []) + sorted(BLOCKED_NAMES),
    }
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]
    llm = get_chat_model(temperature=0.2)
    written: dict[str, Any] | None = None
    trace: list[str] = []
    for turn in range(MAX_TURNS):
        reply = llm.invoke(messages)
        text = str(reply.content)
        messages.append({"role": "assistant", "content": text})
        try:
            action = _parse_action(text)
        except Exception as exc:  # noqa: BLE001
            messages.append({"role": "user", "content": json.dumps({"error": f"not a tool JSON: {exc}. Reply with one JSON object."})})
            continue
        if action.get("done"):
            name = str(action.get("model_name") or (written or {}).get("model") or "")
            if not written or written.get("model") != name:
                messages.append({"role": "user", "content": json.dumps({"error": "call write_generated_model and smoke before done"})})
                continue
            return {
                **written,
                "reason": action.get("reason") or written.get("reason"),
                "used_llm": True,
                "fallback": False,
                "repaired": turn > 0,
                "tool_trace": trace,
            }
        tool = str(action.get("tool") or "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        print(f"  codegen tool {tool} { {k: args.get(k) for k in args if k != 'code'} }", flush=True)
        try:
            result = _run_agent_tool(tool, args, bundle, written)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
        if tool == "write_generated_model" and result.get("ok"):
            written = result
        trace.append(f"{tool}:{'ok' if result.get('ok') else result.get('error')}")
        messages.append({"role": "user", "content": json.dumps({"tool": tool, "result": result}, ensure_ascii=False, default=str)[:12000]})
    if written:
        return {**written, "used_llm": True, "fallback": False, "repaired": True, "tool_trace": trace, "reason": "finished after max tool turns"}
    raise RuntimeError(f"LLM did not write a runnable model in {MAX_TURNS} tool turns: {trace}")
