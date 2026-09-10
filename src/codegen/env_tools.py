"""Tools the model-writing agent can call: inspect env, pip install, restricted terminal."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(\[[A-Za-z0-9_,]+\])?$")
READ_PREFIXES = ("src/models", "src/prior", "src/curator", "src/eval", "src/codegen", "requirements.txt")
DENY_CMD = re.compile(
    r"(rm\s+-rf|sudo\s|shutdown|reboot|mkfs|dd\s+if=|curl\s+[^\n]*\|\s*|wget\s+[^\n]*\|\s*|"
    r"chmod\s+777|>\s*/etc|ssh\s|scp\s|kill\s+-9|:\(\)|forkbomb)",
    re.I,
)
ALLOW_CMD = re.compile(
    r"^(PYTHONPATH=\.\s+)?"
    r"(python3?|pip3?|" + re.escape(sys.executable) + r")"
    r"(\s|$)",
)


def list_installed_packages(query: str = "") -> dict[str, Any]:
    q = query.strip().lower()
    rows = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name") or dist.name
        if q and q not in name.lower():
            continue
        rows.append({"name": name, "version": dist.version})
    rows.sort(key=lambda r: r["name"].lower())
    return {
        "python": sys.executable,
        "n": len(rows),
        "packages": rows[:400],
        "hint": "Use install_python_packages for extra PyPI deps. Generated models still cannot import os/subprocess.",
    }


def install_python_packages(packages: list[str] | str) -> dict[str, Any]:
    names = packages if isinstance(packages, list) else [part.strip() for part in str(packages).split(",") if part.strip()]
    bad = [n for n in names if not PKG_NAME.match(n) or n.startswith("-") or "/" in n or ":" in n]
    if bad:
        return {"ok": False, "error": f"refusing non-PyPI names: {bad}"}
    if not names:
        return {"ok": False, "error": "no packages"}
    cmd = [sys.executable, "-m", "pip", "install", *names]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    return {
        "ok": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


def run_terminal(command: str, timeout: int = 60) -> dict[str, Any]:
    text = (command or "").strip()
    if not text:
        return {"ok": False, "error": "empty command"}
    if DENY_CMD.search(text) or "\n" in text:
        return {"ok": False, "error": "command blocked"}
    if not ALLOW_CMD.match(text):
        return {
            "ok": False,
            "error": "only python/pip in the repo are allowed (optional PYTHONPATH=. prefix)",
        }
    proc = subprocess.run(text, cwd=REPO_ROOT, shell=True, capture_output=True, text=True, timeout=min(int(timeout), 180))
    return {
        "ok": proc.returncode == 0,
        "command": text,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-4000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


def read_repo_file(path: str, max_chars: int = 8000) -> dict[str, Any]:
    rel = path.replace("\\", "/").lstrip("./")
    if not any(rel == p or rel.startswith(p.rstrip("/") + "/") or rel == p for p in READ_PREFIXES):
        if rel != "requirements.txt":
            return {"ok": False, "error": f"read limited to {READ_PREFIXES}"}
    dest = (REPO_ROOT / rel).resolve()
    try:
        dest.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return {"ok": False, "error": "path escapes repo"}
    if not dest.exists():
        return {"ok": False, "error": f"missing {rel}"}
    text = dest.read_text(encoding="utf-8")
    return {"ok": True, "path": rel, "text": text[: max(500, int(max_chars))], "n_chars": len(text)}


TOOL_SPECS = {
    "list_installed_packages": {
        "fn": list_installed_packages,
        "args": {"query": "optional substring, e.g. boost or torch"},
    },
    "install_python_packages": {
        "fn": install_python_packages,
        "args": {"packages": "list of PyPI names, e.g. [\"lightgbm\"]"},
    },
    "run_terminal": {
        "fn": run_terminal,
        "args": {"command": "python/pip command only", "timeout": "seconds, max 180"},
    },
    "read_repo_file": {
        "fn": read_repo_file,
        "args": {"path": "src/models/classic.py", "max_chars": "8000"},
    },
}


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return {"ok": False, "error": f"unknown tool {name}; available {sorted(TOOL_SPECS)}"}
    try:
        return spec["fn"](**{k: v for k, v in (args or {}).items() if k in spec["args"] or k in {"query", "packages", "command", "timeout", "path", "max_chars"}})
    except TypeError as exc:
        return {"ok": False, "error": str(exc)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
