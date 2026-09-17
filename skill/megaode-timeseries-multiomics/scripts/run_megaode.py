#!/usr/bin/env python3
"""Thin wrapper around the packaged MEGAODE LangGraph entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fail_if_no_llm_key() -> None:
    if (os.environ.get("TMO_LLM_API_KEY") or "").strip():
        return
    print(
        "ERROR: TMO_LLM_API_KEY is missing. "
        "Inject the secret via Bohrium Skill credentials before starting MEGAODE. "
        "Optional configurable variables: TMO_LLM_BASE_URL, TMO_LLM_MODEL.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run packaged MEGAODE workflow")
    parser.add_argument("--source", type=Path, required=True, help="processed_data directory")
    parser.add_argument("--artifacts", type=Path, required=True, help="megaode_artifacts directory")
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--direction", default="multimodal")
    args = parser.parse_args()

    fail_if_no_llm_key()

    root = skill_root()
    engine = root / "engine"
    if not (engine / "src").exists():
        print(f"ERROR: packaged engine/src is missing under {engine}", file=sys.stderr)
        return 1

    source = args.source.resolve()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    generated = (args.generated_dir or artifacts.parent / "generated_models").resolve()
    generated.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(engine) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TMO_GENERATED_DIR"] = str(generated)
    if not (env.get("TMO_PRIOR_ROOT") or "").strip():
        for candidate in (Path("/share/TMO-agent-prior"), Path("/bohr-workspace/cache/megaode-prior")):
            if candidate.exists():
                env["TMO_PRIOR_ROOT"] = str(candidate)
                break

    cmd = [
        sys.executable,
        "-m",
        "src.api.run",
        "--source",
        str(source),
        "--artifacts",
        str(artifacts),
        "--direction",
        args.direction or "multimodal",
    ]
    print(f"running MEGAODE: {' '.join(cmd)}", flush=True)
    print(f"TMO_GENERATED_DIR={generated}", flush=True)
    print(f"TMO_PRIOR_ROOT={env.get('TMO_PRIOR_ROOT', '')}", flush=True)
    completed = subprocess.run(cmd, cwd=str(engine), env=env, check=False)
    if completed.returncode != 0:
        print("ERROR: MEGAODE workflow failed. Do not start biological interpretation.", file=sys.stderr)
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
