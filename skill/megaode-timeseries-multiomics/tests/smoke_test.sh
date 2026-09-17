#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${ROOT}/engine"
PYTHONPATH="${ENGINE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH
unset TMO_PRIOR_ROOT || true
unset TMO_LLM_API_KEY || true
unset OPENAI_API_KEY || true

python3 - <<'PY'
import importlib
mods = [
    "src",
    "src.api.run",
    "src.models.registry",
    "src.prior.paths",
    "src.curator.loader",
]
for name in mods:
    importlib.import_module(name)
print("import_ok")
PY

python3 - <<'PY'
for name in (
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "yaml",
    "openpyxl",
    "optuna",
    "langgraph",
    "langchain_openai",
    "langchain_core",
    "shap",
):
    __import__(name)
print("requirements_ok")
PY

python3 "${ROOT}/scripts/validate_processed_data.py" \
  "${ROOT}/tests/fixtures/processed_data" > /tmp/megaode_processed.json
python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("/tmp/megaode_processed.json").read_text(encoding="utf-8"))
assert payload.get("ok") is True, payload
print("validator_ok")
PY

python3 -m src.api.run --help >/tmp/megaode_help.txt
test -s /tmp/megaode_help.txt
echo "cli_help_ok"

set +e
python3 "${ROOT}/scripts/run_megaode.py" \
  --source "${ROOT}/tests/fixtures/processed_data" \
  --artifacts /tmp/megaode_smoke_artifacts > /tmp/megaode_nokey.out 2>/tmp/megaode_nokey.err
rc=$?
set -e
test "$rc" -ne 0
grep -q "TMO_LLM_API_KEY" /tmp/megaode_nokey.err
echo "missing_key_ok"

python3 - <<'PY'
from src.prior.paths import prior_root
root = str(prior_root())
assert "/personal/workspace" not in root, root
print(f"prior_root_ok {root}")
PY

GEN_DIR="$(mktemp -d)"
export TMO_GENERATED_DIR="${GEN_DIR}"
python3 - <<PY
from pathlib import Path
from src.models.registry import generated_dir
dest = generated_dir()
dest.mkdir(parents=True, exist_ok=True)
probe = dest / "writable.txt"
probe.write_text("ok\n", encoding="utf-8")
assert probe.read_text(encoding="utf-8") == "ok\n"
assert dest == Path("${GEN_DIR}")
print("generated_dir_ok")
PY
unset TMO_GENERATED_DIR

python3 "${ROOT}/scripts/collect_megaode_handoff.py" \
  --artifacts "${ROOT}/tests/fixtures/artifacts" \
  --handoff /tmp/megaode_smoke_handoff \
  --processed-data "${ROOT}/tests/fixtures/processed_data" \
  --generated-dir "${ROOT}/tests/fixtures/generated_models" >/tmp/megaode_handoff.json
python3 "${ROOT}/scripts/validate_megaode_handoff.py" /tmp/megaode_smoke_handoff >/tmp/megaode_handoff_val.json
python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("/tmp/megaode_handoff_val.json").read_text(encoding="utf-8"))
assert payload.get("ok") is True, payload
print("handoff_ok")
PY

set +e
python3 "${ROOT}/scripts/validate_final_outputs.py" \
  "${ROOT}/tests/fixtures/final_bad" > /tmp/megaode_final_bad.json
rc=$?
set -e
test "$rc" -ne 0
python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("/tmp/megaode_final_bad.json").read_text(encoding="utf-8"))
assert payload.get("ok") is False, payload
assert any("same-modality" in err for err in payload.get("errors", [])), payload
print("final_same_modality_fail_ok")
PY

echo "SMOKE TEST PASSED"
