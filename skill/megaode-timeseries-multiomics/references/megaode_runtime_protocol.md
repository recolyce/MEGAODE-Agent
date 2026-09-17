# MEGAODE runtime protocol

Do not reimplement MEGAODE. Do not skip nodes. Do not resume from old artifacts.

## Workspace

```text
/bohr-workspace/output/megaode_run/
├── input/
├── processed_data/
├── megaode_artifacts/
├── generated_models/
├── handoff/
├── interpretation/
└── final/
```

Create a new `megaode_run` or empty the specified run directory at the start of every call.

## Credentials

Required before MEGAODE starts:

- `TMO_LLM_API_KEY` (secret, never hard-code, never print)

Optional:

- `TMO_LLM_BASE_URL`
- `TMO_LLM_MODEL`
- `TMO_PRIOR_ROOT`

If `TMO_LLM_API_KEY` is missing, fail fast. Do not start interpretation.

## Prior root

Resolution order:

1. `TMO_PRIOR_ROOT`
2. `/share/TMO-agent-prior` if it exists
3. `/bohr-workspace/cache/megaode-prior` if it exists

If KEGG / STRING / STITCH / pretrained cache is missing, do not fabricate that source.

## Generated models

Set:

```text
TMO_GENERATED_DIR=/bohr-workspace/output/megaode_run/generated_models
```

Never write generated models into the Skill install tree.

## How to run

From the Skill root, after `validate_processed_data.py` succeeds:

```bash
python scripts/run_megaode.py \
  --source /bohr-workspace/output/megaode_run/processed_data \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts \
  --generated-dir /bohr-workspace/output/megaode_run/generated_models
```

This is equivalent to:

```bash
cd engine
python -m src.api.run \
  --source /bohr-workspace/output/megaode_run/processed_data \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts
```

Defaults:

- `task = last_interval`
- `direction = multimodal`
- attribution on
- codegen on

Existing LangGraph remains:

```text
inspect → plan → curate → prior → propose → evaluate → rewrite → contribution → report
```

A non-zero exit code means MEGAODE failed. Stop. Do not search literature.

## After MEGAODE

```bash
python scripts/collect_megaode_handoff.py \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts \
  --handoff /bohr-workspace/output/megaode_run/handoff \
  --processed-data /bohr-workspace/output/megaode_run/processed_data \
  --generated-dir /bohr-workspace/output/megaode_run/generated_models

python scripts/validate_megaode_handoff.py \
  /bohr-workspace/output/megaode_run/handoff
```

Handoff is the only structured input to biological interpretation. Do not recursively guess the meaning of every file under `megaode_artifacts/`.
