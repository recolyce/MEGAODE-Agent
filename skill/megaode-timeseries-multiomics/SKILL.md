---
name: megaode-timeseries-multiomics
version: 1.0.0
type: sandbox
l0: 时间序列蛋白质组和代谢组的动态建模、跨模态归因及生物学发现分析。
l1: >-
  当用户希望对 longitudinal / time-course multi-omics 数据，
  尤其是 proteomics + metabolomics 数据进行动态建模、
  未来时间点预测、跨模态 contribution/attribution、
  protein-metabolite 关系优先级分析、文献验证或潜在生物学
  发现分析时使用。
  每次调用都从用户原始数据开始，依次执行数据标准化、
  MEGAODE workflow 和文献支持的生物学解释。
description: >-
  Run a fixed longitudinal / time-series / temporal multi-omics pipeline
  for proteomics and metabolomics: standardize raw data, run MEGAODE
  Neural ODE / Graph ODE dynamic modeling, cross-modal interaction and
  protein-metabolite contribution/attribution, then literature-backed
  biological discovery. Use when the user provides time-course or
  longitudinal multi-omics data and wants future-time prediction,
  cross-modal attribution, or experimental priority ranking.
---

# megaode-timeseries-multiomics

This Skill is the only orchestration layer. MEGAODE stays inside `engine/`. BioMaster only does pre-MEGAODE data standardization and post-MEGAODE biological interpretation.

Every call must rerun the full pipeline from the user's raw input. Do not detect workflow state. Do not resume from existing `processed_data/`, attribution files, or old artifacts. Old results must not be reused as this run's outputs.

## Hard rules

```text
DO NOT skip preprocessing.
DO NOT reuse previous MEGAODE artifacts.
DO NOT run biological interpretation before MEGAODE finishes.
DO NOT use literature evidence to modify model-derived priority scores.
```

Also:

- Do not reimplement MEGAODE or split the LangGraph into extra Skills.
- Do not invent missing priors, p-values, CIs, folds, replicates, model consistency, or signed direction.
- Do not hard-code `TMO_LLM_API_KEY`.
- Do not write generated models into the Skill install directory.
- If preprocessing is `NOT MODELING READY` or a validator fails, stop and tell the user why.
- If MEGAODE exits non-zero, stop. Do not search literature.

## Workspace

Create or empty:

```text
/bohr-workspace/output/megaode_run/
```

Fixed layout:

```text
megaode_run/
├── input/
├── processed_data/
├── megaode_artifacts/
├── generated_models/
├── handoff/
├── interpretation/
└── final/
```

Copy user files into `input/`. Credentials come from Bohrium Skill configuration:

- secret: `TMO_LLM_API_KEY`
- optional: `TMO_LLM_BASE_URL`, `TMO_LLM_MODEL`, `TMO_PRIOR_ROOT`

## Allowed path

1. Get the user request and load uploaded / fetched files into the sandbox.
2. Copy raw inputs to `megaode_run/input/`.
3. Read `references/data_processing_protocol.md` and `references/processed_data_contract.md`.
4. Actually preprocess the data. Do not only recommend steps.
5. Write the 7 `processed_data/` files. The report must end with `MODELING READY`, `MODELING READY WITH WARNINGS`, or `NOT MODELING READY`.
6. Run:

```bash
python scripts/validate_processed_data.py /bohr-workspace/output/megaode_run/processed_data
```

7. If `ok=false` or status is `NOT MODELING READY`, stop.
8. Run:

```bash
python scripts/run_megaode.py \
  --source /bohr-workspace/output/megaode_run/processed_data \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts \
  --generated-dir /bohr-workspace/output/megaode_run/generated_models
```

9. Confirm MEGAODE exit code is 0.
10. Run:

```bash
python scripts/collect_megaode_handoff.py \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts \
  --handoff /bohr-workspace/output/megaode_run/handoff \
  --processed-data /bohr-workspace/output/megaode_run/processed_data \
  --generated-dir /bohr-workspace/output/megaode_run/generated_models
```

11. Run:

```bash
python scripts/validate_megaode_handoff.py /bohr-workspace/output/megaode_run/handoff
```

12. Read `references/biological_interpretation_protocol.md`, `references/literature_evidence_grading.md`, and `references/final_output_contract.md`.
13. From `handoff/` only, compute and freeze Experimental Priority Score. Missing components are removed and remaining weights are renormalized. Literature never enters EPS.
14. Search literature/databases for high-EPS pairs only. Normalize identifiers first. Do not invent papers.
15. Grade evidence A/B/C/D, write hypotheses and experiment suggestions, and keep MEGAODE DATA SUPPORT / LITERATURE SUPPORT / BIOMASTER HYPOTHESIS separate.
16. Write the three final files under `megaode_run/final/`.
17. Run:

```bash
python scripts/validate_final_outputs.py /bohr-workspace/output/megaode_run/final
```

18. RecordArtifact for `final/` (and preferably `processed_data/` + `handoff/`) only after the final validator passes.
19. Reply with a short summary: preprocessing status, MEGAODE status, best model and test metric, cross-modal pair count, high-priority `known_supported` count, `potential_discovery` count, and the three file paths. Do not paste large CSVs.

This is the only allowed main path. Runtime details are in `references/megaode_runtime_protocol.md`.
