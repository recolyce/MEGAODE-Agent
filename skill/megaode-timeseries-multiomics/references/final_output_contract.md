# Final output contract

Write these three files under `/bohr-workspace/output/megaode_run/final/` and then run `scripts/validate_final_outputs.py`. Record artifacts only after the validator passes.

## 1. `<dataset>_cross_modal_pair_scores.csv`

Must contain every cross-modal pair. Required columns:

```text
dataset
unit
source_modality
source_id
source_name
target_modality
target_id
target_name
direction
models_supporting
raw_attribution_summary
attribution_score
model_consistency_score
context_score
direction_consistency_score
stability_score
experimental_priority_score
experimental_priority_rank
experimental_priority_percentile
selected_for_literature_validation
literature_evidence_level
classification
notes
```

Rules:

- `source_modality != target_modality`
- `experimental_priority_score` is 0–100
- ranks are 1..N with no duplicates or gaps
- missing components are `NA`, not `0`
- `classification` is one of `known_supported`, `potential_discovery`, `uncertain`, `not_literature_reviewed`
- unreviewed pairs have `literature_evidence_level = NA`

## 2. `<dataset>_high_score_literature_and_discoveries.md`

中文报告，重点展示：

- 高分 pair 的筛选阈值
- 已有知识支持的代表性 pair
- 文献及机制证据
- 潜在发现 pair
- mechanistic hypothesis
- 后续实验建议
- 存在冲突或不确定性的高分 pair

Every key claim must be labeled as one of:

1. MEGAODE DATA SUPPORT
2. LITERATURE SUPPORT
3. BIOMASTER HYPOTHESIS

## 3. `<dataset>_cross_modal_biological_report.md`

中文总体报告，至少包括：

- 数据集及时间结构
- BioMaster preprocessing 摘要
- MEGAODE task definition
- 模型库及模型选择结果
- selected model
- test performance
- 使用的 prior
- generated model / rewrite 情况
- contribution methods
- 跨模态 pair 总数
- EPS 公式
- EPS distribution
- cross-unit/context consistency
- literature evidence A/B/C/D distribution
- 模型成功恢复的 known biology
- potential discoveries
- 主要 mechanistic hypotheses
- Tier 1 / Tier 2 / Tier 3 experimental validation candidates
- limitations

Must explicitly state:

```text
attribution ≠ biological causality

source→target 是模型预测/归因方向，
不自动代表真实生物学因果。

literature evidence 与 Experimental Priority Score 完全独立。
```

## Artifact recording

After `validate_final_outputs.py` passes, record at least:

```text
/bohr-workspace/output/megaode_run/final/
```

Prefer also recording:

```text
/bohr-workspace/output/megaode_run/processed_data/
/bohr-workspace/output/megaode_run/handoff/
```

The chat reply should only summarize:

- 数据是否成功预处理
- MEGAODE 是否成功完成
- 最佳模型及主要性能
- 跨模态 pair 数量
- 高优先级 known_supported 数量
- potential_discovery 数量
- 三个最终文件的位置

Do not paste large CSV contents into the chat.
