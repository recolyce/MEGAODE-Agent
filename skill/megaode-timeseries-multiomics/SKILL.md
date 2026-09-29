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
  对纵向 / 时间序列多组学（longitudinal / time-series / temporal multi-omics）
  蛋白质组（proteomics）与代谢组（metabolomics）执行固定流水线：
  标准化原始数据，运行 MEGAODE Neural ODE / Graph ODE 动态建模，
  完成跨模态互作与蛋白–代谢物 contribution/attribution，
  再进行文献支持的生物学发现分析。
  当用户提供时间序列或多时间点多组学数据，并需要未来时间点预测、
  跨模态归因或实验优先级排序时使用。
---

# megaode-timeseries-multiomics

本 Skill 是唯一编排层。MEGAODE 建模逻辑留在 `engine/` 内。BioMaster 只负责 MEGAODE 之前的数据标准化，以及 MEGAODE 之后的生物学解释。

每次调用都必须从用户原始输入完整重跑。不要做 workflow 状态检测。不要从已有 `processed_data/`、attribution 文件或旧 artifacts 断点恢复。旧结果不得当作本次运行输出复用。

**必须启用含有 GPU 的沙箱。** MEGAODE 的 Neural ODE / Graph ODE 等模型依赖 GPU；不要使用仅 CPU 的 sandbox。启动本 Skill 时，请在 Bohrium / BioMaster 中选择或开启 GPU 沙箱后再执行流水线。若当前沙箱没有 GPU，先切换到 GPU 沙箱，不要在 CPU 沙箱中启动 MEGAODE。

## 硬性规则

```text
DO NOT skip preprocessing.
DO NOT reuse previous MEGAODE artifacts.
DO NOT run biological interpretation before MEGAODE finishes.
DO NOT use literature evidence to modify model-derived priority scores.
```

含义如下：

- 不得跳过预处理。
- 不得复用以往 MEGAODE artifacts。
- 不得在 MEGAODE 完成前做生物学解释。
- 不得用文献证据修改模型导出的优先级分数。

另外：

- 不要重新实现 MEGAODE，也不要把现有 LangGraph 拆成多个 Skill。
- 不要虚构缺失的 prior、p-value、置信区间、fold、replicate、模型一致性或符号方向。
- 不要把 `TMO_LLM_API_KEY` 写死在文件里。
- 不要把 generated model 写进 Skill 安装目录。
- 若预处理状态为 `NOT MODELING READY`，或任一 validator 失败，立即停止并向用户说明原因。
- 若 MEGAODE 退出码非 0，立即停止，不要检索文献。
- 必须在含 GPU 的沙箱中运行；未启用 GPU 时不要启动 MEGAODE。

## 运行环境

本 Skill 必须在**含 GPU 的沙箱**中运行。Neural ODE、Graph ODE 以及后续 attribution 需要 GPU。调用前确认：

- Bohrium / BioMaster 已启用 GPU sandbox，而不是默认 CPU 沙箱；
- 当前会话可以访问 GPU 后再进入数据预处理与 MEGAODE。

凭证来自 Bohrium Skill configuration，注入为环境变量：

- 密钥：`TMO_LLM_API_KEY`
- 可选：`TMO_LLM_BASE_URL`、`TMO_LLM_MODEL`、`TMO_PRIOR_ROOT`

## 工作空间

每次创建或清空：

```text
/bohr-workspace/output/megaode_run/
```

固定目录布局：

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

## 唯一允许的主路径

1. 确认已启用含 GPU 的沙箱；然后将用户请求与上传 / 拉取的文件加载到该 sandbox。
2. 将原始输入复制到 `megaode_run/input/`。
3. 阅读 `references/data_processing_protocol.md` 和 `references/processed_data_contract.md`。
4. **实际执行**数据预处理，不要只给出处理建议。
5. 写出 `processed_data/` 下的 7 个必需文件。报告最后一行必须是 `MODELING READY`、`MODELING READY WITH WARNINGS` 或 `NOT MODELING READY`。
6. 运行：

```bash
python scripts/validate_processed_data.py /bohr-workspace/output/megaode_run/processed_data
```

7. 若 `ok=false` 或状态为 `NOT MODELING READY`，停止。
8. 运行：

```bash
python scripts/run_megaode.py \
  --source /bohr-workspace/output/megaode_run/processed_data \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts \
  --generated-dir /bohr-workspace/output/megaode_run/generated_models
```

9. 确认 MEGAODE 退出码为 0。
10. 运行：

```bash
python scripts/collect_megaode_handoff.py \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts \
  --handoff /bohr-workspace/output/megaode_run/handoff \
  --processed-data /bohr-workspace/output/megaode_run/processed_data \
  --generated-dir /bohr-workspace/output/megaode_run/generated_models
```

11. 运行：

```bash
python scripts/validate_megaode_handoff.py /bohr-workspace/output/megaode_run/handoff
```

12. 阅读 `references/biological_interpretation_protocol.md`、`references/literature_evidence_grading.md` 和 `references/final_output_contract.md`。
13. 仅根据 `handoff/` 计算并冻结 Experimental Priority Score（EPS）。缺失的评分项直接移除，剩余权重重新归一化。文献证据不得进入 EPS。
14. 只对高 EPS pair 做文献 / 数据库检索。检索前先做标识符与同义词规范化。禁止虚构论文。
15. 进行 A/B/C/D 证据分级，写出假设和实验建议，并严格区分：MEGAODE 数据支持 / 文献支持 / BioMaster 假设。
16. 在 `megaode_run/final/` 写出三个最终文件。
17. 运行：

```bash
python scripts/validate_final_outputs.py /bohr-workspace/output/megaode_run/final
```

18. 仅在最终 validator 通过后，对 `final/` 执行 RecordArtifact；最好同时保存 `processed_data/` 和 `handoff/`。
19. 向用户返回简短摘要：数据是否成功预处理、MEGAODE 是否成功完成、最佳模型及主要性能、跨模态 pair 数量、高优先级 `known_supported` 数量、`potential_discovery` 数量，以及三个最终文件路径。不要把大型 CSV 粘贴进对话。

这是唯一允许的主路径。运行细节见 `references/megaode_runtime_protocol.md`。
