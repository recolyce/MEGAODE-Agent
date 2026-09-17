# Biological interpretation protocol

请基于**本次运行**生成的 MEGAODE 结果完成一次跨模态归因结果的生物学解释与发现分析。

直接分析：

```text
/bohr-workspace/output/megaode_run/handoff/
/bohr-workspace/output/megaode_run/megaode_artifacts/
/bohr-workspace/output/megaode_run/processed_data/
```

不要访问 GitHub 下载旧 artifacts。不要使用历史运行产物。

上游 MEGAODE 已经完成模型训练和归因分析，因此本任务**不要重新训练模型或重新计算 attribution**。

分析数据集名称从 `handoff/megaode_run_summary.json` 或 `processed_data/dataset_manifest.yaml` 读取。

## 任务目标

只分析**跨模态 source-target pair**，例如 protein→metabolite、metabolite→protein 等。

不要分析同一组学内部的 pair，例如 protein→protein 或 metabolite→metabolite。

请完成以下工作：

1. 读取本次运行的所有跨模态归因结果，确认不同 model、unit、direction 和 attribution 字段的实际含义。先读 `handoff/cross_modal_contributions.csv`、`handoff/model_metrics.csv`、`handoff/generated_models.csv` 和 annotations。
2. 根据实际可用信息建立一个透明、可解释的 **Experimental Priority Score（0–100）**，并计算所有跨模态 pair 的分数和排名。

评分应以 MEGAODE 的归因结果为核心，可根据实际数据考虑：

- attribution strength / percentile；
- 多方法一致性（仅使用真实存在的 method）；
- 多 unit/context 重现性；
- attribution direction consistency（仅当 `signed_score` / `signed_gradient` / `signed_coefficient` 真实存在）；
- stability / robustness。

第一版 contribution 默认只针对 `selected_model`。如果没有对多个模型计算 pair-level attribution，则：

```text
model_consistency_score = NA
```

并从 Experimental Priority Score 权重中移除。不要根据“模型总体 PCC 都不错”推断 pair-level model consistency。

如果无法可靠输出 sign：

```text
direction_consistency_score = NA
```

并从 EPS 中移除。绝对禁止根据 protein→metabolite 的模型方向虚构正/负调控方向。

如果某一类信息不存在，不要虚构，直接从评分体系中移除，剩余权重重新归一化。

3. 文献证据不要进入 Experimental Priority Score。

先完全根据模型数据计算并排序并 **freeze EPS**，再选择高分 pair 做文献验证，这样避免已有知识影响潜在发现的排序。

4. 对高分 pair 系统检索 PubMed、Google Scholar 及可靠生物数据库，例如 KEGG、Reactome、STRING、UniProt、HMDB、ChEBI 等。优先使用 Bohrium 已有论文搜索能力。禁止虚构论文。

检索前必须进行 protein/metabolite identifier 和 synonym normalization。

文献证据等级见 `references/literature_evidence_grading.md`。

5. 将高分结果重点分为：

- `known_supported`：高 Experimental Priority Score + A/B 级文献证据
- `potential_discovery`：高 Experimental Priority Score + C/D 级证据，同时存在合理的 biological/mechanistic hypothesis
- `uncertain`：identifier 歧义、模型结果冲突、文献冲突、或证据不足以解释
- `not_literature_reviewed`：未进行 literature search 的 pair。这类 pair 的 `literature_evidence_level = NA`，绝对不能自动写成 D

没有搜到文献不等于新发现，请使用“本次检索范围内未发现”“尚未充分研究”“潜在发现”“值得进一步实验验证”等谨慎表述，不要轻易声称“首次发现”。

对于最重要的潜在发现，请进一步给出：

- 可能的 mechanistic hypothesis；
- 当前证据链；
- 缺失的关键证据；
- 推荐的后续实验，例如 knockout/knockdown、overexpression、metabolite perturbation、targeted metabolomics/proteomics、enzyme assay、isotope tracing、rescue、time-course 等。

同时明确区分：

- MEGAODE DATA SUPPORT
- LITERATURE SUPPORT
- BIOMASTER HYPOTHESIS

严禁把第三项写成第一项。

文献检索日志写入：

```text
/bohr-workspace/output/megaode_run/interpretation/literature_evidence.csv
/bohr-workspace/output/megaode_run/interpretation/literature_search_log.md
```

## 输出三个文件

写入 `/bohr-workspace/output/megaode_run/final/`：

1. `<数据集名称>_cross_modal_pair_scores.csv`
2. `<数据集名称>_high_score_literature_and_discoveries.md`
3. `<数据集名称>_cross_modal_biological_report.md`

字段、分类和报告章节见 `references/final_output_contract.md`。

需要遵循以下原则：

- attribution ≠ biological causality
- source→target 是模型分析方向，不自动代表真实生物学因果
- 文献证据与实验分数必须分开
- 不要因为某个关系研究很多就给它更高实验分
- metabolite 和 protein 名称在检索前应进行 ID / synonym normalization
- 如果结果与已有文献冲突，也要保留并讨论，不要强行解释
- 不允许虚构不存在的 fold、replicate、p-value、confidence interval 或 biological evidence
- 不存在的 component 使用 `NA`，而不是 `0`
