# Biomaster\-MEGAOD涉及到Biomaster的两部分prompt

## 2\.1 Data Process：

你是一名计算生物学数据科学家。

你的任务是将用户提供的**时间序列多组学数据集**转换为一个**干净、标准化、可直接用于后续机器学习建模**的数据格式。

输入可能包括上传的文件、文件夹、URL、数据库 accession number，或补充数据集。预期的组学模态主要为**蛋白质组学**和**代谢组学**。

请**实际执行整个工作流程**，而不仅仅是描述应该如何处理。

## 任务

1. 获取并检查所有输入文件。
2. 识别以下信息：

   1. 样本元数据
   2. 受试者 ID
   3. 时间点
   4. 条件 / 分组
   5. 批次
   6. 蛋白质组学矩阵
   7. 代谢组学矩阵
3. 标准化样本 ID，并重建纵向时间序列样本结构。
4. 分别处理蛋白质组学和代谢组学数据：

   1. 识别数据类型和数值尺度
   2. 处理无效值以及缺失值编码
   3. 采用有充分依据的数据变换 / 标准化方法
   4. 仅删除明显不可用的特征
   5. 标记潜在异常值或批次效应
5. 在可能的情况下统一特征标识符：

   1. 蛋白质：保留原始 ID，并在可用时映射到 UniProt ID / gene symbol
   2. 代谢物：保留原始 ID，并在可用时映射到 HMDB / KEGG / ChEBI
6. 对齐蛋白质组学和代谢组学中的样本。
7. 执行质量控制，并验证最终矩阵是否适合用于后续建模。

## 不要执行

- 有监督特征选择
- 训练集 / 测试集划分

必须避免数据泄漏。

任何应该**仅在训练数据上拟合**的预处理步骤，例如：

- 学习型缺失值插补
- scaling / 标准化缩放
- PCA
- 特征选择

都不应在整个数据集上全局拟合。应当将这些步骤推迟到后续建模阶段，并在报告中明确记录。

# 必需的输出格式

创建如下目录结构：

```Plain
processed_data/
├── sample_metadata.csv
├── proteomics.csv
├── metabolomics.csv
├── protein_annotations.csv
├── metabolite_annotations.csv
├── dataset_manifest.yaml
└── preprocessing_report.md
```

## sample\_metadata\.csv

至少包含以下字段：

```Plain
sample_id
subject_id
time
time_unit
condition
batch
```

## proteomics\.csv

```Plain
sample_id | protein_1 | protein_2 | ...
```

行 = 样本，列 = 蛋白质。

## metabolomics\.csv

```Plain
sample_id | metabolite_1 | metabolite_2 | ...
```

行 = 样本，列 = 代谢物。

蛋白质组学和代谢组学必须使用**同一套 ****`sample_id`**** 命名体系**。

## dataset\_manifest\.yaml

记录以下内容：

- 受试者数量
- 样本数量
- 时间点数量
- 蛋白质特征数量
- 代谢物特征数量
- 蛋白质组和代谢组配对样本数量
- 所进行的数据变换
- 标准化方法
- 缺失情况
- 批次信息
- 推迟到建模阶段执行的预处理步骤

## preprocessing\_report\.md

简要报告以下内容：

- 原始数据集结构
- 已执行的预处理步骤
- 缺失值情况
- 异常值 / 批次效应
- 样本对齐情况
- 重要的不确定性
- 对后续建模的任何警告

报告最后必须**严格以以下三个状态之一结束**：

`MODELING READY`

`MODELING READY WITH WARNINGS`

或

`NOT MODELING READY`

必须保留原始数据，并记录所有重要的数据处理决策。

当存在不确定性时，应优先采用**保守的预处理策略**，尽可能保留信息，而不是过度过滤数据。

完成预处理后，将整个 `processed_data/` 目录打包为一个 ZIP 压缩包，命名为：

```Plain
processed_data.zip
```

将该 ZIP 文件保存到工作区，并向我提供一个**可直接点击下载的链接**。

## 2\.2 Interpretation

请基于我的 MEGAODE\-Agent 仓库完成一次跨模态归因结果的生物学解释与发现分析。

GitHub 仓库：

https://github\.com/recolyce/MEGAODE\-Agent（这里skill化之后可以直接调用skill的代码库以及skill的返回结果）

分析数据集：

`<数据集名称>`

请首先访问并下载该 GitHub 仓库中与 `<数据集名称>` 相关的归因分析结果及必要的 annotation 文件。优先检查 `artifacts/<数据集名称>/` 下的 attribution、contribution、interpretation 等结果；如果实际目录结构不同，请根据仓库实际内容自行定位。

上游 MEGAODE\-Agent 已经完成模型训练和归因分析，因此本任务**不要重新训练模型或重新计算 attribution**。

任务目标

只分析**跨模态 source\-target pair**，例如 protein→metabolite、metabolite→protein 等。

不要分析同一组学内部的 pair，例如 protein→protein 或 metabolite→metabolite。

请完成以下工作：

1. 读取 `<数据集名称>` 的所有跨模态归因结果，确认不同 model、unit、direction 和 attribution 字段的实际含义。
2. 根据实际可用信息建立一个透明、可解释的 **Experimental Priority Score（0–100）**，并计算所有跨模态 pair 的分数和排名。

评分应以 MEGAODE\-Agent 的归因结果为核心，可根据实际数据考虑：

- attribution strength / percentile；
- 多模型一致性；
- 多 unit/context 重现性；
- attribution direction consistency；
- stability / robustness。

如果某一类信息不存在，不要虚构，直接从评分体系中移除。

3. 文献证据不要进入 Experimental Priority Score。

先完全根据模型数据计算并排序，再选择高分 pair 做文献验证，这样避免已有知识影响潜在发现的排序。

4. 对高分 pair 系统检索 PubMed、Google Scholar 及可靠生物数据库，例如 KEGG、Reactome、STRING、UniProt、HMDB、ChEBI 等。

文献证据分为：

- **A：Direct evidence** — 已有直接实验支持；
- **B：Strong mechanistic evidence** — 无直接验证，但存在明确 pathway / enzyme / signaling / biochemical mechanism；
- **C：Indirect evidence** — 无直接机制，但两者与共同 biological process / phenotype / tissue 等相关；
- **D：Little or no prior evidence** — 本次检索范围内未发现明显直接或间接证据。

优先引用原始研究，并提供标题、年份、期刊、DOI/PMID 或可验证链接。禁止虚构引用。

5. 将高分结果重点分为：

A\. 已有知识支持：高 Experimental Priority Score \+ A/B 级文献证据，这些结果用于说明 MEGAODE\-Agent 的归因结果具有生物学合理性。

B\. 潜在生物学发现：高 Experimental Priority Score，但缺乏直接文献证据，通常为 C/D，同时存在合理的间接生物学机制。

注意：没有搜到文献不等于新发现，请使用“潜在发现”“尚未充分研究的关联”“值得进一步实验验证”等谨慎表述，不要轻易声称“首次发现”。

对于最重要的潜在发现，请进一步给出：

- 可能的 mechanistic hypothesis；
- 当前证据链；
- 缺失的关键证据；
- 推荐的后续实验，例如 knockout/knockdown、overexpression、metabolite perturbation、targeted metabolomics/proteomics、enzyme assay、isotope tracing、rescue、time\-course 等。

同时明确区分：

- MEGAODE 数据直接支持的内容；
- 文献支持的内容；
- BioMaster 提出的 hypothesis。

输出三个文件

1\. `<数据集名称>_cross_modal_pair_scores.csv`

包含**所有跨模态 pair**，建议至少包含：

```Plain
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

其中 `classification` 使用：

```Plain
known_supported
potential_discovery
uncertain
not_literature_reviewed
```

没有进行文献检索的 pair，其 `literature_evidence_level` 应为 NA，而不是默认判为无文献。

2\. `<数据集名称>_high_score_literature_and_discoveries.md`

中文报告，重点展示：

- 高分 pair 的筛选阈值；
- 已有知识支持的代表性 pair；
- 文献及机制证据；
- 潜在发现 pair；
- mechanistic hypothesis；
- 后续实验建议；
- 存在冲突或不确定性的高分 pair。

3\. `<数据集名称>_cross_modal_biological_report.md`

中文总体报告，包括：

- 数据和分析范围；
- Experimental Priority Score 的计算方法及公式；
- score distribution；
- cross\-model / cross\-context consistency；
- 文献证据 A/B/C/D 的总体分布；
- 模型成功恢复的已知 biology；
- 最重要的 potential discoveries；
- Tier 1/2/3 实验验证优先级；
- limitations。

需要遵循以下原则：

- attribution ≠ biological causality；
- source→target 是模型分析方向，不自动代表真实生物学因果；
- 文献证据与实验分数必须分开；
- 不要因为某个关系研究很多就给它更高实验分；
- metabolite 和 protein 名称在检索前应进行 ID / synonym normalization；
- 如果结果与已有文献冲突，也要保留并讨论，不要强行解释；
- 不允许虚构不存在的 fold、replicate、p\-value、confidence interval 或 biological evidence。

# 当前在使用Biomaster过程中的主要问题：

- 目前在网页端使用，在预处理时上传数据容易失败（拖到对话框中被上传100%之后消失）
- 似乎无法直接在我已开的节点上运行程序
