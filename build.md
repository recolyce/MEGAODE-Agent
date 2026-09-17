# MEGAODE BioMaster Skill 构建执行指南

## 一、总体目标

请基于当前服务器上的 `MEGAODE-Agent` 代码仓库，构建一个可以直接上传到 Bohrium / BioMaster 的独立 Skill 文件夹。

不要重新实现 MEGAODE 的建模算法，不要把现有 LangGraph workflow 拆成多个 Skill。

最终只生成一个 Skill：

`megaode-timeseries-multiomics`

用户只需要提供时间序列多组学研究数据和研究需求，BioMaster 检测到该 Skill 后，应当强制完整执行以下固定流水线：

```text
用户原始输入
    ↓
BioMaster 数据理解与预处理
    ↓
processed_data/ 标准模板
    ↓
输入合法性验证
    ↓
现有 MEGAODE LangGraph workflow
    ↓
模型比较 + 生成模型 + prediction/evaluation
    ↓
cross-modal contribution / attribution
    ↓
整理 MEGAODE → BioMaster handoff
    ↓
BioMaster 文献检索与生物学解释
    ↓
Experimental Priority Score
    ↓
literature evidence grading
    ↓
mechanistic hypotheses
    ↓
实验验证建议
    ↓
最终报告和 CSV
```

本 Skill **每次调用都必须完整重新运行**。

不实现 workflow state detection。

不从已有 `processed_data/`、已有 attribution 或旧 artifacts 中断点恢复。

旧结果不得被当作本次运行结果直接复用。

---

# 二、现有 MEGAODE 代码应保持的核心结构

保留当前仓库 `src/` 中的科学建模逻辑。

目前正式建模入口为：

```bash
python -m src.api.run \
    --source <processed_data_directory> \
    --artifacts <artifact_directory>
```

保留现有 LangGraph：

```text
inspect
→ plan
→ curate
→ prior
→ propose
→ evaluate
→ rewrite
→ contribution
→ report
```

不要重新用 BioMaster 实现这些节点。

BioMaster 的职责只负责：

```text
MEGAODE 前：
raw data → processed_data/

MEGAODE 后：
MEGAODE artifacts → biological interpretation
```

MEGAODE 本身继续负责：

```text
last_interval task definition
train/validation/test construction
train-only preprocessing
prior construction
model evaluation
LLM-generated dynamical model
model rewrite
attribution/contribution
modeling report
```

---

# 三、最终 Skill 文件夹

在仓库中建立：

```text
skill/
└── megaode-timeseries-multiomics/
    ├── SKILL.md
    ├── setup.sh
    ├── requirements.txt
    │
    ├── engine/
    │   ├── src/
    │   └── configs/
    │
    ├── scripts/
    │   ├── validate_processed_data.py
    │   ├── run_megaode.py
    │   ├── collect_megaode_handoff.py
    │   ├── validate_megaode_handoff.py
    │   └── validate_final_outputs.py
    │
    ├── references/
    │   ├── data_processing_protocol.md
    │   ├── processed_data_contract.md
    │   ├── megaode_runtime_protocol.md
    │   ├── biological_interpretation_protocol.md
    │   ├── literature_evidence_grading.md
    │   └── final_output_contract.md
    │
    └── tests/
        └── smoke_test.sh
```

`skill/megaode-timeseries-multiomics/` 必须能够脱离原 GitHub 根目录独立运行。

禁止依赖类似：

```text
../../src
../../configs
```

不要使用 symlink。

上传 Skill 文件夹之后，即使原 GitHub repository 不存在，该 Skill 也应能够运行。

因此将运行需要的 `src/` 和 `configs/` 复制进入：

```text
engine/
```

不要复制：

```text
artifacts/
logs/
.env
历史实验产物
```

---

# 四、SKILL.md 的定位

`SKILL.md` 是整个 workflow 的唯一 orchestration 层。

Skill 名称建议：

```yaml
name: megaode-timeseries-multiomics
version: 1.0.0
type: sandbox
```

搜索描述必须明确覆盖：

```text
longitudinal multi-omics
time-series multi-omics
temporal omics
proteomics
metabolomics
dynamic modeling
cross-modal interaction
protein-metabolite dynamics
Neural ODE
Graph ODE
biological discovery
```

建议 `l0`：

```text
时间序列蛋白质组和代谢组的动态建模、跨模态归因及生物学发现分析。
```

建议 `l1` 表达：

```text
当用户希望对 longitudinal / time-course multi-omics 数据，
尤其是 proteomics + metabolomics 数据进行动态建模、
未来时间点预测、跨模态 contribution/attribution、
protein-metabolite 关系优先级分析、文献验证或潜在生物学
发现分析时使用。

每次调用都从用户原始数据开始，依次执行数据标准化、
MEGAODE workflow 和文献支持的生物学解释。
```

必须明确：

```text
DO NOT skip preprocessing.
DO NOT reuse previous MEGAODE artifacts.
DO NOT run biological interpretation before MEGAODE finishes.
DO NOT use literature evidence to modify model-derived priority scores.
```

---

# 五、固定 Workspace 布局

每次任务创建一个新的运行目录：

```text
/bohr-workspace/output/megaode_run/
```

完整布局固定为：

```text
megaode_run/
├── input/
│   └── 用户原始输入文件
│
├── processed_data/
│   ├── sample_metadata.csv
│   ├── proteomics.csv
│   ├── metabolomics.csv
│   ├── protein_annotations.csv
│   ├── metabolite_annotations.csv
│   ├── dataset_manifest.yaml
│   └── preprocessing_report.md
│
├── megaode_artifacts/
│   └── <dataset>/
│       └── last_interval/
│           └── ...
│
├── handoff/
│   ├── megaode_run_summary.json
│   ├── model_metrics.csv
│   ├── cross_modal_contributions.csv
│   ├── generated_models.csv
│   ├── protein_annotations.csv
│   ├── metabolite_annotations.csv
│   └── handoff_manifest.yaml
│
├── interpretation/
│   ├── literature_evidence.csv
│   └── literature_search_log.md
│
└── final/
    ├── <dataset>_cross_modal_pair_scores.csv
    ├── <dataset>_high_score_literature_and_discoveries.md
    └── <dataset>_cross_modal_biological_report.md
```

每次开始运行时创建新的 `megaode_run` 工作空间或清空本次指定运行目录。

不得让旧 artifacts 混入本次分析。

---

# 六、Stage 1：BioMaster 数据预处理

将用户提供的第一段 data processing prompt 基本原样写入, 参见Biomaster-MEGAODE-prompts.md：

```text
references/data_processing_protocol.md
```

保持其科学要求，包括：

```text
识别 sample metadata
subject ID
time
condition
batch
proteomics
metabolomics

统一 sample ID
纵向样本重建
无效值处理
缺失编码
合理的数据变换
annotation harmonization
protein → UniProt / gene symbol
metabolite → HMDB / KEGG / ChEBI
cross-omics sample alignment
QC
```

必须原样保留以下原则：

```text
不做监督特征选择
不做 train/test split

不得在全数据上拟合：
learned imputation
scaling
PCA
feature selection
其他会造成 leakage 的学习型 transformation
```

BioMaster 必须**实际执行数据处理**，而不是仅输出建议。

用户输入可能为：

```text
CSV
TSV
Excel
ZIP / folder
URL
database accession
supplementary files
```

使用 BioMaster/Bohrium 已有的文件和 sandbox 工具完成获取、检查、转换。

Stage 1 最终必须生成：

```text
processed_data/
├── sample_metadata.csv
├── proteomics.csv
├── metabolomics.csv
├── protein_annotations.csv
├── metabolite_annotations.csv
├── dataset_manifest.yaml
└── preprocessing_report.md
```

`preprocessing_report.md` 最后一行必须严格为：

```text
MODELING READY
```

或：

```text
MODELING READY WITH WARNINGS
```

或：

```text
NOT MODELING READY
```

---

# 七、增加 deterministic 数据验证

新增：

```text
scripts/validate_processed_data.py
```

BioMaster 完成自然语言驱动的数据整理后，必须调用该程序。

验证至少包括：

```text
7 个必需文件存在；

sample_metadata.csv 包含：
sample_id
subject_id
time
time_unit
condition
batch；

sample_id 唯一；

time 可用于排序；

proteomics 第一列/索引能映射到 sample_id；

metabolomics 第一列/索引能映射到 sample_id；

protein/metabolite feature 名称无重复；

矩阵可转换为 numeric；

不存在 ±Inf；

至少存在有效 longitudinal structure；

proteomics 与 metabolomics 存在 paired samples；

manifest 中的样本数和特征数与真实文件一致；

preprocessing_report 的最终状态合法。
```

验证成功后返回机器可读 JSON：

```json
{
  "ok": true,
  "modeling_status": "MODELING READY WITH WARNINGS",
  "dataset_name": "...",
  "warnings": []
}
```

失败返回：

```json
{
  "ok": false,
  "errors": []
}
```

如果结果为：

```text
NOT MODELING READY
```

或者 validator `ok=false`：

**不得启动 MEGAODE。**

此时最终回复用户说明阻断原因。

---

# 八、Stage 2：运行现有 MEGAODE

新增：

```text
scripts/run_megaode.py
```

它只是现有 MEGAODE 的包装器，不重写模型逻辑。

逻辑等价于：

```bash
cd engine

python -m src.api.run \
  --source /bohr-workspace/output/megaode_run/processed_data \
  --artifacts /bohr-workspace/output/megaode_run/megaode_artifacts
```

默认：

```text
task = last_interval
direction = multimodal
run attribution = true
run codegen = true
```

保留现有模型库、prior、Optuna、rewrite 和 contribution 行为。

运行失败必须返回非零退出码。

不得在 MEGAODE 部分失败后继续执行 biological interpretation。

---

# 九、必须解决现有代码的可移植性问题

这是 Skill 化过程中必须修改的地方，但修改只针对 runtime portability，不改变科学算法。

## 9.1 移除 `/personal/...` 依赖, Biomaster可能读取不到，但是我在玻尔服务器的/share盘下cp了同一份，可以让biomaster尝试读取

当前 prior root 默认指向服务器路径：

```text
/personal/workspace/TMO-agent-prior
```

Skill 上传 Bohrium 后该路径不存在。

修改 prior path 逻辑，使默认值类似：

```text
/share/TMO-agent-prior或/bohr-workspace/cache/megaode-prior
```

同时继续允许：

```text
TMO_PRIOR_ROOT
```

环境变量覆盖。

如果 KEGG / STRING / STITCH / pretrained cache 不存在：

不要虚构数据。

不要因为某一种 external prior 缺失而伪造其可用性。

---

## 9.2 Generated model 不能写 Skill 安装目录

当前 generated model 写入：

```text
src/models/generated/
```

Skill 包安装位置可能不应被当成运行输出目录。

修改为支持：

```text
TMO_GENERATED_DIR
```

运行时设为：

```text
/bohr-workspace/output/megaode_run/generated_models/
```

`ModelRegistry` 同时从该目录加载本次生成模型。

不要改变生成模型的 AST 安全检查、smoke test 和注册逻辑。

---

## 9.3 LLM credential

当前 MEGAODE 内部需要：

```text
TMO_LLM_API_KEY
TMO_LLM_BASE_URL
TMO_LLM_MODEL
```

不要在：

```text
SKILL.md
setup.sh
.env
代码
日志
artifact
```

中写死 key。

通过 Bohrium Skill configuration / credential 注入环境变量。

保持：

```text
TMO_LLM_API_KEY
```

为 secret。

`TMO_LLM_BASE_URL` 和 `TMO_LLM_MODEL` 可配置。

如果未提供 key，应在真正启动 MEGAODE 前 fail fast，并给出明确错误。

---

# 十、setup.sh

建立幂等：

```text
setup.sh
```

安装当前 requirements：

```text
numpy
pandas
scikit-learn
torch
pyyaml
openpyxl
optuna
langgraph
langchain-openai
langchain-core
shap
```

使用 skill 自己的：

```text
requirements.txt
```

安装脚本必须：

```text
可重复执行
失败立即退出
不得打印 credential
不得安装无关依赖
```

如果某些 MEGAODE prior backend 需要额外包，根据现有代码真实 import 补充，但不要为了“可能需要”批量安装大量库。

---

# 十一、Stage 3：构建 MEGAODE → BioMaster handoff

不要让 BioMaster 自己递归扫描整个 `megaode_artifacts/` 并猜每个文件含义。

新增：

```text
scripts/collect_megaode_handoff.py
```

它负责把所有 unit / direction 的关键结果整理到：

```text
handoff/
```

必须读取并汇总：

```text
evaluation/evaluation.json

contribution/contribution_pairs.csv

contribution/method_scores.csv

contribution/biomaster_handoff.json

interpretation.md

report.md

prior_summary / provenance

本次 generated model 信息
```

生成：

### model_metrics.csv

至少包括：

```text
dataset
unit
direction
model
model_role
test_median_pcc
validation metric fields actually available
selected_model
error
```

禁止生成不存在的：

```text
p-value
confidence interval
replicate
fold metric
```

除非真实 artifact 中存在。

### cross_modal_contributions.csv

从所有 contribution table 中只保留：

```text
source_modality != target_modality
```

也就是只保留：

```text
proteomics → metabolomics
metabolomics → proteomics
```

过滤掉：

```text
proteomics → proteomics
metabolomics → metabolomics
```

至少保留：

```text
dataset
unit
direction
model
source_feature
source_name
source_modality
target_feature
target_name
target_modality
method
score
prior
```

不要在这里做 literature classification。

不要在这里判断 novelty。

---

# 十二、关于 attribution 的几个重要约束

当前代码中的 contribution 包含：

```text
coefficient
occlusion
gradient
permutation
可用时 SHAP
```

BioMaster 后处理必须先读取真实 available methods。

不存在的方法不参与评分。

特别注意：

当前 pipeline 的 contribution 默认针对 `selected_model`。

因此第一版 Skill **不得假装存在 multi-model attribution consistency**。

如果没有对多个模型计算 pair-level attribution：

```text
model_consistency_score = NA
```

并从 Experimental Priority Score 权重中移除。

不要根据“模型总体 PCC 都不错”推断 pair-level model consistency。

如果未来需要 multi-model consistency，再独立扩展 MEGAODE contribution 层。

---

# 十三、建议增加 signed attribution 输出，但不改变现有算法

当前 finite-difference 代码实际上已经计算 signed gradient，但主要 CSV 输出的是 magnitude。

允许 Codex做一个向后兼容的小增强：

保留现有：

```text
score
```

同时在可获得时增加：

```text
signed_score
```

或者单独增加：

```text
signed_gradient
signed_coefficient
```

目的只是让后续 `direction_consistency_score` 有真实数据来源。

如果无法可靠输出 sign：

`direction_consistency_score = NA`

并从 EPS 中移除。

绝对禁止根据 protein→metabolite 的模型方向虚构正/负调控方向。

---

# 十四、Stage 4：BioMaster 生物学解释

将用户提供的第二段 interpretation prompt 放入，参考Biomaster-MEGAODE-prompts.md：

```text
references/biological_interpretation_protocol.md
```

将内容从“访问 GitHub 下载 artifacts”调整为：

> 直接分析本次运行生成的 `/bohr-workspace/output/megaode_run/handoff/` 与 `megaode_artifacts/`。

其他科学约束保持不变。

即 BioMaster 此时应综合：

```text
模型代码
模型性能
selected model
各 unit/context
prior provenance
contribution / attribution
protein annotations
metabolite annotations
MEGAODE modeling reports
```

完成 downstream biological interpretation。

不得重新训练模型。

不得重新执行 MEGAODE。

不得重新计算 attribution。

---

# 十五、Experimental Priority Score

必须先计算纯模型数据驱动的：

```text
Experimental Priority Score
```

然后 freeze。

文献检索只能在 freeze 以后进行。

可根据实际存在的信息使用：

```text
attribution strength / percentile
multi-method consistency
cross-unit/context reproducibility
cross-model consistency
direction consistency
stability / robustness
```

但规则是：

```text
数据里不存在 → 不得虚构 → 该 component 从公式中移除 → 剩余权重重新归一化
```

例如只存在：

```text
attribution
context reproducibility
multi-method consistency
```

则 EPS 只能由这三项决定。

评分必须：

```text
0–100
透明
公式可报告
所有 pair 使用相同规则
```

Literature evidence **绝对不能进入 EPS**。

文献多少也不得影响 rank。

---

# 十六、文献检索阶段

EPS 完成并 freeze 后：

选择高分 pair 做 literature validation。

优先使用 Bohrium 已有论文搜索能力禁止虚构论文。

检索前必须进行 protein/metabolite identifier 和 synonym normalization。

---

# 十七、文献证据等级

将明确规则写入：

```text
references/literature_evidence_grading.md
```

等级固定为：

```text
A — Direct evidence
已有直接实验支持 source-target 关系。

B — Strong mechanistic evidence
没有该 pair 的直接验证，但存在清楚的 enzyme/pathway/
transport/signaling/biochemical mechanism。

C — Indirect evidence
没有直接机制，但双方与共同 pathway/process/phenotype/
tissue/disease 有合理关联。

D — Little or no prior evidence
在本次定义的检索范围内未找到明显直接或间接证据。
```

必须强调：

```text
D != relationship does not exist
D != novel discovery
```

只能表述：

```text
“本次检索范围内未发现”
“尚未充分研究”
“潜在发现”
“值得进一步实验验证”
```

---

# 十八、classification

最终：

```text
known_supported
```

用于：

```text
高 EPS + A/B evidence
```

`potential_discovery` 用于：

```text
高 EPS + C/D evidence
+ 存在合理的 biological/mechanistic hypothesis
```

`uncertain` 用于：

```text
identifier ambiguity
模型结果冲突
literature conflict
证据不足以解释
```

`not_literature_reviewed`：

只用于未进行 literature search 的 pair。

这类 pair：

```text
literature_evidence_level = NA
```

绝对不能自动写为：

```text
D
```

---

# 十九、必须区分三类结论来源

最终报告中所有关键结论必须能区分：

```text
1. MEGAODE DATA SUPPORT

2. LITERATURE SUPPORT

3. BIOMASTER HYPOTHESIS
```

例如：

```text
MEGAODE data support:
该 protein→metabolite pair 在 liver unit 中 attribution percentile 较高。

Literature support:
某研究显示该 protein 所在 enzyme pathway 可影响该代谢物。

BioMaster hypothesis:
该 protein 可能通过 pathway X 改变 metabolite Y 的时间动态。
```

严禁把第三项写成第一项。

---

# 二十、最终三个文件

必须严格生成：

```text
<dataset>_cross_modal_pair_scores.csv

<dataset>_high_score_literature_and_discoveries.md

<dataset>_cross_modal_biological_report.md
```

CSV 至少包含：

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

不存在的 component 使用：

```text
NA
```

而不是 `0`。

---

# 二十一、最终总体报告必须包含

`<dataset>_cross_modal_biological_report.md` 至少说明：

```text
数据集及时间结构

BioMaster preprocessing 摘要

MEGAODE task definition

模型库及模型选择结果

selected model

test performance

使用的 prior

generated model / rewrite 情况

contribution methods

跨模态 pair 总数

EPS 公式

EPS distribution

cross-unit/context consistency

literature evidence A/B/C/D distribution

模型成功恢复的 known biology

potential discoveries

主要 mechanistic hypotheses

Tier 1 / Tier 2 / Tier 3 experimental validation candidates

limitations
```

必须明确声明：

```text
attribution ≠ biological causality

source→target 是模型预测/归因方向，
不自动代表真实生物学因果。

literature evidence 与 Experimental Priority Score 完全独立。
```

---

# 二十二、运行完成后的 Artifact

使用 Bohrium artifact 机制至少保存整个：

```text
/bohr-workspace/output/megaode_run/final/
```

最好同时保存：

```text
processed_data/
handoff/
```

便于 reproducibility。

最终对话回复只需要总结：

```text
数据是否成功预处理
MEGAODE 是否成功完成
最佳模型及主要性能
跨模态 pair 数量
高优先级 known_supported 数量
potential_discovery 数量
三个最终文件的位置
```

不要把巨大的 CSV 内容直接塞进聊天上下文。

---

# 二十三、validate_final_outputs.py

新增最终 validator。

至少检查：

```text
三个最终文件都存在；

cross_modal_pair_scores.csv 不为空；

所有 pair 均满足
source_modality != target_modality；

experimental_priority_score 在 0–100；

rank 无明显重复/缺失错误；

未检索 pair 的 literature_evidence_level == NA；

classification 只能是：
known_supported
potential_discovery
uncertain
not_literature_reviewed；

Markdown 报告不为空；

报告中含 attribution ≠ causality 限制说明。
```

validator 通过以后才 RecordArtifact 并结束 Skill。

---

# 二十四、不要做的事情

不要：

```text
把两个大 prompt 都直接塞进 SKILL.md；

重新实现 MEGAODE；

删除现有 LangGraph；

让 BioMaster 自己实现 train/test split；

在 preprocessing 阶段做全局 scaling/imputation/PCA；

从旧 artifacts 恢复；

在 MEGAODE 运行失败后继续搜文献；

让 literature evidence 改变 EPS；

把未检索 pair 标成 evidence D；

虚构模型 consistency；

虚构 direction consistency；

把 attribution 描述成 causal effect；

写死 API key；

依赖 /personal/... 路径；

依赖原仓库在 Skill 外部存在。
```

---

# 二十五、建议的 SKILL.md 主流程

最终 `SKILL.md` 的工作流应非常简单：

```text
1. 获取用户输入
2. 把用户上传文件加载到 sandbox
3. 读取 references/data_processing_protocol.md
4. 完整执行数据预处理
5. 生成 processed_data/
6. 运行 validate_processed_data.py
7. 若不能建模则停止
8. 运行 scripts/run_megaode.py
9. 确认 MEGAODE 成功
10. 运行 collect_megaode_handoff.py
11. 运行 validate_megaode_handoff.py
12. 读取 references/biological_interpretation_protocol.md
13. 根据 handoff 计算并 freeze EPS
14. 对高分 pair 进行 literature/database search
15. 生成 evidence grading、hypothesis 和 experiments
16. 生成三个最终文件
17. 运行 validate_final_outputs.py
18. RecordArtifact
19. 向用户返回简短结果摘要及文件
```

这是唯一允许的主路径。

---

# 二十六、Smoke test

在：

```text
tests/smoke_test.sh
```

至少验证：

```text
1. Skill 中 engine/src 能正常 import。
2. requirements 已安装。
3. validator 可以识别一个最小合法 processed_data。
4. python -m src.api.run --help 可以执行。
5. TMO_LLM_API_KEY 缺失时给出清晰错误。
6. TMO_PRIOR_ROOT 不再默认指向 /personal/workspace。
7. generated model 输出目录可写。
8. collect_megaode_handoff.py 可读取 fixture artifacts。
9. final validator 对故意加入的 same-modality pair 会失败。
```

---

# 二十七、完成标准

Codex 完成后，请给出：

```text
1. 新建和修改的文件清单；

2. Skill 最终目录 tree；

3. 对原 MEGAODE 源码做了哪些 portability 修改；

4. 一条本地 smoke-test 命令；

5. 一个 zip：
   megaode-timeseries-multiomics.zip
```

zip 解压后的根目录必须直接是：

```text
megaode-timeseries-multiomics/
├── SKILL.md
├── setup.sh
├── ...
```

不要再套多余的：

```text
MEGAODE-Agent/skill/...
```

该 zip 应当能够直接用于 Bohrium Skill 上传。

---

# 最重要的设计原则

整个 Skill 的职责边界必须始终保持：

```text
BioMaster
│
├── 理解用户原始数据
├── 数据标准化
│
▼
processed_data/
│
▼
MEGAODE
│
├── task construction
├── leakage-safe modeling
├── prior
├── model comparison
├── generated model
├── attribution
└── contribution
│
▼
handoff/
│
▼
BioMaster
│
├── Experimental Priority Score
├── literature search
├── evidence grading
├── biological interpretation
├── mechanistic hypothesis
└── experiment design
│
▼
final scientific reports
```

BioMaster 不替代 MEGAODE。

MEGAODE 不承担 literature novelty judgment。

Literature 不反向影响 MEGAODE-derived Experimental Priority Score。
