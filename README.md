# TMO-agent-cursor-data-ready

从 BioMaster 处理好的模板出发：任务定义与训练折内划分（curator）→ 先验库 → 统一模型库。

不读取原始组学文件，不重复 BioMaster 已经做过的 log2 / 器官中位数中心化。

## 输入合约

```text
/personal/data/biomaster-processed/<dataset>/
  sample_metadata.csv
  proteomics.csv
  metabolomics.csv
  protein_annotations.csv
  metabolite_annotations.csv
  dataset_manifest.yaml
  preprocessing_report.md
```

当前可用：`/personal/data/biomaster-processed/gm-aging`（`MODELING READY WITH WARNINGS`）。

先验知识（KEGG 缓存、STRING/STITCH、ESM 权重、Uni-Mol）在 `/personal/workspace/TMO-agent-prior`，不进本仓库。覆盖路径：`export TMO_PRIOR_ROOT=...`。

## 任务：`last_interval`

- 同一 `subject_id` 有多个时间点 → **真滞后**（同人最后一个 interval 做测试）。
- 否则 → **伪滞后**（轨迹键 `tissue|sex|replicate`）。gm-aging 走这条。
- 训练：最后时间点之前的同时刻 + 相邻 interval；内部验证：训练段最后一个 interval；测试：最后一个 interval。
- `Y(t_last)` 不进入预处理拟合。
- 默认按 `tissue` 出 unit（器官=蛋白 batch）。

## 命令

```bash
# 只做肝脏 last_interval（两个方向）
python -m src.curator.run --source /personal/data/biomaster-processed/gm-aging \
  --task last_interval --unit liver

# 冒烟：train_mean / last_value / ridge
python -m src.models.smoke --unit liver --direction proteomics_to_metabolomics
```

产物在 `artifacts/<dataset>/last_interval/<unit>/<direction>/`。模型只读这里的 `ModelingBundle`。

## 模型库

`ModelRegistry().list()` / `build(name)`：

非 ML 对照只保留 `train_mean`, `last_value`, `ridge`。学习模型用完整库：`pls`, `mlp`, `neural_ode`, `feature_chunk_lstm`, `pathway_ridge`, `random_group_ridge`, `laplacian_ridge`, `graph_omics_ode`, `prior_fusion_ridge`, `prior_fusion_mlp`。流水线默认 **两个方向都跑**。

coding 节点写一个**先验注入动力学 ODE**（与 `graph_omics_ode` 同类：图/embedding 向量场 + 对 last_interval Δt 积分）。写模型时 LLM 可以调用环境工具装依赖；生成代码仍禁止 `os`/`subprocess`。

`last_value` 使用配对表里的 `persist_y_sample_id`（X 时刻、同一轨迹的 Y），不再在模型里特判真/伪。

## LangGraph 流水线

`inspect → plan(LLM) → curate → prior → propose → evaluate → (rewrite ≤3) → contribution → report`

```text
inspect      读 BioMaster 模板，判断真/伪滞后，列出可用先验
plan         大模型选 unit / 方向 / prior_sources（STRING / KEGG / 预训练槽）
curate       last_interval 配对 + 训练折内预处理
prior        把先验物化成模型 extra-input（Laplacian、通路分、embedding、pair prior）
propose      写一个先验注入动力学 ODE（无 key 则失败，不再回退 ridge）
evaluate     训练段（train+val）按 subject/轨迹做 5 折 CV 选超参，最后一段 interval 做测试；LLM 最多再改三次结构
contribution 写出各方法归因分数表（不做科学置信度/新颖性综合分）；并写 interpretation.md
report       每个方向一份实验笔记
```

```bash
export TMO_LLM_API_KEY=...          # 不要把 key 写进代码
export TMO_LLM_BASE_URL=https://api.gpugeek.com/v1
export TMO_LLM_MODEL=DeepSeek-V4-Pro

python -m src.api.run --source /personal/data/biomaster-processed/ipop_exercise
```

可调用工具在 `src/api/tools.py`：`inspect_biomaster_dataset`、`curate_last_interval`、`build_prior_network`、`fit_registered_models`、`attribute_model_pairs`。

只对已有 bundle 做归因：

```bash
python -m src.interpretation.run \
  --source /personal/data/biomaster-processed/ipop_exercise \
  --bundle artifacts/ipop_exercise/last_interval/all/proteomics_to_metabolomics \
  --models ridge,mlp
```

对每个蛋白–代谢对写出 coefficient / occlusion / gradient / permutation / 线性 SHAP 分数表，交给 BioMaster 做后续综合。`train_mean` / `last_value` 会跳过。

## 先验库

`PriorRegistry.build(..., backend="name_rule"|"kegg_reactome"|"auto")`

- `name_rule`：基因名 / 代谢物名模块（无网也能用）
- `kegg_reactome`：KEGG/Reactome 缓存 + UniProt/HMDB
- `pretrained`：蛋白走 ESM-2 35M（UniProt 序列）；代谢物走 PubChem SMILES → 本地 Uni-Mol2 84M，失败则 Morgan / 哈希名
