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

无先验：`train_mean`, `last_value`, `ridge`, `pls`, `mlp`, `neural_ode`, `feature_chunk_lstm`

需 `bundle.prior`：`pathway_ridge`, `random_group_ridge`, `laplacian_ridge`, `graph_omics_ode`, `prior_fusion_ridge`, `prior_fusion_mlp`

coding 节点还会在 `src/models/generated/` 注册一个先验注入新模型。写模型时 LLM 可以调用 `list_python_packages` / `install_python_packages` / `run_repo_terminal`（仅 python/pip）查看当前环境并安装额外依赖；生成代码仍禁止 `os`/`subprocess`。

`last_value` 使用配对表里的 `persist_y_sample_id`（X 时刻、同一轨迹的 Y），不再在模型里特判真/伪。

## LangGraph 流水线

`inspect → plan(LLM) → curate → prior → propose → evaluate → (rewrite ≤1) → contribution → report`

```text
inspect      读 BioMaster 模板，判断真/伪滞后，列出可用先验
plan         大模型选 unit / 方向 / prior_sources（STRING / KEGG / 预训练槽）
curate       last_interval 配对 + 训练折内预处理
prior        把先验物化成模型 extra-input（Laplacian、通路分、embedding、pair prior）
propose      参照模型库写一个先验注入新模型（无 key 则写 prior_gated_ridge）
evaluate     GPU 训练 + 小网格调参 + PCC/Spearman/RMSE；LLM 最多再改一次结构
contribution 跨模态 contribution（系数 / 遮挡 / 梯度 / permutation / 可选 SHAP）交给 BioMaster
report       大模型写简短实验笔记
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
  --models ridge,pls
```

对每个蛋白–代谢对计算：系数幅度（线性模型）、输入遮挡、有限差分梯度、bootstrap 稳定性、扰动稳健性、先验支持。`train_mean` / `last_value` 会跳过。

## 先验库

`PriorRegistry.build(..., backend="name_rule"|"kegg_reactome"|"auto")`

- `name_rule`：基因名 / 代谢物名模块（无网也能用）
- `kegg_reactome`：KEGG/Reactome 缓存 + UniProt/HMDB
- `pretrained`：接口位；代谢物在有 SMILES 且安装 rdkit 时做 Morgan fingerprint，不下载 ESM/ChemBERTa
