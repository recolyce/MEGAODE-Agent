# Data processing protocol

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

## 必需的输出格式

将结果写入本次运行目录：

```text
/bohr-workspace/output/megaode_run/processed_data/
├── sample_metadata.csv
├── proteomics.csv
├── metabolomics.csv
├── protein_annotations.csv
├── metabolite_annotations.csv
├── dataset_manifest.yaml
└── preprocessing_report.md
```

用户原始输入复制到：

```text
/bohr-workspace/output/megaode_run/input/
```

## sample_metadata.csv

至少包含以下字段：

```text
sample_id
subject_id
time
time_unit
condition
batch
```

## proteomics.csv

```text
sample_id | protein_1 | protein_2 | ...
```

行 = 样本，列 = 蛋白质。第一列必须是 `sample_id`。

## metabolomics.csv

```text
sample_id | metabolite_1 | metabolite_2 | ...
```

行 = 样本，列 = 代谢物。第一列必须是 `sample_id`。

蛋白质组学和代谢组学必须使用**同一套 `sample_id` 命名体系**。

## dataset_manifest.yaml

记录以下内容：

- dataset_name
- 受试者数量
- 样本数量（`n_samples`）
- 时间点数量
- 蛋白质特征数量（`n_proteins`）
- 代谢物特征数量（`n_metabolites`）
- 蛋白质组和代谢组配对样本数量（`n_paired_samples`）
- 所进行的数据变换
- 标准化方法
- 缺失情况
- 批次信息
- 推迟到建模阶段执行的预处理步骤

## preprocessing_report.md

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

完成后不要把旧 `processed_data/` 或旧 artifacts 当作本次结果。必须实际写出上述 7 个文件。
