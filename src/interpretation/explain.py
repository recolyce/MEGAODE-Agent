"""Write a BioMaster-facing markdown note: attribution tables and how they were computed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_pcc(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "NA"


def _rows_md(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int = 8) -> str:
    if not rows:
        return "_（本次没有可用行）_\n"
    header = "| " + " | ".join(title for _key, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows[:limit]:
        cells = []
        for key, _title in columns:
            val = row.get(key, "")
            if key == "score":
                try:
                    val = f"{float(val):.4f}"
                except (TypeError, ValueError):
                    val = str(val)
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_interpretation_markdown(
    bundle_path: str | Path,
    contribution: dict[str, Any] | None = None,
    attribution: dict[str, Any] | None = None,
    metrics: list[dict[str, Any]] | None = None,
    selected: str = "",
    prior: dict[str, Any] | None = None,
) -> str:
    del attribution
    root = Path(bundle_path)
    dest = root / "interpretation.md"
    contrib = dict(contribution or {})
    if not contrib and (root / "contribution" / "summary.json").exists():
        contrib = json.loads((root / "contribution" / "summary.json").read_text(encoding="utf-8"))
    if not metrics and (root / "evaluation" / "evaluation.json").exists():
        metrics = json.loads((root / "evaluation" / "evaluation.json").read_text(encoding="utf-8")).get("rows") or []

    metric_lines = ["| 模型 | 角色 | 验证中位 PCC | 测试中位 PCC |", "| --- | --- | --- | --- |"]
    for row in metrics or []:
        if row.get("error"):
            continue
        name = str(row.get("model") or "")
        role = "学习模型"
        metric_lines.append(
            f"| {name} | {role} | {_fmt_pcc(row.get('val_median_pcc'))} | {_fmt_pcc(row.get('test_median_pcc'))} |"
        )

    pretrained = (prior or {}).get("pretrained") or {}
    text = f"""# 跨模态归因表说明（给 BioMaster）

这份文件只说明**归因分数表**在哪里、怎么算。任务是多模态下一时刻预测。科学置信度、新颖性、最终每一对综合评分由 **BioMaster 后续分析**，本仓库不写这些列。

- 数据集：`{contrib.get("dataset") or (root.parts[-4] if len(root.parts) >= 4 else root)}`
- 任务：`{contrib.get("task") or "last_interval"}` / unit=`{contrib.get("unit") or ""}` / `{contrib.get("direction") or ""}`
- 滞后：`{contrib.get("lag_mode") or ""}`（true = 同一 `subject_id` 的最后一段 interval）
- 选中模型：`{selected or ""}`
- 写过归因表的模型：{", ".join(str(m) for m in (contrib.get("models") or []))}

## BioMaster 应读哪些文件

| 文件 | 格式 | 用途 |
| --- | --- | --- |
| `contribution/contribution_pairs.csv` | 长表 | 一行 = 模型 × 源特征 × 目标特征 × **一种方法** |
| `contribution/method_scores.csv` | 宽表 | 同一对把各方法分数放在不同列 |
| `contribution/<model>_method_scores.csv` | 宽表 | 单模型 |
| `contribution/biomaster_handoff.json` | JSON | 行数、方法列表、路径 |
| `evaluation/evaluation.json` | JSON | 预测 PCC 等，不是归因分数 |

本任务是多模态：两个组学同时输入，预测下一时刻两个组学。

不要使用旧产物里的 `ScientificConfidence` / `Novelty` / `classification` / `high_confidence_known_pairs.csv` / `novel_candidate_pairs.csv`——那些不再生成。

## 当前预测结果

{chr(10).join(metric_lines) if (metrics or []) else "_本次没有传入 evaluation 行。_"}

先验（若有）：边数 `{(prior or {}).get("n_edges", "")}`，来源 `{",".join((prior or {}).get("sources") or [])}`，预训练 `{pretrained.get("method") or ""}`。

## 当前归因摘要

- 长表行数：`{contrib.get("n_rows", "")}`
- 方法：{ ", ".join(contrib.get("methods") or []) }
- 宽表：`{contrib.get("method_scores_csv") or "contribution/method_scores.csv"}`

### Permutation 分数最高的若干对（仅示例，不是最终排序）

{_rows_md(list(contrib.get("top_permutation") or []), [("model", "模型"), ("source_name", "源"), ("target_name", "目标"), ("score", "permutation")])}

分数是**该目标内相对重要性**（行 L1 归一化），不是因果、也不是浓度变化。综合排序交给 BioMaster。

## 各方法怎么计算

矩阵形状 `(n_targets, n_expr)`。对每个目标（一行）做 L1 归一化，使该目标上各源特征分数之和为 1。只解释表达列，不解释时间差等 context。

### coefficient

线性模型取 `coef_` 绝对值再按目标归一化。MLP / ODE 等没有系数则此方法缺列。

### occlusion / permutation

测试集上打乱第 `i` 个源特征，看每个目标预测的平均绝对变化 `|Ŷ − Ŷ₀|`。两种方法同一思路、不同随机种子。

### gradient

中心有限差分：`ε = 0.01 × std(x_i)`（标准差太小时用 0.01），  
`g = (Ŷ(x+εe_i) − Ŷ(x−εe_i)) / (2ε)`，再对样本取平均绝对梯度并归一化。这是局部灵敏度，不是训练时的反向传播。

### shap

仅线性模型：`shap.LinearExplainer`（interventional）。不用 Kernel SHAP。失败则用 `|x_centered × coef|` 均值。

长表列：`model, source_feature, target_feature, source_name, target_name, source_modality, target_modality, method, score, prior`。  
`prior` 只是网络里是否已有酶–代谢 / 通路共现（1 或 0.5 或 0），不是最终对评分。

## 使用建议

1. 用 `evaluation/metrics_by_method.xlsx` 或 `evaluation.json` 比较各机器学习模型。
2. 把 `contribution_pairs.csv` 或 `method_scores.csv` 读进 BioMaster，在那边做置信度 / 新颖性 / 综合排序。
3. 多模态任务只有一份表：`.../multimodal/contribution/`。
"""
    dest.write_text(text, encoding="utf-8")
    return str(dest)
