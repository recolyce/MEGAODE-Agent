"""Write a short BioMaster note for the selected model only."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any


def _fmt_pcc(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "NA"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_script_path(name: str) -> str:
    if not name:
        return ""
    try:
        from src.models.registry import ModelRegistry

        model = ModelRegistry().build(name)
        path = Path(inspect.getfile(model.__class__)).resolve()
        try:
            return str(path.relative_to(_repo_root()))
        except ValueError:
            return str(path)
    except Exception:
        generated = _repo_root() / "src" / "models" / "generated" / f"{name}.py"
        library = _repo_root() / "src" / "models"
        if generated.exists():
            return str(generated.relative_to(_repo_root()))
        for cand in library.glob("*.py"):
            if name.replace("-", "_") in cand.read_text(encoding="utf-8"):
                return str(cand.relative_to(_repo_root()))
        return ""


def _selected_row(metrics: list[dict[str, Any]] | None, chosen: str) -> dict[str, Any]:
    for row in metrics or []:
        if row.get("error"):
            continue
        if str(row.get("model") or "") == chosen:
            return row
    return {}


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

    chosen = str(selected or (contrib.get("models") or [""])[0] or "")
    row = _selected_row(metrics, chosen)
    script = model_script_path(chosen)
    used = [m for m in (contrib.get("methods") or []) if m in {"occlusion", "gradient", "permutation", "shap", "coefficient"}]
    if (root / "contribution" / "contribution_pairs.csv").exists():
        try:
            import pandas as pd

            used = sorted(set(pd.read_csv(root / "contribution" / "contribution_pairs.csv", usecols=["method"])["method"].astype(str)))
        except Exception:
            used = [m for m in (contrib.get("methods") or []) if m != "coefficient"]
    pretrained = ((prior or {}).get("pretrained") or {})
    prior_line = ""
    if prior:
        prior_line = (
            f"先验：{(prior or {}).get('n_edges', '')} 条边，"
            f"来源 {','.join((prior or {}).get('sources') or [])}，"
            f"嵌入 `{pretrained.get('method') or ''}`。\n"
        )

    text = f"""# 给 BioMaster 的交接说明

只解释**最终选定模型** `{chosen}`。请不要分析库里其他模型。置信度、新颖性和最终对排序由 BioMaster 完成。

任务：同一时刻蛋白质 + 代谢物 → 下一时刻两个组学（`last_interval` / `{contrib.get("direction") or "multimodal"}` / 真滞后）。

- 模型名：`{chosen}`
- 架构脚本：`{script or "（未找到）"}`
- fold1 验证中位 PCC：`{_fmt_pcc(row.get("val_median_pcc"))}`
- 测试中位 PCC：`{_fmt_pcc(row.get("test_median_pcc"))}`（蛋白 `{_fmt_pcc(row.get("test_protein_median_pcc"))}`，代谢 `{_fmt_pcc(row.get("test_metabolite_median_pcc"))}`）
{prior_line}
请先读架构脚本，再读下面两张表：

| 文件 | 说明 |
| --- | --- |
| `{script or "src/models/generated/<model>.py"}` | 选定模型的实现（类、向量场、积分、读出头） |
| `contribution/method_scores.csv` | 宽表：一对源–目标一行，列为 {", ".join(used) or "各方法"} |
| `contribution/contribution_pairs.csv` | 长表：同一对拆成每种方法一行 |

分数是该目标内 L1 归一化后的相对重要性，不是因果。`prior` 列只是网络是否已有该对（1 / 0.5 / 0）。
"""
    dest.write_text(text, encoding="utf-8")
    return str(dest)
