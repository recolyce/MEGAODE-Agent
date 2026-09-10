# 给 BioMaster 的交接说明

只解释**最终选定模型** `embedding_field_ode`。请不要分析库里其他模型。置信度、新颖性和最终对排序由 BioMaster 完成。

任务：同一时刻蛋白质 + 代谢物 → 下一时刻两个组学（`last_interval` / `multimodal` / 真滞后）。

- 模型名：`embedding_field_ode`
- 架构脚本：`src/models/generated/embedding_field_ode.py`
- fold1 验证中位 PCC：`0.235`
- 测试中位 PCC：`0.606`（蛋白 `0.524`，代谢 `0.689`）
先验：8385 条边，来源 name_rule,kegg_reactome,string,stitch,pretrained，嵌入 `esm2_t12_35M+unimol2_84m`。

请先读架构脚本，再读下面两张表：

| 文件 | 说明 |
| --- | --- |
| `src/models/generated/embedding_field_ode.py` | 选定模型的实现（类、向量场、积分、读出头） |
| `contribution/method_scores.csv` | 宽表：一对源–目标一行，列为 gradient, occlusion, permutation |
| `contribution/contribution_pairs.csv` | 长表：同一对拆成每种方法一行 |

分数是该目标内 L1 归一化后的相对重要性，不是因果。`prior` 列只是网络是否已有该对（1 / 0.5 / 0）。
