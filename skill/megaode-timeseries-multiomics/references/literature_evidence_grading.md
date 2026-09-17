# Literature evidence grading

Grade only after Experimental Priority Score is computed and frozen.

Do not invent papers. Prefer primary studies. Record title, year, journal, and DOI/PMID or a verifiable link.

## Grades

### A — Direct evidence

已有直接实验支持 source-target 关系。

### B — Strong mechanistic evidence

没有该 pair 的直接验证，但存在清楚的 enzyme / pathway / transport / signaling / biochemical mechanism。

### C — Indirect evidence

没有直接机制，但双方与共同 pathway / process / phenotype / tissue / disease 有合理关联。

### D — Little or no prior evidence

在本次定义的检索范围内未找到明显直接或间接证据。

## Important

```text
D != relationship does not exist
D != novel discovery
```

Allowed wording:

- 本次检索范围内未发现
- 尚未充分研究
- 潜在发现
- 值得进一步实验验证

Do not write “首次发现” unless a human expert independently confirms it.

## Unreviewed pairs

Pairs that were not searched:

```text
literature_evidence_level = NA
classification = not_literature_reviewed
```

Never auto-label unreviewed pairs as D.

## Classification after grading

- `known_supported`: high EPS + A/B
- `potential_discovery`: high EPS + C/D + a reasonable biological/mechanistic hypothesis
- `uncertain`: identifier ambiguity, model conflict, literature conflict, or insufficient evidence
- `not_literature_reviewed`: no literature search
