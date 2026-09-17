# processed_data contract

`scripts/validate_processed_data.py` 在 MEGAODE 启动前强制检查本合约。验证失败或状态为 `NOT MODELING READY` 时，**不得启动 MEGAODE**。

## Required files

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

## sample_metadata.csv

Required columns:

- `sample_id` — unique
- `subject_id`
- `time` — numeric and sortable
- `time_unit`
- `condition`
- `batch`

Longitudinal structure: at least one `subject_id` with ≥2 timepoints, or ≥2 distinct times.

## Matrices

- First column is `sample_id`.
- Feature names have no duplicates.
- Values convert to numeric.
- No ±Inf.
- Every matrix `sample_id` exists in metadata.
- Proteomics and metabolomics share at least one paired `sample_id`.

## Annotations

- `protein_annotations.csv` and `metabolite_annotations.csv` must exist.
- Keep original feature IDs. Add UniProt / gene symbol or HMDB / KEGG / ChEBI when available.

## dataset_manifest.yaml

Counts must match the real files:

- `n_samples`
- `n_proteins`
- `n_metabolites`
- `n_paired_samples`

## preprocessing_report.md

The last non-empty line must be exactly one of:

```text
MODELING READY
MODELING READY WITH WARNINGS
NOT MODELING READY
```

## Validator JSON

Success:

```json
{"ok": true, "modeling_status": "MODELING READY WITH WARNINGS", "dataset_name": "...", "warnings": []}
```

Failure:

```json
{"ok": false, "errors": []}
```
