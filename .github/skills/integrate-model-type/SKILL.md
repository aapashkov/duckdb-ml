---
name: integrate-model-type
description: "Integrate a new ML model type into this DuckDB extension. Use when adding or modifying algorithm support and you must wire ml_fit, ml_predict, ml_evaluate, and ml_explain consistently with clear unsupported-interface errors and tests."
argument-hint: "Model type and backing library to integrate"
---

# Integrate Model Type

## Goal
Add or update one model type implementation while preserving the MVP SQL contract for all public `ml_*` entry points.

## API Shape To Preserve
- `ml_fit(model STRING, options STRUCT, transforms STRUCT DEFAULT NULL, table TABLE)` returns one row with `model_name`, `model_version`, `model_blob`, `model_timestamp`, `model_options`, `model_transforms`, and `model_table`.
- `ml_predict(model STRING, table TABLE[, version = BIGINT])` returns only prediction columns (no passthrough input columns).
- `ml_evaluate(model STRING, table TABLE[, version = BIGINT])` returns only evaluation metric columns.
- `ml_explain(model STRING, table TABLE[, version = BIGINT])` returns attributions or a clear unsupported error.
- `ml_models()` returns all rows from `duckdb_ml_models` with registry/fit schema columns.
- `ml_fit` persists model blobs in the SQLite registry and returns the persisted blob in the output row.
- `model_table` stores the parsed TABLE-argument SQL expression string when available.
	Parser normalization means exact original formatting/comments are not guaranteed.
- In scoring/evaluation/explain APIs, omitted `version` or `version = 0` means latest stored version.
- For classifiers and regressors, treat omitted `options.label` as defaulting to the literal column name `label`; callers may override `options.label` or alias target columns as `label`.

## Inputs To Collect First
- Model type name for `options.model_type`
- Backing C/C++ ML library and version
- Supported interfaces among `ml_fit`, `ml_predict`, `ml_evaluate`, `ml_explain` (plus visibility in `ml_models`)
- Required options and defaults
- Metrics expected from `ml_evaluate`
- Any training/inference path must avoid loading the full dataset into memory; use chunked or batched execution.

## Procedure
1. Confirm repository constraints in [AGENTS.md](../../../AGENTS.md) and linked docs.
2. Add or update algorithm wrappers in `src/algorithms/` using chunked and batched execution.
3. Add or update SQL glue code in `src/functions/` for bind/init/execute and option parsing.
4. Register all four `ml_*` entry points for the model type.
5. For any unsupported interface, return a clear runtime user-facing error.
6. Ensure table-valued outputs expose only prediction/evaluation columns and ignore extra unrelated input columns.
7. Add SQLLogicTests under `test/sql/` covering happy paths and error paths.
8. Add or update parity tests under `test/python/`.
9. Run both `make test` and `make test_debug`.

## Completion Checklist
- `options.model_type` is required and validated.
- Irrelevant options are ignored safely.
- Required options have sensible defaults where omitted.
- `ml_fit` returns one-row table output with `model_name`, `model_version`, `model_blob`, `model_timestamp`, `model_options`, `model_transforms`, and `model_table`.
- `ml_predict` output row count equals input row count.
- `ml_predict` output contains only prediction columns.
- `ml_predict` errors if any training feature column is missing from inference input.
- `ml_predict`, `ml_evaluate`, and `ml_explain` support optional named `version`; `0` resolves latest.
- Missing model names return a clear error that includes registry path information.
- `ml_evaluate(model, table[, version])` recomputes metrics.
- `ml_evaluate(model, table[, version])` output contains only evaluation metric columns.
- `ml_evaluate(model, table[, version])` errors if any training feature column is missing from evaluation input.
- `ml_explain` returns attributions or clear unsupported error.
- SQL and Python tests include negative and mismatch-schema cases.
- Both release and debug test runs pass.
