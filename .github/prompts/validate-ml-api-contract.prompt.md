---
name: Validate ML API Contract
description: "Validate that ml_fit, ml_predict, ml_evaluate, ml_explain, and ml_models match the repository MVP contract and list concrete gaps with file references and test recommendations."
argument-hint: "Scope to validate (model type, file paths, or full repo)"
agent: "agent"
---

Validate the MVP SQL API contract for this DuckDB ML extension implementation.

Contract to check:
- `ml_fit(model STRING, options STRUCT, transforms STRUCT DEFAULT NULL, table TABLE)` returns exactly one row with `model_name`, `model_version`, `model_blob`, `model_timestamp`, `model_options`, `model_transforms`, and `model_table`, requires `options.model_type`, ignores irrelevant options, applies defaults for required options when omitted.
	For classifiers and regressors, `options.label` defaults to the literal column name `label`; users may set `options.label` or alias the target as `label` in the TABLE query.
- `ml_fit` persists trained model blobs in the SQLite registry; output row schema matches the inserted registry row schema.
- `model_table` stores the parsed TABLE-argument SQL expression string when available.
	Parser normalization means exact original formatting/comments are not guaranteed.
- `ml_predict(model STRING, table TABLE[, version = BIGINT])` is table-valued, preserves input row count, returns prediction columns only, ignores extra unrelated input columns, and errors clearly if a training-time feature column is missing.
- `ml_evaluate(model STRING, table TABLE[, version = BIGINT])` returns one row of recomputed metrics with evaluation columns only, ignores extra unrelated input columns, and errors clearly if a training-time feature column is missing.
- `ml_explain(model STRING, table TABLE[, version = BIGINT])` is table-valued with per-row attributions and clear runtime error for unsupported model types.
- In scoring/evaluation/explain APIs, omitted `version` or `version = 0` means latest stored version.
- `ml_models()` is table-valued with no arguments and returns all rows of `duckdb_ml_models` with the same schema as persisted registry rows.

Output format:
1. Findings ordered by severity with file references.
2. Missing tests and exact test cases to add in `test/sql/` and `test/python/`.
3. Minimal patch plan with affected files.

Requirements:
- Use existing docs in [AGENTS.md](../../AGENTS.md) and linked references.
- Report behavior regressions and schema-mismatch risks.
- Flag any implementation that loads whole datasets into memory instead of using chunked or batched execution.
- Keep suggestions implementation-ready and concise.
- Explicitly validate that both `make test` and `make test_debug` pass.
