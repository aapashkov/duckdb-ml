# AGENTS.md

## Purpose
This repository is a DuckDB out-of-tree extension that will become an ML extension.
Current state is template-like; prioritize building the MVP SQL API and test coverage.

## Read First (Link, Don't Duplicate)
- Project build/test overview: [README.md](README.md)
- Extension tests overview: [test/README.md](test/README.md)
- DuckDB API upgrade workflow: [docs/UPDATING.md](docs/UPDATING.md)

## Immediate Build/Test Commands
- Download pinned DuckDB runtime: `./scripts/setup-duckdb.sh --skip-ml-build`
- Build: `make`
- Faster local rebuilds: `GEN=ninja make`
- Run release tests (SQL + Python): `make test`
- Run debug tests (SQL + Python): `make test_debug`
- Load built extension: `LOAD 'build/release/extension/ml/ml.duckdb_extension';`

## Dependency Sources
- Eigen is fetched by CMake via pinned `FetchContent` URL/hash in `CMakeLists.txt`.
- XGBoost is fetched by CMake via pinned `FetchContent` git commit in `CMakeLists.txt`.
- SHAP is not required by the current build.

## MVP SQL API Contract
Implement and preserve these public SQL entry points and semantics:

1. `ml_fit(model STRING, options STRUCT, transforms STRUCT DEFAULT NULL, table TABLE)`
- Table-valued function returning exactly one row with columns:
  - `model_name STRING`
  - `model_version BIGINT`
  - `model_blob BLOB`
  - `model_timestamp VARCHAR`
  - `model_options VARCHAR`
  - `model_transforms VARCHAR`
  - `model_table VARCHAR`
- Must work on tables, relations, views, and temp tables through the `TABLE` argument.
- The trained model blob is persisted in the SQLite registry and returned in the `model_blob` column.
- `model_table` stores the parsed TABLE-argument SQL expression string when available.
  The parser may normalize formatting/comments, so byte-for-byte original text is not guaranteed.
- `options` is a shared STRUCT across all algorithms.
- `options.model_type` (STRING) is required across all models.
- Ignore irrelevant options for selected model type; use sensible defaults for required options when omitted.
- For classifiers and regressors, `options.label` defaults to the literal column name `label` when omitted.
  Users can either set `options.label` explicitly or alias the target column in the TABLE query (for example `SELECT target AS label, ...`).
- `transforms` is accepted now as a placeholder, even if transformation logic will not be implemented yet.

2. `ml_predict(model STRING, table TABLE[, version = BIGINT])`
- Table-valued function.
- Optional named parameter `version` defaults to `0`, which means latest stored version.
- Output row count must match input row count.
- Output must contain only prediction columns (no passthrough input columns).
- Ignore extra input columns not used by the model.
- Return a clear error if any feature column seen during training is missing in the input table.

3. `ml_evaluate(model STRING, table TABLE[, version = BIGINT])`
- Table-valued function returning one row of recomputed metrics on provided data.
- Optional named parameter `version` defaults to `0`, which means latest stored version.
- Evaluation outputs must contain only evaluation metric columns (no passthrough input columns).
- Ignore extra input columns not used by the model.
- Return a clear error if any feature column seen during training is missing in the input table.

4. `ml_explain(model STRING, table TABLE[, version = BIGINT])`
- Table-valued function returning per-row feature attributions.
- Optional named parameter `version` defaults to `0`, which means latest stored version.
- For unsupported model types, return a clear user-facing error.
- Missing model names must return a clear error that includes the registry path.

5. `ml_models()`
- Table-valued function with no arguments.
- Returns all rows from the SQLite registry table `duckdb_ml_models`.
- Output schema matches the persisted registry row schema and `ml_fit` output columns:
  `model_name`, `model_version`, `model_blob`, `model_timestamp`, `model_options`, `model_transforms`, `model_table`.

## Model Registry Pattern
- `ml_fit` creates immutable versions per model name (`1`, `2`, `3`, ...).
- `ml_predict`, `ml_evaluate`, and `ml_explain` load the latest version by default, or a specific version when `version = N` is provided.
- `ml_models` lists all persisted registry rows for inspection/debugging.
- Persistence is SQLite-backed and resolved from `PROJECT_DB_PATH` or OS defaults:
  - Linux/macOS: `~/.duckdb-ml.db`
  - Windows: `%APPDATA%\\duckdb-ml\\duckdb-ml.db` (fallback `%USERPROFILE%\\.duckdb-ml.db`)
- Extension-owned table name: `duckdb_ml_models`.

Example flow:
- `SELECT * FROM ml_fit('pca_model', {'model_type':'pca', 'num_components':2}, NULL, (SELECT * FROM training_data));`
- `SELECT * FROM ml_predict('pca_model', (SELECT * FROM scoring_data));`
- `SELECT * FROM ml_evaluate('pca_model', (SELECT * FROM scoring_data), version = 1);`

## Implementation Rules (Project-Specific)
- Follow Google's C++ style guide.
- Do not implement ML algorithms from scratch unless explicitly requested.
- Prefer existing C/C++ ML libraries; add dependencies via CMake and/or `third_party/`.
- Do not load full datasets into memory.
- Use DuckDB chunked execution (`DataChunk`) and batched/online processing.
- For every integrated algorithm, expose all `ml_*` interfaces.
  - If an interface is not supported for that algorithm, register it and return a clear error at runtime.

## Code Organization Targets
- Keep source files short (target <200 lines).
- Use `src/functions/` for SQL interface glue (bind/init/execute, input/output handling).
- Use `src/algorithms/` for algorithm implementations and library wrappers.
- Use common C++ files for shared functionality and utilities across functions and algorithms to avoid code duplication.
- Use `src/include/` for headers.
- Place a file header docstring in each source file describing implemented algorithms, referenced libraries, and constraints.

Current template entrypoint is `src/ml_extension.cpp` (registered in [CMakeLists.txt](CMakeLists.txt)).
For MVP alignment, prefer consolidating function registration in `src/ml_extension.c` and update [CMakeLists.txt](CMakeLists.txt) accordingly when that migration is introduced.

## Testing Expectations
- Primary tests: SQLLogicTests in `test/sql/`.
- Add Python parity tests in `test/python/` against reference Python ML behavior.
- Include error-path tests (unsupported model/interface combinations, invalid options, schema mismatch).
- Any feature/change is complete only when both `make test` and `make test_debug` pass.

## Pitfalls
- Do not edit generated artifacts in `build/`, `debug/`, `release/`, or `duckdb_unittest_tempdir/`.
- `duckdb_lib/` is generated/downloaded runtime input; avoid committing large binary churn unless intentionally updating DuckDB version.
- DuckDB C++ internals can change across versions; when bumping versions, follow [docs/UPDATING.md](docs/UPDATING.md).
