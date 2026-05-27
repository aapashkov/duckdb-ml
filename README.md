# Ml

This repository builds the ML extension against a prebuilt DuckDB runtime in `duckdb_lib/`.

---


## Build steps
First, download the pinned DuckDB runtime and headers:

```sh
./scripts/setup-duckdb.sh --skip-ml-build
```

Then build the extension:

```sh
make
```

On the first configure/build, CMake auto-downloads pinned prebuilt dependencies into `build/prebuilt/`:
- Eigen headers
- XGBoost headers + `libxgboost.so`

Dependency setup is handled by `scripts/setup-prebuilt-deps.sh` during CMake configure.
If you want to prefetch dependencies without building, run `make prebuilt_deps`.

Auto-download is pinned and enabled by default on Linux. On unsupported platforms,
set `ML_PREBUILT_AUTO_DOWNLOAD=OFF` and place prebuilt files under `build/prebuilt/`
using the same folder layout.

Main build artifacts:

```sh
./build/release/extension/ml/ml.duckdb_extension
./build/debug/extension/ml/ml.duckdb_extension
```

To build with Ninja:

```sh
GEN=ninja make
```

No submodule checkout is required; `duckdb/`, `extension-ci-tools/`, and third-party dependencies are no longer needed as local git submodules for build/test.

## Running the extension
Load the built extension from DuckDB clients with unsigned extensions enabled:

```python
import duckdb
con = duckdb.connect(':memory:', config={'allow_unsigned_extensions': 'true'})
con.execute("LOAD 'build/release/extension/ml/ml.duckdb_extension'")
```

## ML API
The extension exposes five table functions:

- `ml_fit(model, options, transforms, table)`
- `ml_predict(model, table[, version = <int>])`
- `ml_evaluate(model, table[, version = <int>])`
- `ml_explain(model, table[, version = <int>])`
- `ml_models()`

`model` is a logical model name. `ml_fit` stores an immutable version in a SQLite registry and returns one row:

- `model_name` (VARCHAR)
- `model_version` (BIGINT)
- `model_blob` (BLOB)
- `model_timestamp` (VARCHAR)
- `model_options` (VARCHAR)
- `model_transforms` (VARCHAR)
- `model_table` (VARCHAR)

`ml_predict`, `ml_evaluate`, and `ml_explain` resolve the latest model version by default.
When the optional named `version` argument is provided, `version = 0` also means latest, and `version > 0` resolves that exact stored version.
`ml_models` returns all rows currently persisted in `duckdb_ml_models` with the same seven columns as `ml_fit` output.

### Model Registry Path
The registry database path is resolved in this order:

1. `PROJECT_DB_PATH` environment variable
2. OS default path:
	 - Linux/macOS: `~/.duckdb-ml.db`
	 - Windows: `%APPDATA%\\duckdb-ml\\duckdb-ml.db`, falling back to `%USERPROFILE%\\.duckdb-ml.db`

The extension stores model entries in an extension-owned SQLite table `duckdb_ml_models`.

`model_table` stores the TABLE-argument SQL expression when available.
When parser internals are not exposed by the build headers, the extension stores a fallback JSON payload with source column names/types and a note.

### Example Workflow

```sql
SELECT *
FROM ml_fit(
	'pca_california',
	{'model_type':'pca', 'num_components':2, 'whiten':false},
	NULL,
	(SELECT * FROM california_train)
);

SELECT * FROM ml_predict('pca_california', (SELECT * FROM california_test));
SELECT * FROM ml_evaluate('pca_california', (SELECT * FROM california_test));
SELECT * FROM ml_predict('pca_california', (SELECT * FROM california_test), version = 1);
SELECT * FROM ml_models();
```

## Running the tests
The repository provides SQL and Python parity tests using a local virtualenv (`.venv-test`) that is managed by the Makefile.

Run all tests:

```sh
make test
```

Run debug-profile tests:

```sh
make test_debug
```

Run SQL-only or Python-only suites:

```sh
make test_sql_release
make test_python_release
```

### Installing the deployed binaries
To install your extension binaries from S3, you will need to do two things. Firstly, DuckDB should be launched with the
`allow_unsigned_extensions` option set to true. How to set this will depend on the client you're using. Some examples:

CLI:
```shell
duckdb -unsigned
```

Python:
```python
con = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
```

NodeJS:
```js
db = new duckdb.Database(':memory:', {"allow_unsigned_extensions": "true"});
```

Secondly, you will need to set the repository endpoint in DuckDB to the HTTP url of your bucket + version of the extension
you want to install. To do this run the following SQL query in DuckDB:
```sql
SET custom_extension_repository='bucket.s3.eu-west-1.amazonaws.com/<your_extension_name>/latest';
```
Note that the `/latest` path will allow you to install the latest extension version available for your current version of
DuckDB. To specify a specific version, you can pass the version instead.

After running these steps, you can install and load your extension using the regular INSTALL/LOAD commands in DuckDB:
```sql
INSTALL ml;
LOAD ml;
```

## CLion
Open this repository root (`CMakeLists.txt`) directly in CLion.
Configure `DUCKDB_LIB_DIR` to point to `duckdb_lib` in your CMake profile.
