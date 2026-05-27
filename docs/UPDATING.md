# Extension updating
This repository pins DuckDB through the download script `scripts/setup-duckdb.sh`.
When updating DuckDB, follow this process:

1. Update `DUCKDB_VERSION` in `scripts/setup-duckdb.sh`.
2. Update release artifact checksums in `scripts/setup-duckdb.sh` for each supported platform archive.
3. Update pinned prebuilt dependency sources in `CMakeLists.txt` when bumping dependencies:
   - `EIGEN_PREBUILT_URL` + `EIGEN_PREBUILT_SHA256`
   - `XGBOOST_SOURCE_URL` + `XGBOOST_SOURCE_SHA256` (header source archive)
   - `XGBOOST_WHEEL_URL` + `XGBOOST_WHEEL_SHA256` (prebuilt shared library wheel)
   - Ensure `scripts/setup-prebuilt-deps.sh` still stages files to `build/prebuilt/` with the expected layout.
4. Re-run:
   - `./scripts/setup-duckdb.sh --skip-ml-build`
   - `make release`
   - `make test`
   - `make test_debug`
5. If tests fail due to DuckDB C++ API changes, adjust extension code under `src/` and re-run tests.

Notes:
- Linux prebuilt auto-download is enabled by default.
- On platforms without a pinned wheel URL, configure with `-DML_PREBUILT_AUTO_DOWNLOAD=OFF` and pre-populate `build/prebuilt/` manually.

# API changes
The extension is built against DuckDB C++ APIs exposed by `duckdb_lib/duckdb.hpp`.
This API surface is not guaranteed to be stable across DuckDB versions.
When bumping `DUCKDB_VERSION`, compile and runtime regressions can occur.

Currently, DuckDB does not (yet) provide a specific change log for these API changes, but it is generally not too hard to figure out what has changed.

For investigating API changes, use:
- DuckDB's [Release Notes](https://github.com/duckdb/duckdb/releases)
- DuckDB's history of [Core extension patches](https://github.com/duckdb/duckdb/commits/main/.github/patches/extensions)
- The git history of the relevant C++ Header file of the API that has changed