from pathlib import Path

import duckdb
import numpy as np
import pytest


FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fit_sql() -> str:
    return (
        "SELECT model "
        "FROM ml_fit("
        "{'model_type':'pca', 'num_components':2, 'whiten':false}, NULL, "
        "(SELECT * FROM california_train))"
    )


def _feature_query(table_name: str) -> str:
    return f"SELECT {', '.join(FEATURE_COLUMNS)} FROM {table_name}"


def _collect_chunks_to_numpy(cursor) -> np.ndarray:
    chunks = []
    while True:
        chunk_df = cursor.fetch_df_chunk()
        if chunk_df.empty:
            break
        chunks.append(chunk_df.to_numpy(dtype=np.float64, copy=False))
    if not chunks:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(chunks)


def test_incremental_pca_parity_with_sklearn():
    sklearn_decomp = pytest.importorskip("sklearn.decomposition")
    IncrementalPCA = sklearn_decomp.IncrementalPCA

    root = _repo_root()
    ext_path = root / "build" / "release" / "extension" / "ml" / "ml.duckdb_extension"
    dataset_path = root / "test" / "datasets.duckdb"

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{ext_path.as_posix()}'")
    con.execute("PRAGMA threads=1")
    con.execute(f"ATTACH '{dataset_path.as_posix()}' AS ds")
    con.execute("CREATE TEMP TABLE california AS SELECT * FROM ds.california")
    con.execute("CREATE TEMP TABLE california_train AS SELECT * FROM california LIMIT 16500")
    con.execute("CREATE TEMP TABLE california_test AS SELECT * FROM california LIMIT 4140 OFFSET 16500")

    model_blob = con.execute(_fit_sql()).fetchone()[0]
    assert model_blob is not None

    ext_cursor = con.execute(
        "SELECT principal_component_1, principal_component_2 "
        "FROM ml_predict(?, (SELECT * FROM california_test))",
        [model_blob],
    )
    ext_pred = _collect_chunks_to_numpy(ext_cursor)
    assert ext_pred.shape[0] == 4140
    assert ext_pred.shape[1] == 2

    ipca = IncrementalPCA(n_components=2, whiten=False)
    train_cursor = con.execute(_feature_query("california_train"))
    while True:
        chunk_df = train_cursor.fetch_df_chunk()
        if chunk_df.empty:
            break
        ipca.partial_fit(chunk_df.to_numpy(dtype=np.float64, copy=False))

    sklearn_preds = []
    test_batches = []
    test_cursor = con.execute(_feature_query("california_test"))
    while True:
        chunk_df = test_cursor.fetch_df_chunk()
        if chunk_df.empty:
            break
        chunk_np = chunk_df.to_numpy(dtype=np.float64, copy=False)
        test_batches.append(chunk_np)
        sklearn_preds.append(ipca.transform(chunk_np))

    sk_pred = np.vstack(sklearn_preds)
    x_test = np.vstack(test_batches)

    for component_idx in range(2):
        if np.dot(ext_pred[:, component_idx], sk_pred[:, component_idx]) < 0:
            ext_pred[:, component_idx] *= -1

    assert np.allclose(ext_pred, sk_pred, rtol=1e-4, atol=1e-4)

    ext_train_ratio = con.execute("SELECT ml_evaluate(?)", [model_blob]).fetchone()[0]
    ext_test_ratio = con.execute(
        "SELECT total_explained_variance_ratio "
        "FROM ml_evaluate(?, (SELECT * FROM california_test))",
        [model_blob],
    ).fetchone()[0]

    sk_train_ratio = float(np.sum(ipca.explained_variance_ratio_))
    centered = x_test - x_test.mean(axis=0, keepdims=True)
    projected = centered @ ipca.components_.T
    sk_test_ratio = float(
        projected.var(axis=0, ddof=1).sum() / centered.var(axis=0, ddof=1).sum()
    )

    assert np.isclose(ext_train_ratio, sk_train_ratio, rtol=1e-4, atol=1e-4)
    assert np.isclose(ext_test_ratio, sk_test_ratio, rtol=1e-4, atol=1e-4)
