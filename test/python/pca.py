import os
import tempfile
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


def _fit_sql(model_name: str) -> str:
    return (
        "SELECT model_name, model_version, model_timestamp "
        "FROM ml_fit("
        f"'{model_name}', "
        "{'model_type':'pca', 'num_components':2, 'whiten':false}, NULL, "
        "(SELECT * FROM california_train))"
    )


def _feature_query(table_name: str) -> str:
    return f"SELECT {', '.join(FEATURE_COLUMNS)} FROM {table_name}"


class _RunningMoments:
    def __init__(self, dimensions: int):
        self.n = 0
        self.mean = np.zeros((dimensions,), dtype=np.float64)
        self.m2 = np.zeros((dimensions,), dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        for row in batch:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def total_sample_variance(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sum(self.m2 / (self.n - 1)))


def test_incremental_pca_parity_with_sklearn():
    sklearn_decomp = pytest.importorskip("sklearn.decomposition")
    IncrementalPCA = sklearn_decomp.IncrementalPCA
    model_name = "pca_python_parity"

    root = _repo_root()
    ext_path = root / "build" / "release" / "extension" / "ml" / "ml.duckdb_extension"
    dataset_path = root / "test" / "datasets.duckdb"

    prev_project_db_path = os.environ.get("PROJECT_DB_PATH")
    temp_db_dir = tempfile.TemporaryDirectory(prefix="duckdb_ml_pca_registry_")
    os.environ["PROJECT_DB_PATH"] = str(Path(temp_db_dir.name) / "registry.db")

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    try:
        con.execute(f"LOAD '{ext_path.as_posix()}'")
        con.execute("PRAGMA threads=1")
        con.execute(f"ATTACH '{dataset_path.as_posix()}' AS ds")
        con.execute("CREATE TEMP TABLE california AS SELECT * FROM ds.california")
        con.execute("CREATE TEMP TABLE california_train AS SELECT * FROM california LIMIT 16500")
        con.execute("CREATE TEMP TABLE california_test AS SELECT * FROM california LIMIT 4140 OFFSET 16500")

        fit_row = con.execute(_fit_sql(model_name)).fetchone()
        assert fit_row is not None
        assert fit_row[0] == model_name
        assert fit_row[1] >= 1
        assert fit_row[2] is not None
        model_version = int(fit_row[1])

        pred_schema = con.execute(
            "DESCRIBE SELECT * FROM ml_predict(?, (SELECT * FROM california_test))",
            [model_name],
        ).fetchall()
        assert [row[0] for row in pred_schema] == [
            "principal_component_1",
            "principal_component_2",
        ]

        con.execute(
            "CREATE TEMP TABLE pca_test_with_id AS "
            "SELECT row_number() OVER () AS rid, * FROM california_test"
        )
        con.execute(
            "CREATE TEMP TABLE pca_ext_pred_with_id AS "
            "SELECT row_number() OVER () AS rid, * "
            "FROM ml_predict(?, (SELECT * FROM california_test))",
            [model_name],
        )

        pred_count = con.execute("SELECT COUNT(*) FROM pca_ext_pred_with_id").fetchone()[0]
        assert pred_count == 4140

        ipca = IncrementalPCA(n_components=2, whiten=False)
        train_cursor = con.execute(_feature_query("california_train"))
        while True:
            chunk_df = train_cursor.fetch_df_chunk()
            if chunk_df.empty:
                break
            ipca.partial_fit(chunk_df.to_numpy(dtype=np.float64, copy=False))

        joined_query = (
            "SELECT "
            + ", ".join([f"t.{col}" for col in FEATURE_COLUMNS])
            + ", p.principal_component_1, p.principal_component_2 "
            "FROM pca_test_with_id t "
            "JOIN pca_ext_pred_with_id p USING (rid) "
            "ORDER BY rid"
        )

        dot_sum = np.zeros((2,), dtype=np.float64)
        pass1_cursor = con.execute(joined_query)
        while True:
            chunk_df = pass1_cursor.fetch_df_chunk()
            if chunk_df.empty:
                break
            x_chunk = chunk_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
            ext_chunk = chunk_df[["principal_component_1", "principal_component_2"]].to_numpy(
                dtype=np.float64, copy=False
            )
            sk_chunk = ipca.transform(x_chunk)
            dot_sum += np.sum(ext_chunk * sk_chunk, axis=0)

        signs = np.where(dot_sum < 0.0, -1.0, 1.0)

        original_moments = _RunningMoments(len(FEATURE_COLUMNS))
        projected_moments = _RunningMoments(2)
        total_rows = 0
        pass2_cursor = con.execute(joined_query)
        while True:
            chunk_df = pass2_cursor.fetch_df_chunk()
            if chunk_df.empty:
                break
            x_chunk = chunk_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
            ext_chunk = chunk_df[["principal_component_1", "principal_component_2"]].to_numpy(
                dtype=np.float64, copy=False
            )
            sk_chunk = ipca.transform(x_chunk)
            aligned_ext = ext_chunk * signs
            assert np.allclose(aligned_ext, sk_chunk, rtol=1e-4, atol=1e-4)
            original_moments.update(x_chunk)
            projected_moments.update(sk_chunk)
            total_rows += x_chunk.shape[0]

        assert total_rows == 4140

        ext_train_ratio = con.execute(
            "SELECT total_explained_variance_ratio "
            f"FROM ml_evaluate(?, (SELECT * FROM california_train), version = {model_version})",
            [model_name],
        ).fetchone()[0]
        ext_test_ratio = con.execute(
            "SELECT total_explained_variance_ratio "
            "FROM ml_evaluate(?, (SELECT * FROM california_test))",
            [model_name],
        ).fetchone()[0]

        sk_train_ratio = float(np.sum(ipca.explained_variance_ratio_))
        total_var = original_moments.total_sample_variance()
        explained_var = projected_moments.total_sample_variance()
        sk_test_ratio = float(explained_var / total_var) if total_var > 0.0 else 0.0

        assert np.isclose(ext_train_ratio, sk_train_ratio, rtol=1e-4, atol=1e-4)
        assert np.isclose(ext_test_ratio, sk_test_ratio, rtol=1e-4, atol=1e-4)
    finally:
        con.close()
        if prev_project_db_path is None:
            os.environ.pop("PROJECT_DB_PATH", None)
        else:
            os.environ["PROJECT_DB_PATH"] = prev_project_db_path
        temp_db_dir.cleanup()
