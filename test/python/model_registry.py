import os
import sqlite3
import tempfile
from pathlib import Path

import duckdb
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
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _connect() -> duckdb.DuckDBPyConnection:
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
    return con


def _set_project_db_path(path: str):
    previous = os.environ.get("PROJECT_DB_PATH")
    os.environ["PROJECT_DB_PATH"] = path
    return previous


def _restore_project_db_path(previous):
    if previous is None:
        os.environ.pop("PROJECT_DB_PATH", None)
    else:
        os.environ["PROJECT_DB_PATH"] = previous


def test_registry_persistence_and_versioning_for_pca():
    with tempfile.TemporaryDirectory(prefix="duckdb_ml_registry_") as temp_dir:
        db_path = str(Path(temp_dir) / "registry.db")
        previous = _set_project_db_path(db_path)
        con = _connect()
        try:
            fit_v1 = con.execute(
                "SELECT * FROM ml_fit('registry_pca', {'model_type':'pca', 'num_components':2, 'whiten':false}, NULL, "
                "(SELECT * FROM california_train))"
            ).fetchone()
            fit_v2 = con.execute(
                "SELECT * FROM ml_fit('registry_pca', {'model_type':'pca', 'num_components':2, 'whiten':false}, NULL, "
                "(SELECT * FROM california_train))"
            ).fetchone()

            assert fit_v1[0] == "registry_pca"
            assert fit_v1[1] == 1
            assert fit_v1[2]
            assert fit_v2[1] == 2

            predict_latest_count = con.execute(
                "SELECT COUNT(*) FROM ml_predict('registry_pca', (SELECT * FROM california_test))"
            ).fetchone()[0]
            predict_v1_count = con.execute(
                "SELECT COUNT(*) FROM ml_predict('registry_pca', (SELECT * FROM california_test), version = 1)"
            ).fetchone()[0]
            evaluate_rows = con.execute(
                "SELECT COUNT(*) FROM ml_evaluate('registry_pca', (SELECT * FROM california_test), version = 2)"
            ).fetchone()[0]

            assert predict_latest_count == 4140
            assert predict_v1_count == 4140
            assert evaluate_rows == 1

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT model, version, length(blob), length(timestamp), length(options), transforms "
                    "FROM duckdb_ml_models WHERE model = ? ORDER BY version",
                    ("registry_pca",),
                ).fetchall()

            assert len(rows) == 2
            assert rows[0][0] == "registry_pca"
            assert rows[0][1] == 1
            assert rows[1][1] == 2
            assert rows[0][2] > 0
            assert rows[0][3] > 0
            assert rows[0][4] > 0
            assert rows[0][5] is None
        finally:
            con.close()
            _restore_project_db_path(previous)


def test_unknown_model_errors_and_boosted_explain():
    with tempfile.TemporaryDirectory(prefix="duckdb_ml_registry_") as temp_dir:
        db_path = str(Path(temp_dir) / "registry.db")
        previous = _set_project_db_path(db_path)
        con = _connect()
        try:
            fit_v1 = con.execute(
                "SELECT * FROM ml_fit('registry_btr', "
                "{'model_type':'boosted_tree_regressor', 'label':'MedHouseVal', 'tree_method':'hist', 'max_iterations':5}, "
                "NULL, (SELECT * FROM california_train))"
            ).fetchone()
            fit_v2 = con.execute(
                "SELECT * FROM ml_fit('registry_btr', "
                "{'model_type':'boosted_tree_regressor', 'label':'MedHouseVal', 'tree_method':'hist', 'max_iterations':5}, "
                "NULL, (SELECT * FROM california_train))"
            ).fetchone()

            assert fit_v1[1] == 1
            assert fit_v2[1] == 2

            explain_latest = con.execute(
                "SELECT COUNT(*) FROM ml_explain('registry_btr', "
                "(SELECT MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude "
                "FROM california_test))"
            ).fetchone()[0]
            explain_v1 = con.execute(
                "SELECT COUNT(*) FROM ml_explain('registry_btr', "
                "(SELECT MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude "
                "FROM california_test), version = 1)"
            ).fetchone()[0]

            assert explain_latest == 4140
            assert explain_v1 == 4140

            with pytest.raises(duckdb.Error, match="No trained model named"):
                con.execute("SELECT * FROM ml_predict('unknown_registry_model', (SELECT * FROM california_test))")
            with pytest.raises(duckdb.Error, match="No trained model named"):
                con.execute("SELECT * FROM ml_evaluate('unknown_registry_model', (SELECT * FROM california_test))")
            with pytest.raises(duckdb.Error, match="No trained model named"):
                con.execute("SELECT * FROM ml_explain('unknown_registry_model', "
                            "(SELECT MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude "
                            "FROM california_test))")
        finally:
            con.close()
            _restore_project_db_path(previous)


def test_failed_training_does_not_insert_registry_row():
    with tempfile.TemporaryDirectory(prefix="duckdb_ml_registry_") as temp_dir:
        db_path = str(Path(temp_dir) / "registry.db")
        previous = _set_project_db_path(db_path)
        con = _connect()
        try:
            con.execute("CREATE TEMP TABLE california_empty AS SELECT * FROM california_train WHERE 1=0")
            with pytest.raises(duckdb.Error, match="no valid training rows"):
                con.execute(
                    "SELECT * FROM ml_fit('registry_failed_fit', {'model_type':'pca', 'num_components':2, 'whiten':false}, "
                    "NULL, (SELECT * FROM california_empty))"
                )

            with sqlite3.connect(db_path) as conn:
                table_exists = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='duckdb_ml_models'"
                ).fetchone()[0]
                if table_exists:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM duckdb_ml_models WHERE model = ?", ("registry_failed_fit",)
                    ).fetchone()[0]
                else:
                    count = 0
            assert count == 0
        finally:
            con.close()
            _restore_project_db_path(previous)


def test_incompatible_registry_schema_fails_with_actionable_error():
    with tempfile.TemporaryDirectory(prefix="duckdb_ml_registry_") as temp_dir:
        db_path = str(Path(temp_dir) / "registry.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE duckdb_ml_models (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()

        previous = _set_project_db_path(db_path)
        con = _connect()
        try:
            with pytest.raises(duckdb.Error, match="incompatible schema|unexpected column"):
                con.execute(
                    "SELECT * FROM ml_fit('registry_incompatible', {'model_type':'pca', 'num_components':2, 'whiten':false}, "
                    "NULL, (SELECT * FROM california_train))"
                )
        finally:
            con.close()
            _restore_project_db_path(previous)


@pytest.mark.skipif(os.name == "nt", reason="Linux/macOS fallback path behavior uses HOME")
def test_default_registry_path_uses_home_when_project_db_path_unset():
    with tempfile.TemporaryDirectory(prefix="duckdb_ml_home_") as temp_home:
        previous_project = os.environ.pop("PROJECT_DB_PATH", None)
        previous_home = os.environ.get("HOME")
        os.environ["HOME"] = temp_home
        con = _connect()
        try:
            con.execute(
                "SELECT * FROM ml_fit('registry_home_default', {'model_type':'pca', 'num_components':2, 'whiten':false}, "
                "NULL, (SELECT * FROM california_train))"
            ).fetchone()
            expected_db = Path(temp_home) / ".duckdb-ml.db"
            assert expected_db.exists()
        finally:
            con.close()
            if previous_project is not None:
                os.environ["PROJECT_DB_PATH"] = previous_project
            if previous_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = previous_home
