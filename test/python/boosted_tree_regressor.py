import os
import shutil
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
]
LABEL_COLUMN = "MedHouseVal"


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


def _make_data_iter(xgb_module, con: duckdb.DuckDBPyConnection, table_name: str, cache_prefix: str):
    class NumpyDataIter(xgb_module.DataIter):
        def __init__(self):
            self._con = con
            self._query = (
                f"SELECT {', '.join(FEATURE_COLUMNS)}, {LABEL_COLUMN} "
                f"FROM {table_name}"
            )
            self._cursor = None
            super().__init__(cache_prefix=cache_prefix)

        def reset(self):
            self._cursor = self._con.execute(self._query)

        def next(self, input_data):
            chunk_df = self._cursor.fetch_df_chunk()
            if chunk_df.empty:
                return False
            x_chunk = chunk_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
            y_chunk = chunk_df[LABEL_COLUMN].to_numpy(dtype=np.float32, copy=False)
            input_data(data=x_chunk, label=y_chunk)
            return True

    return NumpyDataIter()


def _fit_extension_model(con: duckdb.DuckDBPyConnection, tree_method: str):
    sql = (
        "SELECT model_name, model_version, model_timestamp FROM ml_fit("
        "'btr_python_parity', "
        f"{{'model_type':'boosted_tree_regressor','label':'{LABEL_COLUMN}','tree_method':'{tree_method}','max_iterations':10}}, "
        "NULL, (SELECT * FROM california_train))"
    )
    return con.execute(sql).fetchone()


def _train_python_xgboost(con: duckdb.DuckDBPyConnection, tree_method: str):
    xgb = pytest.importorskip("xgboost")

    cache_dir = tempfile.mkdtemp(prefix="xgb_extmem_")
    try:
        iterator = _make_data_iter(xgb, con, "california_train", os.path.join(cache_dir, "cache"))
        if tree_method == "hist":
            dtrain = xgb.ExtMemQuantileDMatrix(iterator)
        else:
            dtrain = xgb.DMatrix(iterator)

        params = {
            "seed": 0,
            "verbosity": 0,
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "tree_method": tree_method,
            "num_parallel_tree": 1,
            "max_depth": 6,
            "eta": 0.3,
            "alpha": 0.0,
            "lambda": 1.0,
            "min_child_weight": 1,
            "colsample_bytree": 1.0,
            "colsample_bylevel": 1.0,
            "colsample_bynode": 1.0,
            "gamma": 0.0,
            "subsample": 1.0,
        }
        booster = xgb.train(params, dtrain, num_boost_round=10)
        return booster
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


class _CorrelationAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.n += x.size
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_x2 += float(np.sum(x * x))
        self.sum_y2 += float(np.sum(y * y))
        self.sum_xy += float(np.sum(x * y))

    def finalize(self) -> float:
        num = (self.n * self.sum_xy) - (self.sum_x * self.sum_y)
        den_x = (self.n * self.sum_x2) - (self.sum_x * self.sum_x)
        den_y = (self.n * self.sum_y2) - (self.sum_y * self.sum_y)
        den = float(np.sqrt(max(den_x, 0.0) * max(den_y, 0.0)))
        if den == 0.0:
            return 0.0
        return num / den


class _RegressionAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.sum_sq_log_error = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_err = 0.0
        self.sum_err2 = 0.0
        self.abs_errors = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        err = y_true - y_pred
        abs_err = np.abs(err)
        y_true_safe = np.clip(y_true, 0.0, None)
        y_pred_safe = np.clip(y_pred, 0.0, None)
        log_err = np.log1p(y_true_safe) - np.log1p(y_pred_safe)

        self.n += y_true.size
        self.sum_abs_error += float(np.sum(abs_err))
        self.sum_sq_error += float(np.sum(err**2))
        self.sum_sq_log_error += float(np.sum(log_err**2))
        self.sum_y += float(np.sum(y_true))
        self.sum_y2 += float(np.sum(y_true * y_true))
        self.sum_err += float(np.sum(err))
        self.sum_err2 += float(np.sum(err * err))
        self.abs_errors.extend(abs_err.tolist())

    def finalize(self):
        if self.n == 0:
            return {
                "mean_absolute_error": 0.0,
                "mean_squared_error": 0.0,
                "mean_squared_log_error": 0.0,
                "median_absolute_error": 0.0,
                "r2_score": 0.0,
                "explained_variance": 0.0,
            }

        n = float(self.n)
        y_mean = self.sum_y / n
        ss_tot = max(0.0, self.sum_y2 - n * y_mean * y_mean)
        y_var = max(0.0, (self.sum_y2 / n) - (y_mean * y_mean))
        err_mean = self.sum_err / n
        err_var = max(0.0, (self.sum_err2 / n) - (err_mean * err_mean))

        return {
            "mean_absolute_error": self.sum_abs_error / n,
            "mean_squared_error": self.sum_sq_error / n,
            "mean_squared_log_error": self.sum_sq_log_error / n,
            "median_absolute_error": float(np.median(np.asarray(self.abs_errors, dtype=np.float64))),
            "r2_score": float(1.0 - (self.sum_sq_error / ss_tot) if ss_tot > 0 else 0.0),
            "explained_variance": float(1.0 - (err_var / y_var) if y_var > 0 else 0.0),
        }


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_boosted_tree_regressor_parity_with_python_xgboost(tree_method: str):
    prev_project_db_path = os.environ.get("PROJECT_DB_PATH")
    temp_db_dir = tempfile.TemporaryDirectory(prefix="duckdb_ml_btr_registry_")
    os.environ["PROJECT_DB_PATH"] = str(Path(temp_db_dir.name) / "registry.db")

    con = _connect()
    try:
        fit_row = _fit_extension_model(con, tree_method)
        assert fit_row is not None
        assert fit_row[0] == "btr_python_parity"
        assert fit_row[1] >= 1
        assert fit_row[2] is not None
        model_name = str(fit_row[0])
        model_version = int(fit_row[1])

        con.execute(
            "CREATE TEMP TABLE btr_test_with_id AS "
            "SELECT row_number() OVER () AS rid, * FROM california_test"
        )
        con.execute(
            "CREATE TEMP TABLE btr_ext_pred_with_id AS "
            "SELECT row_number() OVER () AS rid, * "
            "FROM ml_predict(?, (SELECT * FROM california_test))",
            [model_name],
        )

        booster = _train_python_xgboost(con, tree_method)
        xgb = pytest.importorskip("xgboost")

        corr_acc = _CorrelationAccumulator()
        metrics_acc = _RegressionAccumulator()
        comparison_query = (
            "SELECT "
            + ", ".join([f"t.{feature}" for feature in FEATURE_COLUMNS])
            + f", t.{LABEL_COLUMN} AS label, p.{LABEL_COLUMN} AS ext_pred "
            "FROM btr_test_with_id t "
            "JOIN btr_ext_pred_with_id p USING (rid) "
            "ORDER BY rid"
        )
        comparison_cursor = con.execute(comparison_query)
        total_rows = 0
        while True:
            chunk_df = comparison_cursor.fetch_df_chunk()
            if chunk_df.empty:
                break
            x_chunk = chunk_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
            y_chunk = chunk_df["label"].to_numpy(dtype=np.float64, copy=False)
            ext_chunk = chunk_df["ext_pred"].to_numpy(dtype=np.float32, copy=False)
            py_chunk = booster.predict(
                xgb.DMatrix(x_chunk, feature_names=FEATURE_COLUMNS),
                iteration_range=(0, 10),
            ).astype(np.float32)
            corr_acc.update(ext_chunk, py_chunk)
            metrics_acc.update(y_chunk, ext_chunk.astype(np.float64))
            total_rows += x_chunk.shape[0]

        assert total_rows == 4140
        corr = corr_acc.finalize()
        assert corr > 0.98

        explain_cols = ["baseline_prediction_value"] + [f"{name}_attribution" for name in FEATURE_COLUMNS]
        ext_explain = con.execute(
            "SELECT "
            + ", ".join(explain_cols)
            + " FROM ml_explain(?, (SELECT "
            + ", ".join(FEATURE_COLUMNS)
            + " FROM california_test LIMIT 256))",
            [model_name],
        ).fetchdf().to_numpy(dtype=np.float64, copy=False)

        ext_subset_pred = con.execute(
            f"SELECT {LABEL_COLUMN} FROM ml_predict(?, (SELECT * FROM california_test LIMIT 256))",
            [model_name],
        ).fetchnumpy()[LABEL_COLUMN].astype(np.float64)
        assert np.allclose(ext_explain[:, 0] + ext_explain[:, 1:].sum(axis=1), ext_subset_pred, rtol=1e-4, atol=1e-4)

        eval_row = con.execute(
            f"SELECT * FROM ml_evaluate(?, (SELECT * FROM california_test), version = {model_version})",
            [model_name],
        ).fetchone()
        ext_metrics = {
            "mean_absolute_error": float(eval_row[0]),
            "mean_squared_error": float(eval_row[1]),
            "mean_squared_log_error": float(eval_row[2]),
            "median_absolute_error": float(eval_row[3]),
            "r2_score": float(eval_row[4]),
            "explained_variance": float(eval_row[5]),
        }

        ext_metrics_expected = metrics_acc.finalize()
        for key, value in ext_metrics_expected.items():
            assert np.isclose(ext_metrics[key], value, rtol=5e-3, atol=5e-3)
    finally:
        con.close()
        if prev_project_db_path is None:
            os.environ.pop("PROJECT_DB_PATH", None)
        else:
            os.environ["PROJECT_DB_PATH"] = prev_project_db_path
        temp_db_dir.cleanup()
