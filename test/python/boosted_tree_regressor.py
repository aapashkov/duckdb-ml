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


def _load_numpy_data(con: duckdb.DuckDBPyConnection):
    train_query = (
        f"SELECT {', '.join(FEATURE_COLUMNS)}, {LABEL_COLUMN} "
        "FROM california_train"
    )
    test_query = f"SELECT {', '.join(FEATURE_COLUMNS)}, {LABEL_COLUMN} FROM california_test"

    train_df = con.execute(train_query).fetchdf()
    test_df = con.execute(test_query).fetchdf()

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df[LABEL_COLUMN].to_numpy(dtype=np.float32, copy=False)
    x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    y_test = test_df[LABEL_COLUMN].to_numpy(dtype=np.float32, copy=False)
    return x_train, y_train, x_test, y_test


def _make_data_iter(xgb_module, x_data, y_data, batch_size, cache_prefix):
    class NumpyDataIter(xgb_module.DataIter):
        def __init__(self):
            self._x_data = x_data
            self._y_data = y_data
            self._batch_size = batch_size
            self._offset = 0
            super().__init__(cache_prefix=cache_prefix)

        def reset(self):
            self._offset = 0

        def next(self, input_data):
            if self._offset >= self._x_data.shape[0]:
                return False
            end = min(self._offset + self._batch_size, self._x_data.shape[0])
            input_data(data=self._x_data[self._offset:end], label=self._y_data[self._offset:end])
            self._offset = end
            return True

    return NumpyDataIter()


def _fit_extension_model(con: duckdb.DuckDBPyConnection, tree_method: str):
    sql = (
        "SELECT model FROM ml_fit("
        f"{{'model_type':'boosted_tree_regressor','label':'{LABEL_COLUMN}','tree_method':'{tree_method}','max_iterations':10}}, "
        "NULL, (SELECT * FROM california_train))"
    )
    return con.execute(sql).fetchone()[0]


def _train_python_xgboost(tree_method: str, x_train, y_train):
    xgb = pytest.importorskip("xgboost")

    cache_dir = tempfile.mkdtemp(prefix="xgb_extmem_")
    try:
        iterator = _make_data_iter(xgb, x_train, y_train, 2048, os.path.join(cache_dir, "cache"))
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_true - y_pred
    abs_err = np.abs(err)
    y_true_safe = np.clip(y_true, 0.0, None)
    y_pred_safe = np.clip(y_pred, 0.0, None)
    log_err = np.log1p(y_true_safe) - np.log1p(y_pred_safe)

    y_var = np.var(y_true)
    err_var = np.var(err)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return {
        "mean_absolute_error": float(np.mean(abs_err)),
        "mean_squared_error": float(np.mean(err**2)),
        "mean_squared_log_error": float(np.mean(log_err**2)),
        "median_absolute_error": float(np.median(abs_err)),
        "r2_score": float(1.0 - (np.sum(err**2) / ss_tot) if ss_tot > 0 else 0.0),
        "explained_variance": float(1.0 - (err_var / y_var) if y_var > 0 else 0.0),
    }


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_boosted_tree_regressor_parity_with_python_xgboost(tree_method: str):
    con = _connect()
    try:
        x_train, y_train, x_test, y_test = _load_numpy_data(con)

        model_blob = _fit_extension_model(con, tree_method)
        assert model_blob is not None

        ext_pred = con.execute(
            f"SELECT {LABEL_COLUMN} FROM ml_predict(?, (SELECT * FROM california_test))",
            [model_blob],
        ).fetchnumpy()[LABEL_COLUMN].astype(np.float32)

        booster = _train_python_xgboost(tree_method, x_train, y_train)
        xgb = pytest.importorskip("xgboost")
        dtest = xgb.DMatrix(x_test, feature_names=FEATURE_COLUMNS)
        py_pred = booster.predict(dtest, iteration_range=(0, 10)).astype(np.float32)

        corr = float(np.corrcoef(ext_pred, py_pred)[0, 1])
        assert corr > 0.98

        explain_cols = ["baseline_prediction_value"] + [f"{name}_attribution" for name in FEATURE_COLUMNS]
        ext_explain = con.execute(
            "SELECT "
            + ", ".join(explain_cols)
            + " FROM ml_explain(?, (SELECT "
            + ", ".join(FEATURE_COLUMNS)
            + " FROM california_test LIMIT 256))",
            [model_blob],
        ).fetchdf().to_numpy(dtype=np.float64, copy=False)

        ext_subset_pred = con.execute(
            f"SELECT {LABEL_COLUMN} FROM ml_predict(?, (SELECT * FROM california_test LIMIT 256))",
            [model_blob],
        ).fetchnumpy()[LABEL_COLUMN].astype(np.float64)
        assert np.allclose(ext_explain[:, 0] + ext_explain[:, 1:].sum(axis=1), ext_subset_pred, rtol=1e-4, atol=1e-4)

        eval_row = con.execute(
            "SELECT * FROM ml_evaluate(?, (SELECT * FROM california_test))",
            [model_blob],
        ).fetchone()
        ext_metrics = {
            "mean_absolute_error": float(eval_row[0]),
            "mean_squared_error": float(eval_row[1]),
            "mean_squared_log_error": float(eval_row[2]),
            "median_absolute_error": float(eval_row[3]),
            "r2_score": float(eval_row[4]),
            "explained_variance": float(eval_row[5]),
        }

        ext_metrics_expected = _compute_metrics(y_test.astype(np.float64), ext_pred.astype(np.float64))
        for key, value in ext_metrics_expected.items():
            assert np.isclose(ext_metrics[key], value, rtol=5e-3, atol=5e-3)
    finally:
        con.close()
