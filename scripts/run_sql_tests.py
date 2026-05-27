#!/usr/bin/env python3
"""Minimal SQLLogic runner for this repository's test/sql files."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import duckdb


def _format_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _format_rows(rows: List[Tuple[object, ...]]) -> List[str]:
    return ["\t".join(_format_value(v) for v in row) for row in rows]


def _consume_sql_block(lines: List[str], index: int) -> Tuple[str, int, bool]:
    sql_lines: List[str] = []
    saw_separator = False
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if stripped == "----":
            saw_separator = True
            index += 1
            break
        if not stripped:
            break
        sql_lines.append(raw.rstrip("\n"))
        index += 1
    return "\n".join(sql_lines).strip(), index, saw_separator


def _consume_expected_block(lines: List[str], index: int) -> Tuple[List[str], int]:
    expected: List[str] = []
    while index < len(lines):
        stripped = lines[index].rstrip("\n")
        if not stripped.strip():
            break
        expected.append(stripped)
        index += 1
    return expected, index


def _run_file(con: duckdb.DuckDBPyConnection, path: Path, extension_path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        if line.startswith("require "):
            required = line.split(maxsplit=1)[1].strip()
            if required != "ml":
                raise AssertionError(f"Unsupported require in {path}: {required}")
            con.execute(f"LOAD '{extension_path.as_posix()}'")
            continue

        if line.startswith("statement "):
            mode = line.split(maxsplit=1)[1].strip()
            sql, i, has_separator = _consume_sql_block(lines, i)
            if not sql:
                raise AssertionError(f"Empty statement block in {path}")

            if mode == "ok":
                try:
                    con.execute(sql)
                except Exception as exc:  # noqa: BLE001
                    raise AssertionError(f"statement ok failed in {path}:\n{sql}\nerror: {exc}") from exc
            elif mode == "error":
                if not has_separator:
                    raise AssertionError(f"statement error missing expected separator in {path}")
                expected, i = _consume_expected_block(lines, i)
                expected_text = "\n".join(expected).strip()
                if not expected_text:
                    raise AssertionError(f"statement error missing expected text in {path}")
                try:
                    con.execute(sql)
                except Exception as exc:  # noqa: BLE001
                    message = str(exc)
                    if expected_text.lower() not in message.lower():
                        raise AssertionError(
                            f"statement error mismatch in {path}:\n"
                            f"expected substring: {expected_text}\n"
                            f"actual error: {message}\nSQL:\n{sql}"
                        ) from exc
                else:
                    raise AssertionError(f"statement error unexpectedly succeeded in {path}:\n{sql}")
            else:
                raise AssertionError(f"Unsupported statement mode in {path}: {mode}")
            continue

        if line.startswith("query "):
            sql, i, has_separator = _consume_sql_block(lines, i)
            if not sql:
                raise AssertionError(f"Empty query block in {path}")
            if not has_separator:
                raise AssertionError(f"query block missing expected separator in {path}")
            expected, i = _consume_expected_block(lines, i)
            expected = [row.strip() for row in expected]
            try:
                rows = con.execute(sql).fetchall()
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f"query execution failed in {path}:\n{sql}\nerror: {exc}") from exc
            actual = [row.strip() for row in _format_rows(rows)]
            if actual != expected:
                raise AssertionError(
                    f"query result mismatch in {path}:\nSQL:\n{sql}\n"
                    f"expected: {expected}\nactual: {actual}"
                )
            continue

        raise AssertionError(f"Unsupported directive in {path}: {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local SQL tests without DuckDB submodule test harness")
    parser.add_argument("--extension", required=True, help="Path to ml.duckdb_extension")
    parser.add_argument("--tests", default="test/sql/*.test", help="Glob for SQL test files")
    args = parser.parse_args()

    extension_path = Path(args.extension).resolve()
    if not extension_path.exists():
        raise SystemExit(f"Extension not found: {extension_path}")

    test_paths = sorted(Path.cwd().glob(args.tests))
    if not test_paths:
        raise SystemExit(f"No SQL tests matched glob: {args.tests}")

    prev_db_path = os.environ.get("PROJECT_DB_PATH")
    temp_dir = tempfile.TemporaryDirectory(prefix="duckdb_ml_sql_tests_")
    os.environ["PROJECT_DB_PATH"] = str(Path(temp_dir.name) / "registry.db")

    try:
        for path in test_paths:
            con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
            try:
                _run_file(con, path, extension_path)
                print(f"PASS {path}")
            finally:
                con.close()
    finally:
        if prev_db_path is None:
            os.environ.pop("PROJECT_DB_PATH", None)
        else:
            os.environ["PROJECT_DB_PATH"] = prev_db_path
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
