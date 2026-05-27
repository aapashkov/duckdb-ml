# Testing this extension
This directory contains SQL tests in `test/sql` and Python parity tests in `test/python`.

The root Makefile builds the extension and runs tests with a local virtualenv (`.venv-test`).

Run all tests:

```bash
make test
```

Run debug-profile tests:

```bash
make test_debug
```

Run SQL-only:

```bash
make test_sql_release
make test_sql_debug
```

Run Python-only:

```bash
make test_python_release
make test_python_debug
```