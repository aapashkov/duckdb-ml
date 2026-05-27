ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
GEN ?=
DUCKDB_LIB_DIR ?= $(ROOT_DIR)duckdb_lib
DUCKDB_VERSION := $(shell sed -n 's/^DUCKDB_VERSION="\(.*\)"/\1/p' scripts/setup-duckdb.sh)
PYTHON_BIN ?= python3
VENV_DIR ?= .venv-test
VENV_PYTHON := $(VENV_DIR)/bin/python

ifeq ($(GEN),ninja)
GENERATOR := -G Ninja
endif

BUILD_RELEASE_DIR := build/release
BUILD_DEBUG_DIR := build/debug

.PHONY: all release debug reldebug relassert test test_release test_debug test_reldebug \
	ensure_duckdb_python test_python test_python_release test_python_debug test_sql test_sql_release test_sql_debug \
	clean format format-check

all: release

configure_release:
	@cmake $(GENERATOR) -S . -B $(BUILD_RELEASE_DIR) -DCMAKE_BUILD_TYPE=Release -DDUCKDB_LIB_DIR="$(DUCKDB_LIB_DIR)"

configure_debug:
	@cmake $(GENERATOR) -S . -B $(BUILD_DEBUG_DIR) -DCMAKE_BUILD_TYPE=Debug -DDUCKDB_LIB_DIR="$(DUCKDB_LIB_DIR)"

release: configure_release
	@cmake --build $(BUILD_RELEASE_DIR) --config Release

debug: configure_debug
	@cmake --build $(BUILD_DEBUG_DIR) --config Debug

reldebug: release
relassert: release

ensure_duckdb_python:
	@[ -x "$(VENV_PYTHON)" ] || $(PYTHON_BIN) -m venv $(VENV_DIR)
	@$(VENV_PYTHON) -m pip install --quiet --upgrade pip
	@$(VENV_PYTHON) -m pip install --quiet "duckdb==$(DUCKDB_VERSION)" pytest numpy pandas xgboost scikit-learn

test: test_release
test_release: ensure_duckdb_python release test_sql_release test_python_release
test_debug: ensure_duckdb_python debug test_sql_debug test_python_debug
test_reldebug: test_release

test_sql: test_sql_release
test_sql_release: ensure_duckdb_python
	@$(VENV_PYTHON) scripts/run_sql_tests.py --extension "$(ROOT_DIR)build/release/extension/ml/ml.duckdb_extension"

test_sql_debug: ensure_duckdb_python
	@$(VENV_PYTHON) scripts/run_sql_tests.py --extension "$(ROOT_DIR)build/debug/extension/ml/ml.duckdb_extension"

test_python: test_python_release
test_python_release: ensure_duckdb_python
	@ML_EXTENSION_PATH="$(ROOT_DIR)build/release/extension/ml/ml.duckdb_extension" $(VENV_PYTHON) -m pytest $(wildcard test/python/*.py)

test_python_debug: ensure_duckdb_python
	@ML_EXTENSION_PATH="$(ROOT_DIR)build/debug/extension/ml/ml.duckdb_extension" $(VENV_PYTHON) -m pytest $(wildcard test/python/*.py)

format-check:
	@python3 -m ruff format --check src test scripts

format:
	@python3 -m ruff format src test scripts

clean:
	@rm -rf build
