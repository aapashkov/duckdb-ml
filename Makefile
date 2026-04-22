PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=ml
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Append Python parity tests
.PHONY: test_python
test_python:
	@python3 -m pytest $(wildcard test/python/*.py)

test: test_python
test_release: test_python
test_debug: test_python
test_reldebug: test_python
