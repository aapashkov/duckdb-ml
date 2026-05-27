#!/usr/bin/env bash
# Downloads prebuilt DuckDB library for local development.
# This is required because the workspace no longer uses the duckdb "bundled" feature.
#
# Usage: ./scripts/setup-duckdb.sh
# The library is downloaded to duckdb_lib/ and .cargo/config.toml points to it.

set -euo pipefail

DUCKDB_VERSION="1.4.4"
LIB_DIR="duckdb_lib"

# SHA256 checksums for DuckDB v1.4.4 prebuilt library archives
SHA256_LINUX_AMD64="09cc288295964d897b47665d1898e16e8ef176cae9ea615797fc136eae15bd5d"
SHA256_OSX_UNIVERSAL="5e6d79f5fb86bbd4f24d59cee3bc38f113d1433761df35337e3c01c62eeafe26"

cd "$(git rev-parse --show-toplevel)"

if { [ -f "$LIB_DIR/libduckdb.so" ] || [ -f "$LIB_DIR/libduckdb.dylib" ]; } && [ -f "$LIB_DIR/duckdb.h" ]; then
    echo "DuckDB library and headers already present in $LIB_DIR/"
    ls -la "$LIB_DIR"/libduckdb* "$LIB_DIR"/duckdb.h
    exit 0
fi

mkdir -p "$LIB_DIR"

# Detect platform
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)
        ASSET="libduckdb-linux-amd64.zip"
        EXPECTED_SHA256="$SHA256_LINUX_AMD64"
        ;;
    Darwin-x86_64|Darwin-arm64)
        ASSET="libduckdb-osx-universal.zip"
        EXPECTED_SHA256="$SHA256_OSX_UNIVERSAL"
        ;;
    *)
        echo "Unsupported platform: $(uname -s)-$(uname -m)"
        exit 1
        ;;
esac

echo "Downloading DuckDB v${DUCKDB_VERSION} ($ASSET)..."
curl -fsSL "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/${ASSET}" -o /tmp/duckdb_lib.zip

echo "Verifying SHA256 checksum..."
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA256="$(sha256sum /tmp/duckdb_lib.zip | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_SHA256="$(shasum -a 256 /tmp/duckdb_lib.zip | awk '{print $1}')"
else
    echo "ERROR: neither sha256sum nor shasum is available for checksum verification"
    rm -f /tmp/duckdb_lib.zip
    exit 1
fi

if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: SHA256 checksum verification failed for DuckDB download!"
    echo "Expected: $EXPECTED_SHA256"
    echo "Actual:   $ACTUAL_SHA256"
    rm -f /tmp/duckdb_lib.zip
    exit 1
fi

unzip -o /tmp/duckdb_lib.zip -d "$LIB_DIR"
rm /tmp/duckdb_lib.zip

echo "DuckDB library installed to $LIB_DIR/"
ls -la "$LIB_DIR"/libduckdb*
