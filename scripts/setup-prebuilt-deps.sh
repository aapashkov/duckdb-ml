#!/usr/bin/env bash

set -euo pipefail

ROOT=""
EIGEN_URL=""
EIGEN_SHA256=""
XGBOOST_SRC_URL=""
XGBOOST_SRC_SHA256=""
XGBOOST_WHEEL_URL=""
XGBOOST_WHEEL_SHA256=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --eigen-url)
      EIGEN_URL="$2"
      shift 2
      ;;
    --eigen-sha256)
      EIGEN_SHA256="$2"
      shift 2
      ;;
    --xgboost-src-url)
      XGBOOST_SRC_URL="$2"
      shift 2
      ;;
    --xgboost-src-sha256)
      XGBOOST_SRC_SHA256="$2"
      shift 2
      ;;
    --xgboost-wheel-url)
      XGBOOST_WHEEL_URL="$2"
      shift 2
      ;;
    --xgboost-wheel-sha256)
      XGBOOST_WHEEL_SHA256="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ROOT" || -z "$EIGEN_URL" || -z "$EIGEN_SHA256" || -z "$XGBOOST_SRC_URL" || -z "$XGBOOST_SRC_SHA256" || -z "$XGBOOST_WHEEL_URL" || -z "$XGBOOST_WHEEL_SHA256" ]]; then
  echo "Missing required arguments" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required" >&2
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required" >&2
  exit 1
fi

hash_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    echo "Missing sha256 checksum tool (sha256sum/shasum)" >&2
    exit 1
  fi
}

download_checked() {
  local url="$1"
  local expected="$2"
  local output="$3"

  mkdir -p "$(dirname "$output")"
  curl -fsSL "$url" -o "$output"

  local actual
  actual="$(hash_file "$output")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $output" >&2
    echo "Expected: $expected" >&2
    echo "Actual:   $actual" >&2
    exit 1
  fi
}

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

ARCHIVE_DIR="$ROOT/.downloads"
mkdir -p "$ARCHIVE_DIR"

EIGEN_ARCHIVE="$ARCHIVE_DIR/eigen.tar.gz"
XGBOOST_SRC_ARCHIVE="$ARCHIVE_DIR/xgboost-src.tar.gz"
XGBOOST_WHEEL_ARCHIVE="$ARCHIVE_DIR/xgboost.whl"

download_checked "$EIGEN_URL" "$EIGEN_SHA256" "$EIGEN_ARCHIVE"
download_checked "$XGBOOST_SRC_URL" "$XGBOOST_SRC_SHA256" "$XGBOOST_SRC_ARCHIVE"
download_checked "$XGBOOST_WHEEL_URL" "$XGBOOST_WHEEL_SHA256" "$XGBOOST_WHEEL_ARCHIVE"

# Eigen headers
EIGEN_TMP="$TMP_DIR/eigen"
mkdir -p "$EIGEN_TMP"
tar -xzf "$EIGEN_ARCHIVE" -C "$EIGEN_TMP"
EIGEN_SRC_DIR="$(find "$EIGEN_TMP" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [[ ! -d "$EIGEN_SRC_DIR/Eigen" ]]; then
  echo "Could not find Eigen headers in archive" >&2
  exit 1
fi
mkdir -p "$ROOT/eigen3/include"
rm -rf "$ROOT/eigen3/include/Eigen" "$ROOT/eigen3/include/unsupported"
cp -R "$EIGEN_SRC_DIR/Eigen" "$ROOT/eigen3/include/"
cp -R "$EIGEN_SRC_DIR/unsupported" "$ROOT/eigen3/include/"

# XGBoost headers from source archive
XGB_SRC_TMP="$TMP_DIR/xgboost-src"
mkdir -p "$XGB_SRC_TMP"
tar -xzf "$XGBOOST_SRC_ARCHIVE" -C "$XGB_SRC_TMP"
XGB_SRC_DIR="$(find "$XGB_SRC_TMP" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [[ ! -f "$XGB_SRC_DIR/include/xgboost/c_api.h" ]]; then
  echo "Could not find XGBoost headers in source archive" >&2
  exit 1
fi
mkdir -p "$ROOT/xgboost/include"
rm -rf "$ROOT/xgboost/include/xgboost"
cp -R "$XGB_SRC_DIR/include/xgboost" "$ROOT/xgboost/include/"

# Prebuilt libxgboost from wheel
XGB_WHEEL_TMP="$TMP_DIR/xgboost-wheel"
mkdir -p "$XGB_WHEEL_TMP"
unzip -q -o "$XGBOOST_WHEEL_ARCHIVE" "xgboost/lib/libxgboost.so" "xgboost.libs/*" -d "$XGB_WHEEL_TMP"
if [[ ! -f "$XGB_WHEEL_TMP/xgboost/lib/libxgboost.so" ]]; then
  echo "Could not find libxgboost.so in wheel" >&2
  exit 1
fi
mkdir -p "$ROOT/xgboost/lib"
cp "$XGB_WHEEL_TMP/xgboost/lib/libxgboost.so" "$ROOT/xgboost/lib/libxgboost.so"
chmod 755 "$ROOT/xgboost/lib/libxgboost.so"

if [[ -d "$XGB_WHEEL_TMP/xgboost.libs" ]]; then
  mkdir -p "$ROOT/xgboost.libs"
  cp "$XGB_WHEEL_TMP"/xgboost.libs/* "$ROOT/xgboost.libs/"
fi

echo "-- Prebuilt dependencies are ready in: $ROOT"
