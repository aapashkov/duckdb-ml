#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <extension_binary>" >&2
  exit 1
fi

input_file="$1"
workdir="$(mktemp -d)"
cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT

split -b 1M "$input_file" "$workdir/chunk_"

concat_hashes="$workdir/hash_concats"
: > "$concat_hashes"
for chunk in "$workdir"/chunk_*; do
  openssl dgst -binary -sha256 "$chunk" >> "$concat_hashes"
done

openssl dgst -binary -sha256 "$concat_hashes"
