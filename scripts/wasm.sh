#!/usr/bin/env bash
# Cross-check the non-x86_64 scalar-fallback path compiles on wasm32.
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."
cargo check --release --target wasm32-unknown-unknown "$@"
