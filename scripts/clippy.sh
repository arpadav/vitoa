#!/usr/bin/env bash
# Clippy with the macros feature on all targets (lib, examples, benches).
# Uses native AVX-512 IFMA target features to exercise the SIMD code path.
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS='-C target-cpu=native' cargo clippy --release --features macros --all-targets "$@"
