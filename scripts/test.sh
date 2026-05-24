#!/usr/bin/env bash
# Run unit + doctests with the macros feature on the native AVX-512 IFMA path.
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS='-C target-cpu=native' cargo test --release --features macros "$@"
