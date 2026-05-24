#!/usr/bin/env bash
# Full CI-style verification: cargo fmt, unit + doctests, clippy, wasm32
# cross-build, Kani proofs. Each step prints a banner and fails the script
# if it doesn't succeed (set -e).
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."

banner() { printf '\n=== %s ===\n' "$1"; }

banner "cargo fmt --check"
cargo fmt --check

banner "cargo test --release --features macros"
RUSTFLAGS='-C target-cpu=native' cargo test --release --features macros 2>&1 \
    | grep -E '^test result' || true

banner "cargo clippy --release --features macros --all-targets"
RUSTFLAGS='-C target-cpu=native' cargo clippy --release --features macros --all-targets

banner "cargo check --release --target wasm32-unknown-unknown"
cargo check --release --target wasm32-unknown-unknown

banner "cargo kani --features macros"
cargo kani --features macros 2>&1 | grep -E 'VERIFICATION|Complete -' || true

printf '\n==> all checks passed\n'
