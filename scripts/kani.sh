#!/usr/bin/env bash
# Run all Kani proofs under src/kani_proofs.rs.
# Requires: cargo install --locked kani-verifier && cargo kani setup
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."
cargo kani --features macros "$@"
