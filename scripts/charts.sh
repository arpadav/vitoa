#!/usr/bin/env bash
# Regenerate the 4 perf charts: run the digit_curve example, render the JPGs
# into the repo root.
#
# Usage: scripts/charts.sh [TRIALS]
#   TRIALS — bench trials per (width, digits) cell. Default 100.
#
# Requires: .venv with matplotlib + pandas (see scripts/setup_venv.sh)
#
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."

TRIALS="${1:-100}"
CSV_DIR="/tmp"
PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "error: $PY not found — run scripts/setup_venv.sh first" >&2
    exit 1
fi

echo "==> running digit_curve bench ($TRIALS trials)"
rm -f "$CSV_DIR"/digit_curve_*.csv vitoa-times-*.jpg
RUSTFLAGS='-C target-cpu=native' cargo run --release \
    --example digit_curve --features macros -- "$TRIALS" "$CSV_DIR"

echo "==> rendering 4 charts"
"$PY" scripts/plot_times.py "$CSV_DIR/digit_curve_write_string.csv" \
    vitoa-times-write-string.jpg \
    "Write integer → String  ·  ${TRIALS}-trial median (ns/call)"
"$PY" scripts/plot_times.py "$CSV_DIR/digit_curve_writeln_string.csv" \
    vitoa-times-writeln-string.jpg \
    "Writeln integer → String  ·  ${TRIALS}-trial median (ns/call)"
"$PY" scripts/plot_times.py "$CSV_DIR/digit_curve_write_bytes.csv" \
    vitoa-times-write-bytes.jpg \
    "Write integer → &mut [u8]  ·  ${TRIALS}-trial median (ns/call)"
"$PY" scripts/plot_times.py "$CSV_DIR/digit_curve_csv_bytes.csv" \
    vitoa-times-csv-bytes.jpg \
    "Comma-join 4 values → &mut [u8]  ·  ${TRIALS}-trial median (ns/call)"

echo "==> done — 4 JPGs written to repo root"
