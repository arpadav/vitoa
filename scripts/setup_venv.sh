#!/usr/bin/env bash
# Bootstrap the matplotlib venv used by scripts/charts.sh.
# Requires `uv` on PATH.
# Author: aav
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv
uv pip install --python .venv/bin/python matplotlib pandas
echo "==> venv ready at .venv (gitignored)"
