#!/usr/bin/env python3
"""Plot per-digit-count ns curves from a `digit_curve_*.csv`.

Usage: plot_times.py <csv> <out_jpg> <chart_title>

Workload + target group inferred from CSV filename stem. Y-axis sized to
1.2 × max value (no clipping).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(sys.argv[1])
OUT_PATH = Path(sys.argv[2])
TITLE = sys.argv[3] if len(sys.argv) > 3 else CSV_PATH.stem

LABELS_BY_FILE = {
    "digit_curve_write_string": {
        "core_ns": "std::write!",
        "itoa_ns": "itoa::Buffer + push_str",
        "vitoa_ns": "vitoa::write!",
    },
    "digit_curve_writeln_string": {
        "core_ns": "std::writeln!",
        "itoa_ns": "itoa::Buffer + push_str + push '\\n'",
        "vitoa_ns": "vitoa::writeln!",
    },
    "digit_curve_write_bytes": {
        "itoa_ns": "itoa::Buffer::format → &str",
        "vitoa_ns": "vitoa::fmt → &mut [u8]",
    },
    "digit_curve_csv_bytes": {
        "itoa_ns": "itoa::Buffer + copy CSV loop",
        "vitoa_ns": "vitoa::write_joined! → &mut [u8]",
    },
}

PANELS = [
    ("u32", "Digits in u32", 10),
    ("u64", "Digits in u64", 20),
    ("u128", "Digits in u128", 40),
]

LABELS = LABELS_BY_FILE[CSV_PATH.stem]

df = pd.read_csv(CSV_PATH)
y_max = float(df[list(LABELS.keys())].to_numpy().max()) * 1.2

fig, axes = plt.subplots(
    1, 3,
    sharey=True,
    figsize=(13, 4.5),
    gridspec_kw={"width_ratios": [10, 20, 40]},
)

for ax, (width, xlabel, x_max) in zip(axes, PANELS):
    sub = df[df["width"] == width].sort_values("digits")
    for col, label in LABELS.items():
        series = sub[["digits", col]].dropna()
        if series.empty:
            continue
        ax.plot(series["digits"], series[col], marker="o", label=label)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Time per call (ns)")
axes[0].set_ylim(0, y_max)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=len(labels),
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    fontsize=10,
)
fig.suptitle(TITLE, y=1.08, fontsize=12)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
print(f"wrote {OUT_PATH}")
