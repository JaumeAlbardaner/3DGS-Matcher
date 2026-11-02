#!/usr/bin/env python3
"""Recompute Table I (tab:colorpfh_revalidation) from the raw tramway CSV.

Each cell = mean success rate (%) over the full noise sweep (11 levels x 100
seeds = 1100 trials) for a (configuration, perturbation) pair, at the paper's
success criterion of 5 deg / 0.5 m. Run from this directory:

    python3 recompute_table1.py
"""
import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).parent / "noise_doubts_tramway_p500_n100.csv"

# CSV config_label -> paper row label. 'color+cPFH+opacity' appears twice in the
# paper (as the last descriptor row and as the 'centroid (best)' keypoint row).
ROWS = [
    ("colorPFH-only",               "colorPFH only (no RGB)"),
    ("color+colorPFH",              "color + colorPFH"),
    ("colorPFH+opacity",            "colorPFH + opacity"),
    ("color+cPFH+opacity",          "color + colorPFH + opacity  /  centroid (best)"),
    ("color+colorPFH(joint5D)",     "joint-5D"),
    ("color+cPFH+opacity(joint5D)", "joint-5D + opacity"),
]
EXPS = ["baseline", "color_offset", "outliers", "split_half"]

ok = defaultdict(int)
tot = defaultdict(int)
for r in csv.DictReader(open(CSV)):
    k = (r["config_label"], r["experiment"])
    tot[k] += 1
    ok[k] += int(r["success"])

print(f"{'Configuration':46} " + " ".join(f"{e:>13}" for e in EXPS))
print("-" * 106)
for cfg, label in ROWS:
    cells = []
    for e in EXPS:
        k = (cfg, e)
        cells.append(f"{100*ok[k]/tot[k]:5.1f}" if tot[k] else "  n/a")
    print(f"{label:46} " + " ".join(f"{c:>13}" for c in cells))
