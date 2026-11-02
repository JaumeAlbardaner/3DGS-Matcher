#!/usr/bin/env python3
"""Paper Sec. V-B: tramway photometric and scale perturbations.

Reads tramway_perturbations.csv (examples/photometric_scale_perturbations.sh,
100 seeds per condition, joint-5D keypoints) and prints, per condition and
descriptor configuration, the success rate at 5 deg / 0.5 m and the median
translation error.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parent / "tramway_perturbations.csv"

rows = list(csv.DictReader(open(CSV)))
configs = list(dict.fromkeys(r["config"] for r in rows))
conditions = list(dict.fromkeys(r["experiment"] for r in rows))
cell = defaultdict(list)
for r in rows:
    cell[(r["experiment"], r["config"])].append((float(r["rot_err"]), float(r["trans_err"])))

print("success % at 5 deg / 0.5 m   (median translation error, m)")
print(f"{'condition':<14}" + "".join(f"{c:>26}" for c in configs))
for cond in conditions:
    out = []
    for c in configs:
        v = cell[(cond, c)]
        ok = sum(r < 5 and t < 0.5 for r, t in v)
        out.append(f"{100 * ok / len(v):5.0f}  ({st.median(t for _, t in v):.3f})  n={len(v):<3}")
    print(f"{cond:<14}" + "".join(f"{o:>26}" for o in out))
