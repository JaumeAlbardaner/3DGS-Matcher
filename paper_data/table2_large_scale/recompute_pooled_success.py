#!/usr/bin/env python3
"""Recompute the pooled success rates cited in the abstract and conclusion of
the paper, from the CSVs in this folder.

Pooled success = fraction of ALL trials over the nine Mip-NeRF 360 scenes
(n = 9 x 200 = 1800) meeting the paper's success criterion of 5 deg / 0.5 m.

Run from this directory:
    python3 recompute_pooled_success.py

NOTE: the per-scene error/runtime numbers in Table II itself are produced by
gen_large_scale_table.py (also in this folder); run it with `--config on
--centroid centroid_trained_v050_n200_opacityON.csv --joint5d
joint5d_trained_v050_n200_opacityON.csv --gaussreg gaussreg_results.csv`.
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
SCENES = {"bicycle", "bonsai", "counter", "flowers", "garden",
          "kitchen", "room", "stump", "treehill"}
ROT, TRANS = 5.0, 0.5  # paper's stated success criterion


def pooled(path, method, rcol, tcol, mcol=None):
    ok = tot = 0
    for r in csv.DictReader(open(path)):
        if r["scene"] not in SCENES:
            continue
        if mcol and r[mcol] != method:
            continue
        tot += 1
        if float(r[rcol]) < ROT and float(r[tcol]) < TRANS:
            ok += 1
    return 100 * ok / tot, ok, tot


cent = HERE / "centroid_trained_v050_n200_opacityON.csv"
j5 = HERE / "joint5d_trained_v050_n200_opacityON.csv"

print("Pooled success @ 5 deg / 0.5 m  (opacity ON, n=1800)\n")
rows = [
    ("KISS-Matcher            (abstract 6%)",  j5,   "vanilla_kiss",  "rot_error_deg", "trans_error_m", "method"),
    ("Ours centroid coarse    (abstract 45%)", cent, "color_cpfh",    "rot_error_deg", "trans_error_m", "method"),
    ("Ours joint-5D coarse    (abstract 80%)", j5,   "color_cpfh",    "rot_error_deg", "trans_error_m", "method"),
    ("Ours joint-5D +3DGS-ICP (abstract 85%)", j5,   "color_gicp_de", "rot_error_deg", "trans_error_m", "method"),
]
for label, path, method, rc, tc, mc in rows:
    pct, ok, tot = pooled(path, method, rc, tc, mc)
    print(f"  {label:40s} {pct:5.1f}%  ({ok}/{tot})")

pct, ok, tot = pooled(HERE / "gaussreg_results.csv", "full_features",
                      "rot_err_deg", "trans_err_abs", "trial_mode")
print(f"  {'GaussReg                (concl. 62%)':40s} {pct:5.1f}%  ({ok}/{tot})")

pct, ok, tot = pooled(HERE / "colorpcr_trained_nondrone_v050_n100.csv", "colorpcr",
                      "rot_error_deg", "trans_error_m", "method")
print(f"  {'ColorPCR                (concl. 46%)':40s} {pct:5.1f}%  ({ok}/{tot})")
print("\n  Note: ColorPCR is scored at the paper's 5 deg / 0.5 m, like every method")
print("  above, giving 45.9% (rounded to 46% in the paper). Its CSV's own `success`")
print("  column is scored at 1.0 m instead, i.e. the outdated 51.4% figure.")
