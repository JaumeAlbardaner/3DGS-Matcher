#!/usr/bin/env python3
"""Extra experiment 2: score KISS-Matcher on LiDAR maps next to KISS-Matcher on the (filtered) Gaussian maps.

Same scoring as kitti_multirobot_table.py: rotation < 5 deg and ATE over the source
robot's GT cameras < 0.5 m, revisit pairs against kitti/revisit_gt/<scene>.json.
Also reports how often the solver kept any inliers and the loose 5 deg / 2 m rate
(KITTI scan-registration convention).

  python3 scripts/kitti_lidar_kiss_score.py

The Gaussian-map CSVs are read from RESULTS_DIR and the LiDAR CSVs from
LIDAR_RESULTS_DIR (default: RESULTS_DIR).
"""
import csv
import os
from pathlib import Path
import statistics as st

from kitti_multirobot_table import DATA, RESULTS, SCENES
from score_submap_csv import ResidualAt, load_correction

import numpy as np

LIDAR_RESULTS = Path(os.environ.get("LIDAR_RESULTS_DIR", RESULTS))


def kiss(path, base, gt):
    rows = [r for r in csv.DictReader(open(path)) if r["method"] == "vanilla_kiss"]
    ate = ResidualAt("gt_cameras", base, DATA / "trained_gaussians", 180.0, 500.0, gt)
    errs = [(ate.rot(r), ate(r)) for r in rows]
    n = len(rows)
    return dict(n=n, ok=100 * sum(r["solver_ok"] == "1" for r in rows) / n,
                strict=100 * sum(a < 5 and t < 0.5 for a, t in errs) / n,
                loose=100 * sum(a < 5 and t < 2.0 for a, t in errs) / n,
                med_rot=st.median(a for a, _ in errs), med_ate=st.median(t for _, t in errs),
                secs=st.mean(float(r["time_ms"]) / 1000 for r in rows))


def main():
    print("| Pair | Maps | seeds | solver kept inliers | success 5°/0.5 m | 5°/2 m | median rot | median ATE | time |")
    print("|---|---|---|---|---|---|---|---|---|")
    for scene, (label, _, _) in SCENES.items():
        lidar = LIDAR_RESULTS / f"{scene}_robots_lidar_v050.csv"
        if not lidar.exists():
            continue
        corr = DATA / f"kitti/revisit_gt/{scene}.json"
        gt = load_correction(corr) if corr.exists() else np.eye(4)
        for maps, path, base in (("Gaussians (filtered)", RESULTS / f"{scene}_robots_filtered_joint5d_v050.csv",
                                  DATA / "trained_gaussians_filtered"),
                                 ("LiDAR", lidar, DATA / "lidar_maps")):
            s = kiss(path, base, gt)
            print(f"| {label} | {maps} | {s['n']} | {s['ok']:.0f}% | {s['strict']:.0f}% | {s['loose']:.0f}% | "
                  f"{s['med_rot']:.1f}° | {s['med_ate']:.1f} m | {s['secs']:.1f} s |")


if __name__ == "__main__":
    main()
