#!/usr/bin/env python3
"""Collect every KITTI multi-robot experiment into one table (markdown + LaTeX body).

Reads results/<scene>_robots_{unfiltered,filtered}_joint5d_v050.csv (from
examples/run_kitti_robots.sh) and scores them like the rest of the KITTI results:
success = rotation error < 5 deg and ATE over the source robot's ground-truth
cameras < 0.5 m (score_submap_csv.py --at gt_cameras).

Ground truth. KITTI's GT poses can disagree between two passes of one sequence by
metres (GPS/INS height drift), so within-sequence pairs are scored against the
LiDAR-corrected transform in kitti/revisit_gt/<scene>.json when it exists
(scripts/kitti_check_revisit_gt.py). Cross-sequence pairs already use a LiDAR-refined
alignment baked into their poses. The table reports each pair's correction size.

Map overlap = share of source Gaussians within 0.5 m of the target map (filtered
maps), with the source placed by the (corrected) ground truth.

  python3 scripts/kitti_multirobot_table.py [--out results/kitti_multirobot_table] [--no_gt_correction]
"""
import os
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from compute_overlap_metrics import load_ply_xyz
from score_submap_csv import JOIN, ResidualAt, ROT_THRESH, TRANS_THRESH, load_correction

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", DATA / "results"))
PLY = "point_cloud/iteration_30000/point_cloud.ply"
# scene -> (label, how the robots overlap, driving direction on the shared road)
SCENES = {
    "kitti06": ("KITTI 06", "same drive, revisit", "same"),
    "kitti05": ("KITTI 05", "same drive, revisit", "same"),
    "kitti00": ("KITTI 00", "same drive, revisit", "same"),
    "kitti02": ("KITTI 02", "same drive, revisit", "same"),
    "kitti08": ("KITTI 08", "same drive, revisit", "opposite"),
    "kitti00_07": ("KITTI 00 / 07", "different days", "same"),
    "kitti09_10": ("KITTI 09 / 10", "different drives", "opposite"),
}
METHODS = [("KISS-Matcher", "vanilla_kiss", False), ("Ours, coarse", "color_cpfh", False),
           ("+ GICP", "gicp_cpfh", True), ("+ 3DGS-ICP", "color_gicp_de", True)]


def overlap(scene, src, tgt, gt, n=300_000, tau=0.5):
    a = load_ply_xyz(DATA / "trained_gaussians_filtered" / scene / src / PLY)
    b = load_ply_xyz(DATA / "trained_gaussians_filtered" / scene / tgt / PLY)
    q = a if len(a) <= n else a[np.random.default_rng(0).choice(len(a), n, replace=False)]
    q = q @ gt[:3, :3].T + gt[:3, 3]
    return 100.0 * np.mean(cKDTree(b).query(q, distance_upper_bound=tau)[0] <= tau)


def score(path, base, gt):
    rows = list(csv.DictReader(open(path)))
    ate = ResidualAt("gt_cameras", base, DATA / "trained_gaussians", 180.0, 500.0, gt)
    coarse = {tuple(r[k] for k in JOIN): float(r["time_ms"]) / 1000 for r in rows if r["method"] == "color_cpfh"}
    out = {}
    for label, m, refined in METHODS:
        rr = [r for r in rows if r["method"] == m]
        errs = [(ate.rot(r), ate(r)) for r in rr]
        ok = [rot < ROT_THRESH and t < TRANS_THRESH for rot, t in errs]
        secs = [float(r["time_ms"]) / 1000 + (coarse.get(tuple(r[k] for k in JOIN), 0) if refined else 0) for r in rr]
        conv = [t for rot, t in errs if rot < ROT_THRESH]
        out[label] = dict(pct=100 * sum(ok) / len(ok), n=len(ok), time=st.mean(secs),
                          ate=st.median(conv) if conv else float("nan"))
    return rows[0]["split_src"], rows[0]["split_tgt"], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RESULTS / "kitti_multirobot_table")
    ap.add_argument("--no_gt_correction", action="store_true",
                    help="score within-sequence pairs against raw KITTI GT (identity)")
    args = ap.parse_args()

    table = []
    for scene, (label, kind, direction) in SCENES.items():
        paths = {v: RESULTS / f"{scene}_robots_{v}_joint5d_v050.csv" for v in ("unfiltered", "filtered")}
        if not all(p.exists() for p in paths.values()):
            print(f"skip {scene}: results not complete")
            continue
        corr_file = DATA / f"kitti/revisit_gt/{scene}.json"
        if corr_file.exists() and not args.no_gt_correction:
            gt = load_correction(corr_file)
            corr = f"{json.loads(corr_file.read_text())['offset_at_overlap_m']:.2f} m"
        else:
            gt = np.eye(4)
            corr = "LiDAR-aligned poses" if kind.startswith("different") else "none"
        src, tgt, filt = score(paths["filtered"], DATA / "trained_gaussians_filtered", gt)
        _, _, unfilt = score(paths["unfiltered"], DATA / "trained_gaussians", gt)
        table.append((label, kind, direction, corr, overlap(scene, src, tgt, gt), unfilt, filt))
        print(f"scored {scene} ({src} -> {tgt}, GT correction: {corr})")

    cols = [m[0] for m in METHODS]
    header = ["Pair", "Overlap source", "Direction", "GT correction", "Map overlap", "Maps", *cols,
              "3DGS-ICP ATE (median)", "3DGS-ICP time"]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    tex = []
    for label, kind, direction, corr, ov, unfilt, filt in table:
        for maps, res in (("unfiltered", unfilt), ("filtered", filt)):
            cells = " | ".join(f"{res[c]['pct']:.0f}%" for c in cols)
            md.append(f"| {label} | {kind} | {direction} | {corr} | {ov:.0f}% | {maps} | {cells} | "
                      f"{res['+ 3DGS-ICP']['ate']:.2f} m | {res['+ 3DGS-ICP']['time']:.1f} s |")
        tex.append(f"{label} & {ov:.0f}\\% & " + " & ".join(f"{unfilt[c]['pct']:.0f}" for c in cols) + " & "
                   + " & ".join(f"{filt[c]['pct']:.0f}" for c in cols)
                   + f" & {filt['+ 3DGS-ICP']['ate']:.2f} & {filt['+ 3DGS-ICP']['time']:.1f} \\\\")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    args.out.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"\nwrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
