#!/usr/bin/env python3
"""Extra experiment 1: vanilla 3DGS vs 2DGS maps, same benchmark, KISS-Matcher vs ours.

Scoring is the paper's: success = rot < 5 deg and trans < 0.5 m. Mip-NeRF rows use the
residual at the origin (Table II); KITTI rows use ATE over the source robot's GT cameras
(KITTI table). Refined methods' time includes the coarse stage, as elsewhere.
Missing CSVs are skipped, so this can be run while the queue is still going.
"""
import csv
import statistics as st
import sys
import os
from pathlib import Path

D = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", D / "results"))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from score_submap_csv import JOIN, PLY, ROT_THRESH, TRANS_THRESH, ResidualAt  # noqa: E402

# Vanilla baselines re-run with the same (post CIELAB/D65 fix) binary as the 2DGS runs.
ROOM3 = RESULTS / "room_3dgs_joint5d_v050_postfix.csv"
METHODS = [("KISS-Matcher", "vanilla_kiss", False), ("Ours, coarse", "color_cpfh", False),
           ("+ GICP", "gicp_cpfh", True), ("+ 3DGS-ICP", "color_gicp_de", True)]

# (row label, maps, csv, pair filter, trained base, cameras base or None for origin scoring)
ROWS = [
    ("room pos/neg", "3DGS", ROOM3, ("room", "split_pos"), D / "trained_gaussians", None),
    ("room pos/neg", "2DGS", RESULTS / "2dgs_room_joint5d_v050.csv", ("room", "split_pos"), D / "trained_2dgs", None),
    ("room wide", "3DGS", ROOM3, ("room", "split_wide_pos"), D / "trained_gaussians", None),
    ("room wide", "2DGS", RESULTS / "2dgs_room_wide_joint5d_v050.csv", ("room", "split_wide_pos"), D / "trained_2dgs", None),
    ("KITTI 00/07 unfilt.", "3DGS", RESULTS / "kitti00_07_robots_unfiltered_joint5d_v050_d65.csv", None,
     D / "trained_gaussians", D / "trained_gaussians"),
    ("KITTI 00/07 unfilt.", "2DGS", RESULTS / "2dgs_kitti00_07_robots_unfiltered_joint5d_v050.csv", None,
     D / "trained_2dgs", D / "trained_2dgs"),
    ("KITTI 00/07 filt.", "3DGS", RESULTS / "kitti00_07_robots_filtered_joint5d_v050_d65.csv", None,
     D / "trained_gaussians_filtered", D / "trained_gaussians"),
    ("KITTI 00/07 filt.", "2DGS", RESULTS / "2dgs_kitti00_07_robots_filtered_joint5d_v050.csv", None,
     D / "trained_2dgs_filtered", D / "trained_2dgs"),
]


def n_gaussians(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                return 0


def score(path, pair, base, cams):
    rows = [r for r in csv.DictReader(open(path))
            if pair is None or (r["scene"], r["split_src"]) == pair]
    if not rows:
        return None
    if cams is None:
        trans = lambda r: float(r["trans_error_m"])  # noqa: E731
        rot = lambda r: float(r["rot_error_deg"])  # noqa: E731
    else:
        ate = ResidualAt("gt_cameras", base, cams, 180.0, 500.0)
        trans, rot = ate, ate.rot
    coarse = {tuple(r[k] for k in JOIN): float(r["time_ms"]) / 1000 for r in rows if r["method"] == "color_cpfh"}
    out = {}
    for label, m, refined in METHODS:
        rr = [r for r in rows if r["method"] == m]
        errs = [(rot(r), trans(r)) for r in rr]
        ok = [a < ROT_THRESH and t < TRANS_THRESH for a, t in errs]
        secs = [float(r["time_ms"]) / 1000 + (coarse.get(tuple(r[k] for k in JOIN), 0) if refined else 0) for r in rr]
        good = [t for (a, t), o in zip(errs, ok) if o]
        out[label] = dict(pct=100 * sum(ok) / len(ok), n=len(ok), time=st.median(secs),
                          err=st.median(good) if good else float("nan"))
    src, tgt = rows[0]["split_src"], rows[0]["split_tgt"]
    sizes = [n_gaussians(base / rows[0]["scene"] / s / PLY) for s in (src, tgt)]
    return out, sizes


def main():
    cols = [m[0] for m in METHODS]
    md = ["| Pair | Maps | #Gaussians (src/tgt) | n | " + " | ".join(cols) + " | ours 3DGS-ICP err (median) | 3DGS-ICP time |",
          "|" + "---|" * (len(cols) + 6)]
    for label, maps, path, pair, base, cams in ROWS:
        if not Path(path).exists():
            print(f"skip {label} {maps}: {path} missing")
            continue
        res = score(path, pair, base, cams)
        if res is None:
            continue
        out, sizes = res
        md.append(f"| {label} | {maps} | {sizes[0]:,} / {sizes[1]:,} | {out['KISS-Matcher']['n']} | "
                  + " | ".join(f"{out[c]['pct']:.0f}%" for c in cols)
                  + f" | {out['+ 3DGS-ICP']['err']:.2f} m | {out['+ 3DGS-ICP']['time']:.1f} s |")
    text = "\n".join(md) + "\n"
    (RESULTS / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
