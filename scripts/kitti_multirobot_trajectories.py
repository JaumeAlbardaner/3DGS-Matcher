#!/usr/bin/env python3
"""Top-down trajectory plots for the KITTI multi-robot pairs: ground truth vs fused.

For every pair, two panels side by side:
  left   ground truth: robot A (target) and robot B, both in the ground-truth frame
  right  fused: robot A, and robot B placed by 3DGS-Matcher + 3DGS-ICP

Drawing is layered so shared road stays readable: robot B is a wide band and robot A
a thin line on top of it. In the fused panel robot B's ground truth is a thin line
UNDER the fused band, so any visible ground-truth line means placement error.

The fused estimate shown is the MEDIAN seed by ATE (filtered maps), so the plot is
representative rather than a best case. For within-sequence pairs, robot B's ground
truth uses the LiDAR correction in kitti/revisit_gt/<scene>.json, like the table.
Trajectories are the left colour camera of each frame (cameras.json of the maps).

  python3 scripts/kitti_multirobot_trajectories.py
Writes results/figures/kitti_multirobot_trajectories.{png,pdf} and one figure per pair
in results/figures/trajectories/.
"""
import os
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from score_submap_csv import ResidualAt, ROT_THRESH, TRANS_THRESH, load_correction

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
OUT = DATA / "results/figures"

# reference palette, first three categorical slots (validated all-pairs, light surface;
# aqua is below 3:1 contrast, so every trajectory also carries a direct label)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_A, C_B_GT, C_B_FUSED = "#2a78d6", "#eb6834", "#1baf7a"
BAND, THIN, REF = 4.5, 1.6, 1.2  # line widths: robot B band, robot A line, GT reference

# scene, title, source robot (B), target robot (A)
PAIRS = [
    ("kitti00", "KITTI 00 (same drive, revisit)", "robot_b", "robot_a"),
    ("kitti00_07", "KITTI 00 / 07 (different days)", "robot_07", "robot_00"),
    ("kitti06", "KITTI 06 (same drive, revisit)", "robot_b", "robot_a"),
    ("kitti05", "KITTI 05 (same drive, revisit)", "robot_b", "robot_a"),
    ("kitti02", "KITTI 02 (same drive, revisit)", "robot_b", "robot_a"),
    ("kitti08", "KITTI 08 (same drive, opposite direction)", "robot_b", "robot_a"),
    ("kitti09_10", "KITTI 09 / 10 (different drives, opposite direction)", "robot_10", "robot_09"),
]


def left_cameras(scene, robot):
    cams = json.load(open(DATA / f"trained_gaussians/{scene}/{robot}/cameras.json"))
    cams = sorted((c for c in cams if Path(c["img_name"]).stem.endswith("_2")), key=lambda c: c["img_name"])
    return np.array([c["position"] for c in cams])


def placed(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


def pair_data(scene, src, tgt):
    corr = DATA / f"kitti/revisit_gt/{scene}.json"
    gt = load_correction(corr) if corr.exists() else np.eye(4)
    rows = [r for r in csv.DictReader(open(DATA / f"results/{scene}_robots_filtered_joint5d_v050.csv"))
            if r["method"] == "color_gicp_de"]
    ev = ResidualAt("gt_cameras", DATA / "trained_gaussians_filtered", DATA / "trained_gaussians",
                    180.0, 500.0, gt)
    scored = [(ev(r), ev.rot(r), r) for r in rows]
    success = 100 * np.mean([t < TRANS_THRESH and rot < ROT_THRESH for t, rot, _ in scored])
    ate, rot, med = sorted(scored, key=lambda s: s[0])[len(scored) // 2]
    E_raw = np.linalg.inv(ev.gt_inv) @ ev.error(med)   # estimate, in robot A's (raw) frame
    B = left_cameras(scene, src)
    return dict(A=left_cameras(scene, tgt), B_gt=placed(gt, B), B_fused=placed(E_raw, B),
                ate=ate, rot=rot, seed=med["seed"], success=success)


def style(ax, extent):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("x (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("z (m)", color=MUTED, fontsize=8)
    (x0, z0), (x1, z1) = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal", adjustable="box")


def extent(*tracks):
    pts = np.vstack(tracks)[:, [0, 2]]
    lo, hi = pts.min(0), pts.max(0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.12 + 10   # square window, equal scale
    return (c - half, c + half)


def line(ax, P, color, lw, z):
    ax.plot(P[:, 0], P[:, 2], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)


def end_label(ax, P, color, text):
    x, z = P[-1, 0], P[-1, 2]
    ax.plot([x], [z], "o", ms=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=10)
    ax.annotate(text, (x, z), xytext=(6, 6), textcoords="offset points", color=INK, fontsize=8, zorder=11)


def draw(ax_gt, ax_fused, title, d):
    # ground truth: B as a band, A as a thin line on top
    line(ax_gt, d["B_gt"], C_B_GT, BAND, 2)
    line(ax_gt, d["A"], C_A, THIN, 3)
    end_label(ax_gt, d["B_gt"], C_B_GT, "B (GT)")
    end_label(ax_gt, d["A"], C_A, "A")
    # fused: B's GT thin at the bottom (visible only where the placement is wrong),
    # fused B as a band, A thin on top
    line(ax_fused, d["B_gt"], C_B_GT, REF, 1)
    line(ax_fused, d["B_fused"], C_B_FUSED, BAND, 2)
    line(ax_fused, d["A"], C_A, THIN, 3)
    end_label(ax_fused, d["B_fused"], C_B_FUSED, "B (fused)")
    end_label(ax_fused, d["A"], C_A, "A")

    style(ax_gt, extent(d["A"], d["B_gt"]))
    style(ax_fused, extent(d["A"], d["B_gt"], d["B_fused"]))
    ax_gt.set_title(f"{title}\nground truth", loc="left", color=INK, fontsize=9)
    ax_fused.set_title(f"fused by 3DGS-Matcher + 3DGS-ICP: {d['success']:.0f}% of seeds succeed\n"
                       f"median seed {d['seed']}: ATE {d['ate']:.2f} m, rotation error {d['rot']:.2f}°",
                       loc="left", color=INK_2, fontsize=9)


def legend(fig):
    handles = [Line2D([], [], color=C_A, lw=THIN, label="robot A (target map)"),
               Line2D([], [], color=C_B_GT, lw=BAND, label="robot B, ground truth"),
               Line2D([], [], color=C_B_FUSED, lw=BAND, label="robot B, placed by registration (fused)"),
               Line2D([], [], color=C_B_GT, lw=REF, label="robot B ground truth under the fused band"),
               Line2D([], [], ls="none", marker="o", ms=7, color=MUTED, markeredgecolor=SURFACE,
                      markeredgewidth=2, label="end of trajectory")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)


def main():
    (OUT / "trajectories").mkdir(parents=True, exist_ok=True)
    data = [(scene, title, pair_data(scene, src, tgt)) for scene, title, src, tgt in PAIRS]

    height = 5.2 * len(data)
    fig, axes = plt.subplots(len(data), 2, figsize=(11, height), facecolor=SURFACE)
    for row, (scene, title, d) in enumerate(data):
        draw(axes[row, 0], axes[row, 1], title, d)
        print(f"{scene}: success {d['success']:.0f}%, median seed {d['seed']} ATE {d['ate']:.2f} m, rot {d['rot']:.2f} deg")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"kitti_multirobot_trajectories.{ext}", dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    for scene, title, d in data:
        f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.2), facecolor=SURFACE)
        draw(a1, a2, title, d)
        legend(f)
        f.tight_layout(rect=(0, 0, 1, 1 - 0.75 / 6.2))
        f.savefig(OUT / f"trajectories/{scene}.png", dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
        plt.close(f)
    print(f"wrote {OUT / 'kitti_multirobot_trajectories.png'} (+ .pdf) and {OUT / 'trajectories'}/<scene>.png")


if __name__ == "__main__":
    main()
