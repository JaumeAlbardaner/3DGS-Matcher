#!/usr/bin/env python3
"""All KITTI odometry trajectories in one GPS frame, with revisits / overlaps highlighted.

Odometry GT poses start at each sequence's own origin, so a common frame comes from the
raw drives' OXTS GPS (same Mercator projection as kitti_gps_overlap.py). Sequence 03 is
not placed: its raw drive (and GPS) was never published.

A trajectory point is an OVERLAP if it lies within --radius metres of
  * a point of another sequence, or
  * a point of the same sequence at least --min_gap frames away (>= 30 s at 10 Hz),
and is coloured by the relative heading there: same direction (<90 deg) or opposite.

  python3 scripts/kitti_global_trajectories.py [--radius 5] [--out results/figures/kitti_global_overlaps]
"""
import os
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_gps_overlap import SEQUENCES, load_latlon, mercator  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))

# reference palette (dataviz skill, light mode): slots 1-2 validate all-pairs
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
TRACK = "#a8a7a2"          # neutral base trajectory
SAME, OPP = "#2a78d6", "#eb6834"

AREAS = {  # zoom panels: title, sequences shown, (x0, x1, y0, y1) padding added later
    "A": ("Area A · seq 00, 07, 08", ["00", "07", "08"]),
    "B": ("Area B · seq 05, 06", ["05", "06"]),
    "C": ("Area C · seq 02", ["02"]),
    "D": ("Area D · seq 09, 10", ["09", "10"]),
}
NOTES = {  # short direct annotations of the cross-sequence relations
    "A": "00 ↔ 07: different days, same direction\n08: revisits in opposite direction",
    "D": "09 ↔ 10: different drives,\nmostly opposite direction",
}


def heading(p):
    d = np.gradient(p, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def overlaps(xy, radius, min_gap):
    seqs = list(xy)
    trees = {s: cKDTree(p) for s, p in xy.items()}
    head = {s: heading(p) for s, p in xy.items()}
    out = {}
    for s in seqs:
        p = xy[s]
        same = np.zeros(len(p), bool)
        opp = np.zeros(len(p), bool)
        partners = set()
        for t in seqs:
            for i, nb in enumerate(trees[t].query_ball_point(p, radius)):
                for j in nb:
                    if s == t and abs(i - j) < min_gap:
                        continue
                    dh = abs((head[s][i] - head[t][j] + np.pi) % (2 * np.pi) - np.pi)
                    (same if dh < np.pi / 2 else opp)[i] = True
                    partners.add(t)
        out[s] = (same, opp, partners)
    return out


def draw_track(ax, p, same, opp, lw_base, lw_hi):
    ax.plot(p[:, 0], p[:, 1], color=TRACK, lw=lw_base, solid_capstyle="round", zorder=1)
    for mask, col in ((same, SAME), (opp, OPP)):
        seg_mask = mask[:-1] & mask[1:]
        if seg_mask.any():
            segs = np.stack([p[:-1][seg_mask], p[1:][seg_mask]], axis=1)
            ax.add_collection(LineCollection(segs, colors=col, linewidths=lw_hi,
                                             capstyle="round", zorder=3))


def scale_bar(ax, length_m):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx, by = x0 + 0.06 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.plot([bx, bx + length_m], [by, by], color=INK2, lw=2, solid_capstyle="butt", zorder=5)
    ax.text(bx + length_m / 2, by + 0.025 * (y1 - y0), f"{length_m:g} m",
            ha="center", va="bottom", fontsize=8, color=INK2, zorder=5)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#dddcd8")
        sp.set_linewidth(0.8)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", type=Path, default=DATA / "kitti")
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--min_gap", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DATA / "results/figures/kitti_global_overlaps")
    args = ap.parse_args()

    ll = {s: load_latlon(args.kitti_root, *v) for s, v in SEQUENCES.items()}
    scale = np.cos(np.radians(np.mean([a[:, 0].mean() for a in ll.values()])))
    xy = {s: mercator(a, scale) for s, a in ll.items()}
    origin = np.vstack(list(xy.values())).min(0)
    xy = {s: p - origin for s, p in xy.items()}
    ov = overlaps(xy, args.radius, args.min_gap)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.06, hspace=0.16))
    ax_over, ax_a, ax_b, ax_c, ax_d, ax_tab = axes.flat

    # ── overview ──
    for s, p in xy.items():
        draw_track(ax_over, p, *ov[s][:2], lw_base=1.0, lw_hi=1.8)
    style(ax_over, "All sequences, common GPS frame")
    pad = 250
    for key, (_, seqs) in AREAS.items():
        pts = np.vstack([xy[s] for s in seqs])
        (x0, y0), (x1, y1) = pts.min(0) - pad, pts.max(0) + pad
        ax_over.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        ec=INK2, lw=0.8, ls=(0, (3, 2)), zorder=4))
        ax_over.text(x0, y1 + 90, key, fontsize=9, color=INK, fontweight="bold", va="bottom")
    for s in ("01", "04"):
        c = xy[s].mean(0)
        ax_over.text(c[0] + 180, c[1], s, fontsize=8, color=INK2, va="center")
    allp = np.vstack(list(xy.values()))
    ax_over.set_xlim(allp[:, 0].min() - 500, allp[:, 0].max() + 500)
    ax_over.set_ylim(allp[:, 1].min() - 500, allp[:, 1].max() + 700)
    scale_bar(ax_over, 2000)

    # ── zoom panels ──
    for ax, key in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")):
        title, seqs = AREAS[key]
        for s in seqs:
            draw_track(ax, xy[s], *ov[s][:2], lw_base=1.3, lw_hi=3.0)
        pts = np.vstack([xy[s] for s in seqs])
        span = (pts.max(0) - pts.min(0)).max()
        m = 0.08 * span
        ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
        ax.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)
        style(ax, title)
        for s in seqs:
            p0 = xy[s][0]
            ax.plot(*p0, "o", ms=6, mfc=SURFACE, mec=INK, mew=1.2, zorder=6)
            ax.annotate(s, p0, xytext=(5, 5), textcoords="offset points", fontsize=9,
                        color=INK, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        if key in NOTES:
            ax.text(0.98, 0.98, NOTES[key], transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=INK2, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="#dddcd8"))
        x0, x1 = ax.get_xlim()
        bar = {True: 200, False: 100}[(x1 - x0) > 600]
        scale_bar(ax, bar)

    # ── table panel (the chart's data, so nothing relies on colour alone) ──
    ax_tab.axis("off")
    ax_tab.set_title("Share of each trajectory that overlaps", loc="left", fontsize=10, color=INK, pad=6)
    rows = [["Seq", "Length", "Same", "Opp.", "Overlaps with"]]
    for s, p in xy.items():
        same, opp, partners = ov[s]
        L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / 1000
        others = sorted(partners - {s})
        who = ("itself" if s in partners else "") + (", " if s in partners and others else "") + ", ".join(others)
        rows.append([s, f"{L:.1f} km", f"{100*same.mean():.0f}%", f"{100*opp.mean():.0f}%", who or "—"])
    rows.append(["03", "—", "—", "—", "no published GPS"])
    tab = ax_tab.table(cellText=rows[1:], colLabels=rows[0], loc="upper left",
                       cellLoc="left", colLoc="left", bbox=[0.0, 0.08, 1.0, 0.9],
                       colWidths=[0.09, 0.16, 0.12, 0.13, 0.50])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#e6e5e1")
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if r == 0 or c == 0 else INK2)
        if r == 0:
            cell.get_text().set_fontweight("bold")

    handles = [Line2D([], [], color=TRACK, lw=1.5, label="Trajectory"),
               Line2D([], [], color=SAME, lw=3, label="Overlap, same direction"),
               Line2D([], [], color=OPP, lw=3, label="Overlap, opposite direction")]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.006, 0.935), labelcolor=INK, handlelength=2.2)
    fig.subplots_adjust(top=0.86)
    fig.suptitle("KITTI odometry trajectories in one GPS frame: where the vehicle drives a road again",
                 x=0.012, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.012, 0.945,
             f"Overlap = within {args.radius:g} m of another sequence, or of the same sequence "
             f"≥{args.min_gap/10:g} s later. Circles mark sequence start points.",
             fontsize=9, color=INK2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
