#!/usr/bin/env python3
"""Find KITTI odometry sequences that drive through the same places (GPS/OXTS).

Odometry GT poses start at each sequence's own origin, so sequence-to-sequence
overlap can only be found in a frame common to all drives: the raw drives'
OXTS GPS, projected with the KITTI devkit's Mercator formula using ONE shared
reference latitude.

  python3 scripts/kitti_gps_overlap.py --kitti_root kitti [--radius 10] [--plot tracks.png]

Expects <kitti_root>/<date>/<drive>_sync/oxts/data/*.txt (e.g. from
kitti_remote_zip.py --glob '*/oxts/*'). Reports, per pair of sequences, the
share of each trajectory within --radius of the other and where.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# odometry sequence -> (raw drive, first raw frame, last raw frame); 03 is not published
SEQUENCES = {
    "00": ("2011_10_03_drive_0027", 0, 4540), "01": ("2011_10_03_drive_0042", 0, 1100),
    "02": ("2011_10_03_drive_0034", 0, 4660), "04": ("2011_09_30_drive_0016", 0, 270),
    "05": ("2011_09_30_drive_0018", 0, 2760), "06": ("2011_09_30_drive_0020", 0, 1100),
    "07": ("2011_09_30_drive_0027", 0, 1100), "08": ("2011_09_30_drive_0028", 1100, 5170),
    "09": ("2011_09_30_drive_0033", 0, 1590), "10": ("2011_09_30_drive_0034", 0, 1200),
}
EARTH_RADIUS = 6378137.0


def load_latlon(root, drive, first, last):
    d = root / drive[:10] / f"{drive}_sync" / "oxts" / "data"
    rows = [np.loadtxt(d / f"{i:010d}.txt", max_rows=1)[:2] for i in range(first, last + 1)]
    return np.array(rows)  # (N, 2) lat, lon in degrees


def mercator(latlon, scale):
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = scale * lon * np.pi * EARTH_RADIUS / 180.0
    y = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return np.stack([x, y], 1)


def runs(frames):
    if not len(frames):
        return "-"
    out, s, p = [], frames[0], frames[0]
    for k in frames[1:]:
        if k - p > 20:
            out.append((s, p))
            s = k
        p = k
    out.append((s, p))
    return ", ".join(f"{a}-{b}" for a, b in out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=10.0, help="GPS-level proximity, metres")
    ap.add_argument("--plot", type=Path, help="write a PNG of all tracks")
    args = ap.parse_args()

    latlon = {}
    for seq, (drive, first, last) in SEQUENCES.items():
        try:
            latlon[seq] = load_latlon(args.kitti_root, drive, first, last)
        except (FileNotFoundError, OSError) as e:
            print(f"{seq}: skipped ({e})")
    scale = np.cos(np.radians(np.mean([ll[:, 0].mean() for ll in latlon.values()])))
    xy = {seq: mercator(ll, scale) for seq, ll in latlon.items()}
    origin = np.min([p.min(0) for p in xy.values()], axis=0)
    xy = {seq: p - origin for seq, p in xy.items()}

    for seq, p in xy.items():
        length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        print(f"{seq} {SEQUENCES[seq][0]}  {len(p):5d} frames  {length:6.0f} m  "
              f"centre ({p[:, 0].mean():6.0f}, {p[:, 1].mean():6.0f}) m")

    print(f"\npairs sharing road (frames within {args.radius:g} m of the other sequence):")
    found = False
    for a, b in combinations(sorted(xy), 2):
        da, _ = cKDTree(xy[b]).query(xy[a])
        db, _ = cKDTree(xy[a]).query(xy[b])
        na, nb = np.flatnonzero(da < args.radius), np.flatnonzero(db < args.radius)
        if len(na) < 20 and len(nb) < 20:
            continue
        found = True
        print(f"  {a} <-> {b}:  {a} {100 * len(na) / len(da):4.1f}% at frames {runs(na)}  |  "
              f"{b} {100 * len(nb) / len(db):4.1f}% at frames {runs(nb)}")
    if not found:
        print("  none")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        for seq, p in sorted(xy.items()):
            ax.plot(p[:, 0], p[:, 1], lw=1, label=seq)
            ax.annotate(seq, p[0], fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(title="sequence", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
