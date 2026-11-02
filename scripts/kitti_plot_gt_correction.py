#!/usr/bin/env python3
"""Show the KITTI ground-truth corrections on the raw LiDAR: before vs after.

For every pair whose GT was LiDAR-corrected, the two drives' Velodyne scans are
accumulated with their GT poses and drawn together, once with the uncorrected GT
and once with the correction applied:

  revisit pairs (kitti/revisit_gt/<scene>.json):  before = KITTI GT (identity),
                                                   after  = T_correction_B_to_A
  cross-sequence (kitti/align_<src>_to_<tgt>):     before = GPS/IMU fit (T_init_gps),
                                                   after  = GPS + LiDAR ICP (T_final)

Rows: top-down around the overlap, a slice ALONG the road and a slice ACROSS it
(both through the overlap centre, in a road-aligned along / lateral / up frame).
A misaligned GT shows up as doubled walls and poles and two ground planes.

--view opens an interactive 3D viewer per pair with both states side by side
(before left, after right, same camera); --ply writes them for CloudCompare.

  python3 scripts/kitti_plot_gt_correction.py --kitti_root $DATA_ROOT/kitti \
      --out $DATA_ROOT/results/figures/gt_correction [--only kitti02 --view]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kitti_align_sequences as ka
from kitti_validate_alignment import at

# name, JSON (relative to kitti_root), target seq:frames, source seq:frames
REVISIT = ["00", "02", "05", "06", "08"]
CROSS = [("00_07", "align_07_to_00", "00:2590-2900", "07:0-360"),
         ("09_10", "align_10_to_09", "09:1050-1250", "10:550-780")]
COL_A, COL_B = "#5b5b5b", "#d62728"


def frames(spec, stride):
    seq, r = spec.split(":")
    a, b = map(int, r.split("-"))
    return seq, range(a, b + 1, stride)


def load_pairs(root, stride):
    for s in REVISIT:
        j = json.loads((root / "revisit_gt" / f"kitti{s}.json").read_text())
        yield dict(name=f"seq {s} revisit", key=f"kitti{s}",
                   a=(s, frames(f"{s}:{j['pass_a']}", stride)[1]),
                   b=(s, frames(f"{s}:{j['pass_b']}", stride)[1]),
                   label_a=f"pass A ({j['pass_a']})", label_b=f"pass B ({j['pass_b']})",
                   before=np.eye(4), after=np.array(j["T_correction_B_to_A"]),
                   before_name="KITTI GT poses", after_name="LiDAR-corrected GT")
    for key, d, tgt, src in CROSS:
        j = json.loads((root / d / "transform.json").read_text())
        yield dict(name=f"seq {src[:2]} -> seq {tgt[:2]}", key=f"kitti{key}",
                   a=frames(tgt, stride), b=frames(src, stride),
                   label_a=f"seq {tgt[:2]} LiDAR", label_b=f"seq {src[:2]} LiDAR",
                   before=np.array(j["T_init_gps"]), after=np.array(j["T_final"]),
                   before_name="GPS/IMU only", after_name="GPS + LiDAR ICP")


def cloud(root, seq, fr):
    s = ka.Sequence(root, seq)
    have = [f for f in fr if (s.raw / "velodyne_points" / "data" / f"{s.raw_first + f:010d}.bin").exists()]
    pc = s.lidar_map(have)
    return s, np.asarray(pc.points), pc


def ground_offset(pa, pb, cell=2.0):
    """Median over shared cells of (lowest B point - lowest A point); pts in (along, lat, up)."""
    def low(P):
        k = np.floor(P[:, :2] / cell).astype(np.int64)
        order = np.lexsort((P[:, 2], k[:, 1], k[:, 0]))
        k, z = k[order], P[order, 2]
        first = np.ones(len(k), bool)
        first[1:] = np.any(k[1:] != k[:-1], axis=1)
        return {tuple(x): h for x, h in zip(k[first], z[first])}
    la, lb = low(pa), low(pb)
    dz = [lb[c] - la[c] for c in lb if c in la]
    return float(np.median(dz)) if dz else float("nan")


def plot_pair(root, p, out, radius, slab):
    import open3d as o3d
    sa, A, pc_a = cloud(root, *p["a"])
    _, B, pc_b = cloud(root, *p["b"])

    # overlap centre (under the corrected GT) and a road-aligned frame there
    Bc = at(p["after"], B)
    d = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Bc)).compute_point_cloud_distance(pc_a))
    c = Bc[d < 0.5].mean(0)
    cams = sa.poses[list(p["a"][1]), :3, 3]
    i = np.argmin(np.linalg.norm(cams - c, axis=1))
    up = -sa.poses[list(p["a"][1])[i], :3, 1]
    fwd = sa.poses[list(p["a"][1])[i], :3, 2]
    fwd -= up * (fwd @ up); fwd /= np.linalg.norm(fwd)
    lat = np.cross(up, fwd)
    c = cams[i]  # view centred on the road: pass-A camera nearest the overlap centroid
    R = np.stack([fwd, lat, up])  # rows: along, lateral, up
    road = lambda P: (P - c) @ R.T

    a_loc = road(A)
    views = [(p["before_name"], road(at(p["before"], B))), (p["after_name"], road(Bc))]
    ov = Bc[d < 0.5]  # displacement of B's overlap points between before and after
    shift = np.median(np.linalg.norm(ov - at(p["before"] @ np.linalg.inv(p["after"]), ov), axis=1))

    fig, ax = plt.subplots(3, 2, figsize=(13, 11.5), gridspec_kw=dict(height_ratios=[2.2, 0.6, 0.6]))
    rng_np = np.random.default_rng(0)
    for col, (title, b_loc) in enumerate(views):
        gz = ground_offset(a_loc[np.linalg.norm(a_loc[:, :2], axis=1) < radius],
                           b_loc[np.linalg.norm(b_loc[:, :2], axis=1) < radius])
        panels = [  # (mask fn, x idx, y idx, xlabel, ylabel, title)
            (lambda P: np.linalg.norm(P[:, :2], axis=1) < radius, 1, 0,
             "lateral (m)", "along road (m)", f"top-down, {radius:.0f} m around overlap"),
            (lambda P: (np.abs(P[:, 1]) < slab / 2) & (np.abs(P[:, 0]) < radius), 0, 2,
             "along road (m)", "up (m)", f"slice ALONG the road ({slab:.0f} m thick)"),
            (lambda P: (np.abs(P[:, 0]) < slab / 2) & (np.abs(P[:, 1]) < radius), 1, 2,
             "lateral (m)", "up (m)", f"slice ACROSS the road ({slab:.0f} m thick)"),
        ]
        for row, (m, xi, yi, xl, yl, t) in enumerate(panels):
            a = ax[row, col]
            for P, colr, lab, s in ((a_loc, COL_A, p["label_a"], 0.6), (b_loc, COL_B, p["label_b"], 0.6)):
                Q = P[m(P)]
                if row == 0 and len(Q) > 250_000:
                    Q = Q[rng_np.choice(len(Q), 250_000, replace=False)]
                a.scatter(Q[:, xi], Q[:, yi], s=s if row == 0 else 1.2, c=colr, lw=0,
                          alpha=0.5 if row == 0 else 0.8, label=lab, rasterized=True)
            a.set_aspect("equal")
            a.set_xlabel(xl); a.set_ylabel(yl)
            a.set_title(f"{title}: {t}", fontsize=10)
            if row == 0:
                a.legend(loc="upper right", markerscale=12, fontsize=9)
                a.text(0.02, 0.02, f"ground height B-A: {gz:+.2f} m", transform=a.transAxes,
                       fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.8))
    # same height range before/after, robust to stray points (from pass A in the slices)
    for row, m in ((1, lambda P: (np.abs(P[:, 1]) < slab / 2) & (np.abs(P[:, 0]) < radius)),
                   (2, lambda P: (np.abs(P[:, 0]) < slab / 2) & (np.abs(P[:, 1]) < radius))):
        z = np.concatenate([a_loc[m(a_loc), 2]] + [b[m(b), 2] for _, b in views])
        lo, hi = np.percentile(z, [0.5, 99.5])
        for k in (0, 1):
            ax[row, k].set_ylim(lo - 0.5, hi + 0.5)
    fig.suptitle(f"{p['name']}: correction moves B's overlap points by {shift:.2f} m (median)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{p['key']}_gt_correction.png", dpi=130)
    plt.close(fig)
    return shift, a_loc, views


def side_by_side(a_loc, views, radius, gap=10.0):
    """One open3d cloud: 'before' on the left, 'after' shifted right; x lateral, y along, z up."""
    import open3d as o3d
    pts, cols = [], []
    for k, (_, b_loc) in enumerate(views):
        for P, hexc in ((a_loc, COL_A), (b_loc, COL_B)):
            Q = P[np.linalg.norm(P[:, :2], axis=1) < radius][:, [1, 0, 2]]
            Q[:, 0] += k * (2 * radius + gap)
            pts.append(Q)
            cols.append(np.tile(matplotlib.colors.to_rgb(hexc), (len(Q), 1)))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts)))
    pc.colors = o3d.utility.Vector3dVector(np.vstack(cols))
    return pc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="keys to plot, e.g. kitti00 kitti09_10")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--slab", type=float, default=2.0)
    ap.add_argument("--view", action="store_true",
                    help="open a 3D viewer per pair: before (left) | after (right), grey = A, red = B")
    ap.add_argument("--ply", action="store_true", help="also write <key>_before.ply / <key>_after.ply")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for p in load_pairs(args.kitti_root, args.stride):
        if args.only and p["key"] not in args.only:
            continue
        shift, a_loc, views = plot_pair(args.kitti_root, p, args.out, args.radius, args.slab)
        print(f"{p['key']}: {shift:.3f} m at overlap -> {args.out / (p['key'] + '_gt_correction.png')}")
        if args.ply or args.view:
            import open3d as o3d
        if args.ply:
            for k, tag in enumerate(("before", "after")):
                pc = side_by_side(a_loc, [views[k]], args.radius)
                o3d.io.write_point_cloud(str(args.out / f"{p['key']}_{tag}.ply"), pc)
        if args.view:
            print(f"  viewer: LEFT = {views[0][0]}, RIGHT = {views[1][0]} (close the window for the next pair)")
            o3d.visualization.draw_geometries(
                [side_by_side(a_loc, views, args.radius)],
                window_name=f"{p['name']}:  LEFT {views[0][0]}  |  RIGHT {views[1][0]}", width=1600, height=900)


if __name__ == "__main__":
    main()
