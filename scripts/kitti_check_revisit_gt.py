#!/usr/bin/env python3
"""Check whether KITTI's GT poses are consistent between two passes of one sequence.

Within-sequence multi-robot pairs assume the GT transform between the robots'
maps is identity, because both use the same sequence poses. That only holds if
the GT trajectory itself is consistent where the vehicle drives the same road
twice. KITTI GT comes from GPS/INS, so the second pass can be offset from the
first. This script measures that offset with LiDAR:

  both passes' Velodyne scans are placed with the GT poses; point-to-plane ICP
  starts at identity. The refinement ICP needs is the GT inconsistency.

Reports, like kitti_validate_alignment.py: basin (uniqueness), displacement AT THE
DATA (along / lateral / vertical at the overlap centre), ground-height agreement
before/after, and residual.

  python3 scripts/kitti_check_revisit_gt.py --kitti_root kitti --seq 00 \
      --pass_a 559-952 --pass_b 3522-3851 --out kitti/revisit_gt/kitti00.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import kitti_align_sequences as ka
from kitti_validate_alignment import at, ground_heights, icp


def rng_of(spec):
    a, b = map(int, spec.split("-"))
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--pass_a", required=True, metavar="FIRST-LAST", help="first pass (target) frames")
    ap.add_argument("--pass_b", required=True, metavar="FIRST-LAST", help="second pass (source) frames")
    ap.add_argument("--lidar_stride", type=int, default=2)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--out", type=Path, help="write the correction as JSON")
    args = ap.parse_args()
    import open3d as o3d

    s = ka.Sequence(args.kitti_root, args.seq)
    pa, pb = rng_of(args.pass_a), rng_of(args.pass_b)
    map_a = s.lidar_map(range(pa[0], pa[1] + 1, args.lidar_stride))
    map_b = s.lidar_map(range(pb[0], pb[1] + 1, args.lidar_stride))
    pts_a, pts_b = np.asarray(map_a.points), np.asarray(map_b.points)
    up = -s.poses[(pa[0] + pa[1]) // 2, :3, 1]

    nn0 = np.asarray(map_b.compute_point_cloud_distance(map_a))
    T, r = icp(o3d, map_b, map_a, np.eye(4))
    ov_mask = np.asarray(o3d.geometry.PointCloud(map_b).transform(T).compute_point_cloud_distance(map_a)) < 0.5
    c = pts_b[ov_mask].mean(0) if ov_mask.sum() > 100 else pts_b.mean(0)

    rng = np.random.default_rng(0)
    C = np.eye(4); C[:3, 3] = c
    devs = []
    for _ in range(args.restarts):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        P = np.eye(4)
        P[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(ax * np.radians(rng.uniform(-2, 2)))
        P[:3, 3] = rng.uniform(-2, 2, 3)
        T2, _ = icp(o3d, map_b, map_a, C @ P @ np.linalg.inv(C))
        D = T2 @ np.linalg.inv(T)
        devs.append((np.linalg.norm(at(D, c[None])[0] - c), ka.rot_deg(D)))
    devs = np.array(devs)

    d = at(T, c[None])[0] - c
    along = s.poses[pa[1], :3, 3] - s.poses[pa[0], :3, 3]
    along -= up * (along @ up); along /= np.linalg.norm(along)
    lat = np.cross(up, along)
    disp = np.linalg.norm(at(T, pts_b[ov_mask]) - pts_b[ov_mask], axis=1)
    nn1 = np.asarray(o3d.geometry.PointCloud(map_b).transform(T).compute_point_cloud_distance(map_a))

    ga = ground_heights(pts_a, up)
    gz = {}
    for name, TT in (("GT poses", np.eye(4)), ("LiDAR-corrected", T)):
        gb = ground_heights(at(TT, pts_b), up)
        dz = np.array([gb[k] - ga[k] for k in gb if k in ga])
        gz[name] = (float(np.median(dz)), len(dz))

    print(f"seq {args.seq}: pass B {args.pass_b} vs pass A {args.pass_a} "
          f"({len(pts_b):,} / {len(pts_a):,} LiDAR points, 0.2 m voxels)")
    print(f"  ICP from GT (identity): fitness@0.25 m {r.fitness:.3f}, inlier RMSE {r.inlier_rmse:.3f} m; "
          f"basin {args.restarts} restarts max deviation {devs[:, 0].max():.3f} m / {devs[:, 1].max():.3f} deg")
    print(f"  GT INCONSISTENCY at overlap centre: {np.linalg.norm(d):.3f} m "
          f"(along {d @ along:+.3f}, lateral {d @ lat:+.3f}, vertical {d @ up:+.3f}), rotation {ka.rot_deg(T):.3f} deg; "
          f"over overlap points median {np.median(disp):.3f} / max {disp.max():.3f} m")
    print(f"  ground height (B - A): GT poses {gz['GT poses'][0]:+.3f} m -> corrected {gz['LiDAR-corrected'][0]:+.3f} m "
          f"({gz['GT poses'][1]} cells)")
    print(f"  NN distance B->A within 2 m: GT poses median {np.median(nn0[nn0 < 2]):.3f} m -> "
          f"corrected {np.median(nn1[nn1 < 2]):.3f} m")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "seq": args.seq, "pass_a": args.pass_a, "pass_b": args.pass_b,
            "T_correction_B_to_A": T.tolist(), "fitness": r.fitness, "inlier_rmse": r.inlier_rmse,
            "offset_at_overlap_m": float(np.linalg.norm(d)),
            "offset_along_lateral_vertical_m": [float(d @ along), float(d @ lat), float(d @ up)],
            "rotation_deg": ka.rot_deg(T), "basin_max_m": float(devs[:, 0].max()),
            "ground_dz_gt_m": gz["GT poses"][0], "ground_dz_corrected_m": gz["LiDAR-corrected"][0],
        }, indent=2))


if __name__ == "__main__":
    main()
