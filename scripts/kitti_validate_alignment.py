#!/usr/bin/env python3
"""Validate a cross-sequence ground-truth alignment from kitti_align_sequences.py.

The transform T (target <- source) is only trustworthy as ground truth if:
  1. reproducible  : ICP from the GPS init reproduces the saved T_final
  2. unique basin  : ICP restarts from perturbed starts all land on the same solution
                     (a straight street could otherwise let ICP slide along the road)
  3. small residual: after refinement the two LiDAR maps coincide in the overlap
  4. ground agrees : the ground surface of both maps is at the same height in shared
                     cells (independent of ICP's objective weighting, and unambiguous)
  5. on the road   : the two trajectories pass through the same place, and the
                     refinement does not change road angles (a rigid correction of
                     a few tenths of a degree cannot; crossing roads are allowed)
It also reports how far the LiDAR step moved the data relative to GPS alone,
measured AT THE DATA (a small rotation about a far origin inflates |t|).

  python3 scripts/kitti_validate_alignment.py --kitti_root kitti --align_dir kitti/align_07_to_00 \
      --target 00 --source 07 --target_lidar 2590-2900 --source_lidar 0-360 \
      --target_window 2400-2950 --source_window 0-550
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import kitti_align_sequences as ka

BASIN_MAX_M, BASIN_MAX_DEG = 0.10, 0.20
GROUND_MAX_M = 0.20        # |median ground height difference| after refinement
APPROACH_MAX_M = 1.0       # closest approach of the two camera trajectories
ANGLE_CHANGE_MAX_DEG = 2.0  # refinement may not change road angles by more than this
ICP_SCHEDULE = (2.0, 1.0, 0.5, 0.25)


def rng_of(spec):
    a, b = map(int, spec.split("-"))
    return a, b


def icp(o3d, src, tgt, T):
    reg = o3d.pipelines.registration
    for dist in ICP_SCHEDULE:
        r = reg.registration_icp(src, tgt, dist, T, reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(max_iteration=100))
        T = r.transformation
    return T, r


def at(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def ground_heights(P, up, cell=2.0):
    """Lowest point per horizontal cell, as height along `up`."""
    h = P @ up
    horiz = P - np.outer(h, up)
    keys = np.floor(horiz / cell).astype(np.int64)
    out = {}
    for k, z in zip(map(tuple, keys), h):
        if k not in out or z < out[k]:
            out[k] = z
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--align_dir", type=Path, required=True)
    ap.add_argument("--target", required=True, help="target sequence, e.g. 00")
    ap.add_argument("--source", required=True, help="source sequence, e.g. 07")
    ap.add_argument("--target_lidar", required=True, metavar="FIRST-LAST")
    ap.add_argument("--source_lidar", required=True, metavar="FIRST-LAST")
    ap.add_argument("--target_window", required=True, metavar="FIRST-LAST", help="target robot frames")
    ap.add_argument("--source_window", required=True, metavar="FIRST-LAST", help="source robot frames")
    ap.add_argument("--lidar_stride", type=int, default=2)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--perturb_m", type=float, default=2.0)
    ap.add_argument("--perturb_deg", type=float, default=2.0)
    args = ap.parse_args()
    import open3d as o3d

    J = json.loads((args.align_dir / "transform.json").read_text())
    T_init, T_final = np.array(J["T_init_gps"]), np.array(J["T_final"])
    tgt, src = ka.Sequence(args.kitti_root, args.target), ka.Sequence(args.kitti_root, args.source)
    tl, sl = rng_of(args.target_lidar), rng_of(args.source_lidar)
    map_t = tgt.lidar_map(range(tl[0], tl[1] + 1, args.lidar_stride))
    map_s = src.lidar_map(range(sl[0], sl[1] + 1, args.lidar_stride))  # source in its OWN frame
    pts_t, pts_s = np.asarray(map_t.points), np.asarray(map_s.points)
    c = pts_t.mean(0)
    up = -tgt.poses[(tl[0] + tl[1]) // 2, :3, 1]  # KITTI camera y points down
    warn = []

    # 1. reproducibility
    T0, r0 = icp(o3d, map_s, map_t, T_init)
    D = T0 @ np.linalg.inv(T_final)
    rep = np.linalg.norm(at(D, c[None])[0] - c)
    print(f"1. reproduce: differs from saved T_final by {rep:.3f} m at overlap centre, "
          f"{ka.rot_deg(D):.3f} deg (fitness {r0.fitness:.3f})")

    # 2. basin: perturb about the overlap centre so the rotation has no far lever arm
    rng = np.random.default_rng(0)
    C = np.eye(4); C[:3, 3] = c
    devs = []
    for _ in range(args.restarts):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        P = np.eye(4)
        P[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
            ax * np.radians(rng.uniform(-args.perturb_deg, args.perturb_deg)))
        P[:3, 3] = rng.uniform(-args.perturb_m, args.perturb_m, 3)
        T, _ = icp(o3d, map_s, map_t, C @ P @ np.linalg.inv(C) @ T_init)
        D = T @ np.linalg.inv(T0)
        devs.append((np.linalg.norm(at(D, c[None])[0] - c), ka.rot_deg(D)))
    devs = np.array(devs)
    print(f"2. basin ({args.restarts} restarts, +-{args.perturb_m:g} m / +-{args.perturb_deg:g} deg): "
          f"max deviation {devs[:, 0].max():.3f} m, {devs[:, 1].max():.3f} deg, median {np.median(devs[:, 0]):.3f} m")
    if devs[:, 0].max() > BASIN_MAX_M or devs[:, 1].max() > BASIN_MAX_DEG:
        warn.append(f"ICP restarts disagree (> {BASIN_MAX_M} m or {BASIN_MAX_DEG} deg): GT not unique")

    # what the LiDAR step changed w.r.t. GPS alone, measured at the data
    R = T_final @ np.linalg.inv(T_init)
    d = at(R, c[None])[0] - c
    tw = rng_of(args.target_window)
    Ct = tgt.poses[tw[0]:tw[1] + 1, :3, 3]
    along = tgt.poses[tl[1], :3, 3] - tgt.poses[tl[0], :3, 3]
    along -= up * (along @ up); along /= np.linalg.norm(along)
    lat = np.cross(up, along)
    dm = np.linalg.norm(at(R, pts_t) - pts_t, axis=1)
    dc = np.linalg.norm(at(R, Ct) - Ct, axis=1)
    print(f"   LiDAR refinement vs GPS only: at overlap centre {np.linalg.norm(d):.3f} m "
          f"(along {d @ along:+.3f}, lateral {d @ lat:+.3f}, vertical {d @ up:+.3f}) | "
          f"over overlap map median {np.median(dm):.3f} / max {dm.max():.3f} m | "
          f"at target robot cameras median {np.median(dc):.3f} / max {dc.max():.3f} m")

    # 3. residual after refinement, inside the overlap
    nn = np.asarray(o3d.geometry.PointCloud(map_s).transform(T_final).compute_point_cloud_distance(map_t))
    ov = nn[nn < 2.0]
    print(f"3. residual in overlap: NN median {np.median(ov):.3f} m, p75 {np.percentile(ov, 75):.3f} m, "
          f"{100 * np.mean(ov < 0.25):.1f}% < 0.25 m ({len(ov):,} pts)")

    # 4. ground surface agreement in shared cells
    gt = ground_heights(pts_t, up)
    for name, T in (("GPS only", T_init), ("LiDAR refined", T_final)):
        gs = ground_heights(at(T, pts_s), up)
        dz = np.array([gs[k] - gt[k] for k in gs if k in gt])
        med, mad = np.median(dz), np.median(np.abs(dz - np.median(dz)))
        print(f"4. ground height (source - target), {name:13s}: median {med:+.3f} m, MAD {mad:.3f} m "
              f"over {len(dz)} shared 2 m cells")
    if abs(med) > GROUND_MAX_M:
        warn.append(f"ground surfaces differ by {med:+.2f} m after refinement")

    # 5. trajectories meet, and the refinement leaves road angles unchanged
    sw = rng_of(args.source_window)
    fwd_t = tgt.poses[tw[0]:tw[1] + 1, :3, 2]
    src_T = np.tile(np.eye(4), (len(src.poses), 1, 1))
    src_T[:] = src.poses
    stats = {}
    for name, T in (("GPS only", T_init), ("LiDAR refined", T_final)):
        S = (T @ src_T)[sw[0]:sw[1] + 1]
        dist, j = cKDTree(Ct).query(S[:, :3, 3])
        head = np.degrees(np.arccos(np.clip(np.sum(S[:, :3, 2] * fwd_t[j], axis=1), -1, 1)))
        head = np.minimum(head, 180 - head)  # opposite-direction passes count as aligned
        near = dist < 10.0
        stats[name] = (dist.min(), np.median(head[near]) if near.any() else np.nan, int(near.sum()))
        print(f"5. trajectories, {name:13s}: closest approach {dist.min():.2f} m, frames within 10 m {near.sum()}, "
              f"road angle (direction-agnostic) median {stats[name][1]:.1f} deg")
    if stats["LiDAR refined"][0] > APPROACH_MAX_M:
        warn.append(f"trajectories never come within {APPROACH_MAX_M} m: windows may not overlap")
    if abs(stats["LiDAR refined"][1] - stats["GPS only"][1]) > ANGLE_CHANGE_MAX_DEG:
        warn.append("refinement changed road angles, which a small rigid correction cannot do")

    print("RESULT: " + ("PASS" if not warn else "WARN - " + "; ".join(warn)))


if __name__ == "__main__":
    main()
