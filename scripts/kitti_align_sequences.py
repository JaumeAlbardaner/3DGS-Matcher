#!/usr/bin/env python3
"""Ground-truth transform between two KITTI odometry sequences (target <- source).

Each sequence's GT poses live in its own frame (its first camera). To match maps
from two sequences with "identity is the correct answer", the source sequence's
poses are re-expressed in the target sequence's frame:

  1. GPS/IMU init   : IMU positions implied by the GT poses (T_world_cam0 * T_cam0_imu)
                      are rigidly fitted (Umeyama) to OXTS positions in a Mercator
                      frame shared by both drives, over each sequence's window only
                      (keeps GT drift of long sequences out of the fit).
  2. LiDAR refine   : Velodyne scans over the overlap are accumulated per sequence
                      with its own GT poses and aligned with coarse-to-fine
                      point-to-plane ICP. Fitness / inlier RMSE are the GT's accuracy.
  3. Output         : JSON with both transforms + stats, and the source poses in the
                      target frame (drop-in --poses file for kitti_prepare.py).

  python3 scripts/kitti_align_sequences.py --kitti_root kitti \
      --target 00:2400-2950 --source 07:0-550 \
      --target_lidar 2590-2900 --source_lidar 0-360 --out kitti/align_07_to_00
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kitti_gps_overlap import SEQUENCES, mercator


def read_poses(path):
    P = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3] = P
    return T


def read_calib_tr(path):
    for line in open(path):
        if line.startswith("Tr:"):
            T = np.eye(4)
            T[:3] = np.array(line.split()[1:], float).reshape(3, 4)
            return T  # velo -> cam0
    raise ValueError(f"no Tr in {path}")


def read_imu_to_velo(path):
    vals = {}
    for line in open(path):
        key, _, rest = line.partition(":")
        if key in ("R", "T"):
            vals[key] = np.array(rest.split(), float)
    T = np.eye(4)
    T[:3, :3] = vals["R"].reshape(3, 3)
    T[:3, 3] = vals["T"]
    return T  # imu -> velo


def umeyama_rigid(src, dst):
    """R, t minimising |R src + t - dst|."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = md - R @ ms
    return T


class Sequence:
    def __init__(self, root, seq):
        self.seq = seq
        self.drive, self.raw_first, _ = SEQUENCES[seq]
        self.raw = root / self.drive[:10] / f"{self.drive}_sync"
        self.poses = read_poses(root / "dataset" / "poses" / f"{seq}.txt")  # T_seq_cam0
        self.T_cam0_velo = read_calib_tr(root / "dataset" / "sequences" / seq / "calib.txt")
        T_velo_imu = read_imu_to_velo(root / self.drive[:10] / "calib_imu_to_velo.txt")
        self.T_cam0_imu = self.T_cam0_velo @ T_velo_imu

    def latlonalt(self, frames):
        d = self.raw / "oxts" / "data"
        return np.array([np.loadtxt(d / f"{self.raw_first + f:010d}.txt", max_rows=1)[:3] for f in frames])

    def imu_positions(self, frames):
        return (self.poses[frames] @ self.T_cam0_imu)[:, :3, 3]

    def lidar_map(self, frames, T_extra=np.eye(4), voxel=0.2, max_range=40.0):
        import open3d as o3d
        clouds = []
        for f in frames:
            pts = np.fromfile(self.raw / "velodyne_points" / "data" / f"{self.raw_first + f:010d}.bin",
                              np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
            r = np.linalg.norm(pts, axis=1)
            pts = pts[(r > 3.0) & (r < max_range)]
            T = T_extra @ self.poses[f] @ self.T_cam0_velo
            clouds.append(pts @ T[:3, :3].T + T[:3, 3])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(clouds)))
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
        return pc


def parse_window(spec):
    seq, rng = spec.split(":")
    a, b = map(int, rng.split("-"))
    return seq, np.arange(a, b + 1)


def rot_deg(T):
    return np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti_root", type=Path, required=True)
    ap.add_argument("--target", required=True, metavar="SEQ:FIRST-LAST", help="GPS-fit window")
    ap.add_argument("--source", required=True, metavar="SEQ:FIRST-LAST", help="GPS-fit window")
    ap.add_argument("--target_lidar", required=True, metavar="FIRST-LAST")
    ap.add_argument("--source_lidar", required=True, metavar="FIRST-LAST")
    ap.add_argument("--lidar_stride", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import open3d as o3d

    (tseq, twin), (sseq, swin) = parse_window(args.target), parse_window(args.source)
    tgt, src = Sequence(args.kitti_root, tseq), Sequence(args.kitti_root, sseq)

    # 1. GPS/IMU init in one Mercator frame
    lla_t, lla_s = tgt.latlonalt(twin), src.latlonalt(swin)
    scale = np.cos(np.radians(np.r_[lla_t[:, 0], lla_s[:, 0]].mean()))
    origin = mercator(lla_t[:1, :2], scale)[0]
    def world(lla):
        return np.c_[mercator(lla[:, :2], scale) - origin, lla[:, 2] - lla_t[0, 2]]
    fits = {}
    for name, s, win, lla in (("target", tgt, twin, lla_t), ("source", src, swin, lla_s)):
        p_seq, p_w = s.imu_positions(win), world(lla)
        T_w_seq = umeyama_rigid(p_seq, p_w)
        res = np.linalg.norm(p_seq @ T_w_seq[:3, :3].T + T_w_seq[:3, 3] - p_w, axis=1)
        fits[name] = T_w_seq
        print(f"GPS fit {name} seq {s.seq} ({len(win)} frames): residual median {np.median(res):.2f} m, "
              f"p90 {np.percentile(res, 90):.2f} m, max {res.max():.2f} m")
    T_init = np.linalg.inv(fits["target"]) @ fits["source"]
    print(f"T_init ({tseq} <- {sseq}): rotation {rot_deg(T_init):.2f} deg, |t| {np.linalg.norm(T_init[:3, 3]):.2f} m")

    # 2. LiDAR refinement
    tl0, tl1 = map(int, args.target_lidar.split("-"))
    sl0, sl1 = map(int, args.source_lidar.split("-"))
    map_t = tgt.lidar_map(range(tl0, tl1 + 1, args.lidar_stride))
    map_s = src.lidar_map(range(sl0, sl1 + 1, args.lidar_stride), T_extra=T_init)
    print(f"LiDAR maps: target {len(map_t.points):,} pts, source {len(map_s.points):,} pts (0.2 m voxels)")
    T_ref = np.eye(4)
    for dist in (2.0, 1.0, 0.5, 0.25):
        reg = o3d.pipelines.registration.registration_icp(
            map_s, map_t, dist, T_ref,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T_ref = reg.transformation
        print(f"  ICP @ {dist:4.2f} m: fitness {reg.fitness:.3f}, inlier RMSE {reg.inlier_rmse:.3f} m")
    before = o3d.pipelines.registration.evaluate_registration(map_s, map_t, 0.25, np.eye(4))
    T_final = T_ref @ T_init
    print(f"refinement: rotation {rot_deg(T_ref):.3f} deg, |t| {np.linalg.norm(T_ref[:3, 3]):.3f} m | "
          f"fitness@0.25 m {before.fitness:.3f} (GPS only) -> {reg.fitness:.3f} (LiDAR refined)")

    # 3. outputs
    args.out.mkdir(parents=True, exist_ok=True)
    poses_in_target = T_final @ src.poses
    np.savetxt(args.out / f"{sseq}_in_{tseq}.txt", poses_in_target[:, :3].reshape(-1, 12), fmt="%.9e")
    stats = {
        "convention": f"p_{tseq} = T @ p_{sseq} (4x4, row-major)",
        "target": args.target, "source": args.source,
        "T_init_gps": T_init.tolist(), "T_final": T_final.tolist(),
        "refinement_rot_deg": rot_deg(T_ref), "refinement_trans_m": float(np.linalg.norm(T_ref[:3, 3])),
        "icp_fitness_0.25m": reg.fitness, "icp_inlier_rmse_m": reg.inlier_rmse,
        "fitness_0.25m_gps_only": before.fitness,
    }
    (args.out / "transform.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out / 'transform.json'} and {args.out / f'{sseq}_in_{tseq}.txt'}")


if __name__ == "__main__":
    main()
