#!/usr/bin/env python3
"""Build a Velodyne map per KITTI multi-robot window, in the same frame as the robot's 3DGS map.

Sanity baseline for KISS-Matcher: it scores 0 % on the Gaussian maps, so run it on
LiDAR maps of the same windows (same poses, same frame, same GT). Scans are placed
with the poses the robot's COLMAP model was built from (cross-sequence robots use the
LiDAR-aligned pose file), so the benchmark GT and score_submap_csv.py scoring
(gt_cameras, revisit_gt corrections) apply unchanged. Each map is checked against the
robot's cameras.json before writing.

Output: <out>/<scene>/<robot>/point_cloud/iteration_30000/point_cloud.ply (xyz only),
the path layout benchmark_submap expects.

  python3 scripts/kitti_lidar_maps.py [--out $DATA_ROOT/lidar_maps] [SCENE ...]
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np

import kitti_align_sequences as ka

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
K = DATA / "kitti"
# scene -> {robot: (sequence, first, last, pose file or None = sequence GT)}
ROBOTS = {
    "kitti06": {"robot_a": ("06", 0, 500, None), "robot_b": ("06", 600, 1100, None)},
    "kitti05": {"robot_a": ("05", 300, 1000, None), "robot_b": ("05", 1100, 1800, None)},
    "kitti00": {"robot_a": ("00", 450, 1050, None), "robot_b": ("00", 3450, 3950, None)},
    "kitti02": {"robot_a": ("02", 1700, 2150, None), "robot_b": ("02", 4300, 4660, None)},
    "kitti08": {"robot_a": ("08", 0, 600, None), "robot_b": ("08", 1300, 1900, None)},
    "kitti00_07": {"robot_00": ("00", 2400, 2950, None),
                   "robot_07": ("07", 0, 550, K / "align_07_to_00/07_in_00.txt")},
    "kitti09_10": {"robot_09": ("09", 1050, 1240, None),
                   "robot_10": ("10", 550, 780, K / "align_10_to_09/10_in_09.txt")},
}


def write_ply(path, pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n".encode())
        f.write(pts.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "lidar_maps")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--max_range", type=float, default=40.0)
    ap.add_argument("scenes", nargs="*", default=list(ROBOTS))
    args = ap.parse_args()

    for scene in args.scenes:
        robots = ROBOTS[scene]
        for robot, (seq, first, last, pose_file) in robots.items():
            s = ka.Sequence(K, seq)
            if pose_file:
                s.poses = ka.read_poses(pose_file)
            pts = np.asarray(s.lidar_map(range(first, last + 1, args.stride),
                                         voxel=args.voxel, max_range=args.max_range).points)
            # frame check: robot's training cameras (cam2/cam3) sit ~0.06/0.48 m from cam0
            cams = json.load(open(DATA / "trained_gaussians" / scene / robot / "cameras.json"))
            d = [np.linalg.norm(np.array(c["position"]) - s.poses[int(c["img_name"][:6])][:3, 3]) for c in cams]
            if max(d) > 1.0:
                raise SystemExit(f"{scene}/{robot}: LiDAR poses disagree with cameras.json by {max(d):.2f} m")
            write_ply(args.out / scene / robot / "point_cloud/iteration_30000/point_cloud.ply", pts)
            print(f"{scene}/{robot}: {len(pts):,} pts, cameras within {max(d):.2f} m", flush=True)


if __name__ == "__main__":
    main()
