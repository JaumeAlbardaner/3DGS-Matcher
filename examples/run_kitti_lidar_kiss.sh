#!/usr/bin/env bash
# KISS-Matcher on LiDAR maps of the KITTI multi-robot windows (sanity baseline).
#
# Same benchmark_submap call as run_kitti_robots.sh (0.5 m voxel, ±180° / ±500 m,
# no ICP), on the Velodyne maps from scripts/kitti_lidar_maps.py instead of the
# Gaussian maps. Only the vanilla_kiss rows matter; color_cpfh also runs but sees
# constant grey colour.
#
#   N_SEEDS=100 examples/run_kitti_lidar_kiss.sh [SCENE ...]
# Results: results/<SCENE>_robots_lidar_v050.csv; score with scripts/kitti_lidar_kiss_score.py
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$SCRIPT_DIR/../build/benchmark_submap"
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
PLY=point_cloud/iteration_30000/point_cloud.ply
N_SEEDS=${N_SEEDS:-100}
declare -A PAIR=([kitti06]="robot_b robot_a" [kitti05]="robot_b robot_a" [kitti00]="robot_b robot_a"
                 [kitti02]="robot_b robot_a" [kitti08]="robot_b robot_a"
                 [kitti00_07]="robot_07 robot_00" [kitti09_10]="robot_10 robot_09")
SCENES=("$@")
[[ $# -eq 0 ]] && SCENES=(kitti06 kitti05 kitti00 kitti02 kitti08 kitti00_07 kitti09_10)

for scene in "${SCENES[@]}"; do
  read -r src tgt <<< "${PAIR[$scene]}"
  out="$DATA_ROOT/results/${scene}_robots_lidar_v050.csv"
  [[ -e "$out" ]] && { echo "refusing to overwrite $out"; exit 1; }
  echo "=== $scene: $src -> $tgt ($(date +%T))"
  echo "method,scene,split_src,split_tgt,iter,seed,rot_error_deg,trans_error_m,success,solver_ok,tx,ty,tz,qw,qx,qy,qz,time_ms" > "$out"
  for ((seed = 0; seed < N_SEEDS; seed++)); do
    OMP_NUM_THREADS=8 "$BENCH" "$DATA_ROOT/lidar_maps/$scene/$src/$PLY" "$DATA_ROOT/lidar_maps/$scene/$tgt/$PLY" \
      0.5 180 500 "$seed" "$out" 1.0 0 2>/dev/null
  done
done
echo DONE
