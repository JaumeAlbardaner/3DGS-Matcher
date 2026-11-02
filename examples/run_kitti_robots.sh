#!/usr/bin/env bash
# run_kitti_robots.sh SCENE SRC_ROBOT TGT_ROBOT
#
# One KITTI multi-robot experiment, after scripts/kitti_prepare.py has built
# kitti/<SCENE>/<robot>/ for both robots (both in one ground-truth frame):
#   1. train vanilla 3DGS for both robots
#   2. filter both maps with the paper's setting (opacity >= 0.10 + outlier removal)
#   3. benchmark SRC -> TGT on unfiltered and filtered maps (Table II configuration,
#      100 seeds, +-180 deg / +-500 m perturbations)
#   4. score both at 5 deg / 0.5 m with ATE over the source robot's GT cameras
#
# Results: results/<SCENE>_robots_{unfiltered,filtered}_joint5d_v050.csv
# Refuses to overwrite existing result CSVs.

set -euo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 SCENE SRC_ROBOT TGT_ROBOT"; exit 2; }
SCENE="$1" SRC="$2" TGT="$3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
LOGS="$DATA_ROOT/kitti/logs"; mkdir -p "$LOGS"
PLY=point_cloud/iteration_30000/point_cloud.ply

for r in "$SRC" "$TGT"; do
  [[ -f "$DATA_ROOT/kitti/$SCENE/$r/sparse/0/images.bin" ]] || { echo "missing COLMAP model for $SCENE/$r"; exit 1; }
done

echo "=== $SCENE: training $SRC $TGT ($(date +%T))"
SEQ="$SCENE" SPLITS="$SRC $TGT" TRAIN_ONLY=1 "$SCRIPT_DIR/run_kitti_benchmark.sh" > "$LOGS/${SCENE}_train.log" 2>&1

for r in "$SRC" "$TGT"; do
  echo "$r: $(grep -a -m1 'element vertex' "$DATA_ROOT/trained_gaussians/$SCENE/$r/$PLY" | awk '{print $3}') Gaussians"
  OMP_NUM_THREADS=4 python3 "$REPO/scripts/filter_3dgs_ply_batch.py" \
    "$DATA_ROOT/trained_gaussians/$SCENE/$r/point_cloud/iteration_30000" \
    "$DATA_ROOT/trained_gaussians_filtered/$SCENE/$r/point_cloud/iteration_30000" \
    --opacity-threshold 0.1 2>&1 | grep 'Final kept'
done

for v in unfiltered filtered; do
  base="$DATA_ROOT/trained_gaussians"; [[ $v == filtered ]] && base="$DATA_ROOT/trained_gaussians_filtered"
  out="$DATA_ROOT/results/${SCENE}_robots_${v}_joint5d_v050.csv"
  [[ -e "$out" ]] && { echo "refusing to overwrite $out"; exit 1; }
  echo "=== $SCENE: benchmark $v ($(date +%T))"
  SEQ="$SCENE" SPLITS="$SRC $TGT" PAIR_LIST="$SRC:$TGT" TRAINED_BASE="$base" OUT_CSV="$out" \
    "$SCRIPT_DIR/run_kitti_benchmark.sh" > "$LOGS/${SCENE}_benchmark_${v}.log" 2>&1
  grep -a '╚\|Total runtime' "$LOGS/${SCENE}_benchmark_${v}.log"
  python3 "$REPO/scripts/score_submap_csv.py" "$out" --at gt_cameras \
    --base "$base" --cameras_base "$DATA_ROOT/trained_gaussians" | sed -n '/^POOLED/,$p'
done
echo "=== $SCENE done ($(date +%T))"
