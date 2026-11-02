#!/usr/bin/env bash
# run_kitti_benchmark.sh
#
# Train vanilla 3DGS on the KITTI halves built by scripts/kitti_prepare.py, then
# score them with the same harness and configuration as the paper's Mip-NeRF 360
# table (joint-5D, opacity on, 0.5 m voxel, ±180° / ±500 m perturbations).
# Halves share one metric world frame, so the GT transform is identity.
#
# Usage:  ./run_kitti_benchmark.sh            (TRAIN_ONLY=1 to stop after training)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
SEQ="${SEQ:-kitti07}"
SPLITS_DIR="${SPLITS_DIR:-$DATA_ROOT/kitti/$SEQ}"
TRAINED_BASE="${TRAINED_BASE:-$DATA_ROOT/trained_gaussians}"
GS_REPO="${GS_REPO:?set GS_REPO to a graphdeco-inria/gaussian-splatting checkout}"
GS_PY="${GS_PY:-python3}"   # Python env that can run gaussian-splatting/train.py
SPLITS="${SPLITS:-split_neg split_pos split_wide_neg split_wide_pos}"

for s in $SPLITS; do
  out="$TRAINED_BASE/$SEQ/$s"
  if [[ -f "$out/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    echo "already trained: $s"; continue
  fi
  echo "════════ training $SEQ/$s ════════"
  # images stay on the CPU: ~600 KITTI views do not fit next to the Gaussians in 8 GB
  (cd "$GS_REPO" && "$GS_PY" train.py -s "$SPLITS_DIR/$s" -m "$out" \
      --data_device cpu --train_test_exp --test_iterations 30000 --disable_viewer --quiet)
done
[[ "${TRAIN_ONLY:-0}" == 1 ]] && exit 0

USE_JOINT_5D=1 USE_OPACITY=1 \
BASE="$TRAINED_BASE" VOXEL=0.5 SCENE_LIST="$SEQ" \
TARGET_VALID="${TARGET_VALID:-100}" MAX_SEEDS="${MAX_SEEDS:-300}" \
COLOR_WEIGHT_ICP=0.1 RUN_ICP=1 \
  "$SCRIPT_DIR/run_submap_benchmark.sh" "${OUT_CSV:-$DATA_ROOT/results/${SEQ}_joint5d_v050.csv}"
