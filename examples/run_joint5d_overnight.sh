#!/usr/bin/env bash
# run_joint5d_overnight.sh
#
# Overnight sweep for the joint-5D + ColorPFH-weighted-GICP evaluation on
# trained_gaussians (non-drone, 9 scenes).
#
# Binary state (compiled into build/benchmark_submap):
#   - use_joint_5d = true   (colour-aware joint spatial-chromatic downsampling)
#   - methods: vanilla_kiss, color_cpfh, gicp_cpfh, color_gicp_de
#              (color_gicp_cpfh / color_corr_icp / color_full_icp are #if 0'd out)
#
# Produces one CSV per voxel in OUT_DIR:
#   sweep_joint5d_nondrone_v050.csv  (0.5 m)
#   sweep_joint5d_nondrone_v040.csv  (0.4 m)
#   sweep_joint5d_nondrone_v030.csv  (0.3 m)
#
# Compare against the centroid baseline already on disk:
#   results/sweep_trained_nondrone_v050.csv   (centroid, 0.5 m)
#
# Usage:  ./run_joint5d_overnight.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/run_submap_benchmark.sh"

# Dataset/result locations; override with DATA_ROOT (or the individual vars).
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
TRAINED_BASE="${TRAINED_BASE:-$DATA_ROOT/trained_gaussians}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/results}"
mkdir -p "$OUT_DIR"

NON_DRONE="bicycle bonsai counter flowers garden kitchen room stump treehill"
VOXELS=(0.5 0.4 0.3)

TARGET_VALID=${TARGET_VALID:-20}
MAX_SEEDS=${MAX_SEEDS:-60}
COLOR_WEIGHT_ICP=${COLOR_WEIGHT_ICP:-0.1}   # unused by color_gicp_de (auto-sigma) / gicp (geometry)
RUN_ICP=1

voxel_tag() { printf "v%03.0f" "$(echo "$1 * 100" | bc)"; }

# Joint-5D keypoints, opacity off (the binary now takes both from the environment).
export USE_JOINT_5D="${USE_JOINT_5D:-1}" USE_OPACITY="${USE_OPACITY:-0}"

T0=$SECONDS
for v in "${VOXELS[@]}"; do
  tag=$(voxel_tag "$v")
  out="$OUT_DIR/sweep_joint5d_nondrone_${tag}.csv"
  echo ""
  echo "════════ joint5d nondrone  voxel=${v}m  →  $(basename "$out") ════════"
  BASE="$TRAINED_BASE" \
  VOXEL="$v" \
  SCENE_LIST="$NON_DRONE" \
  TARGET_VALID="$TARGET_VALID" \
  MAX_SEEDS="$MAX_SEEDS" \
  COLOR_WEIGHT_ICP="$COLOR_WEIGHT_ICP" \
  RUN_ICP="$RUN_ICP" \
      "$BENCH_SCRIPT" "$out"
done

EL=$(( SECONDS - T0 ))
printf "\nTotal joint5d sweep: %dh %dm %ds  →  %s\n" "$((EL/3600))" "$(((EL%3600)/60))" "$((EL%60))" "$OUT_DIR"
