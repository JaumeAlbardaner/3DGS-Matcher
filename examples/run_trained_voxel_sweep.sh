#!/usr/bin/env bash
# run_trained_voxel_sweep.sh
#
# Voxel-size sweep on trained_gaussians across all 12 scenes.
#
#   Non-drone (9 scenes): voxels 0.05, 0.10, 0.30, 0.50 m
#   Drone/UGV  (3 scenes): voxels 0.10, 0.30, 0.50, 0.70 m
#
# Produces one CSV per (group × voxel) in OUT_DIR:
#   sweep_trained_nondrone_v005.csv   (0.05 m)
#   sweep_trained_nondrone_v010.csv   (0.10 m)
#   ...
#   sweep_trained_drone_v070.csv      (0.70 m)
#
# Runtime hints (approximate, trained_gaussians, TARGET_VALID=20):
#   0.05 m — ~12 h  (large clouds, use cautiously)
#   0.10 m — ~2.5 h
#   0.30 m — ~45 min
#   0.50 m — ~15 min
#   0.70 m — ~5 min  (drone only)
#
# Usage:
#   ./run_trained_voxel_sweep.sh
#   TARGET_VALID=10 ./run_trained_voxel_sweep.sh   # faster smoke test

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/run_submap_benchmark.sh"

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
TRAINED_BASE="${TRAINED_BASE:-$DATA_ROOT/trained_gaussians}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/results}"
mkdir -p "$OUT_DIR"

TARGET_VALID=${TARGET_VALID:-20}
MAX_SEEDS=${MAX_SEEDS:-200}
COLOR_WEIGHT_ICP=${COLOR_WEIGHT_ICP:-0.1}
RUN_ICP=${RUN_ICP:-1}

NON_DRONE="bicycle bonsai counter flowers garden kitchen room stump treehill"
DRONE="drone1_undistorted drone2_undistorted ugv_undistorted"

# ── helpers ──────────────────────────────────────────────────────────────────

voxel_tag() {
    # 0.05 → v005   0.1 → v010   0.3 → v030   0.7 → v070
    printf "v%03.0f" "$(echo "$1 * 100" | bc)"
}

run_group() {
    local label="$1"   # "nondrone" or "drone"
    local scenes="$2"  # space-separated scene names
    shift 2
    local voxels=("$@")

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Group: $label   scenes: $scenes"
    echo "  Voxels: ${voxels[*]}   TARGET_VALID=$TARGET_VALID"
    echo "════════════════════════════════════════════════════════"

    for voxel in "${voxels[@]}"; do
        local tag
        tag=$(voxel_tag "$voxel")
        local outfile="$OUT_DIR/sweep_trained_${label}_${tag}.csv"

        echo ""
        echo "── $label  voxel=${voxel}m  →  $(basename "$outfile")"

        BASE="$TRAINED_BASE" \
        VOXEL="$voxel" \
        SCENE_LIST="$scenes" \
        TARGET_VALID="$TARGET_VALID" \
        MAX_SEEDS="$MAX_SEEDS" \
        COLOR_WEIGHT_ICP="$COLOR_WEIGHT_ICP" \
        RUN_ICP="$RUN_ICP" \
            "$BENCH_SCRIPT" "$outfile"
    done
}

T_START=$SECONDS

# Centroid keypoints, opacity off: the configuration of the paper's voxel figure.
export USE_JOINT_5D="${USE_JOINT_5D:-0}" USE_OPACITY="${USE_OPACITY:-0}"
read -ra NONDRONE_VOXELS <<< "${NONDRONE_VOXELS:-0.5 0.4 0.3 0.2}"
run_group "nondrone" "$NON_DRONE" "${NONDRONE_VOXELS[@]}"
[[ "${SKIP_DRONE:-0}" == 1 ]] || run_group "drone" "$DRONE" 0.3 0.4 0.5

ELAPSED=$(( SECONDS - T_START ))
echo ""
echo "Total sweep runtime: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo "Results in: $OUT_DIR"
