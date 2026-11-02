#!/usr/bin/env bash
#
# No-ground-truth overlap/fitness test on the real cross-platform pairs:
#   drone1 <-> drone2, drone2 <-> gv, drone1 <-> gv
#
# Each seed applies a random ±MAX_ROT/±MAX_TRANS perturbation to the target
# (benchmark_submap protocol), runs the full pipeline (vanilla KISS-Matcher,
# 3DGS-Matcher coarse, +GICP, +3DGS-ICP), and logs the estimated transform.
# scripts/compute_overlap_metrics.py then undoes the perturbation and reports
# fitness / inlier RMSE / reverse-fitness plus the seed-to-seed consensus.
#
# Usage:
#   ./examples/run_overlap_test.sh              # N_SEEDS=5 quick run
#   N_SEEDS=50 ./examples/run_overlap_test.sh   # bigger run later
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/build/benchmark_submap"
HELPER="$ROOT/build/print_perturbation"

DATA="${DATA:-${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}/FullSplats}"
PLY_D1="${PLY_D1:-$DATA/dron1_point_cloud.ply}"
PLY_D2="${PLY_D2:-$DATA/drone2_point_cloud.ply}"
PLY_GV="${PLY_GV:-$DATA/gv_point_cloud.ply}"

N_SEEDS=${N_SEEDS:-5}
VOXEL=${VOXEL:-0.5}
MAX_ROT=${MAX_ROT:-180.0}
MAX_TRANS=${MAX_TRANS:-500.0}
COLOR_WEIGHT_ICP=${COLOR_WEIGHT_ICP:-0.1}
RUN_ICP=${RUN_ICP:-1}
export USE_JOINT_5D=${USE_JOINT_5D:-1}   # production keypoint selection
export USE_OPACITY=${USE_OPACITY:-1}     # production descriptor (opacity on)

OUT="${1:-$ROOT/results/realdata_overlap_n${N_SEEDS}.csv}"
METRICS_OUT="${OUT%.csv}_metrics.csv"

[[ -x "$BENCH" ]] || { echo "ERROR: $BENCH not built"; exit 1; }
if [[ ! -x "$HELPER" || "$ROOT/scripts/print_perturbation.cc" -nt "$HELPER" ]]; then
    echo "Building print_perturbation helper..."
    g++ -O2 -std=c++17 -I/usr/include/eigen3 \
        -o "$HELPER" "$ROOT/scripts/print_perturbation.cc" || exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "method,pair,split_src,split_tgt,iter,seed,rot_error_deg,trans_error_m,success,solver_ok,tx,ty,tz,qw,qx,qy,qz,time_ms" > "$OUT"

run_pair() {
    local name="$1" src="$2" tgt="$3"
    if [[ ! -f "$src" || ! -f "$tgt" ]]; then
        printf "  SKIP %-16s (missing PLY)\n" "$name"; return 0
    fi
    printf "\n== %s (%d seeds) ==\n" "$name" "$N_SEEDS"
    local seed tmp ec
    for seed in $(seq 0 $((N_SEEDS - 1))); do
        tmp=$(mktemp "${TMPDIR:-/tmp}/ov_XXXXXX.csv"); ec=0
        "$BENCH" "$src" "$tgt" "$VOXEL" "$MAX_ROT" "$MAX_TRANS" "$seed" "$tmp" \
            "$COLOR_WEIGHT_ICP" "$RUN_ICP" 2>/dev/null || ec=$?
        if [[ $ec -eq 0 && -s "$tmp" ]]; then
            awk -F',' -v p="$name" 'BEGIN{OFS=","}{$2=p; print}' "$tmp" >> "$OUT"
            printf "  seed %d ok\n" "$seed"
        else
            printf "  seed %d FAILED (exit %d)\n" "$seed" "$ec"
        fi
        rm -f "$tmp"
    done
}

run_pair "drone1-drone2" "$PLY_D1" "$PLY_D2"
run_pair "drone2-gv"     "$PLY_D2" "$PLY_GV"
run_pair "drone1-gv"     "$PLY_D1" "$PLY_GV"

echo ""
echo "Runs done -> $OUT"
echo "Computing overlap/fitness metrics..."
python3 "$ROOT/scripts/compute_overlap_metrics.py" \
    --csv "$OUT" \
    --pair "drone1-drone2:$PLY_D1:$PLY_D2" \
    --pair "drone2-gv:$PLY_D2:$PLY_GV" \
    --pair "drone1-gv:$PLY_D1:$PLY_GV" \
    --max-rot "$MAX_ROT" --max-trans "$MAX_TRANS" \
    --out "$METRICS_OUT"
