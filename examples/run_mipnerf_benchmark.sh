#!/usr/bin/env bash
# run_mipnerf_benchmark.sh [joint5d|centroid ...]
#
# The paper's large-scale Mip-NeRF 360 benchmark (Tables II and III): nine scenes,
# two independently trained split pairs per scene (split_pos/neg and
# split_wide_pos/neg), 100 valid seeds per pair, random +-180 deg / +-500 m start
# pose, 0.5 m voxel, opacity on. Every seed runs all four methods:
#   vanilla_kiss   KISS-Matcher on the Gaussian centroids (FPFH)
#   color_cpfh     3DGS-Matcher coarse (ColorPFH)
#   gicp_cpfh      + geometric GICP
#   color_gicp_de  + 3DGS-ICP (colour-weighted GICP)
#
# Writes $DATA_ROOT/results/{joint5d,centroid}_trained_v050_n200_opacityON.csv,
# the same files as paper_data/table2_large_scale/ (score them with
# paper_data/table2_large_scale/gen_large_scale_table.py --joint5d <csv> --centroid <csv>).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$HERE/.." && pwd)/data}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/results}"
mkdir -p "$OUT_DIR"

MODES=("$@"); [[ ${#MODES[@]} -eq 0 ]] && MODES=(joint5d centroid)
for mode in "${MODES[@]}"; do
  case "$mode" in
    joint5d)  j5d=1 ;;
    centroid) j5d=0 ;;
    *) echo "unknown mode: $mode (joint5d|centroid)"; exit 2 ;;
  esac
  USE_JOINT_5D=$j5d USE_OPACITY=1 \
  BASE="$DATA_ROOT/trained_gaussians" VOXEL=0.5 \
  SCENE_LIST="bicycle bonsai counter flowers garden kitchen room stump treehill" \
  TARGET_VALID=100 MAX_SEEDS=300 COLOR_WEIGHT_ICP=0.1 RUN_ICP=1 \
    "$HERE/run_submap_benchmark.sh" "$OUT_DIR/${mode}_trained_v050_n200_opacityON.csv"
done
