#!/usr/bin/env bash
# run_2dgs.sh — extra experiment 1: does a geometry-regularized 3DGS variant (2DGS)
# change the picture?  Train 2DGS on the same splits as the vanilla-3DGS results, then run
# the identical benchmark (joint-5D, opacity on, 0.5 m voxel, ±180°/±500 m, 100 seeds/pair).
#
#   room        split_pos/split_neg (+ split_wide_* last), same source dirs as Table II
#   kitti00_07  robot_07 -> robot_00, unfiltered + filtered (opacity 0.1), as the KITTI table
#
# Resumable: finished trainings / CSVs are skipped.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${DATA_ROOT:-$R/data}"
GS="${GS2D_REPO:?set GS2D_REPO to a hbb1/2d-gaussian-splatting checkout}"
PY="${GS2D_PY:-python3}"   # Python env that can run 2d-gaussian-splatting/train.py
T=$D/trained_2dgs; TF=$D/trained_2dgs_filtered
LOG=$D/results/2dgs; mkdir -p $LOG
PLY=point_cloud/iteration_30000/point_cloud.ply

train() {  # scene split source [extra args]
  local out=$T/$1/$2; shift 2; local src=$1; shift
  [[ -f $out/$PLY ]] && { echo "trained: $out"; return; }
  echo "=== train $out  $(date +%T)"
  (cd $GS && $PY train.py -s $src -m $out --port 6019 --test_iterations 99999 --quiet "$@") \
      > $LOG/train_$(basename $(dirname $out))_$(basename $out).log 2>&1
  echo "=== done  $out rc=$? $(date +%T)"
}

bench() {  # scene base pairlist csv
  [[ -s $4 ]] && { echo "benchmarked: $4"; return; }
  echo "=== bench $4  $(date +%T)"
  USE_JOINT_5D=1 USE_OPACITY=1 BASE=$2 VOXEL=0.5 SCENE_LIST=$1 PAIR_LIST="$3" \
  TARGET_VALID=100 MAX_SEEDS=300 COLOR_WEIGHT_ICP=0.1 RUN_ICP=1 \
    $R/examples/run_submap_benchmark.sh $4.tmp > $LOG/bench_$(basename $4 .csv).log 2>&1 && mv $4.tmp $4
  echo "=== bench done $4 $(date +%T)"
}

# 1. room pos/neg, then benchmark it on the CPU while the GPU trains KITTI
train room split_pos $D/split_runs/360/room/split_pos
train room split_neg $D/split_runs/360/room/split_neg
bench room $T "split_pos:split_neg" $D/results/2dgs_room_joint5d_v050.csv &

# 2. KITTI 00/07 (images on the CPU, as for vanilla)
train kitti00_07 robot_00 $D/kitti/kitti00_07/robot_00 --data_device cpu
train kitti00_07 robot_07 $D/kitti/kitti00_07/robot_07 --data_device cpu
for r in robot_00 robot_07; do
  if [[ ! -f $TF/kitti00_07/$r/$PLY ]]; then
    python3 $D/scripts/filter_3dgs_ply_batch.py $T/kitti00_07/$r/point_cloud/iteration_30000 \
        $TF/kitti00_07/$r/point_cloud/iteration_30000 --opacity-threshold 0.1 >> $LOG/filter.log 2>&1
  fi
done
wait   # one CPU benchmark at a time
bench kitti00_07 $T  "robot_07:robot_00" $D/results/2dgs_kitti00_07_robots_unfiltered_joint5d_v050.csv &

# 3. room wide pair (completes Table II's n=200 protocol)
train room split_wide_pos $D/split_runs/360/room/split_wide_pos
train room split_wide_neg $D/split_runs/360/room/split_wide_neg
wait
bench kitti00_07 $TF "robot_07:robot_00" $D/results/2dgs_kitti00_07_robots_filtered_joint5d_v050.csv
bench room $T "split_wide_pos:split_wide_neg" $D/results/2dgs_room_wide_joint5d_v050.csv
echo "ALL DONE $(date +%T)"
