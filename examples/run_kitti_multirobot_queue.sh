#!/usr/bin/env bash
# run_kitti_multirobot_queue.sh [EXPERIMENT ...]
#
# Runs the KITTI multi-robot experiments one after another (they share the GPU):
# waits for each experiment's raw images, builds both robots' COLMAP models with
# scripts/kitti_prepare.py (skipped if already built), then runs
# examples/run_kitti_robots.sh (train, filter, benchmark, score).
#
# Robot windows (odometry frames, stride 2) were chosen so that robots share no
# frames and overlap only where the vehicle drives the same road twice
# (or, for 09<->10, where two different drives cover the same road).
#
# Default order: all seven pairs of Table IV

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
K="$DATA_ROOT/kitti"
PY="${GS_PY:-python3}"   # runs kitti_prepare.py (numpy + the colmap CLI on PATH)
WAIT_MAX_S="${WAIT_MAX_S:-14400}"
mkdir -p "$K/logs"

# scene | robot | raw drive | poses file (relative to kitti/) | calib seq | raw offset | first-last
ROBOTS=(
  "kitti05|robot_a|2011_09_30/2011_09_30_drive_0018_sync|dataset/poses/05.txt|05|0|300-1000"
  "kitti05|robot_b|2011_09_30/2011_09_30_drive_0018_sync|dataset/poses/05.txt|05|0|1100-1800"
  "kitti08|robot_a|2011_09_30/2011_09_30_drive_0028_sync|dataset/poses/08.txt|08|1100|0-600"
  "kitti08|robot_b|2011_09_30/2011_09_30_drive_0028_sync|dataset/poses/08.txt|08|1100|1300-1900"
  "kitti00|robot_a|2011_10_03/2011_10_03_drive_0027_sync|dataset/poses/00.txt|00|0|450-1050"
  "kitti00|robot_b|2011_10_03/2011_10_03_drive_0027_sync|dataset/poses/00.txt|00|0|3450-3950"
  "kitti02|robot_a|2011_10_03/2011_10_03_drive_0034_sync|dataset/poses/02.txt|02|0|1700-2150"
  "kitti02|robot_b|2011_10_03/2011_10_03_drive_0034_sync|dataset/poses/02.txt|02|0|4300-4660"
  "kitti06|robot_a|2011_09_30/2011_09_30_drive_0020_sync|dataset/poses/06.txt|06|0|0-500"
  "kitti06|robot_b|2011_09_30/2011_09_30_drive_0020_sync|dataset/poses/06.txt|06|0|600-1100"
  "kitti00_07|robot_00|2011_10_03/2011_10_03_drive_0027_sync|dataset/poses/00.txt|00|0|2400-2950"
  "kitti00_07|robot_07|2011_09_30/2011_09_30_drive_0027_sync|align_07_to_00/07_in_00.txt|07|0|0-550"
  "kitti09_10|robot_09|2011_09_30/2011_09_30_drive_0033_sync|dataset/poses/09.txt|09|0|1050-1240"
  "kitti09_10|robot_10|2011_09_30/2011_09_30_drive_0034_sync|align_10_to_09/10_in_09.txt|10|0|550-780"
)
# scene -> "SRC TGT" benchmark pair
declare -A PAIR=([kitti05]="robot_b robot_a" [kitti08]="robot_b robot_a" [kitti00]="robot_b robot_a"
                 [kitti02]="robot_b robot_a" [kitti09_10]="robot_10 robot_09"
                 [kitti06]="robot_b robot_a" [kitti00_07]="robot_07 robot_00")

wait_for() {  # wait_for <file-producing test command...>; returns 1 on timeout
  local waited=0
  until "$@"; do
    (( waited >= WAIT_MAX_S )) && return 1
    sleep 60; waited=$((waited + 60))
  done
}

images_ready() {  # images_ready raw_dir offset first last
  local raw="$K/$1" off="$2" f
  for ((f = $3; f <= $4; f += 2)); do
    for c in 02 03; do
      [[ -f "$raw/image_$c/data/$(printf '%010d' $((f + off))).png" ]] || return 1
    done
  done
}

EXPERIMENTS=("$@"); [[ ${#EXPERIMENTS[@]} -eq 0 ]] && EXPERIMENTS=(kitti00 kitti00_07 kitti06 kitti05 kitti02 kitti08 kitti09_10)

for scene in "${EXPERIMENTS[@]}"; do
  echo "################ $scene ($(date +%T))"
  ok=1
  for spec in "${ROBOTS[@]}"; do
    IFS='|' read -r s robot raw poses calib off range <<< "$spec"
    [[ "$s" == "$scene" ]] || continue
    first=${range%-*}; last=${range#*-}
    if [[ -f "$K/$scene/$robot/sparse/0/images.bin" ]]; then
      echo "$robot: model exists, skipping prepare"; continue
    fi
    echo "$robot: waiting for poses and raw images ($raw, frames $range, offset $off)"
    if ! wait_for test -f "$K/$poses" || ! wait_for images_ready "$raw" "$off" "$first" "$last"; then
      echo "ABORT $scene: inputs for $robot not ready after ${WAIT_MAX_S}s"; ok=0; break
    fi
    (cd "$K" && "$PY" "$REPO/scripts/kitti_prepare.py" --raw_dir "$raw" --poses "$poses" \
        --calib "dataset/sequences/$calib/calib.txt" --out "$scene" --stride 2 \
        --raw_offset "$off" --robot "$robot:$range") > "$K/logs/${scene}_${robot}_prepare.log" 2>&1 \
      || { echo "ABORT $scene: prepare failed for $robot (see logs/${scene}_${robot}_prepare.log)"; ok=0; break; }
    echo "$robot: $(colmap model_analyzer --path "$K/$scene/$robot/sparse/0" 2>&1 | grep -a 'Registered images\|Mean reprojection' | sed 's/.*\] //' | tr '\n' ' ')"
  done
  (( ok )) || continue
  read -r src tgt <<< "${PAIR[$scene]}"
  "$SCRIPT_DIR/run_kitti_robots.sh" "$scene" "$src" "$tgt" || echo "ABORT $scene: run_kitti_robots.sh failed"
done
echo "################ queue finished ($(date +%T))"
