#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/bin:/bin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAUSSREG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_ROOT=""
WEIGHTS="$GAUSSREG_ROOT/weights/coarse_registration.pth.tar"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_TAG="${RUN_TAG}_job${SLURM_JOB_ID}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_TAG="${RUN_TAG}_arr${SLURM_ARRAY_TASK_ID}"
  fi
fi
OUTPUT_ROOT="$GAUSSREG_ROOT/benchmark_outputs/offset_eval_${RUN_TAG}"
PYTHON_BIN="python"
# Default mixed schedule: 50 xyz-only perturbation trials + 10 full-feature perturbation trials.
XYZ_TRIALS=100
FULL_TRIALS=100
TRIALS=1
LEGACY_TRIALS_SET=0
XYZ_TRIALS_SET=0
FULL_TRIALS_SET=0
XYZ_ONLY_SET=0
SEED_START=42
MAX_TRANSLATION=500.0
MAX_ROT_DEG=180.0
NUM_SAMPLE=30000
ROTATE_SH=1
XYZ_ONLY=0
DRY_RUN=0
RUN_EVAL=0
HEARTBEAT_SEC=30
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SKIP_RANDOMIZE=0
AUTO_SKIP_RANDOMIZE=0
KEEP_TRANSFORMED_PLY=0
KEEP_DEMO_PLY=0
KEEP_RUN_ARTIFACTS=0
# Format: src_split:ref_split:pair_name
PAIRS_CSV="split_neg:split_pos:normal,split_wide_neg:split_wide_pos:wide"

usage() {
  cat <<EOF
Usage: $0 --data-root <root> [options]

Required:
  --data-root <path>          Dataset root with scene subfolders (bicycle, bonsai, ...)

Options:
  --weights <path>            GaussReg weights (.pth.tar) (default: $WEIGHTS)
  --output-root <path>        Output root for runs + CSV
  --python <bin>              Python executable (default: $PYTHON_BIN)
  --xyz-trials <int>          Number of xyz-only trials per pair (default: $XYZ_TRIALS)
  --full-trials <int>         Number of full-feature trials per pair (default: $FULL_TRIALS)
  --trials <int>              Legacy single-mode trial count (see notes below)
  --seed-start <int>          Base seed (default: $SEED_START)
  --max-translation <float>   Max abs translation per axis for random offset (default: $MAX_TRANSLATION)
  --max-rotation-deg <float>  Max abs Euler angle per axis (default: $MAX_ROT_DEG)
  --num-sample <int>          GaussReg demo --num_sample (default: $NUM_SAMPLE)
  --pairs <csv>               Pair list: src:ref:name,src:ref:name (default: $PAIRS_CSV)
  --heartbeat-sec <int>       Seconds between "still running" logs (default: $HEARTBEAT_SEC)
  --shard-index <int>         Scene shard index for SLURM array splitting (default: $SHARD_INDEX)
  --num-shards <int>          Total scene shards for SLURM array splitting (default: $NUM_SHARDS)
  --skip-randomize            Reuse original ref/src PLYs and evaluate against identity GT
  --keep-transformed-ply      Keep per-trial ref/src transformed full PLY files (default: off)
  --keep-demo-ply             Keep demo output PLY files under demo_outputs/ (default: off)
  --keep-run-artifacts        Keep per-run artifacts under output_root/runs/ (default: off)
  --eval                      Also run local error computation script
  --xyz-only                  Run only xyz-only trials (disables full-feature phase)
  --no-rotate-sh              Do not rotate SH coefficients when randomizing
  --dry-run                   Print planned work only
  -h, --help                  Show this help

Notes:
  - Default behavior is mixed mode: --xyz-trials 50 and --full-trials 10.
  - --trials N is a compatibility option:
      with --xyz-only -> runs N xyz-only trials and 0 full-feature trials
      without --xyz-only -> runs 0 xyz-only trials and N full-feature trials

Example:
  $0 --data-root /data/my_splits \
     --weights $GAUSSREG_ROOT/weights/coarse_registration.pth.tar \
     --xyz-trials 50 --full-trials 10 --seed-start 100 --max-translation 500 --max-rotation-deg 180

SLURM array example:
  sbatch --array=0-3 ./scripts/run_gaussreg_offsets.sh \
     --data-root /data/my_splits --shard-index \${SLURM_ARRAY_TASK_ID} --num-shards 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --xyz-trials) XYZ_TRIALS="$2"; XYZ_TRIALS_SET=1; shift 2 ;;
    --full-trials) FULL_TRIALS="$2"; FULL_TRIALS_SET=1; shift 2 ;;
    --trials) TRIALS="$2"; LEGACY_TRIALS_SET=1; shift 2 ;;
    --seed-start) SEED_START="$2"; shift 2 ;;
    --max-translation) MAX_TRANSLATION="$2"; shift 2 ;;
    --max-rotation-deg) MAX_ROT_DEG="$2"; shift 2 ;;
    --num-sample) NUM_SAMPLE="$2"; shift 2 ;;
    --pairs) PAIRS_CSV="$2"; shift 2 ;;
    --heartbeat-sec) HEARTBEAT_SEC="$2"; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --skip-randomize) SKIP_RANDOMIZE=1; shift ;;
    --keep-transformed-ply) KEEP_TRANSFORMED_PLY=1; shift ;;
    --keep-demo-ply) KEEP_DEMO_PLY=1; shift ;;
    --keep-run-artifacts) KEEP_RUN_ARTIFACTS=1; shift ;;
    --eval) RUN_EVAL=1; shift ;;
    --xyz-only) XYZ_ONLY=1; XYZ_ONLY_SET=1; shift ;;
    --no-rotate-sh) ROTATE_SH=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_non_negative_int "$XYZ_TRIALS"; then
  echo "Error: --xyz-trials must be a non-negative integer (got: $XYZ_TRIALS)" >&2
  exit 1
fi
if ! is_non_negative_int "$FULL_TRIALS"; then
  echo "Error: --full-trials must be a non-negative integer (got: $FULL_TRIALS)" >&2
  exit 1
fi
if ! is_non_negative_int "$TRIALS"; then
  echo "Error: --trials must be a non-negative integer (got: $TRIALS)" >&2
  exit 1
fi
if ! is_non_negative_int "$HEARTBEAT_SEC" || [[ "$HEARTBEAT_SEC" -le 0 ]]; then
  echo "Error: --heartbeat-sec must be a positive integer (got: $HEARTBEAT_SEC)" >&2
  exit 1
fi
if ! is_non_negative_int "$SHARD_INDEX"; then
  echo "Error: --shard-index must be a non-negative integer (got: $SHARD_INDEX)" >&2
  exit 1
fi
if ! is_non_negative_int "$NUM_SHARDS" || [[ "$NUM_SHARDS" -le 0 ]]; then
  echo "Error: --num-shards must be a positive integer (got: $NUM_SHARDS)" >&2
  exit 1
fi
if [[ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]]; then
  echo "Error: --shard-index ($SHARD_INDEX) must be < --num-shards ($NUM_SHARDS)" >&2
  exit 1
fi

if [[ "$LEGACY_TRIALS_SET" -eq 1 && ( "$XYZ_TRIALS_SET" -eq 1 || "$FULL_TRIALS_SET" -eq 1 ) ]]; then
  echo "Error: --trials cannot be combined with --xyz-trials or --full-trials" >&2
  exit 1
fi

if [[ "$LEGACY_TRIALS_SET" -eq 1 ]]; then
  if [[ "$XYZ_ONLY" -eq 1 ]]; then
    XYZ_TRIALS="$TRIALS"
    FULL_TRIALS=0
  else
    XYZ_TRIALS=0
    FULL_TRIALS="$TRIALS"
  fi
fi

if [[ "$XYZ_ONLY_SET" -eq 1 && "$FULL_TRIALS_SET" -eq 1 && "$FULL_TRIALS" -gt 0 ]]; then
  echo "Error: --xyz-only conflicts with --full-trials > 0" >&2
  exit 1
fi

if [[ "$XYZ_ONLY" -eq 1 && "$LEGACY_TRIALS_SET" -eq 0 ]]; then
  FULL_TRIALS=0
fi

if "$PYTHON_BIN" - "$MAX_TRANSLATION" "$MAX_ROT_DEG" <<'PY' >/dev/null
import sys
vals = [float(v) for v in sys.argv[1:3]]
raise SystemExit(0 if all(abs(v) <= 1e-12 for v in vals) else 1)
PY
then
  if [[ "$SKIP_RANDOMIZE" -eq 0 ]]; then
    SKIP_RANDOMIZE=1
    AUTO_SKIP_RANDOMIZE=1
  fi
fi

TOTAL_TRIALS_PER_PAIR=$((XYZ_TRIALS + FULL_TRIALS))
if [[ "$TOTAL_TRIALS_PER_PAIR" -eq 0 ]]; then
  echo "Error: total trials per pair is zero (set --xyz-trials and/or --full-trials)" >&2
  exit 1
fi

if [[ -z "$DATA_ROOT" ]]; then
  echo "Error: --data-root is required" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Error: data root not found: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Error: weights not found: $WEIGHTS" >&2
  exit 1
fi
if [[ "$SKIP_RANDOMIZE" -eq 0 && ! -x "$SCRIPT_DIR/randomize_gaussian_pair.py" ]]; then
  echo "Error: helper script missing: $SCRIPT_DIR/randomize_gaussian_pair.py" >&2
  exit 1
fi
if [[ "$RUN_EVAL" -eq 1 && ! -x "$SCRIPT_DIR/evaluate_gaussreg_result.py" ]]; then
  echo "Error: helper script missing: $SCRIPT_DIR/evaluate_gaussreg_result.py" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
RESULTS_CSV="$OUTPUT_ROOT/results.csv"
OUTPUT_ROOT_REAL="$(cd "$OUTPUT_ROOT" && pwd -P)"

# Build CSV header with flattened 4x4 GT and EST matrices.
{
  printf "scene,pair,trial_mode,trial,seed,xyz_only_mode,rotate_sh_mode,ref_split,src_split,status,error_msg"
  printf ",randomize_elapsed_s,demo_elapsed_s"
  printf ",rot_err_deg,trans_err_abs,trans_err_rel,scale_est,scale_err_abs,scale_err_rel"
  printf ",ref_roll_deg,ref_pitch_deg,ref_yaw_deg,ref_tx,ref_ty,ref_tz"
  printf ",src_roll_deg,src_pitch_deg,src_yaw_deg,src_tx,src_ty,src_tz"
  for i in $(seq 0 3); do
    for j in $(seq 0 3); do
      printf ",gt_t%d%d" "$i" "$j"
    done
  done
  for i in $(seq 0 3); do
    for j in $(seq 0 3); do
      printf ",est_t%d%d" "$i" "$j"
    done
  done
  printf ",ref_ply,src_ply,ref_transformed_ply,src_transformed_ply,meta_json,estimated_npz,gaussreg_output_dir,gaussreg_log"
  printf "\n"
} > "$RESULTS_CSV"

find_latest_point_cloud() {
  local split_dir="$1"
  find "$split_dir" -type f -name point_cloud.ply 2>/dev/null | sort -V | tail -n 1
}

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

cleanup_heavy_outputs() {
  local ref_out="$1"
  local src_out="$2"
  local demo_out="$3"

  if [[ "$KEEP_TRANSFORMED_PLY" -eq 0 ]]; then
    rm -f "$ref_out" "$src_out" || true
  fi

  if [[ "$KEEP_DEMO_PLY" -eq 0 ]]; then
    rm -f "$demo_out/point_cloud_ref.ply" \
          "$demo_out/point_cloud_src.ply" \
          "$demo_out/point_cloud_src_org.ply" || true
  fi
}

cleanup_run_artifacts() {
  local run_dir="$1"

  if [[ "$KEEP_RUN_ARTIFACTS" -ne 0 ]]; then
    return
  fi
  if [[ ! -d "$run_dir" ]]; then
    return
  fi

  local run_dir_real
  run_dir_real="$(cd "$run_dir" && pwd -P 2>/dev/null || true)"
  if [[ -z "$run_dir_real" ]]; then
    return
  fi

  case "$run_dir_real" in
    "$OUTPUT_ROOT_REAL"/runs/*)
      rm -rf "$run_dir_real" || true
      ;;
    *)
      echo "[warn] refusing to delete unexpected run dir: $run_dir_real" >&2
      ;;
  esac
}

prepare_identity_run_artifacts() {
  local ref_in="$1"
  local src_in="$2"
  local ref_out="$3"
  local src_out="$4"
  local meta_json="$5"
  local seed="$6"
  local xyz_only_mode="$7"
  local rotate_sh_mode="$8"

  mkdir -p "$(dirname "$ref_out")" "$(dirname "$src_out")" "$(dirname "$meta_json")"
  ln -sfn "$ref_in" "$ref_out"
  ln -sfn "$src_in" "$src_out"

  "$PYTHON_BIN" - "$meta_json" "$ref_in" "$src_in" "$ref_out" "$src_out" "$seed" "$xyz_only_mode" "$rotate_sh_mode" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

meta_path = Path(sys.argv[1])
ref_in = Path(sys.argv[2]).resolve()
src_in = Path(sys.argv[3]).resolve()
ref_out = Path(sys.argv[4]).resolve()
src_out = Path(sys.argv[5]).resolve()
seed = int(sys.argv[6])
xyz_only_mode = bool(int(sys.argv[7]))
rotate_sh_mode = bool(int(sys.argv[8]))

identity = np.eye(4, dtype=np.float64).tolist()
zero3 = [0.0, 0.0, 0.0]

meta = {
    "seed": seed,
    "max_translation": 0.0,
    "max_rotation_deg": 0.0,
    "rotate_sh": rotate_sh_mode,
    "xyz_only": xyz_only_mode,
    "randomize_skipped": True,
    "ref": {
        "input_ply": str(ref_in),
        "output_ply": str(ref_out),
        "euler_deg_xyz": zero3,
        "translation_xyz": zero3,
        "transform": identity,
    },
    "src": {
        "input_ply": str(src_in),
        "output_ply": str(src_out),
        "euler_deg_xyz": zero3,
        "translation_xyz": zero3,
        "transform": identity,
    },
    "T_gt_src_to_ref": identity,
}

meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
PY
}

append_row() {
  local scene="$1"
  local pair_name="$2"
  local trial_mode="$3"
  local trial="$4"
  local seed="$5"
  local xyz_only_mode="$6"
  local rotate_sh_mode="$7"
  local ref_split="$8"
  local src_split="$9"
  local status="${10}"
  local error_msg="${11}"
  local randomize_elapsed_s="${12}"
  local demo_elapsed_s="${13}"
  local ref_ply="${14}"
  local src_ply="${15}"
  local ref_out="${16}"
  local src_out="${17}"
  local meta_json="${18}"
  local est_npz="${19}"
  local metrics_json="${20}"
  local gaussreg_output_dir="${21}"
  local gaussreg_log="${22}"

  "$PYTHON_BIN" - "$RESULTS_CSV" "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$xyz_only_mode" "$rotate_sh_mode" "$ref_split" "$src_split" "$status" "$error_msg" "$randomize_elapsed_s" "$demo_elapsed_s" "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$gaussreg_output_dir" "$gaussreg_log" <<'PY'
import csv
import json
import os
import sys
import numpy as np

csv_path, scene, pair_name, trial_mode, trial, seed, xyz_only_mode, rotate_sh_mode, ref_split, src_split, status, error_msg, randomize_elapsed_s, demo_elapsed_s, ref_ply, src_ply, ref_out, src_out, meta_json, est_npz, metrics_json, gaussreg_output_dir, gaussreg_log = sys.argv[1:]

blank16 = [""] * 16
row = [scene, pair_name, trial_mode, trial, seed, xyz_only_mode, rotate_sh_mode, ref_split, src_split, status, error_msg,
       randomize_elapsed_s, demo_elapsed_s]

# Defaults for missing files/failures
metrics_from_file = {}
meta = {}
if os.path.isfile(meta_json):
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
if os.path.isfile(metrics_json):
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics_from_file = json.load(f)


def as_4x4(mat):
    try:
        arr = np.asarray(mat, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    return arr


def load_estimated_transform(npz_path):
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "estimated_transform" in data:
            est = data["estimated_transform"]
        elif len(data.files) == 1:
            est = data[data.files[0]]
        else:
            return None
        return as_4x4(est)
    except Exception:
        return None


def decompose_similarity(transform):
    linear = transform[:3, :3]
    scale = float(np.sqrt(max((linear @ linear.T)[0, 0], 1e-12)))
    rotation = linear / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = transform[:3, 3]
    return rotation, translation, scale


def rotation_error_deg(r_gt, r_est):
    x = 0.5 * (np.trace(r_est.T @ r_gt) - 1.0)
    x = float(np.clip(x, -1.0, 1.0))
    return float(np.degrees(np.arccos(x)))


derived_metrics = {}
gt_mat_np = as_4x4(meta.get("T_gt_src_to_ref"))
est_mat_np = load_estimated_transform(est_npz)
if gt_mat_np is not None and est_mat_np is not None:
    try:
        r_gt, t_gt, s_gt = decompose_similarity(gt_mat_np)
        r_est, t_est, s_est = decompose_similarity(est_mat_np)
        rot_err = rotation_error_deg(r_gt, r_est)
        trans_err_abs = float(np.linalg.norm(t_gt - t_est))
        trans_norm = float(np.linalg.norm(t_gt))
        trans_err_rel = float(trans_err_abs / max(trans_norm, 1e-12))
        scale_err_abs = float(abs(s_gt - s_est))
        scale_err_rel = float(scale_err_abs / max(abs(s_gt), 1e-12))

        derived_metrics = {
            "rot_err_deg": rot_err,
            "trans_err_abs": trans_err_abs,
            "trans_err_rel": trans_err_rel,
            "scale_est": float(s_est),
            "scale_err_abs": scale_err_abs,
            "scale_err_rel": scale_err_rel,
        }
    except Exception:
        derived_metrics = {}


def metric_value(key):
    if key in derived_metrics:
        return derived_metrics[key]
    return metrics_from_file.get(key, "")


row += [
    metric_value("rot_err_deg"),
    metric_value("trans_err_abs"),
    metric_value("trans_err_rel"),
    metric_value("scale_est"),
    metric_value("scale_err_abs"),
    metric_value("scale_err_rel"),
]

ref_euler = meta.get("ref", {}).get("euler_deg_xyz", ["", "", ""])
ref_t = meta.get("ref", {}).get("translation_xyz", ["", "", ""])
src_euler = meta.get("src", {}).get("euler_deg_xyz", ["", "", ""])
src_t = meta.get("src", {}).get("translation_xyz", ["", "", ""])

row += [
    ref_euler[0] if len(ref_euler) > 0 else "",
    ref_euler[1] if len(ref_euler) > 1 else "",
    ref_euler[2] if len(ref_euler) > 2 else "",
    ref_t[0] if len(ref_t) > 0 else "",
    ref_t[1] if len(ref_t) > 1 else "",
    ref_t[2] if len(ref_t) > 2 else "",
    src_euler[0] if len(src_euler) > 0 else "",
    src_euler[1] if len(src_euler) > 1 else "",
    src_euler[2] if len(src_euler) > 2 else "",
    src_t[0] if len(src_t) > 0 else "",
    src_t[1] if len(src_t) > 1 else "",
    src_t[2] if len(src_t) > 2 else "",
]


def flatten_4x4(mat):
    arr = as_4x4(mat)
    if arr is None:
        return blank16
    out = []
    for i in range(4):
        for j in range(4):
            out.append(float(arr[i, j]))
    return out

row += flatten_4x4(gt_mat_np)
row += flatten_4x4(est_mat_np)

row += [ref_ply, src_ply, ref_out, src_out, meta_json, est_npz, gaussreg_output_dir, gaussreg_log]

with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(row)
PY
}

RUNNING_UNDER_SLURM=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUNNING_UNDER_SLURM=1
fi

echo "========================================"
echo "GaussReg Offset Benchmark"
echo "========================================"
echo "Data root:        $DATA_ROOT"
echo "Weights:          $WEIGHTS"
echo "Output root:      $OUTPUT_ROOT"
echo "Seed start:       $SEED_START"
echo "Max translation:  $MAX_TRANSLATION"
echo "Max rotation deg: $MAX_ROT_DEG"
echo "Num sample:       $NUM_SAMPLE"
echo "XYZ trials:       $XYZ_TRIALS"
echo "Full trials:      $FULL_TRIALS"
echo "Total per pair:   $TOTAL_TRIALS_PER_PAIR"
echo "Rotate SH(full):  $ROTATE_SH"
echo "Heartbeat sec:    $HEARTBEAT_SEC"
echo "Run eval:         $RUN_EVAL"
echo "Pairs:            $PAIRS_CSV"
echo "Sharding:         $SHARD_INDEX / $NUM_SHARDS"
if [[ "$SKIP_RANDOMIZE" -eq 1 && "$AUTO_SKIP_RANDOMIZE" -eq 1 ]]; then
  echo "Skip randomize:   1 (auto: zero perturbation range)"
else
  echo "Skip randomize:   $SKIP_RANDOMIZE"
fi
echo "Keep transformed: $KEEP_TRANSFORMED_PLY"
echo "Keep demo PLY:    $KEEP_DEMO_PLY"
echo "Keep run files:   $KEEP_RUN_ARTIFACTS"
echo "Under SLURM:      $RUNNING_UNDER_SLURM"
if [[ "$RUNNING_UNDER_SLURM" -eq 1 ]]; then
  echo "SLURM job id:     ${SLURM_JOB_ID:-}"
  echo "SLURM array id:   ${SLURM_ARRAY_TASK_ID:-n/a}"
  echo "SLURM host:       ${SLURMD_NODENAME:-$(hostname)}"
  echo "CUDA devices:     ${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
echo "========================================"

IFS=',' read -ra PAIR_ARR <<< "$PAIRS_CSV"
mapfile -t SCENE_DIRS < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

SCENE_INDEX=0
TOTAL_SCENES=${#SCENE_DIRS[@]}
SCENES_PROCESSED=0
SCENES_SKIPPED_BY_SHARD=0
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

for scene_dir in "${SCENE_DIRS[@]}"; do
  [[ -d "$scene_dir" ]] || continue
  scene="$(basename "$scene_dir")"
  scene_dir_real="$(cd "$scene_dir" && pwd -P)"

  if [[ "$scene_dir_real" == "$OUTPUT_ROOT_REAL" ]]; then
    echo "[skip] $scene: output directory detected under data root"
    SCENE_INDEX=$((SCENE_INDEX + 1))
    continue
  fi

  if [[ "$NUM_SHARDS" -gt 1 ]] && [[ $((SCENE_INDEX % NUM_SHARDS)) -ne "$SHARD_INDEX" ]]; then
    SCENES_SKIPPED_BY_SHARD=$((SCENES_SKIPPED_BY_SHARD + 1))
    SCENE_INDEX=$((SCENE_INDEX + 1))
    continue
  fi

  SCENES_PROCESSED=$((SCENES_PROCESSED + 1))
  echo "[$(timestamp)] [scene] processing $scene ($SCENES_PROCESSED/${TOTAL_SCENES})"

  PAIR_INDEX=0
  for pair_spec in "${PAIR_ARR[@]}"; do
    IFS=':' read -r src_split ref_split pair_name <<< "$pair_spec"
    if [[ -z "${src_split:-}" || -z "${ref_split:-}" || -z "${pair_name:-}" ]]; then
      echo "[warn] invalid pair spec: $pair_spec"
      continue
    fi

    src_dir="$scene_dir/$src_split"
    ref_dir="$scene_dir/$ref_split"

    if [[ ! -d "$src_dir" || ! -d "$ref_dir" ]]; then
      echo "[skip] $scene $pair_name: missing split dirs ($src_split or $ref_split)"
      continue
    fi

    src_ply="$(find_latest_point_cloud "$src_dir")"
    ref_ply="$(find_latest_point_cloud "$ref_dir")"

    if [[ -z "$src_ply" || -z "$ref_ply" ]]; then
      echo "[skip] $scene $pair_name: could not find point_cloud.ply"
      continue
    fi

    MODE_INDEX=0
    for mode_spec in "xyz_only:$XYZ_TRIALS:1:0" "full_features:$FULL_TRIALS:0:$ROTATE_SH"; do
      IFS=':' read -r trial_mode mode_trials mode_xyz_only mode_rotate_sh <<< "$mode_spec"
      if [[ "$mode_trials" -eq 0 ]]; then
        MODE_INDEX=$((MODE_INDEX + 1))
        continue
      fi

      for ((trial=0; trial<mode_trials; trial++)); do
        seed=$((SEED_START + SCENE_INDEX * 100000 + PAIR_INDEX * 10000 + MODE_INDEX * 1000 + trial * 10))
        run_id="${scene}__${pair_name}__${trial_mode}__trial_$(printf '%03d' "$trial")"
        run_dir="$OUTPUT_ROOT/runs/$run_id"
        mkdir -p "$run_dir"

        ref_out="$run_dir/ref_transformed.ply"
        src_out="$run_dir/src_transformed.ply"
        meta_json="$run_dir/ground_truth.json"
        demo_out="$run_dir/demo_outputs"
        metrics_json="$run_dir/metrics.json"
        log_file="$run_dir/run.log"
        est_npz="$demo_out/estimated_transform.npz"

        echo "[$(timestamp)] [run] $run_id"
        echo "[$(timestamp)] [run] mode=$trial_mode xyz_only=$mode_xyz_only rotate_sh=$mode_rotate_sh seed=$seed"
        echo "[$(timestamp)] [run] output dir: $run_dir"
        TOTAL_RUNS=$((TOTAL_RUNS + 1))

        randomize_elapsed=""
        demo_elapsed=""

        if [[ "$DRY_RUN" -eq 1 ]]; then
          append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
            "dry_run" "" "$randomize_elapsed" "$demo_elapsed" \
            "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
          cleanup_run_artifacts "$run_dir"
          continue
        fi

        randomize_start=$(date +%s)
        if [[ "$SKIP_RANDOMIZE" -eq 1 ]]; then
          if ! prepare_identity_run_artifacts \
            "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$seed" "$mode_xyz_only" "$mode_rotate_sh" \
            > "$run_dir/randomize.stdout.json" 2> "$run_dir/randomize.stderr.log"; then
            randomize_elapsed=$(( $(date +%s) - randomize_start ))
            FAILED_RUNS=$((FAILED_RUNS + 1))
            err="identity_prepare_failed"
            append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
              "failed" "$err" "$randomize_elapsed" "$demo_elapsed" \
              "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
            cleanup_heavy_outputs "$ref_out" "$src_out" "$demo_out"
            cleanup_run_artifacts "$run_dir"
            continue
          fi
          randomize_elapsed=$(( $(date +%s) - randomize_start ))
          echo "[$(timestamp)] [run] randomize skipped; identity inputs prepared in ${randomize_elapsed}s"
        else
          randomize_cmd=(
            "$PYTHON_BIN" "$SCRIPT_DIR/randomize_gaussian_pair.py"
            --ref-in "$ref_ply"
            --src-in "$src_ply"
            --ref-out "$ref_out"
            --src-out "$src_out"
            --meta-out "$meta_json"
            --seed "$seed"
            --max-translation "$MAX_TRANSLATION"
            --max-rotation-deg "$MAX_ROT_DEG"
          )
          if [[ "$mode_xyz_only" -eq 1 ]]; then
            randomize_cmd+=(--xyz-only)
          else
            if [[ "$mode_rotate_sh" -eq 1 ]]; then
              randomize_cmd+=(--rotate-sh)
            else
              randomize_cmd+=(--no-rotate-sh)
            fi
          fi

          if ! "${randomize_cmd[@]}" > "$run_dir/randomize.stdout.json" 2> "$run_dir/randomize.stderr.log"; then
            randomize_elapsed=$(( $(date +%s) - randomize_start ))
            FAILED_RUNS=$((FAILED_RUNS + 1))
            err="randomize_failed"
            append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
              "failed" "$err" "$randomize_elapsed" "$demo_elapsed" \
              "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
            cleanup_heavy_outputs "$ref_out" "$src_out" "$demo_out"
            cleanup_run_artifacts "$run_dir"
            continue
          fi
          randomize_elapsed=$(( $(date +%s) - randomize_start ))
          echo "[$(timestamp)] [run] randomize done in ${randomize_elapsed}s"
        fi

        mkdir -p "$demo_out"
        echo "[$(timestamp)] [run] GaussReg demo started (log: $log_file)"
        demo_start=$(date +%s)
        (
          cd "$GAUSSREG_ROOT"
          "$PYTHON_BIN" -u experiments/geotransformer.gaussian_splatting.indoor/demo.py \
            --src_file "$src_out" \
            --ref_file "$ref_out" \
            --weights "$WEIGHTS" \
            --output_path "$demo_out" \
            --num_sample "$NUM_SAMPLE"
        ) > "$log_file" 2>&1 &
        demo_pid=$!
        while kill -0 "$demo_pid" 2>/dev/null; do
          sleep "$HEARTBEAT_SEC"
          if kill -0 "$demo_pid" 2>/dev/null; then
            demo_elapsed=$(( $(date +%s) - demo_start ))
            echo "[$(timestamp)] [run] GaussReg demo still running (${demo_elapsed}s): $run_id"
          fi
        done
        if ! wait "$demo_pid"; then
          demo_elapsed=$(( $(date +%s) - demo_start ))
          FAILED_RUNS=$((FAILED_RUNS + 1))
          err="gaussreg_demo_failed"
          append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
            "failed" "$err" "$randomize_elapsed" "$demo_elapsed" \
            "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
          cleanup_heavy_outputs "$ref_out" "$src_out" "$demo_out"
          cleanup_run_artifacts "$run_dir"
          continue
        fi
        demo_elapsed=$(( $(date +%s) - demo_start ))
        echo "[$(timestamp)] [run] GaussReg demo finished in ${demo_elapsed}s"

        if [[ ! -f "$est_npz" ]]; then
          FAILED_RUNS=$((FAILED_RUNS + 1))
          err="estimated_transform_missing"
          append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
            "failed" "$err" "$randomize_elapsed" "$demo_elapsed" \
            "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
          cleanup_heavy_outputs "$ref_out" "$src_out" "$demo_out"
          cleanup_run_artifacts "$run_dir"
          continue
        fi

        if [[ "$RUN_EVAL" -eq 1 ]]; then
          if ! "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_gaussreg_result.py" \
            --gt-json "$meta_json" \
            --estimated-npz "$est_npz" \
            --output-json "$metrics_json" > "$run_dir/eval.stdout.json" 2> "$run_dir/eval.stderr.log"; then
            FAILED_RUNS=$((FAILED_RUNS + 1))
            err="evaluation_failed"
            append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
              "failed" "$err" "$randomize_elapsed" "$demo_elapsed" \
              "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
            cleanup_heavy_outputs "$ref_out" "$src_out" "$demo_out"
            cleanup_run_artifacts "$run_dir"
            continue
          fi
        fi

        SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
        append_row "$scene" "$pair_name" "$trial_mode" "$trial" "$seed" "$mode_xyz_only" "$mode_rotate_sh" "$ref_split" "$src_split" \
          "ok" "" "$randomize_elapsed" "$demo_elapsed" \
          "$ref_ply" "$src_ply" "$ref_out" "$src_out" "$meta_json" "$est_npz" "$metrics_json" "$demo_out" "$log_file"
        cleanup_heavy_outputs "$ref_out" "$src_out" "$demo_out"
        cleanup_run_artifacts "$run_dir"
        echo "[$(timestamp)] [run] completed: $run_id (success=$SUCCESS_RUNS failed=$FAILED_RUNS total=$TOTAL_RUNS)"
      done

      MODE_INDEX=$((MODE_INDEX + 1))
    done

    PAIR_INDEX=$((PAIR_INDEX + 1))
  done

  SCENE_INDEX=$((SCENE_INDEX + 1))
done

echo ""
echo "========================================"
echo "Done"
echo "========================================"
echo "Total scenes:           $TOTAL_SCENES"
echo "Scenes processed:       $SCENES_PROCESSED"
echo "Scenes skipped(shard):  $SCENES_SKIPPED_BY_SHARD"
echo "Total runs:             $TOTAL_RUNS"
echo "Success:                $SUCCESS_RUNS"
echo "Failed:                 $FAILED_RUNS"
echo "Results CSV:            $RESULTS_CSV"
echo "Output root:            $OUTPUT_ROOT"
