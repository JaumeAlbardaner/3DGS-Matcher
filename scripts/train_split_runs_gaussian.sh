#!/usr/bin/env bash
set -euo pipefail

# Batch-train Gaussian Splatting models for all COLMAP scenes under split_runs.
# A valid scene must contain:
#   - images/
#   - sparse/0/{cameras,images,points3D}.{bin|txt}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SPLIT_ROOT="${1:-$PROJECT_ROOT/split_runs}"
OUTPUT_ROOT="${2:-$SPLIT_ROOT/gaussian_splatting_runs}"

if [[ -n "${GS_REPO:-}" ]]; then
  DEFAULT_GS_REPO="$GS_REPO"
else
  DEFAULT_GS_REPO="$(pwd)/gaussian-splatting"   # or set GS_REPO
fi

GAUSSIAN_REPO="${GAUSSIAN_REPO:-$DEFAULT_GS_REPO}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

if [[ ! -d "$SPLIT_ROOT" ]]; then
  echo "ERROR: split root does not exist: $SPLIT_ROOT" >&2
  exit 1
fi

if [[ ! -f "$GAUSSIAN_REPO/train.py" ]]; then
  echo "ERROR: train.py not found at: $GAUSSIAN_REPO/train.py" >&2
  echo "Set GAUSSIAN_REPO=/path/to/gaussian-splatting and rerun." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

has_colmap_model() {
  local scene_dir="$1"
  local sparse_zero="$scene_dir/sparse/0"

  [[ -d "$scene_dir/images" ]] || return 1
  [[ -d "$sparse_zero" ]] || return 1

  [[ -f "$sparse_zero/images.bin" || -f "$sparse_zero/images.txt" ]] || return 1
  [[ -f "$sparse_zero/cameras.bin" || -f "$sparse_zero/cameras.txt" ]] || return 1
  [[ -f "$sparse_zero/points3D.bin" || -f "$sparse_zero/points3D.txt" ]] || return 1
}

has_trained_output() {
  local model_dir="$1"
  find "$model_dir/point_cloud" -mindepth 2 -maxdepth 2 -type f -name point_cloud.ply -print -quit 2>/dev/null | grep -q .
}

# shellcheck disable=SC2206
EXTRA_ARGS=( $TRAIN_EXTRA_ARGS )

while IFS= read -r -d '' sparse_zero; do
  scene_dir="$(dirname "$(dirname "$sparse_zero")")"

  has_colmap_model "$scene_dir" || continue

  rel_path="${scene_dir#"$SPLIT_ROOT"/}"
  [[ "$scene_dir" == "$SPLIT_ROOT" ]] && rel_path="$(basename "$scene_dir")"

  model_dir="$OUTPUT_ROOT/$rel_path"
  train_log="$model_dir/train.log"

  mkdir -p "$model_dir"

  if [[ "$FORCE_RETRAIN" != "1" ]] && has_trained_output "$model_dir"; then
    echo "[$(timestamp)] SKIP existing: $rel_path"
    continue
  fi

  cmd=(
    "$PYTHON_BIN"
    "$GAUSSIAN_REPO/train.py"
    -s "$scene_dir"
    -m "$model_dir"
    --disable_viewer
  )

  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[$(timestamp)] TRAIN $rel_path"
  echo "  source: $scene_dir"
  echo "  model : $model_dir"
  echo -n "  cmd   :"
  printf " %q" "${cmd[@]}"
  echo

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  {
    printf '$'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
    "${cmd[@]}"
  } 2>&1 | tee "$train_log"
done < <(find "$SPLIT_ROOT" -type d -path '*/sparse/0' -print0 | sort -z)

echo "[$(timestamp)] Done. Outputs are under: $OUTPUT_ROOT"
