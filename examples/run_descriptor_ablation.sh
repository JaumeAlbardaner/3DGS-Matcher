#!/bin/bash
#
# Descriptor-family ablation (paper Fig. 4): 31 feature configurations on the
# tramway splat, 0.1 m voxel. Score with
# paper_data/figure4_descriptor_ablation/success_getter.py.
#
# Feature configs:
#   - Exhaustive non-empty combinations of:
#     f1, f2, f3, RGB color, alpha/opacity
#   - Total: 31 configurations
#
# Experiments:
#   - baseline
#   - outliers
#   - color_offset
#   - split_half
#
# color_weight sweep:
#   - Applied only when RGB or opacity is enabled
#   - Geometry-only configurations run once per setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
SPLAT_FILE=""
OUTPUT_FILE="ablation_results.csv"
VOXEL_SIZE=0.1
ITERATIONS=100
NOISE_LEVELS="0,0.025,0.05,0.075,0.1,0.15,0.2"
EXPERIMENTS="baseline,outliers,color_offset,split_half"
COLOR_WEIGHTS="1.0,10.0,50.0,100.0,200.0"
SEED_START=42
MAX_TRANSLATION=500.0
MAX_ROTATION=180.0
CENTROID=1
INCLUDE_VANILLA=0
ONLY_CONFIGS=""
USE_JOINT_5D=0
VOXEL_AB=5.0
MIN_COLOR_BINS=2
USE_CROSS_COLOR_FILTER=0
CROSS_COLOR_MIN_OVERLAP=0.1
USE_DELTA_E_HIST=0
DELTA_E_RADIUS=1.0
DELTA_E_BINS=11
DELTA_E_MAX=50.0
USE_COLOR_PFH=0
COLOR_PFH_RADIUS=1.0
COLOR_PFH_BINS=11
COLOR_PFH_MAX_DE=100.0

BINS=11

while [[ $# -gt 0 ]]; do
  case $1 in
    --splat-file) SPLAT_FILE="$2"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --voxel-size) VOXEL_SIZE="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --noise) NOISE_LEVELS="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --color-weights) COLOR_WEIGHTS="$2"; shift 2 ;;
    --only-configs) ONLY_CONFIGS="$2"; shift 2 ;;
    --seed-start) SEED_START="$2"; shift 2 ;;
    --max-translation) MAX_TRANSLATION="$2"; shift 2 ;;
    --max-rotation) MAX_ROTATION="$2"; shift 2 ;;
    --centroid) CENTROID=1; shift ;;
    --joint-5d) USE_JOINT_5D=1; shift ;;
    --voxel-ab) VOXEL_AB="$2"; shift 2 ;;
    --min-color-bins) MIN_COLOR_BINS="$2"; shift 2 ;;
    --cross-color-filter) USE_CROSS_COLOR_FILTER=1; shift ;;
    --cross-color-min-overlap) CROSS_COLOR_MIN_OVERLAP="$2"; shift 2 ;;
    --delta-e-hist) USE_DELTA_E_HIST=1; shift ;;
    --delta-e-radius) DELTA_E_RADIUS="$2"; shift 2 ;;
    --delta-e-bins) DELTA_E_BINS="$2"; shift 2 ;;
    --delta-e-max) DELTA_E_MAX="$2"; shift 2 ;;
    --color-pfh) USE_COLOR_PFH=1; shift ;;
    --color-pfh-radius) COLOR_PFH_RADIUS="$2"; shift 2 ;;
    --color-pfh-bins) COLOR_PFH_BINS="$2"; shift 2 ;;
    --color-pfh-max-de) COLOR_PFH_MAX_DE="$2"; shift 2 ;;
    --include-vanilla) INCLUDE_VANILLA=1; shift ;;
    --quick)
      ITERATIONS=3
      NOISE_LEVELS="0,0.05,0.1"
      EXPERIMENTS="baseline,outliers,color_offset,split_half"
      shift ;;
    -h|--help)
      echo "Usage: $0 --splat-file <path> [options]"
      echo ""
      echo "  --output <path>           Output CSV file (default: ablation_results.csv)"
      echo "  --voxel-size <float>      Voxel size for downsampling (default: 0.1)"
      echo "  --iterations <int>        Number of iterations per setup (default: 100)"
      echo "  --noise <list>            Comma-separated noise levels"
      echo "  --experiments <list>      baseline,outliers,color_offset,split_half"
      echo "  --color-weights <list>    Color weights for configs with RGB/alpha features"
      echo "  --only-configs <list>     Comma-separated feature configs (e.g., 11111 or f1f2f3+RGBA)"
      echo "  --seed-start <int>        Starting random seed (default: 42)"
      echo "  --max-translation <float> Max translation per axis"
      echo "  --max-rotation <float>    Max rotation per axis in degrees"
      echo "  --centroid                Use centroid voxel sampling"
      echo "  --joint-5d                Enable joint spatial+color 5D voxelization"
      echo "  --voxel-ab <float>        LAB ab color voxel size in ΔE units (default: 5.0)"
      echo "  --min-color-bins <int>    Min distinct colors per spatial voxel (default: 2)"
      echo "  --cross-color-filter      Keep only points in color bins shared by both clouds"
      echo "  --cross-color-min-overlap <float>  Min overlap ratio per bin (default: 0.1)"
      echo "  --include-vanilla         Also run vanilla KISS-Matcher baseline per condition"
      echo "  --quick                   Quick mode"
      exit 0 ;;
    *)
      if [[ -f "$1" ]]; then SPLAT_FILE="$1"; shift
      else echo "Unknown option: $1"; exit 1; fi ;;
  esac
done

if [[ -z "$SPLAT_FILE" ]]; then
  echo "Error: --splat-file is required"
  exit 1
fi
if [[ ! -f "$SPLAT_FILE" ]]; then
  echo "Error: Splat file not found: $SPLAT_FILE"
  exit 1
fi

SINGLE_ABLATION="$BUILD_DIR/single_ablation"
SINGLE_SRC="$SCRIPT_DIR/single_ablation.cc"
NEED_BUILD=0
if [[ ! -x "$SINGLE_ABLATION" ]]; then
  NEED_BUILD=1
elif [[ "$SINGLE_SRC" -nt "$SINGLE_ABLATION" ]]; then
  NEED_BUILD=1
fi

if [[ $NEED_BUILD -eq 1 ]]; then
  echo "Building single_ablation..."
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  cmake .. && make -j$(nproc) single_ablation
  cd - > /dev/null
fi

IFS=',' read -ra NOISE_ARR <<< "$NOISE_LEVELS"
IFS=',' read -ra EXP_ARR <<< "$EXPERIMENTS"
IFS=',' read -ra COLOR_WEIGHT_ARR <<< "$COLOR_WEIGHTS"
IFS=$' \t\n'

normalize_config_bits() {
  local token="$1"
  local u
  u="$(echo "$token" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]')"

  # Direct binary spec: f1f2f3coloropacity
  if [[ "$u" =~ ^[01]{5}$ ]]; then
    if [[ "$u" == "00000" ]]; then
      return 1
    fi
    echo "$u"
    return 0
  fi

  # Readable spec examples:
  # - f1f2f3+RGBA
  # - f1f2f3+RGB
  # - f1f2f3+A
  # - RGB+A
  local f1=0
  local f2=0
  local f3=0
  local color=0
  local opacity=0

  [[ "$u" == *"F1"* ]] && f1=1
  [[ "$u" == *"F2"* ]] && f2=1
  [[ "$u" == *"F3"* ]] && f3=1

  if [[ "$u" == *"RGBA"* ]]; then
    color=1
    opacity=1
  else
    [[ "$u" == *"RGB"* || "$u" == *"COLOR"* ]] && color=1
    [[ "$u" == *"+A"* || "$u" == "A" || "$u" == *"ALPHA"* || "$u" == *"OPACITY"* ]] && opacity=1
  fi

  local bits="${f1}${f2}${f3}${color}${opacity}"
  if [[ "$bits" == "00000" ]]; then
    return 1
  fi

  echo "$bits"
  return 0
}

bits_to_cfg_csv() {
  local bits="$1"
  echo "${bits:0:1},${bits:1:1},${bits:2:1},${bits:3:1},${bits:4:1}"
}

# Build exhaustive non-empty feature configurations as bits:
# f1 f2 f3 color opacity (excluding 00000)
ALL_CONFIG_BITS=()
for f1 in 0 1; do
  for f2 in 0 1; do
    for f3 in 0 1; do
      for color in 0 1; do
        for opacity in 0 1; do
          if [[ $f1 -eq 0 && $f2 -eq 0 && $f3 -eq 0 && $color -eq 0 && $opacity -eq 0 ]]; then
            continue
          fi
          ALL_CONFIG_BITS+=("${f1}${f2}${f3}${color}${opacity}")
        done
      done
    done
  done
done

CONFIGS=()
if [[ -n "$ONLY_CONFIGS" ]]; then
  IFS=',' read -ra CONFIG_TOKEN_ARR <<< "$ONLY_CONFIGS"
  IFS=$' \t\n'

  for token in "${CONFIG_TOKEN_ARR[@]}"; do
    [[ -z "${token// }" ]] && continue

    bits="$(normalize_config_bits "$token")" || {
      echo "Error: invalid --only-configs token '$token'" >&2
      echo "Use 5-bit binary (e.g., 11111) or readable (e.g., f1f2f3+RGBA)." >&2
      exit 1
    }

    valid=0
    for candidate in "${ALL_CONFIG_BITS[@]}"; do
      if [[ "$candidate" == "$bits" ]]; then
        valid=1
        break
      fi
    done
    if [[ $valid -eq 0 ]]; then
      echo "Error: feature config '$token' -> bits '$bits' is not a valid non-empty config." >&2
      exit 1
    fi

    cfg_csv="$(bits_to_cfg_csv "$bits")"
    exists=0
    for existing in "${CONFIGS[@]}"; do
      if [[ "$existing" == "$cfg_csv" ]]; then
        exists=1
        break
      fi
    done
    if [[ $exists -eq 0 ]]; then
      CONFIGS+=("$cfg_csv")
    fi
  done

  if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "Error: --only-configs produced no valid feature configurations." >&2
    exit 1
  fi
else
  for bits in "${ALL_CONFIG_BITS[@]}"; do
    CONFIGS+=("$(bits_to_cfg_csv "$bits")")
  done
fi

NUM_CONFIGS=${#CONFIGS[@]}
NUM_NOISE=${#NOISE_ARR[@]}
NUM_EXP=${#EXP_ARR[@]}
NUM_CW=${#COLOR_WEIGHT_ARR[@]}

TOTAL_TESTS=0
for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r f1 f2 f3 color opacity <<< "$cfg"
  if [[ $color -eq 1 || $opacity -eq 1 ]]; then
    weight_mult=$NUM_CW
  else
    weight_mult=1
  fi
  TOTAL_TESTS=$((TOTAL_TESTS + weight_mult * NUM_NOISE * NUM_EXP * ITERATIONS))
done
if [[ $INCLUDE_VANILLA -eq 1 ]]; then
  TOTAL_TESTS=$((TOTAL_TESTS + NUM_NOISE * NUM_EXP * ITERATIONS))
fi
IFS=$' \t\n'

echo "========================================"
echo "  Old-3DGS-Matcher Full Ablation"
echo "========================================"
echo "  Splat file:      $SPLAT_FILE"
echo "  Output file:     $OUTPUT_FILE"
echo "  Voxel size:      $VOXEL_SIZE"
echo "  Iterations:      $ITERATIONS"
echo "  Seed start:      $SEED_START"
echo "  Max translation: $MAX_TRANSLATION"
echo "  Max rotation:    $MAX_ROTATION deg"
echo "  Bins:            $BINS"
echo "  Noise levels:    ${NOISE_ARR[*]}"
echo "  Experiments:     ${EXP_ARR[*]}"
echo "  Color weights:   ${COLOR_WEIGHT_ARR[*]}"
echo "  Centroid voxel:  $CENTROID"
echo "  Vanilla baseline:$INCLUDE_VANILLA"
if [[ -n "$ONLY_CONFIGS" ]]; then
  echo "  Feature configs: $NUM_CONFIGS selected via --only-configs=$ONLY_CONFIGS"
else
  echo "  Feature configs: $NUM_CONFIGS non-empty combos of (f1,f2,f3,RGB,A)"
fi
echo "  Total tests:     $TOTAL_TESTS"
echo "========================================"
echo ""

echo "config,dimension,f1,f2,f3,color,opacity,bins,noise,color_weight,centroid_voxel,joint_5d,voxel_size_ab,min_color_bins,seed,total_ms,correspondences,rotation_error_deg,translation_error" > "$OUTPUT_FILE"

COMPLETED=0
FAILED=0
START_TIME=$(date +%s)
FIRST_ERROR=""

update_progress() {
  local completed=$1
  local total=$2
  local now=$(date +%s)
  local elapsed=$((now - START_TIME))

  if [[ $completed -gt 0 ]]; then
    local eta=$(( (elapsed * total / completed) - elapsed ))
    printf "\rProgress: %d/%d (%d%%) - ETA: %dm %ds     " \
      "$completed" "$total" $((completed * 100 / total)) $((eta / 60)) $((eta % 60))
  else
    printf "\rProgress: 0/%d (0%%)     " "$total"
  fi
}

echo "Running ablation study..."

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r f1 f2 f3 color opacity <<< "$cfg"
  if [[ $color -eq 0 && $opacity -eq 0 ]]; then
    WEIGHTS=("${COLOR_WEIGHT_ARR[0]}")
  else
    WEIGHTS=("${COLOR_WEIGHT_ARR[@]}")
  fi

  for exp in "${EXP_ARR[@]}"; do
    for noise in "${NOISE_ARR[@]}"; do
      for color_weight in "${WEIGHTS[@]}"; do
        for ((iter=0; iter<ITERATIONS; iter++)); do
          seed=$((SEED_START + iter))

          if result=$("$SINGLE_ABLATION" "$SPLAT_FILE" "$VOXEL_SIZE" \
                       $f1 $f2 $f3 $color $opacity $BINS \
                       "$noise" "$color_weight" "$seed" \
                       "$MAX_TRANSLATION" "$MAX_ROTATION" "$CENTROID" "$exp" "color" \
                       "$USE_JOINT_5D" "$VOXEL_AB" "$MIN_COLOR_BINS" \
                       "$USE_CROSS_COLOR_FILTER" "$CROSS_COLOR_MIN_OVERLAP" \
                       "$USE_DELTA_E_HIST" "$DELTA_E_RADIUS" "$DELTA_E_BINS" "$DELTA_E_MAX" \
                       "$USE_COLOR_PFH" "$COLOR_PFH_RADIUS" "$COLOR_PFH_BINS" "$COLOR_PFH_MAX_DE" 2>/dev/null); then
            if [[ -n "$result" ]]; then
              echo "$result" >> "$OUTPUT_FILE"
            else
              FAILED=$((FAILED + 1))
              if [[ -z "$FIRST_ERROR" ]]; then
                FIRST_ERROR="single_ablation returned success but empty output"
              fi
            fi
          else
            FAILED=$((FAILED + 1))
            if [[ -z "$FIRST_ERROR" ]]; then
              FIRST_ERROR="$result"
            fi
          fi

          COMPLETED=$((COMPLETED + 1))
          update_progress "$COMPLETED" "$TOTAL_TESTS"
        done
      done
    done
  done
done

if [[ $INCLUDE_VANILLA -eq 1 ]]; then
  echo "Running vanilla KISS-Matcher baseline..."
  f1=1; f2=1; f3=1; color=0; opacity=0
  color_weight="${COLOR_WEIGHT_ARR[0]}"
  for exp in "${EXP_ARR[@]}"; do
    for noise in "${NOISE_ARR[@]}"; do
      for ((iter=0; iter<ITERATIONS; iter++)); do
        seed=$((SEED_START + iter))

        if result=$("$SINGLE_ABLATION" "$SPLAT_FILE" "$VOXEL_SIZE" \
                     $f1 $f2 $f3 $color $opacity $BINS \
                     "$noise" "$color_weight" "$seed" \
                     "$MAX_TRANSLATION" "$MAX_ROTATION" "$CENTROID" "$exp" "vanilla" \
                     "$USE_JOINT_5D" "$VOXEL_AB" "$MIN_COLOR_BINS" 2>/dev/null); then
          if [[ -n "$result" ]]; then
            echo "$result" >> "$OUTPUT_FILE"
          else
            FAILED=$((FAILED + 1))
            if [[ -z "$FIRST_ERROR" ]]; then
              FIRST_ERROR="single_ablation returned success but empty output"
            fi
          fi
        else
          FAILED=$((FAILED + 1))
          if [[ -z "$FIRST_ERROR" ]]; then
            FIRST_ERROR="$result"
          fi
        fi

        COMPLETED=$((COMPLETED + 1))
        update_progress "$COMPLETED" "$TOTAL_TESTS"
      done
    done
  done
fi
IFS=$' \t\n'

echo ""
echo ""

echo "========================================"
echo "  Ablation Study Complete"
echo "========================================"
echo "Results saved to: $OUTPUT_FILE"
echo "Tests run:        $COMPLETED"
echo "Tests failed:     $FAILED"
if [[ -n "$FIRST_ERROR" ]]; then
  echo "First error:      $FIRST_ERROR"
fi
echo ""
echo "To visualize results:"
echo "  uv run paper_data/figure4_descriptor_ablation/success_getter.py   (reads ablation_results_RGB.csv next to it)"
echo "========================================"
