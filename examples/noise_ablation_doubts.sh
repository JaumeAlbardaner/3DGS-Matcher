#!/usr/bin/env bash
# noise_ablation_doubts.sh
#
# Focused ablation on tramway.splat to (re)validate three design decisions that
# predate the ColorPFH descriptor and were never tested in the ColorPFH regime:
#
#   1. Opacity — dropped during the June 2026 rewrite, though old raw-4D data
#      showed alpha ~doubled success. Configs 2 vs 4 (and 1 vs 3) test this.
#   2. Raw RGB — kept on top of ColorPFH but never isolated. Configs 1 vs 2 test this.
#   3. joint5D color-aware downsampling — dropped (tramway, old descriptors: no
#      success gain, 3.5x slower). Configs 2 vs 5 re-test it with ColorPFH.
#
# Anchor = production config (color+colorPFH, centroid, opacity off) = config 2.
# Emits a tidy CSV (one row per run) for data-driven plotting.
#
# Usage: ./noise_ablation_doubts.sh [splat_file] [out.csv]

set -uo pipefail

SA="$(dirname "$0")/../build/single_ablation"
SPLAT="${1:-$(dirname "$0")/splats/tramway.splat}"
OUT="${2:-$(cd "$(dirname "$0")/.." && pwd)/results/noise_doubts_tramway.csv}"

SEEDS=($(seq 1 100))
NOISE_LEVELS=(0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20)
# Override with e.g. EXPERIMENTS="outliers split_half" to run/append only those.
read -ra EXPERIMENTS <<< "${EXPERIMENTS:-baseline color_offset outliers split_half}"

# Per-config flags: "color opacity cpfh centroid joint5d"
declare -A CFGS
CFGS["colorPFH-only"]="0 0 1 1 0"
CFGS["color+colorPFH"]="1 0 1 1 0"      # PRODUCTION anchor
CFGS["colorPFH+opacity"]="0 1 1 1 0"
CFGS["color+cPFH+opacity"]="1 1 1 1 0"
CFGS["color+colorPFH(joint5D)"]="1 0 1 0 1"
CFGS["color+cPFH+opacity(joint5D)"]="1 1 1 0 1"
CFG_ORDER=("colorPFH-only" "color+colorPFH" "colorPFH+opacity" "color+cPFH+opacity" "color+colorPFH(joint5D)" "color+cPFH+opacity(joint5D)")

mkdir -p "$(dirname "$OUT")"

# Tidy header: our label/experiment columns + the raw single_ablation CSV columns.
# Append-safe: only write the header when starting a fresh file, so a second run
# with EXPERIMENTS="outliers split_half" merges into the existing CSV.
[[ -f "$OUT" ]] || echo "config_label,experiment,noise,seed,success,rot_err,trans_err,corr,time_ms,raw_name" > "$OUT"

if [[ ! -x "$SA" ]]; then
    echo "ERROR: single_ablation not built at $SA"; exit 1
fi

echo "Output → $OUT"
echo "Splat: $SPLAT   ±180°/±500m   ${#SEEDS[@]} seeds × ${#NOISE_LEVELS[@]} noise × ${#CFG_ORDER[@]} configs × ${#EXPERIMENTS[@]} exp"
T0=$SECONDS

for exp in "${EXPERIMENTS[@]}"; do
  for cfg in "${CFG_ORDER[@]}"; do
    read -ra a <<< "${CFGS[$cfg]}"
    color="${a[0]}"; opacity="${a[1]}"; cpfh="${a[2]}"; centroid="${a[3]}"; joint5d="${a[4]}"
    echo "── exp=$exp  cfg=$cfg  (color=$color opacity=$opacity cpfh=$cpfh centroid=$centroid joint5d=$joint5d)"
    for noise in "${NOISE_LEVELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        line=$("$SA" "$SPLAT" 0.3 \
          0 0 0 "$color" "$opacity" 11 \
          "$noise" 1.0 "$seed" \
          500.0 180.0 "$centroid" "$exp" color \
          "$joint5d" 5.0 2 0 0.1 \
          0 1.0 11 50.0 \
          "$cpfh" 1.0 11 100.0 2>/dev/null) || true
        if [[ -n "$line" ]]; then
          name=$(echo "$line"    | awk -F',' '{print $1}')
          rot=$(echo "$line"     | awk -F',' '{print $(NF-1)}')
          trans=$(echo "$line"   | awk -F',' '{print $NF}')
          corr=$(echo "$line"    | awk -F',' '{print $(NF-2)}')
          tms=$(echo "$line"     | awk -F',' '{print $(NF-3)}')
          ok=0; awk "BEGIN{exit !($rot < 5.0 && $trans < 0.5)}" && ok=1 || true
          echo "$cfg,$exp,$noise,$seed,$ok,$rot,$trans,$corr,$tms,$name" >> "$OUT"
        else
          # crash / failure → record as failed run so success rate denominators stay honest
          echo "$cfg,$exp,$noise,$seed,0,nan,nan,0,nan,CRASH" >> "$OUT"
        fi
      done
    done
  done
done

EL=$(( SECONDS - T0 ))
printf "\nDone in %dm %ds → %s\n" "$(( EL/60 ))" "$(( EL%60 ))" "$OUT"
