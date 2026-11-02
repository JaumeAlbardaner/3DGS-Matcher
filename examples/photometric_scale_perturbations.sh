#!/usr/bin/env bash
# photometric_scale_perturbations.sh
#
# Tramway robustness study (paper Sec. V-B, "Photometric and scale perturbations"):
#   photometric changes a cross-platform map pair can realistically show
#   residual metric-scale error left by the external anchoring
#
# Unlike noise_ablation_doubts.sh (which ties colour noise to the position-noise
# sweep), every condition here applies ONE named, physical change to the target
# copy, with a fixed 2 cm position jitter and no added colour noise, so each row
# reads directly. A `clean` condition from the same binary is the reference.
#
#   clean            rigid transform + position jitter only
#   exposure_0.7/1.4 linear-light gain on all channels (about -0.5 / +0.5 EV)
#   gamma_0.8/1.25   tone-curve change on the encoded values, c' = c^g
#   wb_warm/wb_cool  white balance: red x1.15 / blue x0.85, and the reverse
#   shadow_0.4       half the map (beyond the median plane of its longest axis)
#                    darkened to 40 % in linear light
#   scale_1.01..1.10 target scaled about its centroid by +1/2/5/10 %
#
# Configs (all joint-5D, 0.3 m voxel, as Table I): ColorPFH only; + raw RGB;
# + raw RGB + opacity (production). Success = 5 deg / 0.5 m, 100 seeds each.
#
# Usage: ./photometric_scale_perturbations.sh [out.csv]      (append-safe, resumable)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SA="$REPO/build/single_ablation"
SPLAT="$REPO/examples/splats/tramway.splat"
OUT="${1:-${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}/results/tramway_perturbations.csv}"
NOISE=0.02
SEEDS=($(seq 1 100))

read -ra EXPERIMENTS <<< "${EXPERIMENTS:-clean exposure_0.7 exposure_1.4 gamma_0.8 gamma_1.25 wb_warm wb_cool shadow_0.4 scale_1.01 scale_1.02 scale_1.05 scale_1.10}"

# "color opacity cpfh"   (joint-5D and centroid flags are fixed below)
declare -A CFGS
CFGS["ColorPFH"]="0 0 1"
CFGS["RGB+ColorPFH"]="1 0 1"
CFGS["RGB+ColorPFH+opacity"]="1 1 1"
CFG_ORDER=("ColorPFH" "RGB+ColorPFH" "RGB+ColorPFH+opacity")

[[ -x "$SA" ]] || { echo "ERROR: $SA not built"; exit 1; }
mkdir -p "$(dirname "$OUT")"
[[ -f "$OUT" ]] || echo "config,experiment,seed,success,rot_err,trans_err,corr,time_ms" > "$OUT"

echo "Output -> $OUT"
echo "${#EXPERIMENTS[@]} conditions x ${#CFG_ORDER[@]} configs x ${#SEEDS[@]} seeds"
T0=$SECONDS

for exp in "${EXPERIMENTS[@]}"; do
  for cfg in "${CFG_ORDER[@]}"; do
    read -ra a <<< "${CFGS[$cfg]}"
    color="${a[0]}"; opacity="${a[1]}"; cpfh="${a[2]}"
    done_n=$(awk -F, -v c="$cfg" -v e="$exp" '$1==c && $2==e' "$OUT" | wc -l)
    if (( done_n >= ${#SEEDS[@]} )); then echo "-- skip $exp / $cfg (done)"; continue; fi
    echo "-- $exp / $cfg"
    for seed in "${SEEDS[@]}"; do
      awk -F, -v c="$cfg" -v e="$exp" -v s="$seed" '$1==c && $2==e && $3==s {f=1} END{exit !f}' "$OUT" && continue
      line=$("$SA" "$SPLAT" 0.3 \
        0 0 0 "$color" "$opacity" 11 \
        "$NOISE" 1.0 "$seed" \
        500.0 180.0 0 "$exp" color \
        1 5.0 2 0 0.1 \
        0 1.0 11 50.0 \
        "$cpfh" 1.0 11 100.0 2>/dev/null) || true
      if [[ -n "$line" ]]; then
        rot=$(echo "$line"   | awk -F',' '{print $(NF-1)}')
        trans=$(echo "$line" | awk -F',' '{print $NF}')
        corr=$(echo "$line"  | awk -F',' '{print $(NF-2)}')
        tms=$(echo "$line"   | awk -F',' '{print $(NF-3)}')
        ok=0; awk "BEGIN{exit !($rot < 5.0 && $trans < 0.5)}" && ok=1 || true
        echo "$cfg,$exp,$seed,$ok,$rot,$trans,$corr,$tms" >> "$OUT"
      else
        echo "$cfg,$exp,$seed,0,nan,nan,0,nan" >> "$OUT"   # crash counts as failure
      fi
    done
  done
done

EL=$(( SECONDS - T0 ))
printf "\nDone in %dm %ds -> %s\n" "$(( EL/60 ))" "$(( EL%60 ))" "$OUT"
