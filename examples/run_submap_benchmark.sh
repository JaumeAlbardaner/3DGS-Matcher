#!/usr/bin/env bash
# run_submap_benchmark.sh
#
# Global registration benchmark over all 3DGS submap scene pairs.
# Matches split_pos vs split_neg (and split_wide_pos vs split_wide_neg) for each scene.
#
# One benchmark_submap process per seed — TBB/Eigen heap state is fully reset each time.
# Crashed seeds (exit 139 SIGSEGV) and too-sparse scenes (exit 2) are skipped gracefully.
#
# Usage:
#   ./run_submap_benchmark.sh [output.csv]

set -uo pipefail   # no -e so we can capture non-zero exit codes from the benchmark

BENCH="$(cd "$(dirname "$0")/.." && pwd)/build/benchmark_submap"
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
BASE="${BASE:-$DATA_ROOT/submap_gaussians}"
PLY_SUFFIX="point_cloud/iteration_30000/point_cloud.ply"
OUT="${1:-submap_benchmark_$(date +%Y%m%d_%H%M%S).csv}"

VOXEL=${VOXEL:-0.3}
MAX_ROT=${MAX_ROT:-180.0}
MAX_TRANS=${MAX_TRANS:-500.0}
COLOR_WEIGHT_ICP=${COLOR_WEIGHT_ICP:-1.0}
RUN_ICP=${RUN_ICP:-1}
TARGET_VALID=${TARGET_VALID:-100}  # collect this many valid (non-crashed, non-sparse) seeds per pair
MAX_SEEDS=${MAX_SEEDS:-300}        # upper bound to prevent infinite loops

ROT_THRESH=5.0     # degrees — must match benchmark_submap.cc
TRANS_THRESH=1.0   # metres

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: benchmark binary not found or not executable: $BENCH"
    echo "  Run: cd build && cmake .. && make benchmark_submap"
    exit 1
fi

# CSV header
echo "method,scene,split_src,split_tgt,iter,seed,rot_error_deg,trans_error_m,success,solver_ok,tx,ty,tz,qw,qx,qy,qz,time_ms" > "$OUT"
echo "Benchmark output → $OUT"
echo "Params: voxel=${VOXEL}m  max_rot=${MAX_ROT}°  max_trans=${MAX_TRANS}m  color_weight_icp=${COLOR_WEIGHT_ICP}  run_icp=${RUN_ICP}  target=${TARGET_VALID} valid seeds/pair"
echo ""

# ── run_pair <scene> <src_split> <tgt_split> ─────────────────────────────────
run_pair() {
    local scene="$1" src_split="$2" tgt_split="$3"
    local src="$BASE/$scene/$src_split/$PLY_SUFFIX"
    local tgt="$BASE/$scene/$tgt_split/$PLY_SUFFIX"
    local pair_label="$scene  $src_split → $tgt_split"

    if [[ ! -f "$src" || ! -f "$tgt" ]]; then
        printf "  SKIP %-50s (PLY missing)\n" "$pair_label"
        return 0
    fi

    printf "\n╔══ %s\n" "$pair_label"

    # Per-method success counters (updated from CSV after each seed)
    local valid=0 crashed=0 seed=0
    # Running totals per method: ok counts
    local vk_ok=0 vk_n=0
    local cc_ok=0 cc_n=0
    local gi_ok=0 gi_n=0
    local cg_ok=0 cg_n=0
    local gic_ok=0 gic_n=0
    local cgc_ok=0 cgc_n=0
    local cci_ok=0 cci_n=0
    local cfi_ok=0 cfi_n=0

    local tmp
    while [[ $valid -lt $TARGET_VALID && $seed -lt $MAX_SEEDS ]]; do
        tmp=$(mktemp /tmp/bench_XXXXXX.csv)

        local ec=0
        "$BENCH" "$src" "$tgt" "$VOXEL" "$MAX_ROT" "$MAX_TRANS" "$seed" "$tmp" "$COLOR_WEIGHT_ICP" "$RUN_ICP" \
            2>/dev/null || ec=$?

        case "$ec" in
            0)
                if [[ -s "$tmp" ]]; then
                    # Parse each method's result from this seed's CSV
                    local seed_summary=""
                    while IFS=',' read -r method _scene _ssrc _stgt _iter _seed rot trans ok _sok _tx _ty _tz _qw _qx _qy _qz _ms; do
                        local this_sym="✗"; [[ "$ok" == "1" ]] && this_sym="✓"
                        case "$method" in
                            vanilla_kiss)    vk_n=$((vk_n+1));   [[ "$ok" == "1" ]] && vk_ok=$((vk_ok+1))   || true; seed_summary+=" vk:${this_sym}" ;;
                            color_cpfh)      cc_n=$((cc_n+1));   [[ "$ok" == "1" ]] && cc_ok=$((cc_ok+1))   || true; seed_summary+=" cc:${this_sym}" ;;
                            gicp)            gi_n=$((gi_n+1));   [[ "$ok" == "1" ]] && gi_ok=$((gi_ok+1))   || true; seed_summary+=" gi:${this_sym}" ;;
                            color_gicp)      cg_n=$((cg_n+1));   [[ "$ok" == "1" ]] && cg_ok=$((cg_ok+1))   || true; seed_summary+=" cg:${this_sym}" ;;
                            gicp_cpfh)       gic_n=$((gic_n+1)); [[ "$ok" == "1" ]] && gic_ok=$((gic_ok+1)) || true; seed_summary+=" gic:${this_sym}" ;;
                            color_gicp_cpfh) cgc_n=$((cgc_n+1)); [[ "$ok" == "1" ]] && cgc_ok=$((cgc_ok+1)) || true; seed_summary+=" cgc:${this_sym}" ;;
                            color_corr_icp)  cci_n=$((cci_n+1)); [[ "$ok" == "1" ]] && cci_ok=$((cci_ok+1)) || true; seed_summary+=" cci:${this_sym}" ;;
                            color_full_icp)  cfi_n=$((cfi_n+1)); [[ "$ok" == "1" ]] && cfi_ok=$((cfi_ok+1)) || true; seed_summary+=" cfi:${this_sym}" ;;
                        esac
                    done < "$tmp"

                    cat "$tmp" >> "$OUT"
                    valid=$((valid + 1))
                    printf "  seed %3d: ok   %s   (%d/%d valid)\n" \
                        "$seed" "$seed_summary" "$valid" "$TARGET_VALID"
                fi
                ;;
            2)
                printf "  seed %3d: SPARSE — scene too sparse at voxel=%.1fm, skipping pair\n" "$seed" "$VOXEL"
                rm -f "$tmp"
                return 0
                ;;
            139|134)
                crashed=$((crashed + 1))
                printf "  seed %3d: CRASH (exit %d)\n" "$seed" "$ec"
                ;;
            *)
                crashed=$((crashed + 1))
                printf "  seed %3d: ERROR (exit %d)\n" "$seed" "$ec"
                ;;
        esac

        rm -f "$tmp"
        seed=$((seed + 1))
    done

    # Summary for this pair
    local warn=""
    [[ $valid -lt $TARGET_VALID ]] && warn=" ⚠ only $valid/$TARGET_VALID collected"
    printf "╚══ Done: %d valid, %d crashed, %d seeds attempted%s\n" \
        "$valid" "$crashed" "$seed" "$warn"

    # Per-method success rate
    [[ $vk_n  -gt 0 ]] && printf "  vanilla_kiss      %d/%d (%.0f%%)\n" "$vk_ok"  "$vk_n"  "$(awk "BEGIN{printf \"%.0f\", 100*$vk_ok/$vk_n}")"
    [[ $cc_n  -gt 0 ]] && printf "  color_cpfh        %d/%d (%.0f%%)\n" "$cc_ok"  "$cc_n"  "$(awk "BEGIN{printf \"%.0f\", 100*$cc_ok/$cc_n}")"
    [[ $gi_n  -gt 0 ]] && printf "  gicp              %d/%d (%.0f%%)\n" "$gi_ok"  "$gi_n"  "$(awk "BEGIN{printf \"%.0f\", 100*$gi_ok/$gi_n}")"
    [[ $cg_n  -gt 0 ]] && printf "  color_gicp        %d/%d (%.0f%%)\n" "$cg_ok"  "$cg_n"  "$(awk "BEGIN{printf \"%.0f\", 100*$cg_ok/$cg_n}")"
    [[ $gic_n -gt 0 ]] && printf "  gicp_cpfh         %d/%d (%.0f%%)\n" "$gic_ok" "$gic_n" "$(awk "BEGIN{printf \"%.0f\", 100*$gic_ok/$gic_n}")"
    [[ $cgc_n -gt 0 ]] && printf "  color_gicp_cpfh   %d/%d (%.0f%%)\n" "$cgc_ok" "$cgc_n" "$(awk "BEGIN{printf \"%.0f\", 100*$cgc_ok/$cgc_n}")"
    [[ $cci_n -gt 0 ]] && printf "  color_corr_icp    %d/%d (%.0f%%)\n" "$cci_ok" "$cci_n" "$(awk "BEGIN{printf \"%.0f\", 100*$cci_ok/$cci_n}")"
    [[ $cfi_n -gt 0 ]] && printf "  color_full_icp    %d/%d (%.0f%%)\n" "$cfi_ok" "$cfi_n" "$(awk "BEGIN{printf \"%.0f\", 100*$cfi_ok/$cfi_n}")"
}

# ── Scene list ────────────────────────────────────────────────────────────────
DEFAULT_SCENES=(bicycle bonsai flowers garden stump treehill drone1_undistorted drone2_undistorted ugv_undistorted)
if [[ -n "${SCENE_LIST:-}" ]]; then
    read -ra SCENES <<< "$SCENE_LIST"
else
    SCENES=("${DEFAULT_SCENES[@]}")
fi

T_START=$SECONDS

for scene in "${SCENES[@]}"; do
    # PAIR_LIST="src:tgt ..." overrides the default split pairs (e.g. KITTI robots)
    if [[ -n "${PAIR_LIST:-}" ]]; then
        for pair in $PAIR_LIST; do
            run_pair "$scene" "${pair%%:*}" "${pair##*:}"
        done
        continue
    fi
    run_pair "$scene" "split_pos" "split_neg"
    if [[ -d "$BASE/$scene/split_wide_pos" && -d "$BASE/$scene/split_wide_neg" ]]; then
        run_pair "$scene" "split_wide_pos" "split_wide_neg"
    fi
done

ELAPSED=$(( SECONDS - T_START ))
printf "\nTotal runtime: %dh %dm %ds\n" \
    "$(( ELAPSED / 3600 ))" "$(( (ELAPSED % 3600) / 60 ))" "$(( ELAPSED % 60 ))"

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════ FINAL RESULTS ══════════════════════════════════"
python3 - "$OUT" <<'PYEOF'
import sys, csv, collections

path = sys.argv[1]
data        = collections.defaultdict(lambda: collections.defaultdict(list))
solver_data = collections.defaultdict(lambda: collections.defaultdict(list))

with open(path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            ok  = int(row['success'])
            sok = int(row['solver_ok'])
        except (ValueError, KeyError):
            continue
        if ok < 0:
            continue
        label = f"{row['scene']}  {row['split_src']}→{row['split_tgt']}"
        data[label][row['method']].append(ok)
        if sok >= 0:
            solver_data[label][row['method']].append(sok)

methods = ['vanilla_kiss', 'color_cpfh', 'gicp_cpfh', 'color_gicp_de', 'color_gicp_cpfh', 'color_corr_icp', 'color_full_icp']
labels  = sorted(data.keys())

if not labels:
    print("No results found in", path)
    sys.exit(0)

lw = 52
cw = 17
header = f"{'Scene pair':<{lw}}" + "".join(f"{m:>{cw}}" for m in methods)
sep    = "─" * len(header)
print(header)
print(sep)

for label in labels:
    row_s = f"{label:<{lw}}"
    for m in methods:
        vals = data[label][m]
        if vals:
            pct = sum(vals) / len(vals) * 100
            row_s += f"{pct:>{cw-7}.1f}% (n={len(vals):3d})"
        else:
            row_s += f"{'—':>{cw}}"
    print(row_s)

print(sep)
print(f"{'MEAN (success)':<{lw}}", end="")
for m in methods:
    all_vals = [v for lbl in labels for v in data[lbl][m]]
    if all_vals:
        pct = sum(all_vals) / len(all_vals) * 100
        print(f"{pct:>{cw-7}.1f}% (n={len(all_vals):3d})", end="")
    else:
        print(f"{'—':>{cw}}", end="")
print()

print(sep)
print(f"\n{'Scene pair':<{lw}}" + "".join(f"{m:>{cw}}" for m in methods))
print("  (solver_ok rate)")
print(sep)
for label in labels:
    row_s = f"{label:<{lw}}"
    for m in methods:
        vals = solver_data[label][m]
        if vals:
            pct = sum(vals) / len(vals) * 100
            row_s += f"{pct:>{cw-7}.1f}% (n={len(vals):3d})"
        else:
            row_s += f"{'—':>{cw}}"
    print(row_s)
print(sep)
PYEOF
