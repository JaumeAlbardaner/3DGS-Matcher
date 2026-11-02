#!/usr/bin/env python3
"""Regenerate the per-scene large-scale results table (tab:large_scale) of the
paper from the CSVs shipped alongside this script.

Self-contained version for the shared `paper_data/` folder: it reads the CSVs
sitting in the same directory, defaults to the paper's opacity-ON config, and
emits every row of the paper table -- GaussReg, ColorPCR, KISS-Matcher, and the
six "Ours" rows -- so the printed table matches the paper as published.

Per scene and method it reports the mean +/- std of:
  trans : translation error (m)   <- trans_error_m / trans_err_abs
  rot   : rotation error (deg)    <- rot_error_deg / rot_err_deg
  t     : total pipeline time (s) <- time_ms/1000; refined rows add the coarse stage
plus, as the table's last column, the overall success rate at 5 deg / 0.5 m
pooled over all trials of the nine scenes (n = 1800 per row).

Sources (all co-located in this folder):
  Ours / KISS-Matcher centroid -> centroid_trained_v050_n200_opacityON.csv
  Ours / KISS-Matcher joint-5D -> joint5d_trained_v050_n200_opacityON.csv
  GaussReg baseline            -> gaussreg_results.csv  (trial_mode=full_features)
  ColorPCR baseline            -> colorpcr_trained_nondrone_v050_n100.csv

Run from this folder:
  python3 gen_large_scale_table.py                 # opacity on (paper), all rows
  python3 gen_large_scale_table.py --centroid <csv> --joint5d <csv> ...   # overrides
Output is printed to the terminal and written to large_scale_table_body.tex.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Per-config CSV sets. The paper uses the opacity-ON runs; only those are shipped
# in this folder. "off" is kept for reference but its CSVs are not included here.
CONFIGS = {
    "on": {
        "centroid": "centroid_trained_v050_n200_opacityON.csv",
        "joint5d": "joint5d_trained_v050_n200_opacityON.csv",
    },
    "off": {
        "centroid": "centroid_trained_v050_n200.csv",
        "joint5d": "joint5d_trained_v050_n200.csv",
    },
}
GAUSSREG = HERE / "gaussreg_results.csv"
COLORPCR = HERE / "colorpcr_trained_nondrone_v050_n100.csv"
# Colored-ICP (Park et al., Open3D) refinement baseline, warm-started from the
# corresponding block's coarse poses (generator: scripts/benchmark_colored_icp.py).
# These are the runs behind the PUBLISHED rows. They predate a fix in
# benchmark_colored_icp.py: each scene's split_wide pair was replayed against the
# split_pos clouds (all rows carry split_pos/split_neg). See the repo README, "Known issues".
CICP_CENT = HERE / "coloredicp_from_centroid_v050_n200_crosspair_bug.csv"
CICP_J5 = HERE / "coloredicp_from_joint5d_v050_n200_crosspair_bug.csv"

SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden",
          "kitchen", "room", "stump", "treehill"]

ROT_THRESH = 5.0   # deg
TRANS_THRESH = 0.5  # m  (the paper's criterion -- ColorPCR is scored here too,
                    #     NOT at the 1.0 m of its own CSV `success` column)

# Key used to join a refined row to its coarse (color_cpfh) row for total time.
JOIN = ("scene", "split_src", "split_tgt", "iter", "seed")


def load_submap(path):
    """Return {(method, scene): [row,...]} and {(scene,key): coarse_time_s}.

    Works for the benchmark CSVs (centroid/joint5d) and for the ColorPCR CSV,
    which shares the same schema (method, scene, rot_error_deg, trans_error_m,
    time_ms, ...).
    """
    by_ms = defaultdict(list)
    coarse_time = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_ms[(r["method"], r["scene"])].append(r)
            if r["method"] == "color_cpfh":
                coarse_time[(r["scene"], tuple(r[k] for k in JOIN))] = \
                    float(r["time_ms"]) / 1000.0
    return by_ms, coarse_time


def submap_cell(rows, coarse_time, refined):
    """mean/std for trans, rot, t, success% from a list of benchmark/ColorPCR rows."""
    t = [float(x["trans_error_m"]) for x in rows]
    r = [float(x["rot_error_deg"]) for x in rows]
    succ = [1 if (float(x["rot_error_deg"]) < ROT_THRESH and
                  float(x["trans_error_m"]) < TRANS_THRESH) else 0 for x in rows]
    if refined:  # total time = coarse stage + this stage, matched per seed
        s = []
        for x in rows:
            c = coarse_time.get((x["scene"], tuple(x[k] for k in JOIN)))
            stage = float(x["time_ms"]) / 1000.0
            s.append(stage + c if c is not None else stage)
    else:
        s = [float(x["time_ms"]) / 1000.0 for x in rows]
    return _stats(t, r, s, succ)


def gaussreg_cell(rows):
    t = [float(x["trans_err_abs"]) for x in rows]
    r = [float(x["rot_err_deg"]) for x in rows]
    s = [float(x["gaussreg_solver_runtime_ms"]) / 1000.0 for x in rows]
    succ = [1 if (float(x["rot_err_deg"]) < ROT_THRESH and
                  float(x["trans_err_abs"]) < TRANS_THRESH) else 0 for x in rows]
    return _stats(t, r, s, succ)


def _stats(t, r, s, succ):
    return {
        "t": (st.mean(t), st.pstdev(t)),
        "r": (st.mean(r), st.pstdev(r)),
        "s": (st.mean(s), st.pstdev(s)),
        "pct": 100.0 * st.mean(succ),
        "ok": sum(succ),
        "n": len(t),
    }


# Row spec: (label, source, method, refined, compete_time).
# compete_time=True: this row's time competes for the per-scene "fastest" bold
# (matches the published table, where GaussReg's stump time 1.0 s is bolded as
# the fastest). All rows' trans/rot always compete for bold regardless.
ROWS = [
    ("GaussReg",            "gaussreg", None,            False, True),
    ("ColorPCR",            "colorpcr", "colorpcr",      False, True),
    ("KISS-Matcher",        "joint5d",  "vanilla_kiss",  False, True),
    ("Ours, centroid",      "centroid", "color_cpfh",    False, True),
    # Centroid refinement rows removed for space (2026-07-06, user decision):
    # the refinement comparison is carried by the joint-5D block. Uncomment to
    # restore the full 11-row table.
    # (r"\quad + GICP",       "centroid", "gicp_cpfh",     True,  True),
    # (r"\quad + 3DGS-ICP",   "centroid", "color_gicp_de", True,  True),
    # (r"\quad + Colored ICP", "cicp_cent", "colored_icp", True,  True),
    ("Ours, joint-5D",      "joint5d",  "color_cpfh",    False, True),
    (r"\quad + GICP",       "joint5d",  "gicp_cpfh",     True,  True),
    (r"\quad + 3DGS-ICP",   "joint5d",  "color_gicp_de", True,  True),
    (r"\quad + Colored ICP", "cicp_j5", "colored_icp",   True,  True),
]
# \midrule before the "Ours, centroid" and "Ours, joint-5D" block starts,
# computed from labels so row insertions/skips cannot desynchronize it.
RULE_BEFORE = {i for i, r in enumerate(ROWS)
               if r[0] in ("Ours, centroid", "Ours, joint-5D")}
# Colored-ICP rows are baselines: excluded from the "Ours" bold tie-break.
CICP_SOURCES = {"cicp_cent", "cicp_j5"}


def fmt(val, dec):
    return f"{val:.{dec}f}"


def cell(stat, metric, dec, bold):
    mean, sd = stat[metric]
    m = fmt(mean, dec)
    if bold:
        m = r"\textbf{" + m + "}"
    return (r"\makecell{" + m + r"\\[-3pt]{\scriptsize$\pm$"
            + fmt(sd, dec) + "}}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=("on", "off"), default="on",
                    help="opacity config: 'on' (paper default, shipped) or 'off'")
    ap.add_argument("--centroid", help="override centroid CSV path")
    ap.add_argument("--joint5d", help="override joint-5D CSV path")
    ap.add_argument("--gaussreg", default=str(GAUSSREG),
                    help="override GaussReg CSV path")
    ap.add_argument("--colorpcr", default=str(COLORPCR),
                    help="override ColorPCR CSV path")
    ap.add_argument("--out", help="output .tex path")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    centroid = Path(args.centroid) if args.centroid else HERE / cfg["centroid"]
    joint5d = Path(args.joint5d) if args.joint5d else HERE / cfg["joint5d"]
    gaussreg = Path(args.gaussreg)
    colorpcr = Path(args.colorpcr)
    out = Path(args.out) if args.out else HERE / "large_scale_table_body.tex"

    cent_ms, cent_ct = load_submap(centroid)
    j5_ms, j5_ct = load_submap(joint5d)
    cpcr_ms, _ = load_submap(colorpcr)
    greg = defaultdict(list)
    with open(gaussreg) as f:
        for r in csv.DictReader(f):
            if r["trial_mode"] == "full_features":
                greg[r["scene"]].append(r)

    # Colored-ICP CSVs: a row is emitted only if its CSV exists and covers all
    # nine scenes (lets the generator run while a benchmark is still going).
    cicp = {}
    for tag, path, ct in (("cicp_cent", CICP_CENT, cent_ct),
                          ("cicp_j5", CICP_J5, j5_ct)):
        if path.exists():
            ms, _ = load_submap(path)
            if all(("colored_icp", sc) in ms for sc in SCENES):
                cicp[tag] = (ms, ct)
            else:
                print(f"[skip] {path.name}: not all scenes present")
        else:
            print(f"[skip] {path.name}: missing")
    rows = [r for r in ROWS if r[1] not in CICP_SOURCES or r[1] in cicp]
    rules = {i for i, r in enumerate(rows)
             if r[0] in ("Ours, centroid", "Ours, joint-5D")}

    # Build stats[row_idx][scene] = stat dict
    stats = {}
    for i, (_, src, method, refined, _) in enumerate(rows):
        stats[i] = {}
        for sc in SCENES:
            if src == "gaussreg":
                stats[i][sc] = gaussreg_cell(greg[sc])
            elif src == "colorpcr":
                stats[i][sc] = submap_cell(cpcr_ms[(method, sc)], {}, False)
            elif src in CICP_SOURCES:
                ms, ct = cicp[src]
                stats[i][sc] = submap_cell(ms[(method, sc)], ct, refined)
            else:
                ms, ct = (cent_ms, cent_ct) if src == "centroid" else (j5_ms, j5_ct)
                stats[i][sc] = submap_cell(ms[(method, sc)], ct, refined)

    # Best value per scene/metric (rounded to display precision) for bolding.
    DEC = {"t": 2, "r": 1, "s": 1}
    best = {}
    # "Ours" rows win ties over the baselines (GaussReg/ColorPCR/KISS-Matcher,
    # and the Colored-ICP refinement rows): a baseline t/r/s cell is bolded
    # only if no Ours row also attains the best.
    ours_set = {i for i, r in enumerate(rows)
                if r[1] in ("centroid", "joint5d") and r[2] != "vanilla_kiss"}
    ours = lambda i: i in ours_set
    for sc in SCENES:
        best[sc] = {}
        for metric in ("t", "r", "s"):
            vals, ours_vals = [], []
            for i, (_, _, _, _, compete_time) in enumerate(rows):
                if metric == "s" and not compete_time:
                    continue
                v = round(stats[i][sc][metric][0], DEC[metric])
                vals.append(v)
                if ours(i):
                    ours_vals.append(v)
            best[sc][metric] = min(vals)
            best[sc][metric + "_ours"] = min(ours_vals)
        best[sc]["pct"] = max(round(stats[i][sc]["pct"]) for i in stats)

    def winner(i, sc, metric, dec):
        v = round(stats[i][sc][metric][0], dec)
        if v != best[sc][metric]:
            return False
        # On a tie, do not bold a baseline cell if an Ours row also wins.
        return ours(i) or best[sc][metric + "_ours"] != best[sc][metric]

    # Overall success per row: pooled over ALL trials of the nine scenes
    # (same definition as recompute_pooled_success.py; n = 9 x 200 = 1800).
    overall = {i: 100.0 * sum(stats[i][sc]["ok"] for sc in SCENES)
                  / sum(stats[i][sc]["n"] for sc in SCENES) for i in stats}
    best_overall = max(round(v, 1) for v in overall.values())
    best_overall_ours = max(round(v, 1) for i, v in overall.items() if ours(i))

    def overall_winner(i):
        if round(overall[i], 1) != best_overall:
            return False
        return ours(i) or best_overall_ours != best_overall

    # Emit one readable table per scene; '*' marks the best cell per metric.
    def txt_cell(stat, metric, dec, win):
        mean, sd = stat[metric]
        s = f"{mean:.{dec}f}±{sd:.{dec}f}"
        return f"{s}{'*' if win else ' '}"

    disp = lambda label: label.replace(r"\quad ", "  ")
    LABEL_W = max(len(disp(label)) for label, *_ in ROWS)
    COL_W = 14
    print(f"config=opacity-{args.config}  "
          f"centroid={centroid.name}  joint5d={joint5d.name}  "
          f"colorpcr={colorpcr.name}  (thresh: {ROT_THRESH:g}deg / {TRANS_THRESH:g}m)")

    for sc in SCENES:
        n = stats[0][sc]["n"]
        print(f"\n=== {sc} (n={n}) ===")
        header = (f"{'method':<{LABEL_W}}  " +
                  "".join(h.ljust(COL_W) for h in
                          ("trans (m)", "rot (deg)", "t (s)", "succ%")))
        print(header)
        print("-" * len(header))
        for i, (label, _, _, _, compete_time) in enumerate(rows):
            if i in rules:
                print("-" * len(header))
            s = stats[i][sc]
            tb = winner(i, sc, "t", 2)
            rb = winner(i, sc, "r", 1)
            sb = compete_time and winner(i, sc, "s", 1)
            cells = [
                txt_cell(s, "t", 2, tb),
                txt_cell(s, "r", 1, rb),
                txt_cell(s, "s", 1, sb) if compete_time else "-",
                f"{s['pct']:.1f}",
            ]
            print(f"{disp(label):<{LABEL_W}}  " +
                  "".join(c.ljust(COL_W) for c in cells))

    # Overall (pooled) success per row -- the table's last column.
    print(f"\n=== overall (pooled over all scenes) ===")
    for i, (label, *_) in enumerate(rows):
        if i in rules:
            print("-" * (LABEL_W + 10))
        mark = "*" if overall_winner(i) else " "
        print(f"{disp(label):<{LABEL_W}}  {overall[i]:5.1f}%{mark}")

    # Build the LaTeX table body and write it to the .tex file.
    lines = []
    for i, (label, _, _, _, compete_time) in enumerate(rows):
        if i in rules:
            lines.append(r"\midrule")
        parts = [label]
        for sc in SCENES:
            s = stats[i][sc]
            tb = winner(i, sc, "t", 2)
            rb = winner(i, sc, "r", 1)
            sb = compete_time and winner(i, sc, "s", 1)
            parts += [cell(s, "t", 2, tb), cell(s, "r", 1, rb),
                      cell(s, "s", 1, sb)]
        ov = fmt(overall[i], 1)
        if overall_winner(i):
            ov = r"\textbf{" + ov + "}"
        parts.append(ov)
        lines.append(" & ".join(parts) + r" \\")
    out.write_text("\n".join(lines) + "\n")
    print(f"\n% written to {out}")


if __name__ == "__main__":
    main()
