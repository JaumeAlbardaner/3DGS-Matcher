#!/usr/bin/env python3
"""Accuracy-vs-runtime tradeoff across voxel sizes and methods (large-scale joint-5D).

For each method we plot success (%) against mean wall-time per registration (s, log),
with the voxel size annotated at each point. Refinement runtimes are reported as the
TOTAL pipeline (coarse matcher + refinement). Data is read straight from the sweep CSVs.
"""
import csv, collections, os, pathlib, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
RES = os.environ.get("RESULTS_DIR", os.path.join(DATA_ROOT, "results"))
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
JOINT = {0.5: f"{RES}/sweep_joint5d_nondrone_v050.csv",
         0.4: f"{RES}/sweep_joint5d_nondrone_v040.csv",
         0.3: f"{RES}/sweep_joint5d_nondrone_v030.csv"}
CENTROID = {0.5: f"{RES}/sweep_trained_nondrone_v050.csv",
            0.4: f"{RES}/sweep_trained_nondrone_v040.csv",
            0.3: f"{RES}/sweep_trained_nondrone_v030.csv"}

def load(p):
    try: return list(csv.DictReader(open(p)))
    except FileNotFoundError: return []

def stat(rows, method):
    ok = n = 0; t = []
    for r in rows:
        if r["method"] != method: continue
        try:
            s = int(r["success"])
        except ValueError:
            continue
        if s < 0: continue
        ok += s; n += 1
        try: t.append(float(r["time_ms"]))
        except ValueError: pass
    return (100*ok/n if n else 0.0, st.mean(t)/1000.0 if t else 0.0)  # success %, mean seconds

# Gather per-method series across voxels
voxels = [0.5, 0.4, 0.3]
joint_rows = {v: load(p) for v, p in JOINT.items()}

series = {}  # name -> list of (runtime_s, success, voxel)
# KISS-Matcher (geometry, standalone)
series["KISS-Matcher (geometry)"] = [(*stat(joint_rows[v], "vanilla_kiss")[::-1][::-1], v) for v in voxels]
# 3DGS-Matcher coarse, joint-5D
series["3DGS-Matcher coarse (joint-5D)"] = []
# 3DGS-Matcher + 3DGS-ICP (total pipeline = coarse + refinement)
series["Ours + 3DGS-ICP (joint-5D)"] = []
for v in voxels:
    cc_s, cc_t = stat(joint_rows[v], "color_cpfh")
    de_s, de_t = stat(joint_rows[v], "color_gicp_de")
    series["3DGS-Matcher coarse (joint-5D)"].append((cc_t, cc_s, v))
    series["Ours + 3DGS-ICP (joint-5D)"].append((cc_t + de_t, de_s, v))
# fix KISS series formatting: (success,runtime) -> (runtime,success,voxel)
series["KISS-Matcher (geometry)"] = []
for v in voxels:
    s, t = stat(joint_rows[v], "vanilla_kiss")
    series["KISS-Matcher (geometry)"].append((t, s, v))
# 3DGS-Matcher coarse, centroid (now measured at all three voxels)
series["3DGS-Matcher coarse (centroid)"] = []
for v in voxels:
    cs, ct = stat(load(CENTROID[v]), "color_cpfh")
    series["3DGS-Matcher coarse (centroid)"].append((ct, cs, v))

style = {
    "KISS-Matcher (geometry)":        dict(color="#888888", marker="x", ls="--"),
    "3DGS-Matcher coarse (centroid)": dict(color="#d95f02", marker="D", ls="--"),
    "3DGS-Matcher coarse (joint-5D)": dict(color="#1f78b4", marker="o", ls="-"),
    "Ours + 3DGS-ICP (joint-5D)":     dict(color="#1a9850", marker="s", ls="-"),
}

ann = {  # per-series label offset (pts) and alignment to avoid overlap
    "Ours + 3DGS-ICP (joint-5D)":     ((0, 10), "center"),
    "3DGS-Matcher coarse (joint-5D)": ((0, -15), "center"),
    "KISS-Matcher (geometry)":        ((0, -15), "center"),
    "3DGS-Matcher coarse (centroid)": ((0, -15), "center"),
}
legend_label = {  # short labels to keep the legend compact
    "KISS-Matcher (geometry)":        "KISS-Matcher",
    "3DGS-Matcher coarse (centroid)": "3DGS-Matcher (centroid)",
    "3DGS-Matcher coarse (joint-5D)": "3DGS-Matcher (joint-5D)",
    "Ours + 3DGS-ICP (joint-5D)":     "+ 3DGS-ICP",
}
fig, ax = plt.subplots(figsize=(7.0, 3.7), constrained_layout=True)
for name, pts in series.items():
    pts = sorted(pts)  # by runtime
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, label=legend_label[name], markersize=8, lw=2, **style[name])
    off, ha = ann[name]
    for rt, sc, v in pts:
        ax.annotate(f"{v:g} m", (rt, sc), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8, color=style[name]["color"])

ax.set_xscale("log")
ax.set_xlabel("Mean runtime per registration (s, log scale)", fontsize=11)
ax.set_ylabel("Success rate (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="center right", fontsize=7.5, markerscale=0.75, handlelength=1.4,
          labelspacing=0.3, borderpad=0.3, framealpha=0.9)

FIG_DIR = os.environ.get("FIG_DIR", os.path.join(DATA_ROOT, "results", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)
for out in [os.path.join(FIG_DIR, "voxel_runtime_tradeoff.png"),
            os.path.join(FIG_DIR, "voxel_runtime_tradeoff.pdf")]:
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out)

# print the underlying numbers
print("\nmethod / voxel / runtime_s / success%")
for name, pts in series.items():
    for rt, sc, v in sorted(pts, key=lambda p: -p[2]):
        print(f"  {name:34s} {v:g}m  {rt:7.2f}s  {sc:5.1f}%")
