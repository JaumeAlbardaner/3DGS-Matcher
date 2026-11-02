import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs/outputs live next to this script

# ---- SETTINGS ----
INPUT_CSV = "ablation_results_RGB.csv"
OUTPUT_CSV = "success_rates.csv"   # set to None if you don't want export
OUTPUT_GROUP_CSV = "success_rates_by_group.csv"
ROT_THRESH = 5.0
TRANS_THRESH = 0.5
DROP_LAST_RGB_ROWS = 600  # Ignore trailing legacy LAB+A block in ablation_results_RGB.csv
# ------------------

# Load data
df = pd.read_csv(INPUT_CSV)
if os.path.basename(INPUT_CSV).lower() == "ablation_results_rgb.csv" and DROP_LAST_RGB_ROWS > 0:
    if len(df) > DROP_LAST_RGB_ROWS:
        df = df.iloc[:-DROP_LAST_RGB_ROWS].copy()
        print(f"Ignored last {DROP_LAST_RGB_ROWS} rows from {INPUT_CSV} (legacy LAB+A block).")
    else:
        raise ValueError(
            f"Cannot drop last {DROP_LAST_RGB_ROWS} rows: file has only {len(df)} rows."
        )

# Define success condition
df["success"] = (
    (df["rotation_error_deg"] < ROT_THRESH) &
    (df["translation_error"] < TRANS_THRESH)
)

# Compute success rate per configuration
summary = (
    df.groupby("config")
      .agg(
          total_runs=("success", "count"),
          successful_runs=("success", "sum"),
          success_rate=("success", "mean")
      )
      .reset_index()
)

# Robust feature-family grouping from explicit flags (not config-name heuristics).
flags = (
    df.groupby("config")
      .agg(
          has_geom=("f1", lambda s: s.astype(bool).any())  # placeholder, will combine below
      )
      .reset_index()
)
geom_flags = (
    df.groupby("config")[["f1", "f2", "f3"]]
      .max()
      .reset_index()
)
rgba_flags = (
    df.groupby("config")[["color", "opacity"]]
      .max()
      .reset_index()
)
flags = flags.drop(columns=["has_geom"]).merge(geom_flags, on="config").merge(rgba_flags, on="config")
flags["has_geom"] = flags[["f1", "f2", "f3"]].astype(bool).any(axis=1)
flags["has_rgba"] = flags[["color", "opacity"]].astype(bool).any(axis=1)

def descriptor_group(row):
    has_geom = row["has_geom"]
    has_color = bool(row["color"])
    has_opacity = bool(row["opacity"])

    if has_geom and (has_color or has_opacity):
        return "Mixed"
    if has_geom and (not has_color) and (not has_opacity):
        return "FPFH"
    if (not has_geom) and has_color and (not has_opacity):
        return "RGB"
    if (not has_geom) and (not has_color) and has_opacity:
        return "A"
    if (not has_geom) and has_color and has_opacity:
        return "Mixed"
    if row["has_rgba"]:
        return "RGBA"
    return "Unknown"

flags["group"] = flags.apply(descriptor_group, axis=1)
summary = summary.merge(flags[["config", "group"]], on="config", how="left")

# Print results
print("\nSuccess Rate per Configuration")
print("-" * 40)
for _, row in summary.iterrows():
    print(
        f"{row['config']:15s} | "
        f"{row['successful_runs']:3d}/{row['total_runs']:3d} "
        f"= {row['success_rate']:.3f}"
    )

# Group-level summary:
# - mean_config_success: average of per-config success rates (unweighted by config count)
# - std_config_success: spread across configurations in each group
# - pooled_success_rate: success ratio across all runs in the group
group_order = ["RGB", "A", "FPFH", "Mixed"]
group_summary = (
    summary.copy()
    .assign(group=pd.Categorical(summary["group"], categories=group_order, ordered=True))
    .groupby("group", observed=True)
    .agg(
        n_configs=("config", "count"),
        mean_config_success=("success_rate", "mean"),
        std_config_success=("success_rate", "std"),
        successful_runs=("successful_runs", "sum"),
        total_runs=("total_runs", "sum"),
    )
    .reset_index()
)
group_summary["pooled_success_rate"] = group_summary["successful_runs"] / group_summary["total_runs"]
group_summary = group_summary.sort_values("mean_config_success", ascending=False).reset_index(drop=True)

print("\nGroup-level success summary")
print("-" * 40)
for _, row in group_summary.iterrows():
    print(
        f"{row['group']:6s} | mean={row['mean_config_success']:.3f} "
        f"(std={0.0 if pd.isna(row['std_config_success']) else row['std_config_success']:.3f}, "
        f"n_cfg={int(row['n_configs'])}) | pooled={row['successful_runs']}/{row['total_runs']} "
        f"= {row['pooled_success_rate']:.3f}"
    )

# Optional: save for LaTeX pgfplots
if OUTPUT_CSV:
    summary_sorted_for_csv = (
        summary.copy()
        .sort_values(["success_rate", "group"], ascending=[False, True])
    )
    summary_sorted_for_csv[["config", "group", "success_rate", "successful_runs", "total_runs"]].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved success rates to {OUTPUT_CSV}")
if OUTPUT_GROUP_CSV:
    group_summary.to_csv(OUTPUT_GROUP_CSV, index=False)
    print(f"Saved group summary to {OUTPUT_GROUP_CSV}")

# ---- PDF Plot Generation ----

def plot_success_rate_pdf(group_summary, output_path):
    # Assign color palette per group.
    color_map = {
        "RGB": "#D62828",      # Color (red)
        "A": "#8E8E8E",        # Opacity (neutral gray)
        "FPFH": "#0047FF",     # Geometric (intense blue)
        "Mixed": "#00B050",    # Mixed (greener green)
    }
    label_map = {"RGB": "Color", "A": "Opacity", "FPFH": "Geometric", "Mixed": "Mixed"}

    gs = group_summary.copy()
    gs = gs[gs["group"].isin(group_order)].copy()
    gs = gs.sort_values("mean_config_success", ascending=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(gs))
    y = gs["mean_config_success"].to_numpy()
    yerr = gs["std_config_success"].fillna(0.0).to_numpy()

    for i, row in gs.reset_index(drop=True).iterrows():
        g = row["group"]
        ax.bar(
            x[i],
            y[i],
            yerr=yerr[i],
            capsize=4,
            color=color_map[g],
            edgecolor="#222",
            width=0.72,
        )
        label_y = min(y[i] + yerr[i] + 0.02, 1.03)
        ax.text(
            x[i],
            label_y,
            f"{y[i]:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(g, g) for g in gs["group"].astype(str).tolist()], fontsize=10)
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Descriptor Family')
    ax.set_ylim(0, 1.05)
    ax.set_title('Average Success Rate by Descriptor Family')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # No legend: group names are already on the x-axis.
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

# Generate PDF plot
PDF_PATH = "ablationResults-Geom-LABA.pdf"
plot_success_rate_pdf(group_summary, PDF_PATH)
print(f"Saved PDF plot to {PDF_PATH}")
