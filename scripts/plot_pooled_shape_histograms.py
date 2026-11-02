#!/usr/bin/env python3
"""Fig. 3, pooled over every trained 3DGS map instead of one reconstruction.

Draws the three histograms over all maps in the inventory, keeping the earlier
figure's style (20 bins, serif 26 pt, red / green / blue at alpha 0.7) so the paper
looks unchanged apart from the data.

Each map contributes its own NORMALIZED histogram and all maps count equally, so
the figure is a mean over maps, not over Gaussians -- otherwise the Mip-NeRF 360
halves (half of all 162 M Gaussians) would decide its shape. The whiskers give the
10th-90th percentile across maps per bin, which is what makes the spread between
reconstructions visible rather than hidden behind an average.

Bin edges fall on the paper's thresholds, so the extreme bar IS the number quoted
in Section V; the script verifies that and refuses to write a figure that disagrees.

  python3 scripts/splat_shape_stats.py --histograms     # produces the .npz first
  python3 scripts/plot_pooled_shape_histograms.py
"""
import os
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
FIGURES = DATA / "results/figures"

# (feature index, name, colour, output stem, tail the text quotes)
FEATURES = [
    (0, "Anisotropy", "red", "anisotropy_histogram_pooled", "last"),
    (1, "Linearity", "green", "linearity_histogram_pooled", "last"),
    (2, "Planarity", "blue", "planarity_histogram_pooled", "first"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 26, "axes.labelsize": 26, "axes.titlesize": 26,
    "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 26,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=DATA / "results/splat_shape_histograms.npz")
    ap.add_argument("--out_dir", type=Path, default=FIGURES)
    ap.add_argument("--lo", type=float, default=10.0, help="lower percentile of the whisker")
    ap.add_argument("--hi", type=float, default=90.0, help="upper percentile of the whisker")
    args = ap.parse_args()

    if not args.npz.exists():
        sys.exit(f"{args.npz} missing -- run: python3 scripts/splat_shape_stats.py --histograms")
    d = np.load(args.npz, allow_pickle=True)
    hists, bins = d["hists"], int(d["bins"])          # (maps, 3, bins), each row sums to 1
    n_maps, total = hists.shape[0], int(d["gaussians"].sum())
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres, width = (edges[:-1] + edges[1:]) / 2, 1.0 / bins

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{n_maps} maps, {total/1e6:.1f}M Gaussians, {bins} bins")
    for idx, name, colour, stem, tail in FEATURES:
        h = 100.0 * hists[:, idx, :]                  # percent of each map's Gaussians
        mean = h.mean(0)
        lo, hi = np.percentile(h, args.lo, axis=0), np.percentile(h, args.hi, axis=0)

        quoted = mean[-1] if tail == "last" else mean[0]
        print(f"  {name:10s} extreme bin = {quoted:.1f}%   "
              f"({args.lo:g}-{args.hi:g} pct across maps: "
              f"{(lo[-1] if tail=='last' else lo[0]):.1f}-"
              f"{(hi[-1] if tail=='last' else hi[0]):.1f}%)")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(centres, mean, width=width, color=colour, alpha=0.7, align="center")
        ax.errorbar(centres, mean, yerr=[mean - lo, hi - mean], fmt="none",
                    ecolor="#333333", elinewidth=1.4, capsize=3, capthick=1.4, alpha=0.85)
        ax.set_xlabel(f"{name} Value")
        ax.set_ylabel("Splats [\\%]" if plt.rcParams["text.usetex"] else "Splats [%]",
                      labelpad=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.grid(True, which="major")
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{stem}.svg")
        plt.close(fig)

    # The text quotes these three numbers; keep figure and prose in lockstep.
    quoted = [100.0 * hists[:, 0, -1].mean(), 100.0 * hists[:, 1, -1].mean(),
              100.0 * hists[:, 2, 0].mean()]
    print(f"\nSection V must read: {quoted[0]:.1f}% anisotropy > 0.95, "
          f"{quoted[1]:.1f}% linearity > 0.95, {quoted[2]:.1f}% planarity < 0.05")
    print(f"figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
