"""Run the experiments behind the 3DGS-Matcher paper (IEEE RA-L 2026).

    uv run reproduce list              # every experiment, what it reproduces, what it needs
    uv run reproduce <name> [...]      # run one or more
    uv run reproduce tables            # everything that regenerates from shipped CSVs

Experiments marked "csv" regenerate the paper's numbers from paper_data/ in
seconds and need nothing but this environment. The others re-run the benchmarks
and need the C++ build (scripts/bootstrap.sh) and the data under $DATA_ROOT
(default: ./data). See README.md for the data layout.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# name: (paper item, needs, [commands]). A command is a list of argv strings;
# "{py}" becomes this environment's Python and "{data}" the data root. Leading
# VAR=value tokens set environment variables, as in a shell. Paths are relative
# to the repo root.
EXPERIMENTS = {
    # ── from shipped CSVs ────────────────────────────────────────────────────
    "table1": ("Table I, tramway component ablation", "csv",
               [["{py}", "paper_data/table1_tramway_ablation/recompute_table1.py"]]),
    "table2": ("Table II, per-scene errors and runtimes on Mip-NeRF 360", "csv",
               [["{py}", "paper_data/table2_large_scale/gen_large_scale_table.py"]]),
    "table3": ("Table III / abstract, pooled success rates (5 deg / 0.5 m)", "csv",
               [["{py}", "paper_data/table2_large_scale/recompute_pooled_success.py"]]),
    "fig4": ("Fig. 4, success by descriptor family", "csv",
             [["{py}", "paper_data/figure4_descriptor_ablation/success_getter.py"]]),
    "fig3-plot": ("Fig. 3, pooled splat-shape histograms (84 / 45 / 51 %)", "csv",
                  [["{py}", "scripts/plot_pooled_shape_histograms.py",
                    "--npz", "paper_data/figure3_shape_histograms/splat_shape_histograms.npz"]]),
    "voxel-plot": ("Fig. voxel/runtime trade-off, from the shipped sweeps", "csv",
                   [["RESULTS_DIR=paper_data/voxel_sweep", "{py}", "scripts/plot_voxel_runtime.py"]]),
    "perturbations": ("Sec. V-B, photometric and scale perturbations on the tramway", "csv",
                      [["{py}", "paper_data/photometric_scale/summarize_perturbations.py"]]),
    "restarts": ("Sec. VI, restart convergence on the real aerial-ground pairs", "build/print_perturbation",
                 [["{py}", "paper_data/realdata_restarts.py"]]),
    "table4-score": ("Table IV, KITTI multi-robot table from the shipped benchmark CSVs",
                     "build/print_perturbation + data/trained_gaussians{,_filtered}/kitti* (GT corrections ship in data/kitti)",
                     [["RESULTS_DIR=paper_data/table4_kitti", "{py}", "scripts/kitti_multirobot_table.py",
                       "--out", "{data}/results/kitti_multirobot_table"]]),
    "exp1-score": ("Extra experiment 1 (not in the paper): 2DGS vs 3DGS from the shipped benchmark CSVs",
                   "data/trained_gaussians + data/trained_2dgs cameras",
                   [["RESULTS_DIR=paper_data/extra_experiments/exp1_2dgs", "{py}", "scripts/compare_2dgs.py"]]),
    # ── full re-runs ─────────────────────────────────────────────────────────
    "tramway": ("Table I, re-run (single_ablation, 4 perturbations x 11 noise levels x 100 seeds)", "build",
                [["examples/noise_ablation_doubts.sh", "examples/splats/tramway.splat",
                  "{data}/results/noise_doubts_tramway_p500_n100.csv"]]),
    "descriptor-ablation": ("Fig. 4, re-run the 31 descriptor configurations on the tramway", "build",
                            [["examples/run_descriptor_ablation.sh", "--splat-file", "examples/splats/tramway.splat",
                              "--output", "{data}/results/descriptor_ablation.csv"]]),
    "perturbations-tramway": ("Sec. V-B, photometric and scale perturbations on the tramway, re-run", "build",
                              [["examples/photometric_scale_perturbations.sh"]]),
    "mipnerf": ("Tables II/III, re-run the Mip-NeRF 360 benchmark (joint-5D and centroid)",
                "build + data/trained_gaussians", [["examples/run_mipnerf_benchmark.sh"]]),
    "colored-icp": ("Table II '+ Colored ICP' row (Open3D, warm-started from the coarse poses)",
                    "build + data/trained_gaussians", [["{py}", "scripts/benchmark_colored_icp.py"]]),
    "voxel-sweep": ("Fig. voxel/runtime trade-off (0.5 / 0.4 / 0.3 m)", "build + data/trained_gaussians",
                    [["examples/run_joint5d_overnight.sh"],
                     ["NONDRONE_VOXELS=0.5 0.4 0.3", "SKIP_DRONE=1", "examples/run_trained_voxel_sweep.sh"],
                     ["{py}", "scripts/plot_voxel_runtime.py"]]),
    "fig3": ("Fig. 3, splat-shape statistics over every trained map", "data/trained_gaussians",
             [["{py}", "scripts/splat_shape_stats.py", "--histograms"],
              ["{py}", "scripts/plot_pooled_shape_histograms.py"]]),
    "aerial-ground": ("Sec. VI real aerial-ground pairs (overlap, no ground truth)", "build + data/FullSplats",
                      [["examples/run_overlap_test.sh"]]),
    "kitti": ("Table IV, KITTI multi-robot pairs, re-run (prepare, train 3DGS, filter, benchmark, score)",
              "build + KITTI raw + COLMAP + gaussian-splatting (GS_REPO)",
              [["examples/run_kitti_multirobot_queue.sh"], ["{py}", "scripts/kitti_multirobot_table.py"]]),
    "exp1": ("Extra experiment 1 (not in the paper): train 2DGS on room + KITTI 00/07 and rerun the benchmark",
             "build + GS2D_REPO (2d-gaussian-splatting) + data/split_runs, data/kitti",
             [["scripts/run_2dgs.sh"], ["{py}", "scripts/compare_2dgs.py"]]),
    "exp2-score": ("Extra experiment 2 (not in the paper): KISS-Matcher on LiDAR vs Gaussian maps, from the shipped CSVs",
                   "build/print_perturbation + data/trained_gaussians_filtered/kitti* + data/lidar_maps",
                   [["RESULTS_DIR=paper_data/table4_kitti", "LIDAR_RESULTS_DIR=paper_data/extra_experiments/exp2_kitti_lidar",
                     "{py}", "scripts/kitti_lidar_kiss_score.py"]]),
    "exp2": ("Extra experiment 2 (not in the paper): KISS-Matcher on LiDAR maps of the KITTI windows", "build + KITTI Velodyne",
                    [["{py}", "scripts/kitti_lidar_maps.py"], ["examples/run_kitti_lidar_kiss.sh"],
                     ["{py}", "scripts/kitti_lidar_kiss_score.py"]]),
    "exp3": ("Extra experiment 3 (not in the paper): Gaussians per Mip-NeRF 360 map pair", "data/trained_gaussians",
             [["{py}", "paper_data/extra_experiments/exp3_scene_sizes.py"]]),
}
GROUPS = {"tables": ["table1", "table2", "table3", "fig4", "fig3-plot", "voxel-plot", "perturbations"]}


def data_root():
    return Path(os.environ.get("DATA_ROOT", ROOT / "data"))


def run(name):
    item, needs, cmds = EXPERIMENTS[name]
    print(f"\n=== {name}: {item}", flush=True)
    binary = needs.split()[0] if needs.startswith("build/") else "build/benchmark_submap"
    if "build" in needs and not (ROOT / binary).exists():
        sys.exit(f"{binary} is missing: run scripts/fetch_third_party.sh && scripts/bootstrap.sh --target all")
    env = dict(os.environ, DATA_ROOT=str(data_root()), MPLBACKEND=os.environ.get("MPLBACKEND", "Agg"))
    for cmd in cmds:
        argv = [c.replace("{py}", PY).replace("{data}", str(data_root())) for c in cmd]
        print("$", " ".join(argv), flush=True)
        extra = {}
        while argv and "=" in argv[0] and argv[0].split("=")[0].isupper():
            k, v = argv.pop(0).split("=", 1)
            extra[k] = str(ROOT / v) if (ROOT / v).exists() else v
        r = subprocess.run(argv, cwd=ROOT, env={**env, **extra})
        if r.returncode:
            sys.exit(f"{name}: command failed with exit code {r.returncode}")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "list":
        w = max(map(len, EXPERIMENTS))
        for name, (item, needs, _) in EXPERIMENTS.items():
            print(f"{name:<{w}}  {item}\n{'':<{w}}  needs: {needs}")
        print(f"\ngroups: " + ", ".join(f"{g} = {' '.join(v)}" for g, v in GROUPS.items()))
        return
    names = [n for a in args for n in GROUPS.get(a, [a])]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"unknown experiment(s): {', '.join(unknown)} (see: uv run reproduce list)")
    for n in names:
        run(n)


if __name__ == "__main__":
    main()
