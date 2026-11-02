# 3DGS-Matcher: evaluation

Code and data for every experiment in

> J. Albardaner i Torras, F. Auger, F. Lanthier, À. Santamaria-Navarro,
> **3DGS-Matcher: Revisiting Color-Centric Registration for 3D Gaussian Splats**,
> IEEE Robotics and Automation Letters, 2026.

together with a few extra experiments run during the publication process. This branch holds the exact
pipeline the paper was evaluated with (joint-5D keypoints, ColorPFH, KISS-Matcher
ROBIN/GNC solver, 3DGS-ICP), the benchmark drivers, the baseline runners, and the
raw result CSVs behind the tables.

## Quick start: regenerate the tables (no build, no data)

```bash
uv run reproduce tables          # Tables I-III, Figs. 3-4, voxel figure, Sec. V-B perturbations,
                                 # all from paper_data/
uv run reproduce list            # every experiment and what it needs
```

[uv](https://docs.astral.sh/uv/) creates a Python 3.12 environment from
`pyproject.toml` / `uv.lock` (numpy, scipy, matplotlib, pandas, Open3D 0.19,
Pillow) on first use. Nothing else is installed or modified.

## Full re-runs

### 1. Build the C++ benchmarks

Requirements: CMake >= 3.16, a C++17 compiler, Eigen3, TBB, OpenMP
(`scripts/bootstrap.sh --deps` installs them into an active conda env instead).

```bash
scripts/fetch_third_party.sh          # KISS-Matcher @42154f8 (+ patch), small_gicp @78f2e7a
scripts/bootstrap.sh --target all     # -> build/benchmark_submap, single_ablation,
                                      #    align_two_splats, print_perturbation, ...
```

`third_party/kiss-matcher.patch` adds empty-keypoint guards and an alignment fix
in nanoflann. Both the KISS-Matcher baseline and our pipeline run on the patched
build.

### 2. Data

The trained maps are published separately (DATA_URL). Download the four
archives, extract them into one folder and point `DATA_ROOT` at it (default
`./data`):

```
data/
  trained_gaussians/<scene>/<split>/point_cloud/iteration_30000/point_cloud.ply
      Mip-NeRF 360: bicycle bonsai counter flowers garden kitchen room stump treehill,
      splits split_pos/split_neg and split_wide_pos/split_wide_neg (one 3DGS per camera half)
      KITTI: kitti00 kitti00_07 kitti06 kitti05 kitti02 kitti08 kitti09_10, one dir per robot
  trained_gaussians_filtered/   same layout, opacity >= 0.1 + outlier removal (KITTI)
  FullSplats/                   aerial-ground pairs: dron1_, drone2_, gv_point_cloud.ply
  kitti/                        GT corrections revisit_gt/ and align_*/ (also ship with this branch);
                                KITTI raw drives go here only to re-train the maps
  trained_2dgs{,_filtered}/     extra experiment 1 (2DGS maps of room and KITTI 00/07)
  lidar_maps/                   extra experiment 2 (Velodyne maps of the KITTI 05 / 06 windows)
  results/                      every driver writes its CSVs here
```

| Archive | Size | Needed for |
|---|---|---|
| `3dgs-matcher_mipnerf360_maps.tar` | 20 GB | Tables II/III re-runs, Colored ICP, baselines, voxel figure |
| `3dgs-matcher_kitti_maps.tar` | 2.4 GB | Table IV |
| `3dgs-matcher_farm_maps.tar` | 0.7 GB | Sec. VI |
| `3dgs-matcher_extra_experiments.tar` | 1 GB | extra experiments 1 and 2 |

None of them is needed for `uv run reproduce tables`, which works from the CSVs in
`paper_data/`.

The tramway splat used by Table I and Sec. V-B ships in
`examples/splats/tramway.splat`.

**Building the Mip-NeRF 360 split pairs.** Each scene's COLMAP cameras are cut
at a plane into two halves, and one 3DGS map is trained per half (the
`split_wide_*` variant lets each half extend 10 % into the other):

```bash
uv run scripts/split_and_prepare.py --project_path <scene> --sparse_txt <scene>/sparse/0 \
    --output_root data/split_runs/360/<scene> --split_axis x --split_value 0
GS_REPO=<gaussian-splatting> scripts/train_split_runs_gaussian.sh
```

KITTI maps are filtered with `scripts/filter_3dgs_ply_batch.py
--opacity-threshold 0.1` (plus its default statistical outlier removal, 20
neighbours, 2 std), which `examples/run_kitti_robots.sh` runs for you.

### 3. Run

```bash
uv run reproduce mipnerf         # Tables II/III benchmark (several hours, CPU)
uv run paper_data/table2_large_scale/gen_large_scale_table.py \
    --joint5d data/results/joint5d_trained_v050_n200_opacityON.csv \
    --centroid data/results/centroid_trained_v050_n200_opacityON.csv
uv run reproduce kitti           # Table IV
```

## Experiment index

### In the paper

| Paper item | `reproduce` name | Driver | Output |
|---|---|---|---|
| Table I, tramway component ablation | `table1` (csv), `tramway` | `examples/noise_ablation_doubts.sh` -> `build/single_ablation` | `noise_doubts_tramway_p500_n100.csv` |
| Sec. V-B, photometric and scale perturbations | `perturbations` (csv), `perturbations-tramway` | `examples/photometric_scale_perturbations.sh` -> `build/single_ablation` | `paper_data/photometric_scale/tramway_perturbations.csv` |
| Fig. 3, splat-shape histograms (121 maps) | `fig3-plot` (csv), `fig3` | `scripts/splat_shape_stats.py --histograms`, `scripts/plot_pooled_shape_histograms.py` | `splat_shape_histograms.npz`, SVGs |
| Fig. 4, descriptor families | `fig4` (csv), `descriptor-ablation` | `examples/run_descriptor_ablation.sh` -> `build/single_ablation`, `paper_data/figure4_descriptor_ablation/success_getter.py` | bars + `success_rates_by_group.csv` |
| Tables II/III, Mip-NeRF 360, ours + KISS-Matcher | `table2`, `table3` (csv), `mipnerf` | `examples/run_mipnerf_benchmark.sh` -> `build/benchmark_submap` | `{joint5d,centroid}_trained_v050_n200_opacityON.csv` |
| Table II, + Colored ICP | `colored-icp` | `scripts/benchmark_colored_icp.py` (Open3D, seed-exact replay via `build/print_perturbation`) | `coloredicp_*.csv` |
| Table II, GaussReg | | `baselines/gaussreg/` | `gaussreg_results.csv` |
| Table II, ColorPCR | | `baselines/colorpcr/` | `colorpcr_trained_nondrone_v050_n100.csv` |
| Fig. voxel / runtime trade-off | `voxel-plot` (csv), `voxel-sweep` | `examples/run_joint5d_overnight.sh`, `examples/run_trained_voxel_sweep.sh`, `scripts/plot_voxel_runtime.py` | `sweep_*_nondrone_v0{50,40,30}.csv` |
| Sec. VI, real aerial-ground pairs and restart convergence | `aerial-ground`, `restarts` | `examples/run_overlap_test.sh` -> `scripts/compute_overlap_metrics.py`, `paper_data/realdata_restarts.py` | `realdata_overlap_n100*.csv` |
| Table IV, KITTI multi-robot | `table4-score`, `kitti` | `examples/run_kitti_multirobot_queue.sh` -> `run_kitti_robots.sh`, `scripts/kitti_multirobot_table.py` | `<scene>_robots_{unfiltered,filtered}_joint5d_v050.csv` |

### Extra experiments (not in the final paper)

These experiments were run during the publication process and are not part of
the final article. They are kept here for completeness, and none of the paper's
numbers depend on them.

| Experiment | `reproduce` name | Driver | Output |
|---|---|---|---|
| Exp. 1, 2DGS instead of 3DGS (room, KITTI 00/07) | `exp1-score`, `exp1` | `scripts/run_2dgs.sh` ([2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) @f3e3b9f in `GS2D_REPO`), `scripts/compare_2dgs.py`, `scripts/icp_at_gt_2dgs.py`, `scripts/map_agreement_2dgs.py` | `paper_data/extra_experiments/exp1_2dgs/` |
| Exp. 2, KISS-Matcher on LiDAR maps of the KITTI windows | `exp2-score`, `exp2` | `scripts/kitti_lidar_maps.py`, `examples/run_kitti_lidar_kiss.sh`, `scripts/kitti_lidar_kiss_score.py` | `paper_data/extra_experiments/exp2_kitti_lidar/` |
| Exp. 3, Gaussians per Mip-NeRF 360 map pair | `exp3` | `paper_data/extra_experiments/exp3_scene_sizes.py` | printed |

Success is 5 deg rotation and 0.5 m translation everywhere. On KITTI the
translation is the ATE over the source robot's ground-truth cameras
(`scripts/score_submap_csv.py --at gt_cameras`), because the maps' coordinate
origin can lie hundreds of metres from the data.

## KITTI multi-robot pairs

Robot windows (odometry frames, stride 2) are defined in
`examples/run_kitti_multirobot_queue.sh`:

| Pair | Robot A | Robot B | Overlap from |
|---|---|---|---|
| kitti00 | 450-1050 | 3450-3950 | revisit |
| kitti00_07 | 00: 2400-2950 | 07: 0-550 | different days |
| kitti06 | 0-500 | 600-1100 | revisit |
| kitti05 | 300-1000 | 1100-1800 | revisit |
| kitti02 | 1700-2150 | 4300-4660 | revisit |
| kitti08 | 0-600 | 1300-1900 | revisit, opposite direction |
| kitti09_10 | 09: 1050-1240 | 10: 550-780 | different drives, opposite direction |

KITTI's ground-truth poses disagree between two passes of one drive by up to
3.3 m (seq. 08). `scripts/kitti_check_revisit_gt.py` measures that offset with
LiDAR ICP and writes `kitti/revisit_gt/<scene>.json`. Cross-drive ground truth
comes from `scripts/kitti_align_sequences.py` (GPS init + LiDAR ICP) and is
checked with `scripts/kitti_validate_alignment.py`. Both sets of corrections
used for Table IV are committed under `data/kitti/` (a few hundred kB).

To score Table IV from the shipped CSVs you need the trained KITTI maps, their
filtered copies and `build/print_perturbation`:

```bash
for d in data/trained_gaussians/kitti*/*/point_cloud/iteration_30000; do
  uv run scripts/filter_3dgs_ply_batch.py "$d" "${d/trained_gaussians/trained_gaussians_filtered}" \
      --opacity-threshold 0.1
done
uv run reproduce table4-score    # matches the paper's Table IV exactly
```

The KITTI 06 and 00/07 rows were re-run after the colour-conversion fix below;
their earlier runs are kept as `*_pre_d65.csv`. Raw data is fetched by HTTP
range requests with `scripts/kitti_remote_zip.py`. 3DGS training needs a
[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
checkout (commit 54c035f) in `GS_REPO` and `colmap` on `PATH`.

## Baselines

The learned baselines run in their own CUDA environments (see each upstream
README); this branch ships only the changes needed to run them on 3DGS pairs.

- `baselines/gaussreg/`: patch for
  [GaussReg](https://github.com/GAP-LAB-CUHK-SZ/GaussReg) @85df227, plus the
  pair randomizer, offset runner and evaluator. Run in `full_features` mode.
- `baselines/colorpcr/`: patch for [ColorPCR](https://github.com/mujc2021/ColorPCR)
  @d579a80 and `match_3dgs.py` (copy into `experiments/ColorPCR/`). Pretrained
  weights, no retraining.
- KISS-Matcher runs inside `build/benchmark_submap` (method `vanilla_kiss`).

## Known issues

- **Re-runs use the corrected colour conversion.** The sRGB->XYZ matrix in
  `include/3dgs_matcher/ColorFasterPFH.hpp` had a typo in its Z row
  (0.2591849 instead of 0.1191920), fixed after the results in `paper_data/`
  were produced. Re-running reproduces the paper's trends, not its numbers bit
  for bit. Registration is also not bit-repeatable between runs: TBB's parallel
  sort breaks ties differently, so each run picks slightly different keypoints.
- **+ Colored ICP rows of Table II.** The published rows come from
  `coloredicp_from_*_v050_n200_crosspair_bug.csv`. In that run each scene's
  `split_wide` pair was replayed against the `split_pos` clouds. The current
  `scripts/benchmark_colored_icp.py` registers every pair against its own
  clouds. `uv run reproduce colored-icp` gives the corrected numbers.
- **Voxel figure threshold.** `scripts/plot_voxel_runtime.py` plots each sweep
  CSV's own `success` column, which `benchmark_submap` scores at 5 deg / 1 m.
  At the paper's 0.5 m the same sweeps give 80-82 % coarse and 86 % refined,
  the values quoted in the text.
- **Voxel sizes.** The descriptor ablation (Fig. 4) runs at 0.1 m. The tramway
  component ablation (Table I) and the Sec. V-B perturbations run at 0.3 m
  (`examples/noise_ablation_doubts.sh`, `examples/photometric_scale_perturbations.sh`).
- `paper_data/figure4_descriptor_ablation/ablation_results_RGB.csv` was written
  by an earlier version of the ablation runner (no `centroid_voxel` column).
  `examples/run_descriptor_ablation.sh` re-runs the same 31 configurations with
  the current `build/single_ablation`.
