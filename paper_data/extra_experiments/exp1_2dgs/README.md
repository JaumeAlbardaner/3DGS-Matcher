# Extra experiment 1: geometry-regularized 3DGS (2DGS), KISS-Matcher vs ours

Not part of the final paper. Kept for completeness.

Run 2026-09-17. Question: on a geometry-regularized variant, does geometric matching recover,
and does our colour-based pipeline still work?

## Setup
* **Variant**: 2DGS (hbb1/2d-gaussian-splatting, default 30k iters, default λ_normal=0.05,
  λ_dist=0). Installed at `~/Documents/git/2d-gaussian-splatting`, conda env `2dgs`
  (CUDA 12.8.1 toolkit, torch 2.11+cu128, gcc 13; `<cstdint>`/`<cfloat>` includes added to
  the two submodules to build on this machine).
* **Scenes** (same splits/images/poses as the vanilla maps, so GT = identity):
  * Mip-NeRF 360 **room** (best vanilla scene: 100 % for all our variants), both split pairs.
  * KITTI **00/07** (two different sequences/days, robot_07 → robot_00), unfiltered and
    filtered (opacity 0.1 + SOR), exactly as the KITTI multi-robot table.
* **Benchmark**: unchanged `benchmark_submap`, joint-5D, opacity on, 0.5 m voxel,
  ±180° / ±500 m, 100 seeds per pair. Scoring as in the paper: 5° / 0.5 m; room at the
  origin (Table II), KITTI as ATE over the source robot's GT cameras.
* 2DGS has no `--train_test_exp` (per-image exposure compensation) — the vanilla maps used it.
* A different session was running CPU benchmarks during part of this; runtimes are indicative.

Reproduce: `scripts/run_2dgs.sh` (resumable) then `python3 scripts/compare_2dgs.py`.

## Results (success at 5° / 0.5 m, n = 100 per row)

All rows use the same `benchmark_submap` build (post CIELAB/D65 fix, 2026-09-14 19:42). Vanilla
baselines: room re-run 2026-09-17 (`results/room_3dgs_joint5d_v050_postfix.csv`; the June Table II
CSV predates the fix and gives KISS 4 % / 11 %), KITTI from the `_d65` re-runs used by Table IV.

| Pair | Maps | #Gaussians (src/tgt) | KISS-Matcher | Ours, coarse | + GICP | + 3DGS-ICP | 3DGS-ICP err (median, successes) |
|---|---|---|---|---|---|---|---|
| room pos/neg | 3DGS | 973k / 1.23M | 3% | 100% | 100% | 100% | 0.05 m |
| room pos/neg | **2DGS** | 655k / 900k | **21%** | 100% | 100% | 100% | 0.08 m |
| room wide | 3DGS | 1.19M / 1.19M | 9% | 100% | 100% | 100% | 0.05 m |
| room wide | **2DGS** | 796k / 839k | **35%** | 100% | 100% | 100% | 0.05 m |
| KITTI 00/07 unfilt. | 3DGS | 573k / 286k | 0% | 6% | 19% | 63% | 0.38 m |
| KITTI 00/07 unfilt. | **2DGS** | 348k / 155k | 0% | 8% | 0% | 17% | 0.41 m |
| KITTI 00/07 filt. | 3DGS | 220k / 119k | 0% | 3% | 80% | 96% | 0.30 m |
| KITTI 00/07 filt. | **2DGS** | 238k / 112k | 0% | 3% | 0% | 1% | 0.48 m |

KITTI, 2DGS maps, relaxed threshold (ATE, % of seeds): coarse basin (rot<5°) 94 % unfilt. / 100 % filt.;
+GICP and +3DGS-ICP basin 100 %; within 1 m: +GICP 93 / 28 %, +3DGS-ICP 93 / 78 % (unfilt. / filt.).
Vanilla (`_d65`) +3DGS-ICP within 1 m: 97 / 100 %.

ICP between the robot maps started at GT (`scripts/icp_at_gt_2dgs.py`, output `icp_at_gt.txt`):
correction ATE 2DGS 0.76 m / 0.85° vs 3DGS 0.27 m / 0.28° at 1 m correspondence radius
(0.36 vs 0.15 m at 0.5 m radius); fitness 0.40 vs 0.50.

## Reading
1. **Mip-NeRF (room)**: regularized geometry does help the geometric baseline, 3→21 % and
   9→35 %. It does not close the gap: our pipeline stays at
   100 % on both pairs, with the same accuracy and ~30 % less runtime (fewer Gaussians).
2. **KITTI 00/07**: KISS-Matcher stays at 0 %. Our **coarse stage is unaffected** (reaches the
   correct basin in 94–100 % of seeds, same accuracy distribution as on 3DGS). The drop in the
   refined columns is a **fine-accuracy / threshold effect caused by the maps, not a matching failure**:
   * every refined seed lands in the correct basin (100 %), but at ~0.8 m ATE instead of ~0.3 m;
   * plain point-to-plane ICP between the two robot maps **started at the GT pose** has to move the
     2DGS maps by 0.76 m ATE / 0.85° to agree (vanilla: 0.27 m / 0.28°), with lower overlap
     fitness (0.40 vs 0.50). The two independently trained 2DGS reconstructions are less
     consistent with each other in the GT frame, so any geometry-driven refinement settles
     ~0.4–0.8 m from GT.
   * Plausible causes (not isolated): no exposure compensation across two capture days; 2DGS
     surfels on sparse forward-driving views giving biased depth.
3. Net for the claim: geometric matchability improves somewhat on a well-observed object-centric
   scene, but not enough to rival colour, and not at all on the robotic outdoor pair.

## Why the renders look good but KITTI scores drop
`scripts/map_agreement_2dgs.py` → `map_agreement.txt`, on the shared road at the GT pose:
surface-to-surface NN distance median/p90 = 0.41/1.18 m (3DGS) vs 0.54/1.40 m (2DGS); per-cell
ground-height difference has the same median bias but a wider IQR (1.12 → 1.46 m). With the ICP-at-GT
result, the two 2DGS maps are ~2.5x looser in mutual agreement while each renders well: rendering is
only constrained from the training views, and a depth error along the ray is nearly invisible at the
small baselines of forward driving. Refinement converges to where the maps agree (~0.8 m from GT),
which is past the 0.5 m criterion but still within 1 m. 2DGS also kept 348k vs 573k Gaussians
(room: 655k vs 973k), i.e. sparser centroids for the matcher. Room, with orbital capture and wide
baselines, shows no such effect (100 % for ours, KISS improving) — the difference is capture geometry,
not render quality.
