r"""Run ColorPCR registration on 3DGS-Matcher submap pairs.

Mirrors the protocol of 3DGS-Matcher/examples/benchmark_submap.cc:
  - For each scene, match  split_pos -> split_neg  and  split_wide_pos -> split_wide_neg.
  - The *target* split is perturbed by a random rigid transform capped at
    (max_rot deg, max_trans m); the registrar must recover it.
  - Ground-truth relative pose between the two co-located splits is identity, so
    the transform that aligns src -> perturbed_tgt is exactly the perturbation.

3DGS .ply -> colored point cloud:
  - color = clamp01(0.5 + C0 * f_dc)   (C0 = 0.28209479177387814), same as the C++ loader.

Output CSV columns match benchmark_submap.cc:
  method,scene,split_src,split_tgt,iter,seed,rot_error_deg,trans_error_m,
  success,solver_ok,tx,ty,tz,qw,qx,qy,qz,time_ms
"""
import argparse
import csv
import os
import os.path as osp
import time

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from skimage import color as skcolor

from config import make_cfg
from model import create_model
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.modules.registration.metrics import isotropic_transform_error

SH_C0 = 0.28209479177387814
PLY_SUFFIX = "point_cloud/iteration_30000/point_cloud.ply"
ROT_THRESH = 5.0    # deg  (must match benchmark_submap)
TRANS_THRESH = 1.0  # m
NEIGHBOR_LIMITS = [38, 36, 36, 38]  # 3DMatch-calibrated caps the model was trained with

DEFAULT_SCENES = [
    "bicycle", "bonsai", "flowers", "garden", "stump", "treehill",
    "drone1_undistorted", "drone2_undistorted", "ugv_undistorted",
]


def read_ply_colored(path):
    """Read a 3DGS .ply -> (full xyz Nx3 float64, rgb Nx3 float64 in [0,1]).

    This mirrors what the C++ benchmark does *outside* its timer: PLY parsing and
    the SH-DC -> RGB conversion. The (timed) downsampling happens in downsample_colored.
    """
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    names = v.data.dtype.names
    if {"red", "green", "blue"} <= set(names):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64)
        if rgb.max() > 1.0:
            rgb /= 255.0
    else:
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
        rgb = 0.5 + SH_C0 * f_dc
    return xyz, np.clip(rgb, 0.0, 1.0)


def downsample_colored(xyz, rgb, voxel, max_points, rng):
    """Voxel-downsample a colored cloud (timed: the analog of the C++ in-estimate downsample)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if max_points and pts.shape[0] > max_points:
        idx = rng.permutation(pts.shape[0])[:max_points]
        pts, cols = pts[idx], cols[idx]
    return pts.astype(np.float32), cols.astype(np.float32)


def random_rotation(rng, max_angle_deg):
    """Uniform-axis rotation with angle capped at max_angle_deg (axis-angle)."""
    axis = rng.normal(size=3)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])
    angle = np.deg2rad(rng.uniform(0.0, max_angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def random_translation(rng, max_t):
    """Random translation with norm <= max_t."""
    d = rng.normal(size=3)
    d /= max(np.linalg.norm(d), 1e-8)
    return d * rng.uniform(0.0, max_t)


def rotmat_to_quat(R):
    """3x3 -> (qw, qx, qy, qz)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return qw, qx, qy, qz


def run_one_seed(cfg, model, ref_pts, ref_hsv, src_pts, src_hsv, R_true, t_true, s):
    """One registration attempt. Returns (rre_deg, rte_real, success, solver_ok, R_est, t_est_real, n_corr, time_ms)."""
    T_true = np.eye(4, dtype=np.float32)
    T_true[:3, :3] = R_true
    T_true[:3, 3] = t_true
    ref_pts_pert = (ref_pts @ R_true.T + t_true).astype(np.float32)
    data_dict = {
        "ref_points": ref_pts_pert,
        "src_points": src_pts,
        "ref_feats": np.ones((ref_pts.shape[0], 1), dtype=np.float32),
        "src_feats": np.ones((src_pts.shape[0], 1), dtype=np.float32),
        "ref_hsv": ref_hsv,
        "src_hsv": src_hsv,
        "transform": T_true,
    }
    # Timed region: KPConv subsample + neighbor search (collate) + host->GPU copy + forward.
    # This is the analog of the C++ in-estimate() downsample + features + solve.
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        collated = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, NEIGHBOR_LIMITS,
        )
        collated = to_cuda(collated)
        with torch.no_grad():
            output = model(collated)
        est = output["estimated_transform"]
        torch.cuda.synchronize()
        time_ms = (time.time() - t0) * 1000.0
        rre, rte = isotropic_transform_error(
            torch.from_numpy(T_true).cuda(), est, reduction="none")
        rre, rte = float(rre), float(rte) / s
        est = release_cuda(est)
        R_est, t_est = est[:3, :3], est[:3, 3] / s
        success = int(rre < ROT_THRESH and rte < TRANS_THRESH)
        n_corr = int(output["corr_scores"].shape[0])
        solver_ok = 1
    except Exception as e:
        time_ms = (time.time() - t0) * 1000.0
        solver_ok, success, rre, rte, n_corr = 0, 0, -1.0, -1.0, 0
        R_est, t_est = np.eye(3), np.zeros(3)
        print(f"    [error] {type(e).__name__}: {e}")
    torch.cuda.empty_cache()
    return rre, rte, success, solver_ok, R_est, t_est, n_corr, time_ms


def run_pair(cfg, model, base, scene, src_split, tgt_split, args, writer, fout):
    src_path = osp.join(base, scene, src_split, PLY_SUFFIX)
    tgt_path = osp.join(base, scene, tgt_split, PLY_SUFFIX)
    if not (osp.isfile(src_path) and osp.isfile(tgt_path)):
        print(f"  SKIP {scene} {src_split}->{tgt_split} (PLY missing)")
        return
    print(f"  [load] src: {src_path}", flush=True)
    print(f"  [load] tgt: {tgt_path}", flush=True)

    # UNTIMED (matches C++ exclusions): PLY parse + SH->RGB conversion.
    src_xyz, src_rgb_full = read_ply_colored(src_path)
    tgt_xyz, tgt_rgb_full = read_ply_colored(tgt_path)

    # TIMED prep (once per pair, deterministic across seeds; added to every row's time_ms):
    # voxel downsample (the analog of the C++ in-estimate downsample) + scale + rgb->hsv.
    # ColorPCR uses a fixed 0.025 voxel tuned for 3DMatch rooms (~3 unit extent), so we
    # rescale the scene to a comparable extent (rotation is scale-free; translation error
    # is converted back to real units inside run_one_seed). Fixed RNG -> identical point set.
    t_prep = time.time()
    load_rng = np.random.default_rng(0)
    src_pts, src_rgb = downsample_colored(src_xyz, src_rgb_full, args.voxel, args.max_points, load_rng)
    tgt_pts, tgt_rgb = downsample_colored(tgt_xyz, tgt_rgb_full, args.voxel, args.max_points, load_rng)
    diag = float(np.linalg.norm(src_pts.max(0) - src_pts.min(0)))
    s = args.target_diag / max(diag, 1e-6)
    src_pts = (src_pts * s).astype(np.float32)
    tgt_pts = (tgt_pts * s).astype(np.float32)
    src_hsv = skcolor.rgb2hsv(src_rgb).astype(np.float32)
    tgt_hsv = skcolor.rgb2hsv(tgt_rgb).astype(np.float32)
    prep_ms = (time.time() - t_prep) * 1000.0  # constant per-registration cost, added to each row

    # ColorPCR estimates src -> ref; we want it to recover the perturbation, so ref = perturbed tgt.
    n_ok, rres, rtes, times = 0, [], [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        R_true = random_rotation(rng, args.max_rot)
        t_true = random_translation(rng, args.max_trans) * s  # real <= max_trans, scaled
        rre, rte, success, solver_ok, R_est, t_est, n_corr, reg_ms = run_one_seed(
            cfg, model, tgt_pts, tgt_hsv, src_pts, src_hsv, R_true, t_true, s)
        time_ms = prep_ms + reg_ms  # downsample/prep + collate + to_cuda + forward
        qw, qx, qy, qz = rotmat_to_quat(R_est)
        writer.writerow(["colorpcr", scene, src_split, tgt_split, 0, seed,
                         f"{rre:.4f}", f"{rte:.4f}", success, solver_ok,
                         f"{t_est[0]:.4f}", f"{t_est[1]:.4f}", f"{t_est[2]:.4f}",
                         f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}",
                         f"{time_ms:.1f}"])
        n_ok += success
        if success:
            rres.append(rre); rtes.append(rte)
        times.append(time_ms)
    fout.flush()
    rate = 100.0 * n_ok / max(args.seeds, 1)
    mrre = np.mean(rres) if rres else float("nan")
    mrte = np.mean(rtes) if rtes else float("nan")
    print(f"  {scene:>10s} {src_split}->{tgt_split}: {n_ok:3d}/{args.seeds} "
          f"({rate:5.1f}%)  mean(success) RRE={mrre:5.2f}deg RTE={mrte:5.3f}m  "
          f"time={np.mean(times):6.1f}ms (prep={prep_ms:5.1f}+reg)  "
          f"(N_src={src_pts.shape[0]}, N_tgt={tgt_pts.shape[0]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="../../weights/weights.pth.tar")
    ap.add_argument("--base", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "trained_gaussians"))
    ap.add_argument("--scenes", nargs="*", default=None, help="default: all 9 scenes")
    ap.add_argument("--out", default="colorpcr_submap_results.csv")
    ap.add_argument("--voxel", type=float, default=0.08, help="input voxel downsample (scene units)")
    ap.add_argument("--max_points", type=int, default=25000)
    ap.add_argument("--target_diag", type=float, default=3.0,
                    help="rescale each scene to this diagonal extent (3DMatch-like) for the model")
    ap.add_argument("--max_rot", type=float, default=180.0)
    ap.add_argument("--max_trans", type=float, default=500.0)
    ap.add_argument("--seeds", type=int, default=1, help="number of perturbation seeds per pair (0..N-1)")
    args = ap.parse_args()

    scenes = args.scenes if args.scenes else DEFAULT_SCENES

    cfg = make_cfg()
    model = create_model(cfg).cuda()
    state = torch.load(args.snapshot, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    print(f"Loaded ColorPCR from {args.snapshot}")
    print(f"Params: voxel={args.voxel}  max_points={args.max_points}  "
          f"max_rot={args.max_rot}deg  max_trans={args.max_trans}m  seeds={args.seeds}")

    # Warmup: trigger one-time CUDA/cuDNN JIT so it doesn't land in any reported time_ms.
    # Use a thin surface-like slab (not a dense volume) so it mirrors real scans and won't OOM.
    print("Warming up CUDA kernels...", flush=True)
    wp = np.random.default_rng(0).random((4000, 3)).astype(np.float32) * np.float32([3.0, 3.0, 0.05])
    wc = np.random.default_rng(1).random((4000, 3)).astype(np.float32)
    whsv = skcolor.rgb2hsv(wc).astype(np.float32)
    run_one_seed(cfg, model, wp, whsv, wp.copy(), whsv.copy(), np.eye(3), np.zeros(3), 1.0)
    print("Warmup done.\n", flush=True)

    fout = open(args.out, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["method", "scene", "split_src", "split_tgt", "iter", "seed",
                     "rot_error_deg", "trans_error_m", "success", "solver_ok",
                     "tx", "ty", "tz", "qw", "qx", "qy", "qz", "time_ms"])
    for scene in scenes:
        run_pair(cfg, model, args.base, scene, "split_pos", "split_neg", args, writer, fout)
        run_pair(cfg, model, args.base, scene, "split_wide_pos", "split_wide_neg", args, writer, fout)
    fout.close()
    print(f"\nResults -> {args.out}")


if __name__ == "__main__":
    main()
