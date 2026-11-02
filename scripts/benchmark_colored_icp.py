#!/usr/bin/env python3
"""Colored-ICP (Park et al., ICCV 2017; Open3D) refinement baseline for Table II.

For every (scene, seed) of the published joint-5D benchmark, warm-starts
Open3D's registration_colored_icp from the SAME coarse 3DGS-Matcher pose
stored in the benchmark CSV (color_cpfh rows), against the SAME perturbed
target (seed-exact RNG replay via build/print_perturbation), at the same
0.5 m operating point as the other refinement rows (max_corr = 3x voxel).

Output rows share the benchmark_submap CSV schema (method "colored_icp"),
so gen_large_scale_table*.py can ingest them after adding the row spec.

Usage:
  python3 scripts/benchmark_colored_icp.py [--scenes bicycle bonsai ...]
      [--csv paper_data/table2_large_scale/joint5d_trained_v050_n200_opacityON.csv]
      [--out results/coloredicp_trained_v050_n200.csv] [--limit N]
"""
import os
import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data")) / "trained_gaussians"
HELPER = REPO / "build/print_perturbation"
DEFAULT_CSV = REPO / "paper_data/table2_large_scale/joint5d_trained_v050_n200_opacityON.csv"

VOXEL = 0.5
MAX_CORR = 3.0 * VOXEL          # matches gicp_cfg.max_correspondence_distance
MAX_ITER = 50                   # matches gicp_cfg.max_iterations
# --multiscale: coarse-to-fine schedule (generous variant, per the original
# colored-ICP usage); each level (voxel, iters), max_corr = 3x voxel.
MS_LEVELS = [(0.5, 50), (0.25, 30)]
ROT_T, TRANS_T = 5.0, 0.5       # paper success criterion (deg, m)
SH_C0 = 0.28209479177387814

PLY_TYPES = {"float": "f4", "double": "f8", "uchar": "u1", "int": "i4",
             "uint": "u4", "short": "i2", "ushort": "u2", "char": "i1"}


def load_splat_ply(path):
    """xyz + RGB (from f_dc SH) as float64 arrays."""
    with open(path, "rb") as f:
        assert f.readline().strip().lower() == b"ply"
        fields, n, in_v = [], 0, False
        while True:
            tok = f.readline().decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "end_header":
                break
            if tok[0] == "format":
                assert tok[1] == "binary_little_endian", tok
            elif tok[0] == "element":
                in_v = tok[1] == "vertex"
                if in_v:
                    n = int(tok[2])
            elif tok[0] == "property" and in_v:
                fields.append((tok[2], PLY_TYPES[tok[1]]))
        d = np.fromfile(f, dtype=np.dtype(fields), count=n)
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    rgb = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1).astype(np.float64)
    rgb = np.clip(SH_C0 * rgb + 0.5, 0.0, 1.0)
    return xyz, rgb


def make_cloud(xyz, rgb, voxel):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel,
                                                             max_nn=30))
    return pc


def perturbation(seed):
    out = subprocess.run([str(HELPER), str(seed), "180", "500"],
                         capture_output=True, text=True, check=True).stdout.split()
    v = np.array([float(x) for x in out])
    T = np.eye(4)
    T[:3, :3] = v[:9].reshape(3, 3)
    T[:3, 3] = v[9:12]
    return T


def row_T(r):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        [float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]).as_matrix()
    T[:3, 3] = [float(r["tx"]), float(r["ty"]), float(r["tz"])]
    return T


def errors(T_est, T_pert):
    dR = T_est[:3, :3] @ T_pert[:3, :3].T
    re = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    te = np.linalg.norm(T_est[:3, 3] - T_pert[:3, 3])
    return re, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DATA.parent / "results/coloredicp_trained_v050_n200.csv"))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="seeds per scene")
    ap.add_argument("--multiscale", action="store_true",
                    help="coarse-to-fine schedule (method colored_icp_ms)")
    args = ap.parse_args()
    levels = MS_LEVELS if args.multiscale else [(VOXEL, MAX_ITER)]
    method = "colored_icp_ms" if args.multiscale else "colored_icp"

    coarse = defaultdict(list)
    for r in csv.DictReader(open(args.csv)):
        if r["method"] == "color_cpfh":
            coarse[r["scene"]].append(r)
    scenes = args.scenes or sorted(coarse)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    fresh = not out_path.exists()
    out = open(out_path, "a", newline="")
    w = csv.writer(out)
    if fresh:
        w.writerow(["method", "scene", "split_src", "split_tgt", "iter", "seed",
                    "rot_error_deg", "trans_error_m", "success", "solver_ok",
                    "tx", "ty", "tz", "qw", "qx", "qy", "qz", "time_ms"])

    perts = {}
    for sc in scenes:
      # Each scene holds TWO independently registered pairs (split_pos/neg and
      # split_wide_pos/neg, 100 seeds each): run every pair's seeds against
      # that pair's own clouds.
      pairs = {}
      for r in coarse[sc]:
          pairs.setdefault((r["split_src"], r["split_tgt"]), []).append(r)
      for (src_split, tgt_split), rows in sorted(pairs.items()):
        if args.limit:
            rows = rows[:args.limit]
        base = DATA / sc
        src_raw = load_splat_ply(
            base / src_split / "point_cloud/iteration_30000/point_cloud.ply")
        tgt_raw = load_splat_ply(
            base / tgt_split / "point_cloud/iteration_30000/point_cloud.ply")
        pyr = [(make_cloud(*src_raw, v), make_cloud(*tgt_raw, v), v, it)
               for v, it in levels]
        src, tgt0 = pyr[0][0], pyr[0][1]
        print(f"[{sc}:{src_split}->{tgt_split}] src={len(src.points)} "
              f"tgt={len(tgt0.points)} pts (level0 {levels[0][0]} m, "
              f"{len(levels)} level(s)), {len(rows)} seeds", file=sys.stderr)

        # protocol validation: our replayed perturbation must reproduce the
        # CSV's stored errors for the coarse rows (to ~1e-2 tolerance).
        for r in rows[:3]:
            s = int(r["seed"])
            perts.setdefault(s, perturbation(s))
            re, te = errors(row_T(r), perts[s])
            dre = abs(re - float(r["rot_error_deg"]))
            dte = abs(te - float(r["trans_error_m"]))
            if dre > 0.05 or dte > 0.05:
                sys.exit(f"PROTOCOL MISMATCH {sc} seed {s}: "
                         f"recomputed {re:.3f}deg/{te:.3f}m vs CSV "
                         f"{r['rot_error_deg']}/{r['trans_error_m']}")

        n_ok = 0
        for r in rows:
            s = int(r["seed"])
            perts.setdefault(s, perturbation(s))
            T_pert = perts[s]
            t0 = time.perf_counter()
            T_est, ok = row_T(r), 1
            for src_l, tgt_l, v, it in pyr:
                tgt = o3d.geometry.PointCloud(tgt_l)
                tgt.transform(T_pert)
                try:
                    res = o3d.pipelines.registration.registration_colored_icp(
                        src_l, tgt, 3.0 * v, T_est,
                        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=it))
                    T_est = np.asarray(res.transformation)
                except RuntimeError:    # e.g. no correspondences at this level
                    ok = 0
                    break
            ms = (time.perf_counter() - t0) * 1e3
            re, te = errors(T_est, T_pert)
            succ = int(re < ROT_T and te < TRANS_T)
            n_ok += succ
            q = Rotation.from_matrix(T_est[:3, :3]).as_quat()  # x,y,z,w
            w.writerow([method, sc, src_split, tgt_split, r["iter"], s,
                        f"{re:.6g}", f"{te:.6g}", succ, ok,
                        f"{T_est[0,3]:.6g}", f"{T_est[1,3]:.6g}", f"{T_est[2,3]:.6g}",
                        f"{q[3]:.6g}", f"{q[0]:.6g}", f"{q[1]:.6g}", f"{q[2]:.6g}",
                        f"{ms:.6g}"])
            out.flush()
        print(f"[{sc}:{src_split}->{tgt_split}] done: {n_ok}/{len(rows)} success",
              file=sys.stderr)
    out.close()
    print(f"written -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
