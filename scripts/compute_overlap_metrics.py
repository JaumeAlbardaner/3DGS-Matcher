#!/usr/bin/env python3
"""No-ground-truth registration quality metrics for the real-data pairs.

Reads the CSV produced by examples/run_overlap_test.sh (benchmark_submap rows),
undoes each seed's random perturbation (via build/print_perturbation, which
replays the benchmark's RNG), and evaluates the alignment T_align = T_pert^-1
* T_est between the ORIGINAL clouds:

  - fitness src->tgt at each tau: fraction of downsampled source points whose
    nearest target neighbour lies within tau (Open3D-style fitness)
  - inlier RMSE at each tau
  - fitness tgt->src (reverse direction, same taus)
  - per-(pair, method) seed consensus: max/median pairwise rotation and
    translation difference between the per-seed T_align estimates

An identity row per pair gives the "no alignment" reference fitness.
"""
import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]

PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
}


def load_ply_xyz(path):
    """Minimal binary_little_endian PLY reader; returns (N,3) float32 xyz."""
    with open(path, "rb") as f:
        if f.readline().strip().lower() != b"ply":
            raise ValueError(f"not a PLY: {path}")
        fields, n_vertex, in_vertex = [], 0, False
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "end_header":
                break
            if tok[0] == "format" and tok[1] != "binary_little_endian":
                raise ValueError(f"unsupported PLY format {tok[1]}: {path}")
            if tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError("list property in vertex element")
                fields.append((tok[2], PLY_TYPES[tok[1]]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=n_vertex)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def voxel_downsample(pts, grid):
    keys = np.floor(pts / grid).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def perturbation(helper, seed, max_rot, max_trans):
    out = subprocess.run([str(helper), str(seed), str(max_rot), str(max_trans)],
                         capture_output=True, text=True, check=True).stdout.split()
    v = np.array([float(x) for x in out])
    T = np.eye(4)
    T[:3, :3] = v[:9].reshape(3, 3)
    T[:3, 3] = v[9:12]
    return T


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="output of run_overlap_test.sh")
    ap.add_argument("--pair", action="append", required=True, metavar="NAME:SRC:TGT",
                    help="pair name and its src/tgt PLY paths (repeatable)")
    ap.add_argument("--helper", default=str(REPO / "build/print_perturbation"))
    ap.add_argument("--max-rot", type=float, default=180.0)
    ap.add_argument("--max-trans", type=float, default=500.0)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--grid", type=float, default=0.2,
                    help="voxel size for the metric point sets (m)")
    ap.add_argument("--out", default=None, help="per-run metrics CSV")
    args = ap.parse_args()

    pairs = {}
    for spec in args.pair:
        name, src, tgt = spec.split(":")
        pairs[name] = (Path(src).expanduser(), Path(tgt).expanduser())

    clouds, trees = {}, {}
    for name, (src, tgt) in pairs.items():
        for p in (src, tgt):
            if p not in clouds:
                pts = voxel_downsample(load_ply_xyz(p), args.grid)
                clouds[p] = pts
                trees[p] = cKDTree(pts)
                print(f"[load] {p.name}: {len(pts)} pts at grid={args.grid}m",
                      file=sys.stderr)

    rows = list(csv.DictReader(open(args.csv)))
    T_pert = {}
    for r in rows:
        s = int(r["seed"])
        if s not in T_pert:
            T_pert[s] = perturbation(args.helper, s, args.max_rot, args.max_trans)

    def metrics(pts_src, tree_tgt, T):
        moved = pts_src @ T[:3, :3].T + T[:3, 3]
        d, _ = tree_tgt.query(moved, workers=-1)
        out = {}
        for tau in args.taus:
            m = d < tau
            out[f"fit@{tau}"] = float(m.mean())
            out[f"rmse@{tau}"] = float(np.sqrt((d[m] ** 2).mean())) if m.any() else float("nan")
        return out

    results, aligns = [], {}
    for r in rows:
        pair, method, seed = r["pair"], r["method"], int(r["seed"])
        if pair not in pairs:
            continue
        src, tgt = pairs[pair]
        q = [float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
        T_est = np.eye(4)
        T_est[:3, :3] = Rotation.from_quat(q).as_matrix()
        T_est[:3, 3] = [float(r["tx"]), float(r["ty"]), float(r["tz"])]
        T_align = np.linalg.inv(T_pert[seed]) @ T_est
        aligns.setdefault((pair, method), {})[seed] = T_align

        fwd = metrics(clouds[src], trees[tgt], T_align)
        rev = metrics(clouds[tgt], trees[src], np.linalg.inv(T_align))
        row = {"pair": pair, "method": method, "seed": seed,
               "solver_ok": r["solver_ok"], "time_ms": float(r["time_ms"])}
        row.update({f"src2tgt_{k}": v for k, v in fwd.items()})
        row.update({f"tgt2src_{k}": v for k, v in rev.items()})
        results.append(row)

    # identity reference (are the raw frames pre-aligned at all?)
    for pair, (src, tgt) in pairs.items():
        fwd = metrics(clouds[src], trees[tgt], np.eye(4))
        row = {"pair": pair, "method": "identity_ref", "seed": -1,
               "solver_ok": "-", "time_ms": 0.0}
        row.update({f"src2tgt_{k}": v for k, v in fwd.items()})
        results.append(row)

    if args.out:
        cols = sorted({k for r in results for k in r}, key=lambda c: (c not in
                      ("pair", "method", "seed", "solver_ok", "time_ms"), c))
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(results)
        print(f"[out] {args.out}", file=sys.stderr)

    # ── report ────────────────────────────────────────────────────────────────
    tau_mid = args.taus[len(args.taus) // 2]
    print(f"\n== per-run fitness (src->tgt fraction within tau; tau={tau_mid}m shown) ==")
    for r in results:
        if r["method"] == "identity_ref":
            continue
        print(f"  {r['pair']:14s} {r['method']:13s} seed {r['seed']}: "
              f"fit={r[f'src2tgt_fit@{tau_mid}']:.3f} "
              f"rmse={r[f'src2tgt_rmse@{tau_mid}']:.3f} "
              f"rev-fit={r[f'tgt2src_fit@{tau_mid}']:.3f} "
              f"({r['time_ms']/1000:.1f}s)")

    print(f"\n== summary: mean±std over seeds (tau={tau_mid}m) ==")
    for (pair, method), per_seed in sorted(aligns.items()):
        sel = [r for r in results if r["pair"] == pair and r["method"] == method]
        f = np.array([r[f"src2tgt_fit@{tau_mid}"] for r in sel])
        rv = np.array([r[f"tgt2src_fit@{tau_mid}"] for r in sel])
        # seed-to-seed consensus of T_align
        ds = []
        for a, b in itertools.combinations(per_seed.values(), 2):
            d = np.linalg.inv(a) @ b
            ds.append((rot_angle_deg(d[:3, :3]), np.linalg.norm(d[:3, 3])))
        if ds:
            dr = np.array([x[0] for x in ds]); dt = np.array([x[1] for x in ds])
            cons = (f"consensus: drot med/max {np.median(dr):.2f}/{dr.max():.2f} deg, "
                    f"dtrans med/max {np.median(dt):.2f}/{dt.max():.2f} m")
        else:
            cons = "consensus: n/a (1 seed)"
        print(f"  {pair:14s} {method:13s} fit {f.mean():.3f}±{f.std():.3f} "
              f"rev {rv.mean():.3f}±{rv.std():.3f} | {cons}")

    print(f"\n== identity reference (no alignment, tau={tau_mid}m) ==")
    for r in results:
        if r["method"] == "identity_ref":
            print(f"  {r['pair']:14s} fit={r[f'src2tgt_fit@{tau_mid}']:.3f}")


if __name__ == "__main__":
    main()
