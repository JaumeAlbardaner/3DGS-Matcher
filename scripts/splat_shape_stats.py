#!/usr/bin/env python3
"""Fig. 3 splat-shape statistics over every trained 3DGS map on disk.

For each Gaussian, the covariance is Sigma = R S S^T R^T. R is orthonormal, so
Sigma's eigenvalues are the squared linear scales and the rotation drops out:
    lambda1 >= lambda2 >= lambda3 = sorted(s**2)
    anisotropy = (l1 - l3) / l1,  linearity = (l1 - l2) / l1,  planarity = (l2 - l3) / l1
Reported per map as in Fig. 3 / Section V:
    % anisotropy > 0.95,  % linearity > 0.95,  % planarity < 0.05

Scale storage differs by format, and getting it wrong silently corrupts every feature:
    .ply   (3DGS training output)  scale_0..2 are LOG-scales  -> exp() applied here
    .splat (antimatter15 format)   scales are already linear
(LiDAR-3DGS-Relocalization/.../gs_operations.py::read_gs_ply does NOT exponentiate,
so it is only correct for .splat input -- which is what Fig. 3 used.)

Only the three scale columns are read, via a memory map, so multi-GB PLYs are cheap.

  python3 scripts/splat_shape_stats.py --validate          # reproduce Fig. 3 (87/46/52 %)
  python3 scripts/splat_shape_stats.py                     # full inventory -> CSV + summary
"""
import os
import argparse
import csv
import hashlib
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
# Extra tree scanned for third-party *.splat maps (the paper pooled maps from both).
GIT = Path(os.environ.get("SPLAT_ROOT", HOME / "Documents/git"))
OUT_CSV = DATA / "results/splat_shape_stats.csv"
FIG3_SOURCE = GIT / "LiDAR-3DGS-Relocalization/pc-to-gs-alignment/data/gs/moia_drone1.splat"
BINS = 20                      # matches the paper's Fig. 3 (bin edges hit 0.05 / 0.95)
OUT_NPZ = DATA / "results/splat_shape_histograms.npz"

MIPNERF = {"bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"}
OURS_HALVES = {"drone1_undistorted", "drone2_undistorted", "ugv_undistorted"}

PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


# ── readers: return an (N, 3) array of LINEAR scales ────────────────────────────
def ply_scales(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header += line
        offset = f.tell()
    lines = header.decode("ascii", "replace").splitlines()
    fmt = next(l.split()[1] for l in lines if l.startswith("format"))
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"{path}: unsupported PLY format {fmt}")
    end = "<" if fmt == "binary_little_endian" else ">"
    n, props, in_vertex = 0, [], False
    for l in lines:
        t = l.split()
        if t[:1] == ["element"]:
            in_vertex = t[1] == "vertex"
            if in_vertex:
                n = int(t[2])
        elif t[:1] == ["property"] and in_vertex:
            if t[1] == "list":
                raise ValueError(f"{path}: list property in vertex element")
            props.append((t[2], end + PLY_TYPES[t[1]]))
    names = [p[0] for p in props]
    if not all(k in names for k in ("scale_0", "scale_1", "scale_2")):
        raise ValueError(f"{path}: no scale_0..2 (not a 3DGS map)")
    mm = np.memmap(path, dtype=np.dtype(props), mode="r", offset=offset, shape=(n,))
    log_s = np.stack([mm["scale_0"], mm["scale_1"], mm["scale_2"]], axis=1).astype(np.float64)
    return np.exp(log_s)


def splat_scales(path):
    dt = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    mm = np.memmap(path, dtype=dt, mode="r")
    return mm["scale"].astype(np.float64)


def shape_features(scales):
    """(anisotropy, linearity, planarity) per Gaussian, degenerate ones dropped."""
    lam = np.sort(scales ** 2, axis=1)[:, ::-1]          # descending eigenvalues
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    ok = l1 > 0
    l1, l2, l3 = l1[ok], l2[ok], l3[ok]
    return (l1 - l3) / l1, (l1 - l2) / l1, (l2 - l3) / l1


def shape_stats(scales):
    aniso, lin, plan = shape_features(scales)
    return (len(aniso),
            100.0 * np.mean(aniso > 0.95),
            100.0 * np.mean(lin > 0.95),
            100.0 * np.mean(plan < 0.05))


def shape_histograms(scales, bins=BINS):
    """Per-map NORMALIZED 20-bin histograms of the three features over [0, 1].

    Normalizing per map before averaging is what makes the pooled figure a mean
    over maps rather than over Gaussians: without it the few largest maps (the
    Mip-NeRF 360 halves are half of all Gaussians) would decide the shape.
    The bin edges land on the paper's thresholds (0.95 / 0.05), so the last and
    first bins are exactly the percentages reported in the text.
    """
    out = []
    for f in shape_features(scales):
        h, _ = np.histogram(f, bins=bins, range=(0.0, 1.0))
        out.append(h / max(h.sum(), 1))
    return np.stack(out)


# ── inventory ───────────────────────────────────────────────────────────────────
def group_of(path):
    s = str(path)
    if "_filtered/" in s or "_cube" in s or "_cropped" in s or "/_work/" in s:
        return None                                         # derived copies
    if s.endswith(".ply"):
        if "/trained_gaussians/" in s:
            scene = path.parts[path.parts.index("trained_gaussians") + 1]
            if scene in MIPNERF:
                return "Mip-NeRF 360 halves"
            if scene in OURS_HALVES:
                return "Our dataset, aerial/ground halves"
            if scene.startswith("kitti"):
                return "KITTI robot maps"
        if "/submap_gaussians/" in s:
            return "Older submap training runs"
        if "/FullSplats/" in s:
            return "Our dataset, full maps"
        return None
    name = path.name
    if name.startswith("moia_drone"):
        return "Our drone (.splat, Fig. 3 source)"
    if name == "tramway.splat":
        return "Tramway (Table I)"
    return "Third-party .splat (other trainers)"


def fingerprint(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return (path.stat().st_size, h.hexdigest())


def inventory():
    cands = sorted(DATA.rglob("iteration_30000/point_cloud.ply"))
    cands += sorted(p for p in GIT.rglob("*.splat"))
    seen, maps = {}, []
    for p in cands:
        g = group_of(p)
        if g is None:
            continue
        fp = fingerprint(p)
        if fp in seen:                                      # byte-identical copy
            continue
        seen[fp] = p
        maps.append((g, p))
    return maps


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="reproduce Fig. 3 on its source map")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--histograms", type=Path, nargs="?", const=OUT_NPZ,
                    help="also save per-map normalized histograms for the pooled Fig. 3")
    args = ap.parse_args()

    if args.validate:
        n, a, l, pl = shape_stats(splat_scales(FIG3_SOURCE))
        print(f"Fig. 3 source ({FIG3_SOURCE.name}): n={n:,}  aniso>0.95 {a:.1f}%  "
              f"lin>0.95 {l:.1f}%  plan<0.05 {pl:.1f}%   (paper: 709,401 / 87 / 46 / 52)")
        tram = Path(__file__).resolve().parents[1] / "examples/splats/tramway.splat"
        n, a, l, pl = shape_stats(splat_scales(tram))
        print(f"tramway.splat: n={n:,}  aniso>0.95 {a:.1f}%  lin>0.95 {l:.1f}%  plan<0.05 {pl:.1f}%")
        return

    maps = inventory()
    print(f"{len(maps)} distinct maps", file=sys.stderr)
    rows, hists = [], []
    for g, p in maps:
        try:
            sc = splat_scales(p) if p.suffix == ".splat" else ply_scales(p)
            n, a, l, pl = shape_stats(sc)
            if args.histograms:
                hists.append(shape_histograms(sc))
        except Exception as e:                              # report, never silently drop
            print(f"  SKIP {p}: {e}", file=sys.stderr)
            continue
        label = str(p.relative_to(DATA) if str(p).startswith(str(DATA)) else p.relative_to(GIT))
        rows.append({"group": g, "map": label, "gaussians": n,
                     "aniso_gt_095": round(a, 2), "lin_gt_095": round(l, 2), "plan_lt_005": round(pl, 2)})
        print(f"  {a:5.1f} {l:5.1f} {pl:5.1f}  {n:>9,}  {label}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    keys = ("aniso_gt_095", "lin_gt_095", "plan_lt_005")

    def line(label, rs):
        cells = []
        for k in keys:
            v = [r[k] for r in rs]
            cells.append(f"{st.mean(v):5.1f} [{min(v):5.1f}-{max(v):5.1f}]")
        tot = sum(r["gaussians"] for r in rs)
        return f"{label:38s} {len(rs):4d} {tot/1e6:8.1f}M   " + "   ".join(cells)

    print("\nMean over maps [min-max across maps]")
    print(f"{'group':38s} {'maps':>4s} {'splats':>9s}   {'aniso>0.95':^18s}   {'lin>0.95':^18s}   {'plan<0.05':^18s}")
    for g in sorted(by):
        print(line(g, by[g]))
    print(line("ALL MAPS (map-averaged)", rows))
    pooled = [sum(r[k] * r["gaussians"] for r in rows) / sum(r["gaussians"] for r in rows) for k in keys]
    print(f"{'ALL SPLATS (splat-weighted)':38s} {'':4s} {'':9s}   " + "   ".join(f"{v:5.1f}{'':13s}" for v in pooled))
    print(f"\nPer-map CSV -> {args.out}")
    if args.histograms:
        np.savez_compressed(
            args.histograms, hists=np.stack(hists),      # (maps, 3, BINS), row-normalized
            groups=np.array([r["group"] for r in rows]),
            names=np.array([r["map"] for r in rows]),
            gaussians=np.array([r["gaussians"] for r in rows]), bins=BINS)
        print(f"Per-map histograms -> {args.histograms}")


if __name__ == "__main__":
    main()
