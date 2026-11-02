#!/usr/bin/env python3
"""Score a benchmark_submap CSV at the paper's success criterion (5 deg / 0.5 m).

Same rules as docs/.../paper_data/table2_large_scale/gen_large_scale_table.py,
but for any scene set (e.g. KITTI), reported per split pair and pooled:
  * success = rotation error < 5 deg and translation error < 0.5 m  (NOT the binary's 1 m column)
  * refined methods' time = coarse color_cpfh time + own stage, matched per seed

WHERE the translation error is measured (--at):
  origin            the CSV's trans_error_m, i.e. the residual at the maps' coordinate
                    origin (default; what Table II uses). Fine when the origin sits in
                    the map, misleading when it is far away: a 0.1 deg residual adds
                    ~0.7 m at 400 m.
  gt_cameras        ATE-style: RMS displacement of the source map's ground-truth camera
                    positions (cameras.json of the trained model) under the estimated
                    registration, with no re-alignment. Recommended for KITTI.
  map_centroid      residual at the source map's centroid
  overlap_centroid  residual at the centroid of source Gaussians within 0.5 m of the target
All but origin replay each seed's perturbation with build/print_perturbation and need
--base (the trained_gaussians root the benchmark read); gt_cameras reads cameras.json
from --cameras_base (default --base; filtered trees have no cameras.json).

--gt_correction FILE: the benchmark assumes the true source->target transform is
identity. When the maps' ground-truth frames are known to disagree (e.g. KITTI GT
poses drift between two passes of a sequence), pass the JSON written by
kitti_check_revisit_gt.py; rotation AND translation errors are then measured
against that corrected transform (the CSV's rot_error_deg is not used).

  python3 scripts/score_submap_csv.py results/kitti07_joint5d_v050.csv
  python3 scripts/score_submap_csv.py results/kitti00_07_robots_filtered_joint5d_v050.csv \
      --at gt_cameras --base $DATA_ROOT/trained_gaussians_filtered \
      --cameras_base $DATA_ROOT/trained_gaussians
"""
import argparse
import csv
import json
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

ROT_THRESH, TRANS_THRESH = 5.0, 0.5
JOIN = ("scene", "split_src", "split_tgt", "iter", "seed")
PLY = "point_cloud/iteration_30000/point_cloud.ply"
REPO = Path(__file__).resolve().parents[1]
METHODS = [  # (label, method, refined) -- row order of the paper's Table II
    ("KISS-Matcher", "vanilla_kiss", False),
    ("Ours, joint-5D", "color_cpfh", False),
    ("  + GICP", "gicp_cpfh", True),
    ("  + 3DGS-ICP", "color_gicp_de", True),
]


def load_correction(path):
    import numpy as np
    return np.array(json.loads(Path(path).read_text())["T_correction_B_to_A"], float)


class ResidualAt:
    """Errors of a benchmark row, evaluated inside the maps instead of at the origin.

    Calling returns the translation error; .rot(row) returns the rotation error in
    degrees, both against `gt` (identity unless a ground-truth correction is given).
    """

    def __init__(self, at, base, cameras_base, max_rot, max_trans, gt=None):
        import numpy as np
        self.np, self.at = np, at
        self.base, self.cameras_base = Path(base), Path(cameras_base)
        self.max_rot, self.max_trans = max_rot, max_trans
        self.gt_inv = np.eye(4) if gt is None else np.linalg.inv(gt)
        self.points, self.perts = {}, {}

    def _points(self, r):
        """(N, 3) points whose displacement is measured, in the ground-truth frame."""
        key = (r["scene"], r["split_src"], r["split_tgt"])
        if key not in self.points:
            np = self.np
            if self.at == "gt_cameras":
                cams = json.load(open(self.cameras_base / r["scene"] / r["split_src"] / "cameras.json"))
                self.points[key] = np.array([c["position"] for c in cams], float)
            else:
                from compute_overlap_metrics import load_ply_xyz
                src = load_ply_xyz(self.base / r["scene"] / r["split_src"] / PLY)
                if self.at == "map_centroid":
                    self.points[key] = src.mean(0)[None]
                else:
                    from scipy.spatial import cKDTree
                    tgt = load_ply_xyz(self.base / r["scene"] / r["split_tgt"] / PLY)
                    d, _ = cKDTree(tgt).query(src, distance_upper_bound=TRANS_THRESH)
                    self.points[key] = src[d <= TRANS_THRESH].mean(0)[None]
        return self.points[key]

    def _pert(self, seed):
        if seed not in self.perts:
            out = subprocess.run([str(REPO / "build/print_perturbation"), str(seed),
                                  str(self.max_rot), str(self.max_trans)],
                                 capture_output=True, text=True, check=True).stdout.split()
            T = self.np.eye(4)
            T[:3, :3] = self.np.array(out[:9], float).reshape(3, 3)
            T[:3, 3] = self.np.array(out[9:], float)
            self.perts[seed] = T
        return self.perts[seed]

    def error(self, r):
        """4x4 residual transform; identity for a perfect estimate."""
        np = self.np
        from scipy.spatial.transform import Rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([float(r[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
        T[:3, 3] = [float(r[k]) for k in ("tx", "ty", "tz")]
        return self.gt_inv @ np.linalg.inv(self._pert(int(r["seed"]))) @ T

    def rot(self, r):
        np = self.np
        E = self.error(r)
        return float(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))

    def __call__(self, r):
        np = self.np
        E = self.error(r)
        X = self._points(r)
        disp = X @ E[:3, :3].T + E[:3, 3] - X
        return float(np.sqrt(np.mean(np.sum(disp ** 2, axis=1))))  # RMS; a single point for centroids


def summarize(rows, coarse_time, refined, trans_of, rot_of):
    t = [trans_of(r) for r in rows]
    rot = [rot_of(r) for r in rows]
    succ = [a < ROT_THRESH and b < TRANS_THRESH for a, b in zip(rot, t)]
    secs = []
    for r in rows:
        s = float(r["time_ms"]) / 1000.0
        if refined:
            s += coarse_time.get(tuple(r[k] for k in JOIN), 0.0)
        secs.append(s)
    return (f"{100 * sum(succ) / len(succ):5.1f}% ({sum(succ):4d}/{len(succ):4d})  "
            f"trans {st.mean(t):8.2f}±{st.pstdev(t):<8.2f} rot {st.mean(rot):6.1f}±{st.pstdev(rot):<6.1f} "
            f"time {st.mean(secs):6.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--scenes", nargs="+", help="only these scenes (default: all in the CSV)")
    ap.add_argument("--at", choices=("origin", "gt_cameras", "map_centroid", "overlap_centroid"),
                    default="origin", help="where the translation error is measured (see above)")
    ap.add_argument("--base", help="trained_gaussians root; needed unless --at origin")
    ap.add_argument("--cameras_base", help="root holding <scene>/<split>/cameras.json (default --base)")
    ap.add_argument("--gt_correction", help="JSON with T_correction_B_to_A (kitti_check_revisit_gt.py)")
    ap.add_argument("--max_rot", type=float, default=180.0, help="benchmark perturbation, deg")
    ap.add_argument("--max_trans", type=float, default=500.0, help="benchmark perturbation, m")
    args = ap.parse_args()
    if args.at == "origin":
        if args.gt_correction:
            ap.error("--gt_correction needs an --at other than origin")
        trans_of = lambda r: float(r["trans_error_m"])
        rot_of = lambda r: float(r["rot_error_deg"])
    else:
        if not args.base:
            ap.error(f"--at {args.at} needs --base")
        gt = load_correction(args.gt_correction) if args.gt_correction else None
        trans_of = ResidualAt(args.at, args.base, args.cameras_base or args.base,
                              args.max_rot, args.max_trans, gt)
        rot_of = trans_of.rot

    groups = defaultdict(list)  # (pair label, method) -> rows
    coarse_time = {}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if args.scenes and r["scene"] not in args.scenes:
                continue
            groups[(f'{r["scene"]} {r["split_src"]}->{r["split_tgt"]}', r["method"])].append(r)
            groups[("POOLED", r["method"])].append(r)
            if r["method"] == "color_cpfh":
                coarse_time[tuple(r[k] for k in JOIN)] = float(r["time_ms"]) / 1000.0

    pairs = sorted({p for p, _ in groups} - {"POOLED"}) + ["POOLED"]
    gt_note = f", ground truth corrected by {args.gt_correction}" if args.gt_correction else ""
    print(f"success @ {ROT_THRESH:g} deg / {TRANS_THRESH:g} m, translation measured at: {args.at}{gt_note}\n")
    for pair in pairs:
        print(pair)
        for label, method, refined in METHODS:
            if groups.get((pair, method)):
                print(f"  {label:16s} {summarize(groups[(pair, method)], coarse_time, refined, trans_of, rot_of)}")
        print()


if __name__ == "__main__":
    main()
