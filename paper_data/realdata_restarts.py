#!/usr/bin/env python3
"""Paper Sec. VI: restart convergence on the real aerial-ground pairs.

From the real aerial-ground run (examples/run_overlap_test.sh, 100 restarts per
pair, paper_data/realdata_overlap_n100*.csv). Each restart's recovered pose is
inv(P_seed) @ T_est. A restart converges when it lies within 10 deg / 5 m of the
maximum-fitness restart. Needs build/print_perturbation.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
PD = REPO / "paper_data"
DATA = Path(os.environ.get("DATA_ROOT", REPO / "data"))
PAIRS = {"drone1-drone2": "AV1-AV2", "drone2-gv": "AV2-GV", "drone1-gv": "AV1-GV"}
PLY = "point_cloud/iteration_30000/point_cloud.ply"


def perturbation(seed):
    out = subprocess.run([str(REPO / "build/print_perturbation"), str(seed), "180", "500"],
                         capture_output=True, text=True, check=True).stdout.split()
    T = np.eye(4)
    T[:3, :3] = np.array(out[:9], float).reshape(3, 3)
    T[:3, 3] = np.array(out[9:], float)
    return T


def restarts():
    poses = pd.read_csv(PD / "realdata_overlap_n100.csv")
    fit = pd.read_csv(PD / "realdata_overlap_n100_metrics.csv")
    df = poses.merge(fit, on=["method", "pair", "seed"])
    df = df[df.method == "color_gicp_de"]
    print("restarts converging to the maximum-fitness pose (within 10 deg / 5 m), 3DGS-ICP")
    for pair, g in df.groupby("pair"):
        E = {}
        for _, r in g.iterrows():
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
            T[:3, 3] = [r.tx, r.ty, r.tz]
            E[r.seed] = np.linalg.inv(perturbation(int(r.seed))) @ T
        best = g.loc[g["tgt2src_fit@0.5"].idxmax()]
        Eb_inv = np.linalg.inv(E[best.seed])
        n = 0
        for e in E.values():
            D = Eb_inv @ e
            ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
            n += ang < 10 and np.linalg.norm(D[:3, 3]) < 5
        print(f"  {PAIRS[pair]:8s} {n:3d}/{len(E)}  best restart: target-map fitness "
              f"{best['tgt2src_fit@0.5']:.2f}, source-map fitness {best['src2tgt_fit@0.5']:.2f}, "
              f"source inlier RMSE {best['src2tgt_rmse@0.5']:.2f} m")



if __name__ == "__main__":
    restarts()
