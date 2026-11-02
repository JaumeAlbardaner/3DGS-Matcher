#!/usr/bin/env python3
"""Extra experiment 1 (2DGS): how well do the two KITTI 00/07 robot maps agree with each other at the GT pose?
Surface-to-surface NN distance on the shared road, and per-cell ground-height difference.
Shows 2DGS maps are mutually looser than 3DGS ones although each renders well."""
import numpy as np
import os
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree
D=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data")); PLY="point_cloud/iteration_30000/point_cloud.ply"
def load(p):
    v=PlyData.read(str(p))["vertex"]; return np.stack([v["x"],v["y"],v["z"]],1).astype(float)
for tag,base in [("3DGS",D/"trained_gaussians_filtered"),("2DGS",D/"trained_2dgs_filtered")]:
    s=load(base/"kitti00_07/robot_07"/PLY); t=load(base/"kitti00_07/robot_00"/PLY)
    # restrict to the shared road: source points with a target point within 2 m (GT = identity)
    tree=cKDTree(t); d,_=tree.query(s, distance_upper_bound=2.0); m=np.isfinite(d)
    print(f"{tag}: overlap pts {m.sum():,}/{len(s):,}  NN dist p50/p75/p90 (m) {np.percentile(d[m],[50,75,90]).round(3)}")
    # ground height in the overlap, per 2 m cell: 5th percentile of the vertical axis (KITTI cam frame: y down)
    so=s[m]; to_=t[tree.query(so)[1]]
    g={}
    for pts,name in ((so,"src"),(load(base/"kitti00_07/robot_00"/PLY),"tgt")):
        key=np.floor(pts[:,[0,2]]/2.0).astype(int)
        idx={}
        for k,y in zip(map(tuple,key),pts[:,1]): idx.setdefault(k,[]).append(y)
        g[name]={k:np.percentile(v,95) for k,v in idx.items() if len(v)>=20}  # y down => ground = high y
    common=set(g["src"])&set(g["tgt"])
    diff=np.array([g["src"][k]-g["tgt"][k] for k in common])
    print(f"      ground-height difference over {len(common)} cells: median {np.median(diff):+.3f} m  IQR {np.percentile(diff,75)-np.percentile(diff,25):.3f} m")
