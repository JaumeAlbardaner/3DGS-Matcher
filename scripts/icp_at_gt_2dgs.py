#!/usr/bin/env python3
"""Extra experiment 1 (2DGS) check: point-to-plane ICP between KITTI 00/07 robot maps started AT the GT pose (identity).
The correction ICP applies, measured as ATE over robot_07 GT cameras, is how far the two maps disagree.
Run with the 2dgs env python (needs open3d + plyfile)."""
import json, numpy as np, open3d as o3d
import os
from pathlib import Path
D=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data")); PLY="point_cloud/iteration_30000/point_cloud.ply"
def load(p):
    from plyfile import PlyData
    v=PlyData.read(str(p))["vertex"]; return np.stack([v["x"],v["y"],v["z"]],1).astype(float)
for tag,base,cams in [("3DGS filt","trained_gaussians_filtered","trained_gaussians"),("2DGS filt","trained_2dgs_filtered","trained_2dgs"),("2DGS unfilt","trained_2dgs","trained_2dgs")]:
    s=load(D/base/"kitti00_07/robot_07"/PLY); t=load(D/base/"kitti00_07/robot_00"/PLY)
    ps=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s)).voxel_down_sample(0.2)
    pt=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t)).voxel_down_sample(0.2)
    pt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0,max_nn=30))
    cam=np.array([c["position"] for c in json.load(open(D/cams/"kitti00_07/robot_07/cameras.json"))])
    for thr in (1.0,0.5):
        r=o3d.pipelines.registration.registration_icp(ps,pt,thr,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T=r.transformation; disp=cam@T[:3,:3].T+T[:3,3]-cam
        ang=np.degrees(np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1)))
        print(f"{tag:12s} thr{thr}: fitness {r.fitness:.2f} rmse {r.inlier_rmse:.3f}  correction ATE {np.sqrt((disp**2).sum(1).mean()):.2f} m  rot {ang:.2f} deg  mean disp {disp.mean(0).round(2)}")
