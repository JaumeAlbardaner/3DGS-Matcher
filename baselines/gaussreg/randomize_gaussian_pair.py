#!/usr/bin/env python3
"""Apply independent random rigid offsets to a ref/src Gaussian pair.

This script preserves Gaussian structure fields in point_cloud.ply and updates:
- xyz
- per-Gaussian quaternion rotation
- SH coefficients (optional, enabled by default)

It writes transformed PLYs plus a JSON metadata file with the generated transforms
and the ground-truth relative transform from transformed src to transformed ref.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gs_fusion import (  # noqa: E402
    load_ply,
    save_ply,
    sh_rotation,
    quaternion_to_matrix,
    matrix_to_quaternion,
)


def euler_xyz_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build rotation matrix from intrinsic XYZ euler angles (degrees)."""
    rx = np.deg2rad(roll_deg)
    ry = np.deg2rad(pitch_deg)
    rz = np.deg2rad(yaw_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    ry_m = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rz_m = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Same convention used in multiple scripts in this repo (Rz @ Ry @ Rx).
    return rz_m @ ry_m @ rx_m


def sample_random_transform(rng: np.random.Generator, max_translation: float, max_rotation_deg: float):
    euler_deg = rng.uniform(-max_rotation_deg, max_rotation_deg, size=3)
    translation = rng.uniform(-max_translation, max_translation, size=3)
    rotation = euler_xyz_to_matrix(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2]))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, rotation, translation, euler_deg


def apply_transform_to_gaussian_ply(
    input_ply: Path,
    output_ply: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    rotate_sh: bool,
    xyz_only: bool,
):
    xyz, features_dc, features_extra, opacities, scales, rots = load_ply(str(input_ply))

    xyz_t = xyz @ rotation.T + translation[None, :]

    if xyz_only:
        # Fast mode for GaussReg demo: only xyz is used for geometry, and
        # features are read directly from stored coefficients.
        rots_t = rots
        features_extra_t = features_extra
    else:
        rot_matrix_src = quaternion_to_matrix(torch.tensor(rots, dtype=torch.float32))
        rot_matrix_t = torch.matmul(torch.tensor(rotation[None, :, :], dtype=torch.float32), rot_matrix_src)
        rots_t = matrix_to_quaternion(rot_matrix_t).cpu().numpy()

        if rotate_sh:
            features_extra_t = sh_rotation(features_extra, features_dc, rotation)
        else:
            features_extra_t = features_extra

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    save_ply(
        xyz_t,
        features_dc.reshape(xyz.shape[0], -1),
        features_extra_t.reshape(xyz.shape[0], -1),
        opacities,
        scales,
        rots_t,
        str(output_ply),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Randomize one ref/src Gaussian pair with independent rigid transforms.")
    parser.add_argument("--ref-in", required=True, type=Path, help="Input reference point_cloud.ply")
    parser.add_argument("--src-in", required=True, type=Path, help="Input source point_cloud.ply")
    parser.add_argument("--ref-out", required=True, type=Path, help="Output transformed reference point_cloud.ply")
    parser.add_argument("--src-out", required=True, type=Path, help="Output transformed source point_cloud.ply")
    parser.add_argument("--meta-out", required=True, type=Path, help="Output JSON with generated transforms and GT")
    parser.add_argument("--seed", required=True, type=int, help="Random seed")
    parser.add_argument("--max-translation", type=float, default=500.0, help="Max abs translation per axis")
    parser.add_argument("--max-rotation-deg", type=float, default=180.0, help="Max abs euler angle (deg) per axis")
    parser.add_argument(
        "--xyz-only",
        action="store_true",
        help="Fast mode: transform xyz only, keep all other Gaussian fields unchanged",
    )
    parser.add_argument(
        "--rotate-sh",
        dest="rotate_sh",
        action="store_true",
        help="Rotate SH coefficients consistently with rigid transform (default)",
    )
    parser.add_argument(
        "--no-rotate-sh",
        dest="rotate_sh",
        action="store_false",
        help="Do not rotate SH coefficients",
    )
    parser.set_defaults(rotate_sh=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.ref_in.is_file():
        raise FileNotFoundError(f"ref input not found: {args.ref_in}")
    if not args.src_in.is_file():
        raise FileNotFoundError(f"src input not found: {args.src_in}")

    rng = np.random.default_rng(args.seed)

    t_ref, r_ref, trans_ref, euler_ref = sample_random_transform(
        rng,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
    )
    t_src, r_src, trans_src, euler_src = sample_random_transform(
        rng,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
    )

    apply_transform_to_gaussian_ply(
        input_ply=args.ref_in,
        output_ply=args.ref_out,
        rotation=r_ref,
        translation=trans_ref,
        rotate_sh=args.rotate_sh,
        xyz_only=args.xyz_only,
    )
    apply_transform_to_gaussian_ply(
        input_ply=args.src_in,
        output_ply=args.src_out,
        rotation=r_src,
        translation=trans_src,
        rotate_sh=args.rotate_sh,
        xyz_only=args.xyz_only,
    )

    # If ref/src start in same original coordinate system, this is the exact GT.
    t_gt_src_to_ref = t_ref @ np.linalg.inv(t_src)

    meta = {
        "seed": int(args.seed),
        "max_translation": float(args.max_translation),
        "max_rotation_deg": float(args.max_rotation_deg),
        "rotate_sh": bool(args.rotate_sh),
        "xyz_only": bool(args.xyz_only),
        "ref": {
            "input_ply": str(args.ref_in.resolve()),
            "output_ply": str(args.ref_out.resolve()),
            "euler_deg_xyz": [float(x) for x in euler_ref],
            "translation_xyz": [float(x) for x in trans_ref],
            "transform": t_ref.tolist(),
        },
        "src": {
            "input_ply": str(args.src_in.resolve()),
            "output_ply": str(args.src_out.resolve()),
            "euler_deg_xyz": [float(x) for x in euler_src],
            "translation_xyz": [float(x) for x in trans_src],
            "transform": t_src.tolist(),
        },
        "T_gt_src_to_ref": t_gt_src_to_ref.tolist(),
    }

    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps({
        "ref_out": str(args.ref_out.resolve()),
        "src_out": str(args.src_out.resolve()),
        "meta_out": str(args.meta_out.resolve()),
    }))


if __name__ == "__main__":
    main()
