#!/usr/bin/env python3
"""Compare GaussReg estimated transform against generated GT transform.

Inputs:
- metadata JSON from randomize_gaussian_pair.py
- estimated_transform.npz from demo.py

Outputs JSON metrics (stdout and optionally file).
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate estimated transform against GT.")
    parser.add_argument("--gt-json", required=True, type=Path, help="Metadata JSON from randomization step")
    parser.add_argument("--estimated-npz", required=True, type=Path, help="NPZ produced by GaussReg demo.py")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


def decompose_similarity(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    linear = transform[:3, :3]
    scale = float(np.sqrt(max((linear @ linear.T)[0, 0], 1e-12)))
    rotation = linear / scale

    # Project back to SO(3) for stable angle computation.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    translation = transform[:3, 3]
    return rotation, translation, scale


def rotation_error_deg(r_gt: np.ndarray, r_est: np.ndarray) -> float:
    x = 0.5 * (np.trace(r_est.T @ r_gt) - 1.0)
    x = float(np.clip(x, -1.0, 1.0))
    return float(np.degrees(np.arccos(x)))


def load_estimated_transform(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "estimated_transform" in data:
        est = data["estimated_transform"]
    elif len(data.files) == 1:
        est = data[data.files[0]]
    else:
        raise KeyError(f"Could not locate transform array in {npz_path}; keys={data.files}")
    est = np.asarray(est, dtype=np.float64)
    if est.shape != (4, 4):
        raise ValueError(f"Estimated transform must be 4x4, got shape {est.shape}")
    return est


def main():
    args = parse_args()

    if not args.gt_json.is_file():
        raise FileNotFoundError(f"GT metadata JSON not found: {args.gt_json}")
    if not args.estimated_npz.is_file():
        raise FileNotFoundError(f"Estimated transform NPZ not found: {args.estimated_npz}")

    meta = json.loads(args.gt_json.read_text(encoding="utf-8"))
    gt = np.asarray(meta["T_gt_src_to_ref"], dtype=np.float64)
    est = load_estimated_transform(args.estimated_npz)

    r_gt, t_gt, s_gt = decompose_similarity(gt)
    r_est, t_est, s_est = decompose_similarity(est)

    rot_err = rotation_error_deg(r_gt, r_est)
    trans_err_abs = float(np.linalg.norm(t_gt - t_est))
    trans_norm = float(np.linalg.norm(t_gt))
    trans_err_rel = float(trans_err_abs / max(trans_norm, 1e-12))

    scale_err_abs = float(abs(s_gt - s_est))
    scale_err_rel = float(scale_err_abs / max(abs(s_gt), 1e-12))

    result = {
        "rot_err_deg": rot_err,
        "rotation_error_deg": rot_err,
        "trans_err_abs": trans_err_abs,
        "translation_error_m": trans_err_abs,
        "trans_err_rel": trans_err_rel,
        "scale_gt": float(s_gt),
        "scale_est": float(s_est),
        "scale_err_abs": scale_err_abs,
        "scale_error_abs": scale_err_abs,
        "scale_err_rel": scale_err_rel,
        "scale_error_rel": scale_err_rel,
        "T_gt_src_to_ref": gt.tolist(),
        "T_est_src_to_ref": est.tolist(),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result))


if __name__ == "__main__":
    main()
