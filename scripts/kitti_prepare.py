#!/usr/bin/env python3
"""Build 3DGS training sets from one KITTI odometry sequence, in ONE metric frame.

Camera poses are the official KITTI GT poses (metric, loop-closed), so every
output shares the sequence's world frame and the GT transform between any two
maps built here is identity. COLMAP only triangulates the sparse points used to
initialise 3DGS. Both rectified colour cameras (image_02, image_03) are views.

Two modes:

  robots  (--robot NAME:FIRST-LAST, repeatable) -- multi-robot mapping. Each robot
          is a frame range of the trajectory and is reconstructed INDEPENDENTLY
          (own images, own feature database, own triangulation), as separate
          robots would. Overlap between robots should come from the vehicle
          driving through the same place twice, not from shared frames.

  halves  (default) -- one joint model cut into two halves by frame index with a
          shared overlap of --overlap / --wide_overlap of the frames
          (split_{neg,pos}, split_wide_{neg,pos}). Kept to reproduce the first
          seq-07 benchmark.

Frame numbers are odometry frames. --raw_offset maps them to raw-drive image
numbers (raw = odometry + offset); only sequence 08 needs it (1100).

  python3 scripts/kitti_prepare.py \
      --raw_dir  kitti/2011_09_30/2011_09_30_drive_0020_sync \
      --poses    kitti/dataset/poses/06.txt \
      --calib    kitti/dataset/sequences/06/calib.txt \
      --out      kitti/kitti06 --stride 2 \
      --robot robot_a:0-550 --robot robot_b:551-1100

Output: <out>/<NAME>/{images,database.db,sparse_known,sparse/0}; halves mode also
writes <out>/full/sparse_txt.
"""
import argparse
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

CAMS = (2, 3)  # rectified colour cameras


def read_calib(path):
    P = {}
    for line in open(path):
        key, *vals = line.split()
        if key.startswith("P"):
            P[int(key[1])] = np.array(vals, float).reshape(3, 4)
    return P


def read_poses(path):
    """KITTI odometry poses: one 3x4 T_world_cam0 per line."""
    out = []
    for line in open(path):
        T = np.eye(4)
        T[:3] = np.array(line.split(), float).reshape(3, 4)
        out.append(T)
    return out


def rotmat2qvec(R):
    """COLMAP convention (w, x, y, z)."""
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def image_names(frames):
    return [f"{fr:06d}_{c}.png" for fr in frames for c in CAMS]


def raw_image(raw_dir, name, raw_offset):
    frame, cam = name[:-4].split("_")
    return raw_dir / f"image_0{cam}" / "data" / f"{int(frame) + raw_offset:010d}.png"


def link_images(names, raw_dir, dst, raw_offset=0):
    # hard links, not symlinks: COLMAP stores a symlink's resolved target as the
    # image name, which breaks the <frame>_<cam>.png naming the splits rely on
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        link = dst / name
        if not link.exists():
            os.link(raw_image(raw_dir, name, raw_offset), link)


def write_known_model(db_path, poses, P, width, height, out_dir):
    """Posed, point-free model whose ids match the COLMAP database."""
    db = sqlite3.connect(db_path)
    images = db.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    frame_of = dict(db.execute(
        "SELECT data_id, frame_id FROM frame_data WHERE sensor_type = 0").fetchall())  # 0 = CAMERA
    db.close()
    assert {c for _, _, c in images} == {1}, "expected a single shared camera"

    K = P[2][:, :3]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")
    (out_dir / "rigs.txt").write_text("1 1 CAMERA 1\n")
    (out_dir / "points3D.txt").write_text("")

    img_lines, frame_lines = [], []
    for image_id, name, cam_id in sorted(images):
        frame, cam = map(int, name[:-4].split("_"))
        # rectified camera k sits at x_k = x_0 + t_k, with t_k = K^-1 P_k[:, 3]
        T_ck_c0 = np.eye(4)
        T_ck_c0[:3, 3] = np.linalg.solve(K, P[cam][:, 3])
        T_ck_w = T_ck_c0 @ np.linalg.inv(poses[frame])
        pose = " ".join(f"{v:.17g}" for v in (*rotmat2qvec(T_ck_w[:3, :3]), *T_ck_w[:3, 3]))
        img_lines.append(f"{image_id} {pose} {cam_id} {name}\n\n")
        frame_lines.append(f"{frame_of[image_id]} 1 {pose} 1 CAMERA 1 {image_id}\n")
    (out_dir / "images.txt").write_text("".join(img_lines))
    (out_dir / "frames.txt").write_text("".join(frame_lines))


def build_model(model_dir, names, raw_dir, poses, P, raw_offset=0, text_copy=False):
    """GT-posed COLMAP model of `names`: features, matches, triangulation."""
    from PIL import Image
    # size from the robot's own first frame: a window extraction may not include frame 0
    width, height = Image.open(raw_image(raw_dir, names[0], raw_offset)).size
    link_images(names, raw_dir, model_dir / "images", raw_offset)
    db = model_dir / "database.db"
    if db.exists():
        db.unlink()
    K = P[2]
    run(["colmap", "feature_extractor", "--database_path", db,
         "--image_path", model_dir / "images",
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
         "--ImageReader.camera_params", f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}"])
    run(["colmap", "sequential_matcher", "--database_path", db,
         "--SequentialMatching.overlap", "20"])
    write_known_model(db, poses, P, width, height, model_dir / "sparse_known")
    (model_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    run(["colmap", "point_triangulator", "--database_path", db,
         "--image_path", model_dir / "images", "--input_path", model_dir / "sparse_known",
         "--output_path", model_dir / "sparse" / "0",
         "--Mapper.ba_refine_focal_length", "0",
         "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.ba_refine_extra_params", "0"])
    if text_copy:
        (model_dir / "sparse_txt").mkdir(exist_ok=True)
        run(["colmap", "model_converter", "--input_path", model_dir / "sparse" / "0",
             "--output_path", model_dir / "sparse_txt", "--output_type", "TXT"])


def read_text_model(txt_dir):
    lines = (txt_dir / "images.txt").read_text().splitlines(keepends=True)
    body = [l for l in lines if not l.startswith("#")]
    images = {}  # name -> (id, line1, line2)
    for l1, l2 in zip(body[0::2], body[1::2]):
        parts = l1.split()
        images[parts[9]] = (int(parts[0]), l1, l2)
    return images


def write_half(full, images, names, raw_dir, dst, raw_offset=0):
    if dst.exists():
        shutil.rmtree(dst)
    link_images(names, raw_dir, dst / "images", raw_offset)
    sparse = dst / "sparse" / "0"
    sparse.mkdir(parents=True)
    keep = {images[n][0] for n in names}
    txt = full / "sparse_txt"
    for f in ("cameras.txt", "rigs.txt"):
        shutil.copy(txt / f, sparse / f)
    with open(sparse / "images.txt", "w") as f:
        for n in names:
            f.write(images[n][1] + images[n][2])
    with open(sparse / "frames.txt", "w") as f:
        for line in open(txt / "frames.txt"):
            if not line.startswith("#") and int(line.split()[-1]) in keep:
                f.write(line)
    # a half keeps only the points it observes at least twice itself
    n_pts = 0
    with open(sparse / "points3D.txt", "w") as f:
        for line in open(txt / "points3D.txt"):
            if line.startswith("#"):
                continue
            parts = line.split()
            track = [(parts[i], parts[i + 1]) for i in range(8, len(parts), 2)
                     if int(parts[i]) in keep]
            if len(track) >= 2:
                f.write(" ".join(parts[:8] + [x for t in track for x in t]) + "\n")
                n_pts += 1
    print(f"  {dst.name}: {len(names)} images, {n_pts} points")


def parse_robot(spec):
    name, rng = spec.split(":")
    first, last = map(int, rng.split("-"))
    return name, first, last


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2, help="use every n-th frame")
    ap.add_argument("--raw_offset", type=int, default=0,
                    help="raw image number = odometry frame + offset (seq 08: 1100)")
    ap.add_argument("--robot", action="append", type=parse_robot, default=[],
                    metavar="NAME:FIRST-LAST",
                    help="robots mode: an independently reconstructed frame range (repeatable)")
    ap.add_argument("--overlap", type=float, default=0.1,
                    help="halves mode: fraction of frames shared by split_neg/split_pos")
    ap.add_argument("--wide_overlap", type=float, default=0.3,
                    help="halves mode: fraction of frames shared by split_wide_neg/split_wide_pos")
    ap.add_argument("--skip_colmap", action="store_true",
                    help="halves mode: reuse an existing <out>/full/sparse_txt and only re-split")
    args = ap.parse_args()

    spans = sorted((first, last, name) for name, first, last in args.robot)
    for (_, last, a), (first, _, b) in zip(spans, spans[1:]):
        if first <= last:
            raise SystemExit(f"robots {a} and {b} share frames; overlap must come from revisits")

    raw_dir, out = args.raw_dir.resolve(), args.out.resolve()
    poses, P = read_poses(args.poses), read_calib(args.calib)

    if args.robot:
        for name, first, last in args.robot:
            frames = list(range(first, last + 1, args.stride))
            print(f"== {name}: frames {frames[0]}-{frames[-1]} ({len(frames)} frames, "
                  f"raw offset {args.raw_offset})", flush=True)
            build_model(out / name, image_names(frames), raw_dir, poses, P, args.raw_offset)
        return

    full = out / "full"
    frames = list(range(0, len(poses), args.stride))
    if not args.skip_colmap:
        build_model(full, image_names(frames), raw_dir, poses, P, args.raw_offset, text_copy=True)

    images = read_text_model(full / "sparse_txt")
    n = len(frames)
    for tag, overlap in (("split", args.overlap), ("split_wide", args.wide_overlap)):
        h = round(overlap * n / 2)
        neg, pos = frames[: n // 2 + h], frames[n // 2 - h:]
        print(f"{tag}: overlap {overlap:.0%} -> frames {neg[0]}-{neg[-1]} | {pos[0]}-{pos[-1]}")
        for side, fr in (("neg", neg), ("pos", pos)):
            write_half(full, images, image_names(fr), raw_dir, out / f"{tag}_{side}", args.raw_offset)


if __name__ == "__main__":
    main()
