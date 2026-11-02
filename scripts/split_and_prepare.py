import os
import numpy as np
import shutil
import argparse

# -------------------------
# Math utils
# -------------------------

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# -------------------------
# Parse COLMAP
# -------------------------

def read_images_txt(path):
    images = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or len(line.strip()) == 0:
            i += 1
            continue

        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        name = parts[9]

        R = qvec2rotmat(qvec)
        C = -R.T @ tvec

        images.append({
            "id": image_id,
            "name": name,
            "qvec": qvec,
            "tvec": tvec,
            "center": C,
            "line1": line,
            "line2": lines[i+1]
        })

        i += 2

    return images

def read_points3D_txt(path):
    with open(path) as f:
        return f.readlines()

def read_cameras_txt(path):
    with open(path) as f:
        return f.readlines()

def read_optional_txt(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.readlines()

# -------------------------
# Copy images preserving structure
# -------------------------

def copy_images(src_root, dst_root, selected):
    for root, _, files in os.walk(os.path.join(src_root, "images")):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f),
                                  os.path.join(src_root, "images"))

            if rel in selected:
                src = os.path.join(root, f)
                dst = os.path.join(dst_root, "images", rel)

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

# -------------------------
# Filter sparse model
# -------------------------

def filter_sparse(images, points3D_lines, cameras_lines, rigs_lines, frames_lines, selected_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    selected_images = [im for im in images if im["name"] in selected_names]
    selected_ids = {im["id"] for im in selected_images}
    image_id_map = {
        im["id"]: new_id
        for new_id, im in enumerate(selected_images, start=1)
    }

    # -------------------------
    # images.txt
    # -------------------------
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        for im in selected_images:
            parts = im["line1"].split()
            parts[0] = str(image_id_map[im["id"]])
            f.write(" ".join(parts) + "\n")
            f.write(im["line2"])

    # -------------------------
    # cameras.txt
    # -------------------------
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.writelines(cameras_lines)

    # -------------------------
    # rigs.txt
    # -------------------------
    if rigs_lines is not None:
        with open(os.path.join(out_dir, "rigs.txt"), "w") as f:
            f.writelines(rigs_lines)

    # -------------------------
    # frames.txt
    # -------------------------
    if frames_lines is not None:
        with open(os.path.join(out_dir, "frames.txt"), "w") as f:
            next_frame_id = 1
            for line in frames_lines:
                if line.startswith("#") or len(line.strip()) == 0:
                    f.write(line)
                    continue

                parts = line.strip().split()
                num_data_ids = int(parts[9])
                data_tokens = parts[10:]

                kept_data = []
                for i in range(num_data_ids):
                    offset = 3 * i
                    sensor_type = data_tokens[offset]
                    sensor_id = data_tokens[offset + 1]
                    data_id = int(data_tokens[offset + 2])

                    if data_id in selected_ids:
                        kept_data.extend([sensor_type, sensor_id, str(image_id_map[data_id])])

                if kept_data:
                    header = [str(next_frame_id)] + parts[1:9] + [str(len(kept_data) // 3)]
                    f.write(" ".join(header + kept_data) + "\n")
                    next_frame_id += 1

    # -------------------------
    # points3D.txt
    # -------------------------
    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        for line in points3D_lines:
            if line.startswith("#"):
                f.write(line)
                continue

            parts = line.strip().split()

            # Base fields
            header = parts[:8]
            track = parts[8:]

            new_track = []

            # Keep ONLY valid observations
            for i in range(0, len(track), 2):
                img_id = int(track[i])
                pt2d_id = track[i+1]

                if img_id in selected_ids:
                    new_track.extend([str(image_id_map[img_id]), pt2d_id])

            # Keep point ONLY if >= 2 observations
            if len(new_track) >= 4:
                f.write(" ".join(header + new_track) + "\n")

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", required=True)
    parser.add_argument("--sparse_txt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split_axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--split_value", type=float, default=0.0)
    parser.add_argument("--wide_margin_ratio", type=float, default=0.1)
    args = parser.parse_args()

    images = read_images_txt(os.path.join(args.sparse_txt, "images.txt"))
    points = read_points3D_txt(os.path.join(args.sparse_txt, "points3D.txt"))
    cameras = read_cameras_txt(os.path.join(args.sparse_txt, "cameras.txt"))
    rigs = read_optional_txt(os.path.join(args.sparse_txt, "rigs.txt"))
    frames = read_optional_txt(os.path.join(args.sparse_txt, "frames.txt"))

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_to_index[args.split_axis]
    coords = np.array([im["center"][axis_idx] for im in images])

    cmin, cmax = coords.min(), coords.max()
    if args.wide_margin_ratio < 0.0 or args.wide_margin_ratio > 1.0:
        raise ValueError("--wide_margin_ratio must be between 0.0 and 1.0")

    # Expand each half around split_value by a fraction of the opposite-side span.
    left_span = max(0.0, args.split_value - cmin)
    right_span = max(0.0, cmax - args.split_value)
    wide_pos_min = args.split_value - args.wide_margin_ratio * left_span
    wide_neg_max = args.split_value + args.wide_margin_ratio * right_span

    print(f"{args.split_axis.upper()} range: {cmin:.3f} → {cmax:.3f}")
    print(f"Split plane: {args.split_axis} = {args.split_value:.3f}")
    print(f"Wide margin ratio: {args.wide_margin_ratio:.3f}")
    print(
        f"Wide thresholds: wide_pos >= {wide_pos_min:.3f}, "
        f"wide_neg <= {wide_neg_max:.3f}"
    )

    splits = {
        "pos": [],
        "neg": [],
        "wide_pos": [],
        "wide_neg": []
    }

    for im in images:
        coord = im["center"][axis_idx]
        name = im["name"]

        if coord >= args.split_value:
            splits["pos"].append(name)
        if coord <= args.split_value:
            splits["neg"].append(name)
        if coord >= wide_pos_min:
            splits["wide_pos"].append(name)
        if coord <= wide_neg_max:
            splits["wide_neg"].append(name)

    for split, names in splits.items():
        print(f"{split}: {len(names)} images")

        split_dir = os.path.join(args.output_root, f"split_{split}")
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)

        copy_images(args.project_path, split_dir, set(names))

        filter_sparse(
            images, points, cameras, rigs, frames,
            set(names),
            os.path.join(split_dir, "sparse_filtered")
        )

if __name__ == "__main__":
    main()
