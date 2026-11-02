#!/usr/bin/env python3
"""Scale a 3DGS ply to metric, preserving every property.

Two things must change together, which is the part that is easy to get wrong:
  positions   xyz      *= s
  gaussian sizes scale_0/1/2 += log(s)     (3DGS stores them as logs)
Colours (f_dc/f_rest), opacity and rotations are scale-invariant and stay put.
Everything else in the file is copied through byte-for-byte.

Either give the factor directly, or measure it off something of known length in
the scene (a tape, a metre stick) by clicking its two ends:

  # known factor
  python3 scripts/scale_splat.py in.ply --scale 2.104705 --out out.ply

  # derive it: click the two ends of a 1 m tape
  python3 scripts/scale_splat.py in.ply --measure 1.0 --out out.ply

  # just measure a distance, change nothing
  python3 scripts/scale_splat.py in.ply --ruler
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PLY_T = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
         "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
         "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
         "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1"}
MAX_DISPLAY = 600_000


def read_ply(path):
    """-> (structured array, raw header bytes). Binary little-endian only."""
    with open(path, "rb") as f:
        header, props, n = b"", [], 0
        while True:
            line = f.readline()
            header += line
            s = line.split()
            if line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                n = int(s[-1])
            elif line.startswith(b"property"):
                if s[1] == b"list":
                    raise SystemExit("list properties are not supported")
                props.append((s[-1].decode(), PLY_T[s[1].decode()]))
        if b"binary_little_endian" not in header:
            raise SystemExit("only binary_little_endian plys are supported")
        return np.fromfile(f, dtype=np.dtype(props), count=n), header


def pick_two(xyz, rgb):
    """Click two points; return their world coordinates."""
    import open3d as o3d
    disp = (np.random.default_rng(0).choice(len(xyz), MAX_DISPLAY, replace=False)
            if len(xyz) > MAX_DISPLAY else np.arange(len(xyz)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[disp])
    pcd.colors = o3d.utility.Vector3dVector(rgb[disp])
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="click the TWO ENDS of the known length (shift+click), q when done",
                      width=1280, height=860)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    print("  shift+left-click each end, shift+right-click undoes, q when done")
    vis.run()
    vis.destroy_window()
    p = list(vis.get_picked_points())
    if len(p) < 2:
        raise SystemExit(f"need 2 points, got {len(p)}")
    if len(p) > 2:
        print(f"  {len(p)} points picked -- using the last two")
    return xyz[disp][p[-2]], xyz[disp][p[-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--scale", type=float, help="multiply positions by this")
    ap.add_argument("--measure", type=float, metavar="TRUE_LENGTH_M",
                    help="click two points, derive the factor from their true separation")
    ap.add_argument("--ruler", action="store_true", help="measure a distance and exit")
    ap.add_argument("--out", help="output ply")
    args = ap.parse_args()

    arr, header = read_ply(args.input)
    names = arr.dtype.names
    for need in ("x", "y", "z"):
        if need not in names:
            raise SystemExit(f"ply has no '{need}' property")
    print(f"{Path(args.input).name}: {len(arr):,} points, {len(names)} properties")

    if args.ruler or args.measure:
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
        if "f_dc_0" in names:
            rgb = np.clip(np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], 1)
                          * 0.28209479177387814 + 0.5, 0, 1).astype(np.float64)
        else:
            rgb = np.full((len(xyz), 3), 0.6)
        a, b = pick_two(xyz, rgb)
        d = float(np.linalg.norm(a - b))
        print(f"  measured separation: {d:.6f} file units")
        if args.ruler:
            print("  (--ruler: nothing changed)")
            return 0
        scale = args.measure / d
        print(f"  true length {args.measure} m -> scale factor {scale:.6f}")
    elif args.scale:
        scale = args.scale
    else:
        ap.error("give --scale, --measure, or --ruler")

    if not args.out:
        print(f"\nfactor is {scale:.6f}; rerun with --out to write")
        return 0

    for k in ("x", "y", "z"):
        arr[k] = (arr[k].astype(np.float64) * scale).astype(arr[k].dtype)
    ln = np.log(scale)
    sc = [k for k in names if k.startswith("scale_")]
    for k in sc:
        arr[k] = (arr[k].astype(np.float64) + ln).astype(arr[k].dtype)
    print(f"  xyz *= {scale:.6f}   {', '.join(sc) if sc else '(no scale_* properties)'}"
          f"{f' += log(s) = {ln:+.6f}' if sc else ''}")
    if not sc:
        print("  WARNING: no scale_* properties found -- gaussian sizes could not be "
              "updated. Fine for point-cloud matching, wrong for rendering.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(header)
        arr.tofile(f)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
