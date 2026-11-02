#!/usr/bin/env python3
"""Visualize the disruption cases that the tramway.splat benchmark runs.

Renders, for each case, a side-by-side "clean | perturbed" view of the tramway
splat so the effect of each disruption is visible against the original cloud.

Cases (matching examples/single_ablation.cc, noise level = 0.20):
    base          no disruption (reference cloud only)
    noise         baseline: position noise + per-point CIELAB/alpha noise
    outliers      noise + 30% of points get very strong xyz/colour/alpha outliers
    color_offset  global CIELAB tint shift + small per-point colour noise
    split_half    cloud cut along x into two overlapping halves (partial overlap)

For each case it writes:
    <out>/<case>.png        side-by-side still from a fixed viewpoint
    <out>/<case>.mp4        turntable orbit (clean and perturbed spin together)

The perturbation maths mirror single_ablation.cc. The random rigid transform
used in the benchmark is intentionally NOT applied here: both panels share the
same canonical pose so only the disruption itself differs between them.

Usage:
    python scripts/visualize_tramway_disruptions.py \
        [--splat examples/splats/tramway.splat] [--out results/tramway_disruptions] \
        [--noise 0.20] [--png-only|--video-only] [--frames 120] [--max-points N]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

try:
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except Exception:
    _FONT = ImageFont.load_default()


def add_caption(img, text, bar_h=48, bg=255, fg=(30, 30, 30)):
    """Prepend a caption bar with centred `text` above `img` (uint8 HxWx3)."""
    h, w = img.shape[:2]
    bar = Image.new("RGB", (w, bar_h), (bg, bg, bg))
    d = ImageDraw.Draw(bar)
    tb = d.textbbox((0, 0), text, font=_FONT)
    d.text(((w - (tb[2] - tb[0])) / 2, (bar_h - (tb[3] - tb[1])) / 2 - tb[1]),
           text, fill=fg, font=_FONT)
    return np.concatenate([np.asarray(bar, np.uint8), img], axis=0)

# ---------------------------------------------------------------------------
# .splat loader (antimatter15 32-byte record: pos[3]f, scale[3]f, rgba[4]u8,
# quat[4]u8). Colours are stored directly (not SH-encoded), so we read RGBA.
# ---------------------------------------------------------------------------
SPLAT_DTYPE = np.dtype([
    ("pos", "<f4", 3),
    ("scale", "<f4", 3),
    ("rgba", "u1", 4),
    ("quat", "u1", 4),
])


def load_splat(path):
    raw = np.fromfile(path, dtype=SPLAT_DTYPE)
    pts = raw["pos"].astype(np.float64)
    rgb = raw["rgba"][:, :3].astype(np.float64) / 255.0
    alpha = raw["rgba"][:, 3].astype(np.float64) / 255.0
    return pts, rgb, alpha


# ---------------------------------------------------------------------------
# sRGB <-> CIELAB (D65), vectorised. Disruptions in the C++ code happen in Lab.
# ---------------------------------------------------------------------------
_M_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
_M_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    c = np.clip(rgb, 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T
    t = xyz / _WHITE
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def lab_to_rgb(lab):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = np.stack([fx, fy, fz], axis=1)
    t3 = f ** 3
    t = np.where(t3 > 0.008856, t3, (f - 16.0 / 116.0) / 7.787)
    xyz = t * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 0) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Perturbations (mirror examples/single_ablation.cc)
# ---------------------------------------------------------------------------
def add_position_noise(pts, std, rng):
    if std <= 0:
        return pts
    return pts + rng.normal(0.0, std, size=pts.shape)


def add_color_noise(rgb, noise, rng):
    # CIELAB std = noise*10 (alpha handled separately / ignored for rendering).
    if noise <= 0:
        return rgb
    lab = rgb_to_lab(rgb)
    lab = lab + rng.normal(0.0, noise * 10.0, size=lab.shape)
    lab[:, 0] = np.clip(lab[:, 0], 0.0, 100.0)
    return lab_to_rgb(lab)


def add_outliers(pts, rgb, ratio, xyz_std, lab_std, rng):
    n = pts.shape[0]
    k = max(1, int(round(ratio * n)))
    idx = rng.integers(0, n, size=k)
    pts = pts.copy()
    pts[idx] += rng.normal(0.0, xyz_std, size=(k, 3))
    lab = rgb_to_lab(rgb)
    lab[idx] += rng.normal(0.0, lab_std, size=(k, 3))
    lab[:, 0] = np.clip(lab[:, 0], 0.0, 100.0)
    return pts, lab_to_rgb(lab)


def apply_global_color_offset(rgb, offset_std, per_point_std, rng):
    lab = rgb_to_lab(rgb)
    offset = rng.normal(0.0, offset_std, size=3)
    lab = lab + offset + rng.normal(0.0, per_point_std, size=lab.shape)
    lab[:, 0] = np.clip(lab[:, 0], 0.0, 100.0)
    return lab_to_rgb(lab)


def split_half(pts, rgb, axis=0, overlap=0.25):
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    span = max(1e-6, hi - lo)
    mid = 0.5 * (lo + hi)
    half_ov = 0.5 * overlap * span
    src_mask = pts[:, axis] <= mid + half_ov
    tgt_mask = pts[:, axis] >= mid - half_ov
    return (pts[src_mask], rgb[src_mask]), (pts[tgt_mask], rgb[tgt_mask])


# ---------------------------------------------------------------------------
# Rendering (legacy Open3D Visualizer, offscreen via visible=False)
# ---------------------------------------------------------------------------
def make_pcd(pts, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    return pcd


class Renderer:
    def __init__(self, w, h, point_size, bg, anchor_pts=None):
        self.w, self.h = w, h
        self.bg = np.asarray(bg)
        # Fixed framing reference: 8 bbox corners rendered in the background
        # colour (invisible) so every panel — including ones whose own cloud is
        # blown up by outliers — shares the exact same camera framing.
        self.anchor = None
        if anchor_pts is not None:
            self.anchor = make_pcd(anchor_pts, np.tile(self.bg, (len(anchor_pts), 1)))
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=w, height=h, visible=False)
        opt = self.vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = self.bg
        opt.light_on = False

    def render(self, pcd, front, up, lookat, zoom):
        self.vis.clear_geometries()
        if self.anchor is not None:
            self.vis.add_geometry(self.anchor, reset_bounding_box=True)
            self.vis.add_geometry(pcd, reset_bounding_box=False)
        else:
            self.vis.add_geometry(pcd, reset_bounding_box=True)
        ctr = self.vis.get_view_control()
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_front(front)
        ctr.set_zoom(zoom)
        self.vis.poll_events()
        self.vis.update_renderer()
        buf = self.vis.capture_screen_float_buffer(do_render=True)
        return (np.asarray(buf) * 255.0 + 0.5).astype(np.uint8)

    def close(self):
        self.vis.destroy_window()


def rotate_about_axis(vec, axis, angle):
    """Rodrigues rotation of `vec` about unit `axis` by `angle` radians."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    return (vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1 - c))


def hstack_pad(imgs, gap=16, bg=255):
    h = max(im.shape[0] for im in imgs)
    sep = np.full((h, gap, 3), bg, np.uint8)
    parts = []
    for i, im in enumerate(imgs):
        if i:
            parts.append(sep)
        parts.append(im)
    out = np.concatenate(parts, axis=1)
    # ffmpeg needs even dimensions
    if out.shape[1] % 2:
        out = np.concatenate([out, np.full((out.shape[0], 1, 3), bg, np.uint8)], axis=1)
    if out.shape[0] % 2:
        out = np.concatenate([out, np.full((1, out.shape[1], 3), bg, np.uint8)], axis=0)
    return out


def write_png(path, img):
    o3d.io.write_image(path, o3d.geometry.Image(np.ascontiguousarray(img)))


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
           "-framerate", str(fps), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


# ---------------------------------------------------------------------------
def build_cases(pts, rgb, noise, rng):
    """Return ordered list of (name, [(label, pts, rgb), ...]) panels."""
    cases = []

    # base: reference only
    cases.append(("base", [("clean", pts, rgb)]))

    # noise (baseline at the given level)
    npts = add_position_noise(pts, noise, rng)
    nrgb = add_color_noise(rgb, noise, rng)
    cases.append(("noise", [("clean", pts, rgb), (f"noise std={noise:g}", npts, nrgb)]))

    # outliers: color noise then 30% strong outliers
    opts = add_position_noise(pts, noise, rng)
    orgb = add_color_noise(rgb, noise, rng)
    opts, orgb = add_outliers(opts, orgb, 0.30,
                              max(1e-3, noise * 50.0),
                              max(1e-2, noise * 80.0), rng)
    cases.append(("outliers", [("clean", pts, rgb), ("outliers", opts, orgb)]))

    # color_offset: global tint + per-point
    cpts = add_position_noise(pts, noise, rng)
    crgb = apply_global_color_offset(rgb, max(1e-3, noise * 6.0),
                                     max(1e-4, noise * 2.0), rng)
    cases.append(("color_offset", [("clean", pts, rgb), ("color offset", cpts, crgb)]))

    # split_half: two overlapping halves, tinted to show partial overlap
    (sp, sc), (tp, tc) = split_half(pts, rgb, axis=0, overlap=0.25)
    src_tint = np.tile([0.95, 0.55, 0.15], (sp.shape[0], 1))   # orange
    tgt_tint = np.tile([0.20, 0.55, 0.95], (tp.shape[0], 1))   # blue
    half_pts = np.vstack([sp, tp])
    half_rgb = np.vstack([src_tint, tgt_tint])
    cases.append(("split_half", [("clean", pts, rgb),
                                 ("split halves (overlap=0.25)", half_pts, half_rgb)]))
    return cases


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat", default=os.path.join(here, "examples/splats/tramway.splat"))
    ap.add_argument("--out", default=os.path.join(here, "results/tramway_disruptions"))
    ap.add_argument("--noise", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--panel", type=int, default=900, help="per-panel pixel size")
    ap.add_argument("--point-size", type=float, default=1.5)
    ap.add_argument("--frames", type=int, default=120, help="orbit frames per video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zoom", type=float, default=0.62)
    ap.add_argument("--max-points", type=int, default=0,
                    help="random subsample to this many points (0 = keep all)")
    ap.add_argument("--front", type=float, nargs=3, default=[0.0, 0.0, -1.0])
    ap.add_argument("--up", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    ap.add_argument("--png-only", action="store_true")
    ap.add_argument("--video-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.splat} ...")
    pts, rgb, _alpha = load_splat(args.splat)
    print(f"  {len(pts):,} splats")
    if args.max_points and len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, rgb = pts[sel], rgb[sel]
        print(f"  subsampled to {len(pts):,}")

    rng = np.random.default_rng(args.seed)
    cases = build_cases(pts, rgb, args.noise, rng)

    center = pts.mean(axis=0)
    front = np.asarray(args.front, float)
    up = np.asarray(args.up, float)
    bg = (1.0, 1.0, 1.0)

    # 8 corners of the clean cloud's bbox = fixed framing reference for all panels.
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    anchor = np.array([[x, y, z] for x in (lo[0], hi[0])
                       for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)

    r = Renderer(args.panel, args.panel, args.point_size, bg, anchor_pts=anchor)
    try:
        for name, panels in cases:
            pcds = [(label, make_pcd(p, c)) for (label, p, c) in panels]

            if not args.video_only:
                imgs = [add_caption(r.render(pcd, front, up, center, args.zoom), label)
                        for label, pcd in pcds]
                png = os.path.join(args.out, f"{name}.png")
                write_png(png, hstack_pad(imgs))
                print(f"  wrote {png}")

            if not args.png_only:
                frames = []
                for k in range(args.frames):
                    ang = 2 * np.pi * k / args.frames
                    f_k = rotate_about_axis(front, up, ang)
                    imgs = [add_caption(r.render(pcd, f_k, up, center, args.zoom), label)
                            for label, pcd in pcds]
                    frames.append(hstack_pad(imgs))
                mp4 = os.path.join(args.out, f"{name}.mp4")
                write_mp4(mp4, frames, args.fps)
                print(f"  wrote {mp4}")
    finally:
        r.close()

    print("Done.")


if __name__ == "__main__":
    main()
