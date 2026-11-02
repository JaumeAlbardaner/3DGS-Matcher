#!/usr/bin/env python3
"""
Batch filter 3DGS PLY files.

For every `.ply` under an input directory, this script:
1) removes points below an opacity threshold
2) removes statistical outliers
3) writes the filtered cloud to an output directory with mirrored structure

The output path is always:
    <output_dir>/<relative path from input_dir to source .ply>

This mirrors layouts like:
    scene/split_*/point_cloud/iteration_30000/point_cloud.ply
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. Install with: pip install open3d")
    sys.exit(1)

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("ERROR: plyfile not installed. Install with: pip install plyfile")
    sys.exit(1)


@dataclass
class FilterStats:
    input_points: int
    after_opacity: int
    after_statistical: int
    removed_by_opacity: int
    removed_by_statistical: int
    wrote_file: bool
    skipped_statistical: bool


def sigmoid(x: np.ndarray) -> np.ndarray:
    # 3DGS stores opacity logits in many pipelines; clamp for numerical stability.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply opacity + statistical filtering to all PLY files in a folder tree."
    )
    parser.add_argument("input_dir", type=Path, help="Root folder to scan recursively for .ply files.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root folder where filtered files are written with mirrored relative paths.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ply",
        help="Glob pattern used with recursive search (default: *.ply).",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.01,
        help="Opacity threshold (default: 0.01). Interpreted by --opacity-space.",
    )
    parser.add_argument(
        "--opacity-space",
        choices=["sigmoid", "raw"],
        default="sigmoid",
        help=(
            "Interpret opacity values as sigmoid(alpha) or raw stored values. "
            "For 3DGS point_cloud.ply this is usually 'sigmoid'."
        ),
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="Neighbors used by statistical outlier removal (default: 20).",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Std-dev ratio used by statistical outlier removal (default: 2.0).",
    )
    parser.add_argument(
        "--skip-statistical",
        action="store_true",
        help="Skip statistical outlier removal and apply only opacity filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining files if one file fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating output files.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open an interactive before/after preview window for each processed file.",
    )
    parser.add_argument(
        "--visualize-max-points",
        type=int,
        default=250000,
        help="Maximum total points shown per visualization window (default: 250000, <=0 means all).",
    )
    parser.add_argument(
        "--visualize-point-size",
        type=float,
        default=1.5,
        help="Point size in visualization windows (default: 1.5).",
    )
    parser.add_argument(
        "--visualize-seed",
        type=int,
        default=0,
        help="Random seed for visualization subsampling (default: 0).",
    )
    return parser.parse_args()


def find_ply_files(input_dir: Path, pattern: str) -> list[Path]:
    files = [p for p in input_dir.rglob(pattern) if p.is_file() and p.suffix.lower() == ".ply"]
    return sorted(files)


def build_opacity_mask(vertex_data: np.ndarray, threshold: float, opacity_space: str) -> np.ndarray:
    names = set(vertex_data.dtype.names or ())
    if "opacity" not in names:
        raise RuntimeError("PLY vertex element has no 'opacity' property.")

    opacity = vertex_data["opacity"].astype(np.float32, copy=False)
    if opacity_space == "sigmoid":
        alpha = sigmoid(opacity)
    else:
        alpha = opacity
    return alpha >= float(threshold)


def run_statistical_filter(
    vertex_data: np.ndarray,
    candidate_indices: np.ndarray,
    nb_neighbors: int,
    std_ratio: float,
) -> tuple[np.ndarray, bool]:
    if candidate_indices.size == 0:
        return candidate_indices, True

    if candidate_indices.size <= nb_neighbors:
        return candidate_indices, True

    points = np.column_stack(
        [
            vertex_data["x"][candidate_indices],
            vertex_data["y"][candidate_indices],
            vertex_data["z"][candidate_indices],
        ]
    ).astype(np.float64, copy=False)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _, inlier_rel = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    inlier_rel = np.asarray(inlier_rel, dtype=np.int64)
    return candidate_indices[inlier_rel], False


def allocate_sample_budgets(counts: list[int], max_total: int) -> list[int]:
    if max_total <= 0:
        return counts

    total = int(sum(counts))
    if total <= max_total:
        return counts

    counts_arr = np.asarray(counts, dtype=np.int64)
    budgets = np.zeros_like(counts_arr)

    positive = np.flatnonzero(counts_arr > 0)
    baseline = min(int(positive.size), int(max_total))
    if baseline > 0:
        order = positive[np.argsort(counts_arr[positive])[::-1]]
        budgets[order[:baseline]] = 1

    remaining = int(max_total - int(budgets.sum()))
    if remaining <= 0:
        return budgets.tolist()

    capacity = counts_arr - budgets
    capacity_total = int(capacity.sum())
    if capacity_total <= 0:
        return budgets.tolist()

    raw_add = remaining * (capacity.astype(np.float64) / float(capacity_total))
    add_floor = np.floor(raw_add).astype(np.int64)
    add_floor = np.minimum(add_floor, capacity)
    budgets += add_floor

    remaining = int(max_total - int(budgets.sum()))
    if remaining > 0:
        fractional = raw_add - add_floor.astype(np.float64)
        order = np.argsort(-fractional)
        for idx in order:
            if remaining <= 0:
                break
            if budgets[idx] < counts_arr[idx]:
                budgets[idx] += 1
                remaining -= 1

    return budgets.tolist()


def sample_indices_from_mask(mask: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask).astype(np.int64, copy=False)
    if idx.size == 0 or budget <= 0:
        return np.empty((0,), dtype=np.int64)
    if idx.size <= budget:
        return idx
    return rng.choice(idx, size=budget, replace=False).astype(np.int64, copy=False)


def points_from_indices(vertex_data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(
        [vertex_data["x"][indices], vertex_data["y"][indices], vertex_data["z"][indices]]
    ).astype(np.float64, copy=False)


def build_colored_cloud(points: np.ndarray, color: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(color[None, :], (points.shape[0], 1)).astype(np.float64, copy=False)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def visualize_filter_result(
    source_file: Path,
    vertex_data: np.ndarray,
    keep_mask: np.ndarray,
    opacity_mask: np.ndarray,
    max_points: int,
    point_size: float,
    seed: int,
) -> None:
    removed_opacity_mask = ~opacity_mask
    removed_statistical_mask = opacity_mask & ~keep_mask

    counts = [
        int(keep_mask.sum()),
        int(removed_opacity_mask.sum()),
        int(removed_statistical_mask.sum()),
    ]
    budgets = allocate_sample_budgets(counts, max_points)

    rng = np.random.default_rng(seed)

    keep_idx = sample_indices_from_mask(keep_mask, budgets[0], rng)
    opacity_idx = sample_indices_from_mask(removed_opacity_mask, budgets[1], rng)
    stat_idx = sample_indices_from_mask(removed_statistical_mask, budgets[2], rng)

    clouds: dict[str, o3d.geometry.PointCloud] = {}
    labels = {
        "keep": f"Kept (after filters): {counts[0]:,}",
        "opacity": f"Removed by opacity: {counts[1]:,}",
        "stat": f"Removed by statistical: {counts[2]:,}",
    }

    if keep_idx.size > 0:
        clouds["keep"] = build_colored_cloud(points_from_indices(vertex_data, keep_idx), np.array([0.10, 0.85, 0.25]))
    if opacity_idx.size > 0:
        clouds["opacity"] = build_colored_cloud(
            points_from_indices(vertex_data, opacity_idx), np.array([0.95, 0.60, 0.15])
        )
    if stat_idx.size > 0:
        clouds["stat"] = build_colored_cloud(points_from_indices(vertex_data, stat_idx), np.array([0.95, 0.15, 0.15]))

    if not clouds:
        print(f"Visualization skipped for {source_file}: no points to display.")
        return

    shown_points = int(keep_idx.size + opacity_idx.size + stat_idx.size)
    total_points = int(vertex_data.shape[0])
    print("\nVisualization controls:")
    print(f"  File: {source_file}")
    print(f"  Total points (before): {total_points:,}")
    print(f"  {labels['keep']}")
    print(f"  {labels['opacity']}")
    print(f"  {labels['stat']}")
    print(f"  Displaying {shown_points:,} sampled points (max={max_points if max_points > 0 else 'all'})")
    print("  Keys: B=Before(all), A=After(kept only), R=Removed only, K/O/S=toggle categories")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"3DGS Filter Preview: {source_file.name}", width=1600, height=960)

    for geom in clouds.values():
        vis.add_geometry(geom)

    render_opts = vis.get_render_option()
    render_opts.point_size = float(point_size)
    render_opts.background_color = np.array([0.04, 0.04, 0.04], dtype=np.float64)

    state = {name: True for name in clouds}

    def set_visible(name: str, visible: bool) -> None:
        if name not in clouds:
            return
        if state[name] == visible:
            return
        if visible:
            vis.add_geometry(clouds[name], reset_bounding_box=False)
        else:
            vis.remove_geometry(clouds[name], reset_bounding_box=False)
        state[name] = visible

    def toggle(name: str):
        def callback(_vis):
            if name in clouds:
                set_visible(name, not state[name])
                _vis.update_renderer()
            return False

        return callback

    def show_before(_vis):
        for name in clouds:
            set_visible(name, True)
        _vis.update_renderer()
        return False

    def show_after(_vis):
        set_visible("keep", True)
        set_visible("opacity", False)
        set_visible("stat", False)
        _vis.update_renderer()
        return False

    def show_removed(_vis):
        set_visible("keep", False)
        set_visible("opacity", True)
        set_visible("stat", True)
        _vis.update_renderer()
        return False

    vis.register_key_callback(ord("B"), show_before)
    vis.register_key_callback(ord("A"), show_after)
    vis.register_key_callback(ord("R"), show_removed)
    vis.register_key_callback(ord("K"), toggle("keep"))
    vis.register_key_callback(ord("O"), toggle("opacity"))
    vis.register_key_callback(ord("S"), toggle("stat"))

    vis.run()
    vis.destroy_window()


def mirror_output_path(input_root: Path, output_root: Path, source_file: Path) -> Path:
    relative = source_file.relative_to(input_root)
    return output_root / relative


def write_filtered_ply(
    ply: PlyData,
    output_file: Path,
    selected_indices: np.ndarray,
    dry_run: bool,
    overwrite: bool,
) -> bool:
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output_file}")

    if dry_run:
        return False

    output_file.parent.mkdir(parents=True, exist_ok=True)

    vertex_data = ply["vertex"].data
    selected_vertices = vertex_data[selected_indices]

    elements: list[PlyElement] = [PlyElement.describe(selected_vertices, "vertex")]
    for elem in ply.elements:
        if elem.name != "vertex":
            elements.append(elem)

    PlyData(
        elements,
        text=bool(ply.text),
        byte_order=str(ply.byte_order),
        comments=list(ply.comments),
        obj_info=list(ply.obj_info),
    ).write(str(output_file))
    return True


def filter_one_file(source_file: Path, output_file: Path, args: argparse.Namespace) -> FilterStats:
    ply = PlyData.read(str(source_file), mmap="r")
    if "vertex" not in ply:
        raise RuntimeError("PLY has no vertex element.")

    vertex_data = ply["vertex"].data
    names = set(vertex_data.dtype.names or ())
    missing = {"x", "y", "z"} - names
    if missing:
        raise RuntimeError(f"PLY missing required fields: {', '.join(sorted(missing))}")

    input_points = int(vertex_data.shape[0])
    opacity_mask = build_opacity_mask(vertex_data, args.opacity_threshold, args.opacity_space)
    opacity_indices = np.flatnonzero(opacity_mask).astype(np.int64, copy=False)

    if args.skip_statistical:
        final_indices = opacity_indices
        skipped_statistical = True
    else:
        final_indices, skipped_statistical = run_statistical_filter(
            vertex_data=vertex_data,
            candidate_indices=opacity_indices,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )

    keep_mask = np.zeros(input_points, dtype=bool)
    keep_mask[final_indices] = True
    removed_by_opacity = int((~opacity_mask).sum())
    removed_by_statistical = int((opacity_mask & ~keep_mask).sum())

    if args.visualize:
        try:
            visualize_filter_result(
                source_file=source_file,
                vertex_data=vertex_data,
                keep_mask=keep_mask,
                opacity_mask=opacity_mask,
                max_points=args.visualize_max_points,
                point_size=args.visualize_point_size,
                seed=args.visualize_seed,
            )
        except Exception as exc:
            print(f"WARNING: visualization failed for {source_file}: {exc}")

    wrote = write_filtered_ply(
        ply=ply,
        output_file=output_file,
        selected_indices=final_indices,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    return FilterStats(
        input_points=input_points,
        after_opacity=int(opacity_indices.size),
        after_statistical=int(final_indices.size),
        removed_by_opacity=removed_by_opacity,
        removed_by_statistical=removed_by_statistical,
        wrote_file=wrote,
        skipped_statistical=skipped_statistical,
    )


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return 1

    files = find_ply_files(input_dir, args.pattern)
    if not files:
        print(f"No .ply files found in {input_dir} (pattern={args.pattern!r})")
        return 1

    print(f"Found {len(files)} PLY file(s)")
    print(f"Input root : {input_dir}")
    print(f"Output root: {output_dir}")
    print(
        f"Filter config: opacity>={args.opacity_threshold} ({args.opacity_space}), "
        f"statistical={'off' if args.skip_statistical else f'on (k={args.nb_neighbors}, std={args.std_ratio})'}"
    )
    if args.dry_run:
        print("Dry run enabled: no files will be written.")

    total_in = 0
    total_after_opacity = 0
    total_after_stat = 0
    total_removed_opacity = 0
    total_removed_statistical = 0
    written_count = 0
    error_count = 0

    for idx, source_file in enumerate(files, start=1):
        output_file = mirror_output_path(input_dir, output_dir, source_file)
        rel = source_file.relative_to(input_dir)
        try:
            stats = filter_one_file(source_file, output_file, args)
            total_in += stats.input_points
            total_after_opacity += stats.after_opacity
            total_after_stat += stats.after_statistical
            total_removed_opacity += stats.removed_by_opacity
            total_removed_statistical += stats.removed_by_statistical
            if stats.wrote_file:
                written_count += 1

            skipped_label = " (stat skipped)" if stats.skipped_statistical else ""
            print(
                f"[{idx}/{len(files)}] {rel} -> kept {stats.after_statistical:,}/{stats.input_points:,} "
                f"(after opacity: {stats.after_opacity:,}, "
                f"removed: opacity {stats.removed_by_opacity:,}, statistical {stats.removed_by_statistical:,})"
                f"{skipped_label}"
            )
        except Exception as exc:
            error_count += 1
            print(f"[{idx}/{len(files)}] ERROR {rel}: {exc}")
            if not args.continue_on_error:
                return 1

    print("\nSummary")
    print(f"  Files processed      : {len(files)}")
    print(f"  Files written        : {written_count}" + (" (dry-run)" if args.dry_run else ""))
    print(f"  Files with errors    : {error_count}")
    print(f"  Total input points   : {total_in:,}")
    print(f"  After opacity filter : {total_after_opacity:,}")
    print(f"  Removed by opacity   : {total_removed_opacity:,}")
    print(f"  Removed statistical  : {total_removed_statistical:,}")
    print(f"  Final kept points    : {total_after_stat:,}")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
