#!/usr/bin/env python3
"""Extra experiment 3: Gaussians per Mip-NeRF 360 map pair.

Counts the Gaussians of the two split halves of each scene from the PLY headers
under $DATA_ROOT/trained_gaussians, to set against the coarse runtimes of
Table II (gen_large_scale_table.py). Skipped if the maps are missing.
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
PLY = "point_cloud/iteration_30000/point_cloud.ply"


def vertex_count(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    raise ValueError(f"no vertex count in {path}")


def scene_sizes():
    base = DATA / "trained_gaussians"
    scenes = ["bonsai", "counter", "kitchen", "room", "bicycle", "flowers", "garden", "stump", "treehill"]
    if not all((base / s / h / PLY).exists() for s in scenes for h in ("split_pos", "split_neg")):
        print(f"\nscene sizes skipped: Mip-NeRF 360 maps not found under {base}")
        return
    print("\nGaussians per map pair (split_pos + split_neg); coarse runtimes are Table II's")
    for s in scenes:
        n = sum(vertex_count(base / s / h / PLY) for h in ("split_pos", "split_neg"))
        print(f"  {s:9s} {n / 1e6:5.2f} M")



if __name__ == "__main__":
    scene_sizes()
