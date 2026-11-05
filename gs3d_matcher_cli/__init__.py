"""`uv run align_two_splats ...`: build the C++ tool on first use, check the inputs, run it.

Without conda: uv provides CMake and TBB from PyPI (see pyproject.toml), and
scripts/bootstrap.sh builds the pinned KISS-Matcher and small_gicp against them.
Only a C++17 compiler, make and git are needed from the system.

The tool's defaults are the configuration evaluated in the paper, so every
option is passed through unchanged.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "align_two_splats"
PLY_FIELDS = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")
SPLAT_RECORD = 32  # bytes: position 3f, scale 3f, RGBA 4B, rotation 4B


def fail(msg):
    sys.exit(f"align_two_splats: {msg}")


def check_map(path):
    """Fail early, with a clear message, on inputs the C++ loader would reject."""
    p = Path(path)
    if not p.is_file():
        fail(f"{path}: file not found")
    ext = p.suffix.lower()
    if ext == ".splat":
        size = p.stat().st_size
        if size == 0 or size % SPLAT_RECORD:
            fail(f"{path}: {size} bytes is not a whole number of {SPLAT_RECORD}-byte .splat records")
    elif ext == ".ply":
        with p.open("rb") as f:
            if f.readline().strip() != b"ply":
                fail(f"{path}: not a PLY file")
            fmt, props, in_vertex = None, [], False
            for raw in f:
                tok = raw.decode("ascii", "replace").split()
                if not tok:
                    continue
                if tok[0] == "format":
                    fmt = tok[1]
                elif tok[0] == "element":
                    in_vertex = tok[1] == "vertex"
                elif tok[0] == "property" and in_vertex:
                    props.append(tok[-1])
                elif tok[0] == "end_header":
                    break
        if fmt not in ("ascii", "binary_little_endian"):
            fail(f"{path}: PLY format '{fmt}' is not supported (ascii or binary_little_endian)")
        missing = [k for k in PLY_FIELDS if k not in props]
        if missing:
            fail(f"{path}: not a 3DGS PLY, missing vertex properties {', '.join(missing)} "
                 "(expected the output of 3D Gaussian Splatting training: "
                 "point_cloud/iteration_*/point_cloud.ply)")
    else:
        fail(f"{path}: expected a .ply or .splat 3DGS map")


def build():
    env = dict(os.environ)
    env["DEPS_PREFIX"] = sys.prefix  # this uv environment: CMake + TBB wheels
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    if not (ROOT / "third_party" / "KISS-Matcher").is_dir():
        subprocess.run(["bash", str(ROOT / "scripts" / "fetch_third_party.sh")], check=True, env=env)
    print("align_two_splats: building the C++ tool (first run only) ...", file=sys.stderr)
    subprocess.run(["bash", str(ROOT / "scripts" / "bootstrap.sh"), "--target", "align_two_splats"],
                   check=True, env=env)


def main():
    if not (ROOT / "CMakeLists.txt").is_file():
        fail("run this from a clone of the repository (uv run align_two_splats ...)")
    args = sys.argv[1:]
    if args and args[0] == "--rebuild":
        args = args[1:]
        BIN.unlink(missing_ok=True)
    maps = [a for a in args[:2] if not a.startswith("-")]
    if "-h" not in args and "--help" not in args:
        if len(maps) < 2:
            fail("usage: uv run align_two_splats <source.ply|.splat> <target.ply|.splat> [options]")
        for m in maps:
            check_map(m)
    if not BIN.is_file():
        try:
            build()
        except subprocess.CalledProcessError as e:
            fail(f"build failed (exit {e.returncode}); see the output above and the README's Troubleshooting")
    os.execv(str(BIN), [str(BIN), *args])
