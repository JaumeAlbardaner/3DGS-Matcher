#!/usr/bin/env bash

set -euo pipefail

# Locate the repository from this script's own path so bootstrap.sh can be run
# from any working directory, not just the repo root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INVOCATION_DIR="$PWD"

# Relative paths passed on the command line are resolved against the directory
# the user invoked us from, since we chdir to REPO_ROOT before building.
to_abs() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s\n' "${INVOCATION_DIR%/}/$1" ;;
  esac
}

# Pinned checkouts from scripts/fetch_third_party.sh take precedence over siblings.
if [[ -d "${REPO_ROOT}/third_party/KISS-Matcher" ]]; then
  DEFAULT_KISS_DIR="${REPO_ROOT}/third_party/KISS-Matcher"
else
  DEFAULT_KISS_DIR="$(cd -- "${REPO_ROOT}/.." && pwd)/KISS-Matcher"
fi

# nproc is GNU coreutils; macOS/BSD need sysctl, and minimal images may have neither.
detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif sysctl -n hw.ncpu >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  elif getconf _NPROCESSORS_ONLN >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    echo 4
  fi
}

BUILD_DIR="${REPO_ROOT}/build"
BUILD_DIR_EXPLICIT=0
BUILD_TYPE="Release"
JOBS="${JOBS:-$(detect_jobs)}"
INSTALL_DEPS=0
INSTALL_PCL=0
DO_CLEAN=0
TARGET="align_two_splats"
KISS_DIR="$(to_abs "${KISS_DIR:-$DEFAULT_KISS_DIR}")"
SMALL_GICP_DIR="${SMALL_GICP_DIR:-}"
[[ -z "$SMALL_GICP_DIR" && -d "${REPO_ROOT}/third_party/small_gicp" ]] && SMALL_GICP_DIR="${REPO_ROOT}/third_party/small_gicp"
KISS_BUILD_DIR=""
KISS_BUILD_DIR_EXPLICIT=0
KISS_INSTALL_DIR=""
CLONE_KISS=0
SKIP_KISS_BUILD=0
KISS_URL="https://github.com/MIT-SPARK/KISS-Matcher.git"
TBB_CMAKE_DIR=""
KISS_MATCHER_CMAKE_DIR=""

patch_kiss_matcher_config_eigen_requirement() {
  local cfg="$1"
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    return 0
  fi

  if ! grep -Eq 'find_dependency\(Eigen3[[:space:]]+3\.3[[:space:]]+REQUIRED\)' "$cfg"; then
    return 0
  fi

  echo "[bootstrap] Patching kiss_matcherConfig.cmake Eigen3 constraint..."
  local tmp="${cfg}.tmp"
  awk '
    {
      if ($0 ~ /find_dependency\(Eigen3[[:space:]]+3\.3[[:space:]]+REQUIRED\)/) {
        print "find_dependency(Eigen3 REQUIRED)"
      } else {
        print
      }
    }
  ' "$cfg" > "$tmp"
  mv "$tmp" "$cfg"
}

patch_robin_graph_core_header() {
  local hdr
  hdr="$(find "$KISS_BUILD_DIR" -path "*/_deps/robin-src/include/robin/graph_core.hpp" -print -quit 2>/dev/null || true)"
  if [[ -z "$hdr" || ! -f "$hdr" ]]; then
    return 0
  fi

  if grep -Eq '^[[:space:]]*#include[[:space:]]*<cassert>' "$hdr"; then
    return 0
  fi

  echo "[bootstrap] Applying ROBIN compatibility patch (<cassert>)..."
  local tmp="${hdr}.tmp"
  awk '
    BEGIN { inserted = 0 }
    {
      print
      if ($0 ~ /^#include <robin\/views\.hpp>/ && inserted == 0) {
        print "#include <cassert>"
        inserted = 1
      }
    }
    END {
      if (inserted == 0) {
        # no-op here; handled after awk by prepending
      }
    }
  ' "$hdr" > "$tmp"

  if ! grep -Eq '^[[:space:]]*#include[[:space:]]*<cassert>' "$tmp"; then
    {
      echo "#include <cassert>"
      cat "$tmp"
    } > "${tmp}.2"
    mv "${tmp}.2" "$tmp"
  fi

  mv "$tmp" "$hdr"
}

build_kiss_core_with_retry() {
  if cmake --build "$KISS_BUILD_DIR" -j "$JOBS" --target kiss_matcher_core; then
    return 0
  fi

  echo "[bootstrap] KISS core build failed once. Re-applying ROBIN patch and retrying..."
  patch_robin_graph_core_header
  cmake --build "$KISS_BUILD_DIR" -j "$JOBS" --target kiss_matcher_core
}

# A CMake build tree records both the source it was generated from and its own
# location. Either can go stale when a checkout is copied between machines or
# moved, and CMake then refuses to reuse the cache.
# Package configs generated inside a build tree (robin, xenium) can export include
# directories that the build never creates, and every consumer then fails with
# "includes non-existent path ... in its INTERFACE_INCLUDE_DIRECTORIES". The headers
# themselves live elsewhere in the tree, so materialising the empty directory is
# enough. Only paths under the given root are touched.
# robin is fetched into the KISS build tree when no system robin exists, and the
# package config generated there bakes in absolute build paths. The copy installed
# alongside KISS is relocatable (${_IMPORT_PREFIX}/include), so make sure it exists.
# Installing robin needs the full KISS tree built, not just kiss_matcher_core.
# find_package also consults the CMake user package registry (~/.cmake/packages),
# which robin populates via export(PACKAGE robin) for every build tree it configures.
# A registered tree whose exported include dir was never created breaks the
# KISS-Matcher configure itself, before any of our own CMake runs - and the tree it
# points at need not be the one we are building.
repair_registered_package_trees() {
  local pkg reg entry dir
  for pkg in "$@"; do
    reg="${HOME}/.cmake/packages/${pkg}"
    [[ -d "$reg" ]] || continue
    for entry in "$reg"/*; do
      [[ -f "$entry" ]] || continue
      dir="$(head -n1 "$entry")"
      [[ -n "$dir" ]] || continue
      if [[ ! -d "$dir" ]]; then
        echo "[bootstrap] Dropping stale ${pkg} package-registry entry: $dir"
        rm -f "$entry"
        continue
      fi
      ensure_dep_include_dirs "$dir"
    done
  done
}

ensure_robin_installed() {
  [[ -d "$KISS_BUILD_DIR/_deps/robin-build" ]] || return 0
  local d
  for d in "$KISS_INSTALL_DIR/lib/cmake/robin" "$KISS_INSTALL_DIR/lib64/cmake/robin"; do
    [[ -f "$d/robinConfig.cmake" ]] && return 0
  done
  echo "[bootstrap] robin was fetched but not installed; building the full KISS tree to install it..."
  cmake --build "$KISS_BUILD_DIR" -j "$JOBS" || return 0
  cmake --install "$KISS_BUILD_DIR" --prefix "$KISS_INSTALL_DIR" || return 0
}

ensure_dep_include_dirs() {
  local root="$1" targets_file inc_path parent
  [[ -d "$root" ]] || return 0
  while IFS= read -r targets_file; do
    while IFS= read -r inc_path; do
      [[ -n "$inc_path" ]] || continue
      case "$inc_path" in "$root"|"$root"/*) ;; *) continue ;; esac
      # Only generated <dep>-build/include dirs: a missing <dep>-src/include means a
      # broken checkout, and should keep failing loudly rather than be papered over.
      [[ "$(basename "$inc_path")" == "include" ]] || continue
      parent="$(dirname "$inc_path")"
      case "$parent" in *-build) ;; *) continue ;; esac
      if [[ ! -d "$inc_path" ]]; then
        echo "[bootstrap] Creating include dir missing from a package config: $inc_path"
        mkdir -p "$inc_path"
      fi
    done < <(sed -n 's/.*INTERFACE_INCLUDE_DIRECTORIES "\([^"]*\)".*/\1/p' "$targets_file" | tr ';' '\n')
  done < <(find "$root" -name '*Targets.cmake' 2>/dev/null)
}

check_build_dir_source() {
  local dir="$1" expected="$2" explicit="$3" label="$4"
  local cache="$dir/CMakeCache.txt" cached_src cached_dir real reason
  [[ -f "$cache" ]] || return 0
  real="$(cd -- "$dir" && pwd)"
  cached_src="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | head -n1)"
  cached_dir="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache" | head -n1)"
  if [[ -n "$cached_src" && "$cached_src" != "$expected" ]]; then
    reason="configured for a different source tree ($cached_src)"
  elif [[ -n "$cached_dir" && "$cached_dir" != "$real" ]]; then
    reason="created in a different directory ($cached_dir)"
  else
    return 0
  fi
  if [[ "$explicit" -eq 0 ]]; then
    echo "[bootstrap] $label build dir $real was $reason; reconfiguring."
    rm -rf "$dir"
  else
    echo "$label build directory cannot be reused: it was $reason." >&2
    echo "  build dir      : $real" >&2
    echo "  expected source: $expected" >&2
    echo "Remove it, or point at a different build directory." >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]

Options:
  --deps                    Install required packages into active conda env (no sudo)
  --conda-deps              Alias of --deps
  --with-pcl                Also install PCL in conda env (optional)
  --build-dir <dir>         CMake build directory for this repo (default: build)
  --build-type <type>       CMake build type (default: Release)
  --target <name>           Build target: align_two_splats | all (default: align_two_splats)
  --kiss-dir <dir>          Path to KISS-Matcher checkout (default: <repo>/../KISS-Matcher)
  --small-gicp-dir <dir>    Path to small_gicp checkout (default: <repo>/../small_gicp)
  --kiss-build-dir <dir>    KISS-Matcher build directory (default: <kiss-dir>/build)
  --kiss-install-dir <dir>  KISS-Matcher local install prefix (default: <kiss-build-dir>/install)
  --clone-kiss              Clone KISS-Matcher if --kiss-dir does not exist
  --skip-kiss-build         Skip building KISS-Matcher
  -j, --jobs <n>            Parallel build jobs (default: detected CPU count)
  --clean                   Remove build directory before configuring
  -h, --help                Show this help

Examples:
  scripts/bootstrap.sh --deps --clone-kiss
  scripts/bootstrap.sh --conda-deps --clone-kiss
  scripts/bootstrap.sh --target align_two_splats -j 8
  scripts/bootstrap.sh --kiss-dir ../KISS-Matcher --kiss-build-dir ../KISS-Matcher/build
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deps)
      INSTALL_DEPS=1
      shift
      ;;
    --conda-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --with-pcl)
      INSTALL_PCL=1
      shift
      ;;
    --build-dir)
      BUILD_DIR="$(to_abs "${2:-}")"
      BUILD_DIR_EXPLICIT=1
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="${2:-}"
      shift 2
      ;;
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --kiss-dir)
      KISS_DIR="$(to_abs "${2:-}")"
      shift 2
      ;;
    --small-gicp-dir)
      SMALL_GICP_DIR="$(to_abs "${2:-}")"
      shift 2
      ;;
    --kiss-build-dir)
      KISS_BUILD_DIR="$(to_abs "${2:-}")"
      KISS_BUILD_DIR_EXPLICIT=1
      shift 2
      ;;
    --kiss-install-dir)
      KISS_INSTALL_DIR="$(to_abs "${2:-}")"
      shift 2
      ;;
    --clone-kiss)
      CLONE_KISS=1
      shift
      ;;
    --skip-kiss-build)
      SKIP_KISS_BUILD=1
      shift
      ;;
    -j|--jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd -- "$REPO_ROOT"

if [[ ! -f "CMakeLists.txt" ]] || [[ ! -d "include/3dgs_matcher" ]]; then
  echo "Not a 3DGS-Matcher checkout: $REPO_ROOT" >&2
  echo "(expected CMakeLists.txt and include/3dgs_matcher alongside scripts/)" >&2
  exit 1
fi

if [[ "$TARGET" != "align_two_splats" && "$TARGET" != "all" ]]; then
  echo "Invalid --target '$TARGET'. Expected align_two_splats|all." >&2
  exit 1
fi

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected. Activate one first, then re-run." >&2
    echo "Example: conda create -n 3dgs-feature -y && conda activate 3dgs-feature" >&2
    exit 1
  fi

  CONDA_TOOL=""
  if command -v mamba >/dev/null 2>&1; then
    CONDA_TOOL="mamba"
  elif command -v micromamba >/dev/null 2>&1; then
    CONDA_TOOL="micromamba"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_TOOL="conda"
  else
    echo "Could not find mamba/micromamba/conda in PATH." >&2
    exit 1
  fi

  # OpenMP runtime package name differs by platform.
  # Linux toolchain from conda-forge typically provides it via compiler packages,
  # while macOS often needs llvm-openmp explicitly.
  OPENMP_PKG=""
  if [[ "$(uname -s)" == "Darwin" ]]; then
    OPENMP_PKG="llvm-openmp"
  fi

  CONDA_PKGS=(
    "cmake<4"
    ninja
    make
    pkg-config
    cxx-compiler
    "eigen<4"
    tbb
    tbb-devel
    lz4-c
    flann
    git
  )
  if [[ -n "$OPENMP_PKG" ]]; then
    CONDA_PKGS+=("$OPENMP_PKG")
  fi

  echo "[bootstrap] Installing conda dependencies into: $CONDA_PREFIX"
  "$CONDA_TOOL" install -y -c conda-forge "${CONDA_PKGS[@]}"

  if [[ "$INSTALL_PCL" -eq 1 ]]; then
    "$CONDA_TOOL" install -y -c conda-forge pcl
  fi
fi

# Prefix holding the build dependencies: the active conda env, or DEPS_PREFIX
# (set by the uv entry point to its virtual environment, see pyproject.toml).
DEPS_PREFIX="${DEPS_PREFIX:-${CONDA_PREFIX:-}}"
if [[ -n "$DEPS_PREFIX" ]]; then
  if [[ -f "${DEPS_PREFIX}/lib/cmake/TBB/TBBConfig.cmake" ]]; then
    TBB_CMAKE_DIR="${DEPS_PREFIX}/lib/cmake/TBB"
  elif [[ -f "${DEPS_PREFIX}/lib/cmake/tbb/TBBConfig.cmake" ]]; then
    TBB_CMAKE_DIR="${DEPS_PREFIX}/lib/cmake/tbb"   # PyPI tbb-devel wheel layout
  else
    echo "[bootstrap] Warning: TBBConfig.cmake not found under ${DEPS_PREFIX}."
    echo "[bootstrap] If CMake cannot find TBB, run: conda install -c conda-forge tbb-devel"
  fi
fi

if [[ -z "$KISS_BUILD_DIR" ]]; then
  KISS_BUILD_DIR="$KISS_DIR/build"
fi
if [[ -z "$KISS_INSTALL_DIR" ]]; then
  KISS_INSTALL_DIR="$KISS_BUILD_DIR/install"
fi

if [[ ! -d "$KISS_DIR" ]]; then
  if [[ "$CLONE_KISS" -eq 1 ]]; then
    echo "[bootstrap] Cloning KISS-Matcher into $KISS_DIR ..."
    git clone "$KISS_URL" "$KISS_DIR"
  else
    echo "KISS-Matcher not found at: $KISS_DIR" >&2
    echo "Re-run with --clone-kiss or provide --kiss-dir." >&2
    exit 1
  fi
fi

if [[ ! -f "$KISS_DIR/cpp/kiss_matcher/CMakeLists.txt" ]]; then
  echo "Invalid KISS-Matcher checkout: $KISS_DIR (missing cpp/kiss_matcher/CMakeLists.txt)" >&2
  exit 1
fi

KISS_CMAKE_PREFIX_ARGS=()
if [[ -n "$DEPS_PREFIX" ]]; then
  KISS_CMAKE_PREFIX_ARGS+=("-DCMAKE_PREFIX_PATH=${DEPS_PREFIX}")
fi
if [[ -n "$TBB_CMAKE_DIR" ]]; then
  KISS_CMAKE_PREFIX_ARGS+=("-DTBB_DIR=${TBB_CMAKE_DIR}")
fi
KISS_CMAKE_PREFIX_ARGS+=("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")

if [[ "$SKIP_KISS_BUILD" -eq 0 ]]; then
  # KISS-Matcher has its own build tree, untouched by --clean; a copied checkout
  # leaves a cache here naming the old machine's paths.
  check_build_dir_source "$KISS_BUILD_DIR" "$KISS_DIR/cpp/kiss_matcher" \
    "$KISS_BUILD_DIR_EXPLICIT" "KISS-Matcher"

  # Must happen before this configure: KISS itself does find_package(robin), and a
  # broken registered build tree fails at generate time right here.
  repair_registered_package_trees robin pmc
  ensure_dep_include_dirs "$KISS_BUILD_DIR"

  echo "[bootstrap] Configuring KISS-Matcher dependency..."
  cmake -S "$KISS_DIR/cpp/kiss_matcher" -B "$KISS_BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    "${KISS_CMAKE_PREFIX_ARGS[@]}"

  patch_robin_graph_core_header

  echo "[bootstrap] Building KISS-Matcher core..."
  build_kiss_core_with_retry

  echo "[bootstrap] Installing KISS-Matcher package locally to $KISS_INSTALL_DIR ..."
  if ! cmake --install "$KISS_BUILD_DIR" --prefix "$KISS_INSTALL_DIR"; then
    echo "[bootstrap] KISS install failed once; building full KISS tree and retrying install..."
    cmake --build "$KISS_BUILD_DIR" -j "$JOBS"
    cmake --install "$KISS_BUILD_DIR" --prefix "$KISS_INSTALL_DIR"
  fi
fi

if [[ "$SKIP_KISS_BUILD" -eq 0 ]]; then
  ensure_robin_installed
fi
ensure_dep_include_dirs "$KISS_BUILD_DIR"

for candidate in \
  "$KISS_INSTALL_DIR/lib/cmake/kiss_matcher" \
  "$KISS_INSTALL_DIR/lib64/cmake/kiss_matcher" \
  "$KISS_BUILD_DIR"; do
  if [[ -f "$candidate/kiss_matcherConfig.cmake" || -f "$candidate/kiss_matcher-config.cmake" ]]; then
    KISS_MATCHER_CMAKE_DIR="$candidate"
    break
  fi
done
if [[ -z "$KISS_MATCHER_CMAKE_DIR" ]]; then
  echo "[bootstrap] Warning: could not locate kiss_matcher CMake package config." >&2
  echo "[bootstrap] Expected in: $KISS_INSTALL_DIR/lib/cmake/kiss_matcher" >&2
else
  if [[ -f "$KISS_MATCHER_CMAKE_DIR/kiss_matcherConfig.cmake" ]]; then
    patch_kiss_matcher_config_eigen_requirement "$KISS_MATCHER_CMAKE_DIR/kiss_matcherConfig.cmake"
  elif [[ -f "$KISS_MATCHER_CMAKE_DIR/kiss_matcher-config.cmake" ]]; then
    patch_kiss_matcher_config_eigen_requirement "$KISS_MATCHER_CMAKE_DIR/kiss_matcher-config.cmake"
  fi
fi

# Prefer the installed, relocatable robin over the build-tree config.
ROBIN_CMAKE_DIR=""
for candidate in \
  "$KISS_INSTALL_DIR/lib/cmake/robin" \
  "$KISS_INSTALL_DIR/lib64/cmake/robin" \
  "$KISS_BUILD_DIR/_deps/robin-build"; do
  if [[ -f "$candidate/robinConfig.cmake" ]]; then
    ROBIN_CMAKE_DIR="$candidate"
    break
  fi
done
PREFIX_PATH="$KISS_BUILD_DIR"
if [[ -n "$DEPS_PREFIX" ]]; then
  PREFIX_PATH="${DEPS_PREFIX};${PREFIX_PATH}"
fi
if [[ -d "$KISS_INSTALL_DIR" ]]; then
  PREFIX_PATH="${KISS_INSTALL_DIR};${PREFIX_PATH}"
fi
if [[ -d "$ROBIN_CMAKE_DIR" ]]; then
  PREFIX_PATH="$PREFIX_PATH;$ROBIN_CMAKE_DIR"
fi

if [[ "$DO_CLEAN" -eq 1 ]]; then
  echo "[bootstrap] Cleaning build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

check_build_dir_source "$BUILD_DIR" "$REPO_ROOT" "$BUILD_DIR_EXPLICIT" "Project"

echo "[bootstrap] Repo:      $REPO_ROOT"
echo "[bootstrap] Build dir: $BUILD_DIR"
echo "[bootstrap] Configuring 3DGS-Matcher..."
PROJECT_CMAKE_ARGS=(
  "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
  "-DCMAKE_PREFIX_PATH=$PREFIX_PATH"
)
if [[ -n "$TBB_CMAKE_DIR" ]]; then
  PROJECT_CMAKE_ARGS+=("-DTBB_DIR=$TBB_CMAKE_DIR")
fi
if [[ -n "$KISS_MATCHER_CMAKE_DIR" ]]; then
  PROJECT_CMAKE_ARGS+=("-Dkiss_matcher_DIR=$KISS_MATCHER_CMAKE_DIR")
fi
if [[ -n "$ROBIN_CMAKE_DIR" ]]; then
  echo "[bootstrap] robin: $ROBIN_CMAKE_DIR"
  PROJECT_CMAKE_ARGS+=("-Drobin_DIR=$ROBIN_CMAKE_DIR")
fi
if [[ -n "$SMALL_GICP_DIR" ]]; then
  PROJECT_CMAKE_ARGS+=("-DSMALL_GICP_DIR=$SMALL_GICP_DIR")
fi
cmake -S . -B "$BUILD_DIR" "${PROJECT_CMAKE_ARGS[@]}"

echo "[bootstrap] Building target(s)..."
if [[ "$TARGET" == "all" ]]; then
  cmake --build "$BUILD_DIR" -j "$JOBS"
else
  cmake --build "$BUILD_DIR" -j "$JOBS" --target "$TARGET"
fi

echo
echo "[bootstrap] Done."
if [[ "$TARGET" == "align_two_splats" || "$TARGET" == "all" ]]; then
  echo "Align two splats with:"
  echo "  $BUILD_DIR/align_two_splats <source.splat|.ply> <target.splat|.ply>"
fi
