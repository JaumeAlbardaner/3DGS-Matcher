#!/usr/bin/env bash
# fetch_third_party.sh
#
# Clone the C++ dependencies at the exact commits the paper's results were
# produced with, into third_party/. scripts/bootstrap.sh picks them up from there.
#
#   KISS-Matcher  42154f8  + third_party/kiss-matcher.patch
#                          (empty-keypoint guards, nanoflann alignment fix)
#   small_gicp    78f2e7a  (header + align helper used by benchmark_submap)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"

fetch() {  # fetch <dir> <url> <commit>
  local dir="$TP/$1"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$2" "$dir"
  fi
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$3"
  echo "$1 at $(git -C "$dir" rev-parse --short HEAD)"
}

fetch KISS-Matcher https://github.com/MIT-SPARK/KISS-Matcher.git 42154f8
if git -C "$TP/KISS-Matcher" apply --check "$TP/kiss-matcher.patch" 2>/dev/null; then
  git -C "$TP/KISS-Matcher" apply "$TP/kiss-matcher.patch"
  echo "applied kiss-matcher.patch"
else
  git -C "$TP/KISS-Matcher" apply --reverse --check "$TP/kiss-matcher.patch" \
    && echo "kiss-matcher.patch already applied"
fi

fetch small_gicp https://github.com/koide3/small_gicp.git 78f2e7a
