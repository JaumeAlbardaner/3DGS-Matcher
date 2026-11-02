#!/usr/bin/env python3
"""Read members of a remote zip over HTTP range requests, without downloading it.

KITTI raw drives ship as one zip per drive (1-21 GB), mostly Velodyne scans.
The S3 bucket honours Range requests, so only the zip's central directory and
the wanted members (colour images, OXTS) are fetched.

  python3 scripts/kitti_remote_zip.py list    <url> [--glob PATTERN]
  python3 scripts/kitti_remote_zip.py extract <url> <out_dir> --glob '*/image_0[23]/data/*'

Extraction mirrors the zip layout under out_dir (the same layout `unzip` gives),
checks every member's CRC, and skips files already present with the right size,
so an interrupted run can simply be restarted.
"""
import argparse
import fnmatch
import io
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOCAL_HEADER = 30   # fixed part of a zip local file header
EXTRA_SLACK = 256   # over-read so header + data usually arrive in one request


def fetch(url, start, end, tries=5):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise IOError(f"expected HTTP 206, got {r.status}")
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges, enough for zipfile's directory scan."""

    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        data = fetch(self.url, self.pos, min(self.pos + len(buf), self.size) - 1)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def members(url, pattern, frames=None):
    """Matching file members; `frames` = (first, last) keeps only numbered files in range."""
    def in_range(name):
        stem = Path(name).stem
        return frames is None or not stem.isdigit() or frames[0] <= int(stem) <= frames[1]
    with zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 20)) as zf:
        return [i for i in zf.infolist() if not i.is_dir()
                and fnmatch.fnmatch(i.filename, pattern) and in_range(i.filename)]


def extract_one(url, info, out_dir):
    dst = out_dir / info.filename
    if dst.exists() and dst.stat().st_size == info.file_size:
        return 0
    want = LOCAL_HEADER + len(info.filename.encode()) + EXTRA_SLACK + info.compress_size
    blob = fetch(url, info.header_offset, info.header_offset + want - 1)
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = LOCAL_HEADER + name_len + extra_len
    data = blob[start:start + info.compress_size]
    if len(data) < info.compress_size:  # extra field longer than the slack
        begin = info.header_offset + start
        data = fetch(url, begin, begin + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{info.filename}: unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise IOError(f"{info.filename}: CRC mismatch")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dst)
    return info.compress_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("list", "extract"))
    ap.add_argument("url")
    ap.add_argument("out_dir", nargs="?", type=Path)
    ap.add_argument("--glob", default="*", help="fnmatch pattern on member names")
    ap.add_argument("--frames", metavar="FIRST-LAST",
                    help="only numbered files (e.g. 0000002599.png) in this raw frame range")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    frames = tuple(map(int, args.frames.split("-"))) if args.frames else None
    infos = members(args.url, args.glob, frames)
    if args.mode == "list":
        groups = defaultdict(lambda: [0, 0])
        for i in infos:
            key = "/".join(i.filename.split("/")[:-1])
            groups[key][0] += 1
            groups[key][1] += i.compress_size
        for key, (n, size) in sorted(groups.items()):
            print(f"{size / 1e9:8.3f} GB  {n:6d} files  {key}")
        print(f"{sum(i.compress_size for i in infos) / 1e9:8.3f} GB  {len(infos):6d} files  TOTAL matching '{args.glob}'")
        return

    if args.out_dir is None:
        sys.exit("extract needs out_dir")
    total = sum(i.compress_size for i in infos)
    print(f"extracting {len(infos)} files ({total / 1e9:.2f} GB) -> {args.out_dir}", flush=True)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for k, n in enumerate(pool.map(lambda i: extract_one(args.url, i, args.out_dir), infos), 1):
            done += n
            if k % 500 == 0 or k == len(infos):
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {k}/{len(infos)} files, {done / 1e9:.2f} GB fetched, {rate:.1f} MB/s", flush=True)


if __name__ == "__main__":
    main()
