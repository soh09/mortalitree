#!/usr/bin/env python3
"""
Fetch the NeonTreeEvaluation benchmark (Weinstein et al. 2021, PLoS Comput Biol;
Zenodo record 5914554, v0.2.2) -- only the parts needed to score LiDAR crowns:

  OUT_DIR/annotations/*.xml          crown boxes (Pascal VOC, RGB pixel coords)
  OUT_DIR/evaluation/CHM/*_CHM.tif   1 m canopy height per 40 x 40 m plot
  OUT_DIR/evaluation/RGB/*.tif       0.1 m RGB, only for plots that have annotations
  OUT_DIR/training/{CHM,RGB}/*.tif   with --training: the large RGB-only-annotated tiles

The zips are mostly hyperspectral and point clouds (~8 GB), so instead of downloading
them whole this reads each zip's central directory over HTTP range requests and pulls
only the members above. Every member is CRC-checked by zipfile as it is extracted.

    uv run python baselines/watershed_eval/neontree_fetch.py OUT_DIR [--training]
"""
import argparse, http.client, io, json, time, urllib.request, zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/5914554"
BLOCK = 1 << 16          # minimum range-request size (zip directory reads are small and scattered)


class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a URL via HTTP range requests, with one block of cache."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.buf_start, self.buf = 0, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def prefetch(self, start, n):
        """Cache bytes [start, start+n) with one request (one zip member = one request)."""
        self.pos = start
        self._fill(n)

    def _fill(self, n):
        end = min(self.pos + max(n, BLOCK), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    self.buf_start, self.buf = self.pos, r.read()
                return
            except (OSError, http.client.HTTPException) as e:      # dropped connections, timeouts
                if attempt == 5:
                    raise
                print(f"    retry {attempt + 1} after {type(e).__name__}")
                time.sleep(2 ** attempt)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        off = self.pos - self.buf_start
        if not (0 <= off and off + n <= len(self.buf)):
            self._fill(n); off = 0
        out = self.buf[off:off + n]
        self.pos += len(out)
        return out


def extract(z, members, out, prefix=""):
    done = 0
    for info in members:
        dest = out / prefix / info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        # local header (30 B + name + extra, extra <= 64 KiB) then the compressed data
        z.fp.prefetch(info.header_offset, 30 + len(info.orig_filename.encode()) + (1 << 16) + info.compress_size)
        tmp.write_bytes(z.read(info))           # raises BadZipFile on CRC mismatch
        tmp.rename(dest)
        done += 1
        if done % 50 == 0:
            print(f"    {done}/{len(members)}")
    return done


def open_zip(files, name):
    return zipfile.ZipFile(HttpFile(files[name]["links"]["self"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--training", action="store_true", help="also fetch training-tile CHM + RGB (~0.7 GB)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = {f["key"]: f for f in json.load(urllib.request.urlopen(RECORD, timeout=60))["files"]}

    def wanted(z, keep):
        return [i for i in z.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX")
                and keep(i.filename)]

    z = open_zip(files, "annotations.zip")
    ann = wanted(z, lambda f: f.startswith("annotations/") and f.endswith(".xml"))
    print(f"annotations: {len(ann)} xml, {extract(z, ann, out)} new")
    annotated = {Path(i.filename).stem for i in ann}

    z = open_zip(files, "evaluation.zip")
    chm = wanted(z, lambda f: f.startswith("evaluation/CHM/") and f.endswith("_CHM.tif"))
    rgb = wanted(z, lambda f: f.startswith("evaluation/RGB/") and f.endswith(".tif")
                 and Path(f).stem in annotated)
    print(f"evaluation: {len(chm)} CHM, {len(rgb)} annotated RGB "
          f"({sum(i.file_size for i in rgb) / 1e6:.0f} MB)")
    print(f"  {extract(z, chm + rgb, out)} new")

    if a.training:
        z = open_zip(files, "training.zip")
        tr = wanted(z, lambda f: f.split("/")[0] in ("CHM", "RGB") and f.endswith(".tif"))
        print(f"training: {len(tr)} tifs ({sum(i.file_size for i in tr) / 1e6:.0f} MB)")
        print(f"  {extract(z, tr, out, 'training')} new")


if __name__ == "__main__":
    main()
