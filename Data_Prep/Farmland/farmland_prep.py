"""
farmland_prep.py - turn ESA WorldCover 2021 (v200) tiles into the cropland
files the Flood Impact Model reads. Usually started through farmland_prep.sh.

For each 3x3 degree ESA tile:
  1. download   the tile's GeoTIFF from ESA's public AWS bucket, unless the
                output already exists or the tile is available locally
  2. convert    count cropland pixels (class 40) in blocks of BLOCK_PX x
                BLOCK_PX pixels (15 px ~ 150 m) and store one point per block
                at the block centre, with the block's exact cropland area
                (each pixel's area computed on the WGS84 ellipsoid)
  3. validate   read the new file back completely (../OSM/validate_parquet.py)
  4. checksum   write <file>.sha256
  5. clean up   delete the downloaded GeoTIFF (unless --keep-tif)

Output: <work-dir>/Farmland/<tile>.parquet, e.g. Farmland/N21W081.parquet
Columns: geometry (block-centre point), area_m2, bbox (covering).
Tiles with no cropland get an empty (0-row) file, so a rerun skips them too.

Unlike the old polygon process, nothing is traced, simplified or filtered:
every cropland pixel's area is counted. At an AOI border a block's whole area
goes to the AOI holding its centre (at most ~110 m off for 15 px blocks).
Tested on a real tile against a per-pixel count over GEOGLOWS catchments:
  block 30 px: per-catchment error median 2.2%, max 22%   (~8 GB globally)
  block 15 px: per-catchment error median 1.0%, max 6.9%  (~20 GB globally)
  block 10 px: per-catchment error median 0.5%, max 3.5%  (~35-40 GB globally)
Totals are exact for every block size.

Safe to rerun: finished tiles are skipped.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio.windows import Window

BUCKET_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
PREFIX = "v200/2021/map/"
CROPLAND_CLASS = 40
TILE_RE = re.compile(r"ESA_WorldCover_10m_2021_v200_([NS]\d{2}[EW]\d{3})_Map\.tif$")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "OSM"))
from validate_parquet import check_file          # noqa: E402  (shared with the OSM prep)


# ---------------------------------------------------------------------------
# Tile names and the ESA tile list
# ---------------------------------------------------------------------------
def tile_file(code):
    return f"ESA_WorldCover_10m_2021_v200_{code}_Map.tif"


def tile_bounds(code):
    """N21W081 -> (-81, 21, -78, 24): the tile's lower-left corner + 3 degrees."""
    lat = int(code[1:3]) * (1 if code[0] == "N" else -1)
    lon = int(code[4:7]) * (1 if code[3] == "E" else -1)
    return lon, lat, lon + 3, lat + 3


def list_tiles(cache, refresh=False):
    """All tile codes in ESA's bucket (about 2,650 land tiles). The list is
    cached in a text file so later runs don't need to ask again."""
    if cache.exists() and not refresh:
        return cache.read_text().split()
    codes, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX}
        if token:
            q["continuation-token"] = token
        url = f"{BUCKET_URL}/?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for key in root.iter(f"{S3_NS}Key"):
            m = TILE_RE.search(key.text)
            if m:
                codes.append(m.group(1))
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            break
        token = root.findtext(f"{S3_NS}NextContinuationToken")
    codes = sorted(set(codes))
    if not codes:
        raise RuntimeError(f"No tiles found under {BUCKET_URL}/{PREFIX}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(codes) + "\n")
    return codes


def download(code, dest, tries=4):
    """Download one tile to dest (via dest.partial), checking the byte count."""
    url = f"{BUCKET_URL}/{PREFIX}{tile_file(code)}"
    part = dest.with_name(dest.name + ".partial")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                expected = int(r.headers.get("Content-Length", -1))
                while chunk := r.read(8 << 20):
                    f.write(chunk)
            if expected >= 0 and part.stat().st_size != expected:
                raise IOError(f"got {part.stat().st_size} of {expected} bytes")
            part.replace(dest)
            return
        except Exception as e:
            if attempt == tries:
                raise RuntimeError(f"download failed for {code}: {e}") from e
            time.sleep(10 * attempt)


# ---------------------------------------------------------------------------
# Raster -> cropland blocks
# ---------------------------------------------------------------------------
def pixel_area_m2(dlon, dlat, lat):
    """Area (m^2) of a dlon x dlat degree pixel centred at `lat`, WGS84 ellipsoid."""
    phi = np.radians(lat)
    w2 = 1.0 - WGS84_E2 * np.sin(phi) ** 2
    m_n = (WGS84_A ** 2) * (1.0 - WGS84_E2) / (w2 ** 2)
    return dlon * dlat * (math.pi / 180.0) ** 2 * m_n * np.cos(phi)


def cropland_blocks(tif, block_px):
    """(lng, lat, area_m2) of every block with any cropland, block centres."""
    out_x, out_y, out_a = [], [], []
    with rasterio.open(tif) as src:
        t = src.transform
        if src.crs is None or not src.crs.is_geographic or t.b != 0 or t.d != 0:
            raise ValueError(f"{tif}: expected a north-up lon/lat raster")
        W, H = src.width, src.height
        dlon, dlat = t.a, -t.e
        nbx = math.ceil(W / block_px)
        strip = block_px * max(1, 3600 // block_px)       # ~3,600 rows per read
        for r0 in range(0, H, strip):
            h = min(strip, H - r0)
            data = src.read(1, window=Window(0, r0, W, h))
            mask = data == CROPLAND_CLASS
            if not mask.any():
                continue
            # Pad so rows/columns divide evenly into blocks (only the last
            # block of a non-multiple raster is partial; padding counts 0).
            ph, pw = math.ceil(h / block_px) * block_px, nbx * block_px
            if (ph, pw) != mask.shape:
                padded = np.zeros((ph, pw), dtype=bool)
                padded[:h, :W] = mask
                mask = padded
            # Cropland pixels per row, per block column ...
            per_row = mask.reshape(ph, nbx, block_px).sum(axis=2, dtype=np.int32)
            # ... times each row's pixel area (depends on latitude) ...
            row_lat = t.f + (r0 + np.arange(ph) + 0.5) * t.e
            per_row = per_row * pixel_area_m2(dlon, dlat, row_lat)[:, None]
            # ... summed over the rows of each block.
            area = per_row.reshape(ph // block_px, block_px, nbx).sum(axis=1)
            bi, bj = np.nonzero(area > 0)
            out_a.append(area[bi, bj])
            out_x.append(t.c + (bj * block_px + block_px / 2) * t.a)
            out_y.append(t.f + (r0 + bi * block_px + block_px / 2) * t.e)
    if not out_a:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(out_x), np.concatenate(out_y), np.concatenate(out_a)


def write_blocks(path, lng, lat, area):
    """GeoParquet 1.1 points with a covering bbox column (lon/lat)."""
    geom = shapely.to_wkb(shapely.points(lng, lat)) if len(lng) else np.array([], dtype=object)
    bbox = pa.StructArray.from_arrays(
        [pa.array(lng, pa.float64()), pa.array(lat, pa.float64()),
         pa.array(lng, pa.float64()), pa.array(lat, pa.float64())],
        names=["xmin", "ymin", "xmax", "ymax"])
    table = pa.table({"geometry": pa.array(list(geom), pa.binary()),
                      "area_m2": pa.array(area, pa.float64()),
                      "bbox": bbox})
    geo = {"version": "1.1.0", "primary_column": "geometry",
           "columns": {"geometry": {
               "encoding": "WKB", "geometry_types": ["Point"],
               "covering": {"bbox": {k: ["bbox", k] for k in ("xmin", "ymin", "xmax", "ymax")}}}}}
    if len(lng):
        geo["columns"]["geometry"]["bbox"] = [float(lng.min()), float(lat.min()),
                                              float(lng.max()), float(lat.max())]
    # No "crs" key = OGC:CRS84, i.e. lon/lat WGS84 (GeoParquet default).
    table = table.replace_schema_metadata({"geo": json.dumps(geo)})
    pq.write_table(table, path, compression="zstd", row_group_size=100_000)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# One tile, start to finish (runs in a worker process)
# ---------------------------------------------------------------------------
def process_tile(code, out_dir, tif_dir, local_tifs, block_px, keep_tif):
    out = out_dir / f"{code}.parquet"
    if out.exists() and Path(f"{out}.sha256").exists():
        return code, "skipped (already done)", 0
    t0 = time.time()

    downloaded = False
    tif = local_tifs.get(code)
    if tif is None:
        tif = tif_dir / tile_file(code)
        if not tif.exists():
            download(code, tif)
            downloaded = True

    lng, lat, area = cropland_blocks(tif, block_px)
    partial = out.with_name(out.name + ".partial")      # not *.parquet: never read half-done
    write_blocks(partial, lng, lat, area)
    problems = check_file(partial)
    if problems:
        raise RuntimeError(f"{code}: validation failed: {problems}")
    partial.replace(out)
    Path(f"{out}.sha256").write_text(f"{sha256_file(out)}  {out.name}\n")

    if (downloaded or tif.parent == tif_dir) and not keep_tif:
        tif.unlink()                                     # only ever our own downloads
    return code, f"{len(area):,} blocks, {area.sum() / 1e6:,.1f} km2 cropland", time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default="/Users/maugh24/Flood_Impact_Model/Files",
                    help="outputs go to WORK_DIR/Farmland, downloads to WORK_DIR/Farmland_download")
    ap.add_argument("--workers", type=int, default=4, help="tiles processed at once (~2 GB RAM each)")
    ap.add_argument("--block-px", type=int, default=15, help="block size in 10 m pixels (15 = ~150 m)")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                    help="only tiles overlapping this lon/lat box (e.g. for a test)")
    ap.add_argument("--tiles", nargs="+", help="only these tile codes, e.g. N21W081")
    ap.add_argument("--local-tifs", help="folder with ESA tiles already on disk (searched recursively); "
                                         "these are used instead of downloading and are never deleted")
    ap.add_argument("--keep-tif", action="store_true", help="keep downloaded tiles")
    ap.add_argument("--refresh-list", action="store_true", help="re-read the tile list from ESA")
    args = ap.parse_args()

    work = Path(args.work_dir)
    out_dir, tif_dir = work / "Farmland", work / "Farmland_download"
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_dir.mkdir(parents=True, exist_ok=True)

    local = {}
    if args.local_tifs:
        for p in Path(args.local_tifs).rglob("*.tif"):
            m = TILE_RE.search(p.name)
            if m and not p.name.startswith("."):
                local[m.group(1)] = p
        print(f"{len(local)} tile(s) found locally in {args.local_tifs}")

    if args.tiles:
        codes = sorted(args.tiles)
    elif args.local_tifs and not args.bbox:
        codes = sorted(local)              # just the local tiles, no download needed
    else:
        codes = list_tiles(out_dir / "_tile_list.txt", args.refresh_list)
    if args.bbox:
        codes = [c for c in codes if not (tile_bounds(c)[2] <= args.bbox[0] or tile_bounds(c)[0] >= args.bbox[2]
                                          or tile_bounds(c)[3] <= args.bbox[1] or tile_bounds(c)[1] >= args.bbox[3])]
    print(f"Farmland prep -> {out_dir}")
    print(f"{len(codes)} tile(s), block {args.block_px} px, {args.workers} worker(s)")

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_tile, c, out_dir, tif_dir, local, args.block_px, args.keep_tif): c
                for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            code = futs[f]
            try:
                _, msg, secs = f.result()
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(codes)} {code}: {msg}"
                      + (f" ({secs:.0f}s)" if secs else ""), flush=True)
            except Exception as e:
                failed.append(code)
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(codes)} {code}: FAILED - {e}", flush=True)

    if failed:
        print(f"\n{len(failed)} tile(s) failed: {' '.join(failed)}\nRerun to retry them.")
        return 1
    print("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
