"""
population_prep.py - turn the WorldPop global 1 km population raster into the
population file the Flood Impact Model reads. Usually started through
population_prep.sh.

  1. download   WorldPop's global mosaic for --year (R2025A release,
                constrained, UN-adjusted, 1 km) into <work-dir>/Population/,
                unless it is already there
  2. convert    one point per populated pixel, at the pixel centre, carrying
                the pixel's population (pop_value)
  3. validate   read the new file back completely and check its total
                population equals the raster's
  4. checksum   write <file>.sha256
  5. clean up   delete the raster if this script downloaded it (--keep-tif keeps it)

Output: <work-dir>/Population/worldpop_<year>_1km.parquet
Columns: geometry (pixel-centre point), pop_value, bbox (covering).

The intersection run assigns population by pixel centre, so points give the
same results as the old pixel polygons (population_raster_to_parquet.py), in
a smaller file. Safe to rerun: a finished year is skipped.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio.windows import Window

# Name sent with every download (servers can refuse generic client names).
USER_AGENT = "FloodImpactModel/1.0"


def _request(url):
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

# WorldPop "Global2" R2025A mosaics. {year} is filled in; override with --url
# if WorldPop moves the file.
TIF_NAME = "global_pop_{year}_CN_1km_R2025A_UA_v1.tif"
URL_TEMPLATE = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/"
                "{year}/0_Mosaicked/v1/1km_ua/constrained/" + TIF_NAME)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "OSM"))
from validate_parquet import check_file          # noqa: E402  (shared with the OSM prep)


def download(url, dest, tries=4):
    part = dest.with_name(dest.name + ".partial")
    for attempt in range(1, tries + 1):
        try:
            print(f"downloading {url}", flush=True)
            with urllib.request.urlopen(_request(url), timeout=120) as r, open(part, "wb") as f:
                expected = int(r.headers.get("Content-Length", -1))
                got, last = 0, time.time()
                while chunk := r.read(8 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if time.time() - last > 30:
                        print(f"  {got / 1e6:,.0f} / {expected / 1e6:,.0f} MB", flush=True)
                        last = time.time()
            if expected >= 0 and part.stat().st_size != expected:
                raise IOError(f"got {part.stat().st_size} of {expected} bytes")
            part.replace(dest)
            return
        except Exception as e:
            if attempt == tries:
                raise SystemExit(
                    f"\nDownload failed: {e}\n"
                    f"Download the file yourself (WorldPop hub, global mosaic, 1 km, constrained,\n"
                    f"UN-adjusted, R2025A) and put it at:\n  {dest}\nthen rerun, or pass --url.")
            print(f"  attempt {attempt} failed ({e}); retrying", flush=True)
            time.sleep(15 * attempt)


def convert(tif, out, strip_rows=1024):
    """Write populated pixels as points. Returns (rows, raster total)."""
    geo = {"version": "1.1.0", "primary_column": "geometry",
           "columns": {"geometry": {
               "encoding": "WKB", "geometry_types": ["Point"],
               "covering": {"bbox": {k: ["bbox", k] for k in ("xmin", "ymin", "xmax", "ymax")}}}}}
    schema = pa.schema([("geometry", pa.binary()), ("pop_value", pa.float64()),
                        ("bbox", pa.struct([(k, pa.float64()) for k in ("xmin", "ymin", "xmax", "ymax")]))],
                       metadata={"geo": json.dumps(geo)})   # no "crs" key = lon/lat WGS84
    rows, total = 0, 0.0
    with rasterio.open(tif) as src, pq.ParquetWriter(out, schema, compression="zstd") as w:
        t = src.transform
        if src.crs is None or not src.crs.is_geographic or t.b != 0 or t.d != 0:
            raise SystemExit(f"{tif}: expected a north-up lon/lat raster")
        nodata = src.nodata
        for r0 in range(0, src.height, strip_rows):
            h = min(strip_rows, src.height - r0)
            v = src.read(1, window=Window(0, r0, src.width, h)).astype(np.float64)
            ok = np.isfinite(v) & (v > 0)
            if nodata is not None:
                ok &= v != nodata
            r, c = np.nonzero(ok)
            if len(r) == 0:
                continue
            pop = v[r, c]
            lng = t.c + (c + 0.5) * t.a
            lat = t.f + (r0 + r + 0.5) * t.e
            bbox = pa.StructArray.from_arrays([pa.array(lng), pa.array(lat), pa.array(lng), pa.array(lat)],
                                              names=["xmin", "ymin", "xmax", "ymax"])
            w.write_table(pa.table({"geometry": pa.array(list(shapely.to_wkb(shapely.points(lng, lat))), pa.binary()),
                                    "pop_value": pa.array(pop), "bbox": bbox}, schema=schema),
                          row_group_size=1_000_000)
            rows += len(pop)
            total += pop.sum()
    return rows, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2024, help="WorldPop year (default 2024)")
    ap.add_argument("--work-dir", default=str(Path.home() / "Flood_Impact_Model" / "Files"),
                    help="raster and output go in WORK_DIR/Population")
    ap.add_argument("--url", help="download from this URL instead of the WorldPop default")
    ap.add_argument("--tif", help="use this raster file instead of downloading (never deleted)")
    ap.add_argument("--keep-tif", action="store_true", help="keep the downloaded raster")
    args = ap.parse_args()

    pop_dir = Path(args.work_dir) / "Population"
    pop_dir.mkdir(parents=True, exist_ok=True)
    out = pop_dir / f"worldpop_{args.year}_1km.parquet"
    sha = Path(f"{out}.sha256")
    print(f"Population prep -> {out}")
    if out.exists() and sha.exists():
        print("already done, skipping")
        return 0

    downloaded = False
    if args.tif:
        tif = Path(args.tif)
        if not tif.is_file():
            raise SystemExit(f"File not found: {tif}")
    else:
        tif = pop_dir / TIF_NAME.format(year=args.year)
        if tif.is_file():
            print(f"using existing {tif}")
        else:
            download(args.url or URL_TEMPLATE.format(year=args.year), tif)
            downloaded = True

    t0 = time.time()
    partial = out.with_name(out.name + ".partial")       # not *.parquet: never read half-done
    rows, raster_total = convert(tif, partial)
    problems = check_file(partial)
    file_total = pq.read_table(partial, columns=["pop_value"]).column(0).to_numpy().sum()
    if not np.isclose(file_total, raster_total, rtol=1e-9):
        problems.append(f"total {file_total:,.0f} != raster total {raster_total:,.0f}")
    if problems:
        raise SystemExit(f"validation failed: {problems}")
    partial.replace(out)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    sha.write_text(f"{h.hexdigest()}  {out.name}\n")
    print(f"wrote {rows:,} populated pixels, total population {raster_total:,.0f}, "
          f"{out.stat().st_size / 1e6:,.0f} MB in {time.time() - t0:,.0f}s")

    if downloaded and not args.keep_tif:
        tif.unlink()
        print(f"deleted {tif.name} (use --keep-tif to keep it)")
    print("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
