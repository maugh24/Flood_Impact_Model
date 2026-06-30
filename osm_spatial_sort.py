"""
osm_spatial_sort.py

One-time preprocessing: rewrite the OSM continent parquet files so that rows
are physically ordered along a Z-order (Morton) space-filling curve, with small
row groups. This makes the existing GeoParquet covering-bbox column actually
*prune* row groups during a bbox query, instead of forcing a full-file scan.

Why this is needed
------------------
The OSM files already carry a valid covering bbox column, but the rows are in
OSM-id order, so every row group's bbox spans nearly the whole continent. A
small VPU bbox therefore overlaps every row group and PyArrow can skip none of
them - it reads and decodes the entire file every time. After this rewrite,
each row group covers a small, compact area, so a VPU query touches only the
handful of row groups near it.

What it does, per file
----------------------
  1. Reads the GeoParquet 'geo' footer metadata (to find the covering bbox
     column - handles both 'bbox' and 'geometry_bbox' naming - and the file's
     overall bounds).
  2. Uses DuckDB to ORDER BY a Morton key built from the bbox-center coords
     (computed purely from the cheap bbox struct columns - the heavy polygon
     geometry is never decoded for sorting). DuckDB spills to disk, so files
     far larger than RAM (Asia ~28 GB) sort fine.
  3. Writes a new parquet with small row groups and the SAME 'geo' metadata
     re-attached via KV_METADATA, so your impact code reads it with no change.
  4. Verifies row count and reports the row-group bbox tightening.

Non-destructive: originals are left untouched; output goes to a sibling folder.
Resumable: a file already present (and complete) in the output folder is
skipped.

Requirements: duckdb, pyarrow  (both already in the hydroinformatics env;
                                 `conda install -c conda-forge duckdb` if not)

No DuckDB extensions are required - the Morton key is plain integer bit-ops.
"""

import json
import os
import sys
import glob
import time

import duckdb
import pyarrow.parquet as pq


# ----------------------------------------------------------------------------
# CONFIG - edit these paths/values for your machine.
# ----------------------------------------------------------------------------
SRC_DIR = "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/OSM/Parquet"
DST_DIR = "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/OSM/Parquet_sorted"

# Rows per row group in the output. Smaller -> tighter per-group bbox -> better
# pruning, at the cost of slightly more metadata. 50k is a good balance for
# dense building data.
ROW_GROUP_SIZE = 50_000

# DuckDB resource limits. memory_limit caps RAM; anything beyond spills to
# TEMP_DIR. Put TEMP_DIR on the big external drive (not the system disk) so the
# 28 GB sorts have room to spill.
MEMORY_LIMIT = "8GB"
THREADS = 4
TEMP_DIR = os.path.join(DST_DIR, ".duckdb_tmp")

# Output compression. zstd gives a good size/speed tradeoff.
COMPRESSION = "zstd"
# ----------------------------------------------------------------------------


def _sql_str(s: str) -> str:
    """Escape a Python string for use as a single-quoted SQL literal."""
    return s.replace("'", "''")


def _spread16(expr: str) -> str:
    """SQL expression interleaving the low 16 bits of `expr` with zero bits
    (the per-axis half of a 32-bit Morton code), via the classic magic-mask
    method. `expr` must evaluate to an integer."""
    x = f"({expr} & 65535)"
    for shift, mask in [(8, 16711935), (4, 252645135), (2, 858993459), (1, 1431655765)]:
        x = f"(({x} | ({x} << {shift})) & {mask})"
    return x


def _quant(center_expr: str, lo: float, hi: float) -> str:
    """SQL expression quantizing a coordinate to a 0..65535 integer bucket.
    Constants are parenthesized so negative bounds don't produce a literal
    '--' (which SQL reads as a line comment)."""
    return (f"CAST(least(65535, greatest(0, "
            f"floor(({center_expr} - ({lo})) / (({hi}) - ({lo})) * 65535.0))) AS BIGINT)")


def _morton_expr(xcol, Xcol, ycol, Ycol, bounds):
    """Build the Morton(Z-order) ORDER BY expression from the covering bbox
    columns and the file's (minx, miny, maxx, maxy) bounds."""
    minx, miny, maxx, maxy = bounds
    qx = _quant(f"(({xcol} + {Xcol}) / 2.0)", minx, maxx)
    qy = _quant(f"(({ycol} + {Ycol}) / 2.0)", miny, maxy)
    return f"({_spread16(qx)} | ({_spread16(qy)} << 1))"


def _read_geo(src):
    """Return (geo_json_str, geo_dict) from a parquet footer, or (None, None)."""
    kv = pq.ParquetFile(src).metadata.metadata or {}
    if b"geo" not in kv:
        return None, None
    s = kv[b"geo"].decode()
    return s, json.loads(s)


def _covering_cols(geo):
    """Return dotted paths (xmin, ymin, xmax, ymax) of the covering bbox
    column, or None if the file declares no covering metadata."""
    pc = geo.get("primary_column", "geometry")
    cov = geo.get("columns", {}).get(pc, {}).get("covering", {}).get("bbox")
    if not cov:
        return None
    return (".".join(cov["xmin"]), ".".join(cov["ymin"]),
            ".".join(cov["xmax"]), ".".join(cov["ymax"]))


def _file_bounds(geo, con, src, cols):
    """File (minx, miny, maxx, maxy): prefer the declared geo bbox, else read
    min/max of the covering columns (cheap - uses column statistics)."""
    pc = geo.get("primary_column", "geometry")
    bb = geo.get("columns", {}).get(pc, {}).get("bbox")
    if bb and len(bb) == 4:
        return bb
    xcol, ycol, Xcol, Ycol = cols
    row = con.execute(
        f"SELECT min({xcol}), min({ycol}), max({Xcol}), max({Ycol}) "
        f"FROM read_parquet('{_sql_str(src)}')"
    ).fetchone()
    return list(row)


def _rg_widths(path):
    """Median (x-width, y-width) of the per-row-group covering bbox, in degrees.
    A small number means the row groups are spatially compact (pruning works)."""
    md = pq.ParquetFile(path).metadata
    paths = [md.row_group(0).column(i).path_in_schema for i in range(md.row_group(0).num_columns)]

    def find(suffix):
        return next(i for i, n in enumerate(paths) if n.endswith(suffix))

    ix, iX = find("xmin"), find("xmax")
    iy, iY = find("ymin"), find("ymax")
    if md.row_group(0).column(ix).statistics is None:
        return None, None
    xs, ys = [], []
    for r in range(md.num_row_groups):
        rg = md.row_group(r)
        xs.append(rg.column(iX).statistics.max - rg.column(ix).statistics.min)
        ys.append(rg.column(iY).statistics.max - rg.column(iy).statistics.min)
    xs.sort(); ys.sort()
    return xs[len(xs) // 2], ys[len(ys) // 2]


def sort_one(con, src, dst):
    geo_str, geo = _read_geo(src)
    if geo is None:
        print(f"  SKIP (no geo metadata): {os.path.basename(src)}")
        return False

    cov = _covering_cols(geo)
    if cov is None:
        print(f"  SKIP (no covering bbox column): {os.path.basename(src)}")
        return False
    xcol, ycol, Xcol, Ycol = cov

    bounds = _file_bounds(geo, con, src, cov)
    morton = _morton_expr(xcol, Xcol, ycol, Ycol, bounds)

    n_src = pq.ParquetFile(src).metadata.num_rows
    xw0, yw0 = _rg_widths(src)

    t0 = time.time()
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{_sql_str(src)}') ORDER BY {morton}) "
        f"TO '{_sql_str(dst)}' "
        f"(FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE}, "
        f"COMPRESSION '{COMPRESSION}', KV_METADATA {{geo: '{_sql_str(geo_str)}'}});"
    )
    dt = time.time() - t0

    # ---- verify ----
    md = pq.ParquetFile(dst).metadata
    n_dst = md.num_rows
    has_geo = b"geo" in (md.metadata or {})
    xw1, yw1 = _rg_widths(dst)
    ok = (n_dst == n_src) and has_geo

    print(f"  rows {n_src:,} -> {n_dst:,}  geo={'ok' if has_geo else 'MISSING'}  "
          f"row_groups -> {md.num_row_groups}  ({dt:.0f}s)")
    if xw0 and xw1:
        print(f"  median row-group bbox: {xw0:.1f}x{yw0:.1f} deg  ->  {xw1:.2f}x{yw1:.2f} deg")
    if not ok:
        print(f"  !! VERIFY FAILED for {os.path.basename(src)} - leaving it; do not use.")
    return ok


def main():
    if not os.path.isdir(SRC_DIR):
        sys.exit(f"SRC_DIR not found: {SRC_DIR}")
    os.makedirs(DST_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}';")
    con.execute(f"SET temp_directory='{_sql_str(TEMP_DIR)}';")
    con.execute("SET preserve_insertion_order=false;")  # let the sort stream/spill freely

    def _set_threads(n):
        con.execute(f"SET threads={n};")

    _set_threads(THREADS)

    # Smallest files first: fast feedback and early error detection.
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.parquet")), key=os.path.getsize)
    print(f"Found {len(files)} parquet files in {SRC_DIR}")
    print(f"Writing sorted copies to {DST_DIR}  (row_group_size={ROW_GROUP_SIZE})\n")

    done = skipped = failed = 0
    for src in files:
        name = os.path.basename(src)
        dst = os.path.join(DST_DIR, name)
        # Resume: skip if a complete copy already exists.
        if os.path.exists(dst):
            try:
                if pq.ParquetFile(dst).metadata.num_rows == pq.ParquetFile(src).metadata.num_rows:
                    print(f"[skip] {name} (already done)")
                    skipped += 1
                    continue
            except Exception:
                pass  # partial/corrupt -> rewrite
        print(f"[sort] {name}  ({os.path.getsize(src)/1e9:.1f} GB)")
        # Try at the configured thread count; on an out-of-memory error, retry
        # with fewer threads. Each thread pins buffers that can't be spilled, so
        # dropping to 2 then 1 thread shrinks the un-spillable memory and lets
        # large sorts finish (slower, but it completes) without needing more RAM.
        thread_ladder = sorted({t for t in (THREADS, 2, 1) if t <= THREADS}, reverse=True)
        succeeded = False
        for attempt, nthreads in enumerate(thread_ladder):
            if attempt > 0:
                print(f"  out of memory - retrying with threads={nthreads}")
            _set_threads(nthreads)
            try:
                succeeded = sort_one(con, src, dst)
                break
            except Exception as e:
                msg = str(e)
                if "Out of Memory" in msg or "failed to pin" in msg:
                    if attempt < len(thread_ladder) - 1:
                        continue  # step down the ladder
                    print(f"  ERROR: still out of memory at threads=1. "
                          f"Raise MEMORY_LIMIT if your Mac has spare RAM, then rerun.")
                else:
                    print(f"  ERROR on {name}: {e}")
                break
        _set_threads(THREADS)  # restore for the next file
        done += succeeded
        failed += (not succeeded)
        print()

    print("=" * 60)
    print(f"done={done}  skipped={skipped}  failed={failed}")
    print(f"Sorted files in: {DST_DIR}")
    print("Next: point building_source / transportation_source at DST_DIR")
    print("(or swap it in for the original Parquet folder once you've spot-checked it).")


if __name__ == "__main__":
    main()
