"""
Simplify_Cropland.py

Preprocessing pass that walks every ESA cropland tile in INPUT_FOLDER,
simplifies polygon geometries and drops tiny noise polygons, then writes
each tile to OUTPUT_FOLDER with brotli compression and a per-row covering
bbox column.

ROW-GROUP STREAMING
-------------------
Each tile is processed one row group at a time:
  - Read a single row group of WKB geometry via pyarrow.iter_batches.
  - Deserialize WKB -> shapely, simplify, area-filter.
  - Serialize the survivors back to WKB and write them straight out as
    a new row group in the output parquet via ParquetWriter.
  - Track the running global covering bbox as we go.
  - On close, attach the GeoParquet 1.1 file metadata, including the
    global covering bbox and the per-row bbox-column pointer.

Peak memory per process is bounded by ONE row group's worth of geometry
(controlled by BATCH_SIZE), not the whole tile. The simplified output
is written incrementally so it never sits all in memory at once either.

Other behavior matches the previous version:
  - Tile-at-a-time across the folder, no inner parallelism.
  - Tiles whose output already exists are skipped (resumable).
  - Failed tiles have their partial output cleaned up.
"""
import glob
import json
import math
import os
import time
import warnings
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ---- Paths ----
INPUT_FOLDER  = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet"
OUTPUT_FOLDER = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\bbox"

# ---- Simplification parameters (CEA meters - cropland tiles were saved
# in cylindrical equal-area by cropland_raster_to_parquet.py) ----
TOLERANCE   = 10       # max distance (m) a vertex may move during simplification
MIN_AREA    = 500      # drop polygons smaller than this (m^2)
COMPRESSION = 'brotli'

# Row-group streaming knob. Lower this if you're still hitting memory
# pressure; raise it if you have RAM to spare and want to amortize the
# per-batch overhead. Each "batch" is one pyarrow row group's worth of
# geometries deserialized to shapely + processed.
BATCH_SIZE  = 50000

# Thread-pool knob. 1 means fully sequential (no pool). Raise to use a
# bounded ThreadPoolExecutor over batches within a single tile.
#
# Threads, not processes: Shapely 2.x releases the GIL for from_wkb,
# simplify, area and bounds, so threads get real CPU parallelism without
# the pickling overhead a multiprocessing pool would impose on millions
# of shapely objects.
#
# Memory: with N workers, peak per-tile in-flight memory scales with
# (N+1) batches because we keep at most ~2*N futures queued at any time.
# Bump this carefully - 2 to 4 is the usual sweet spot. ParquetWriter is
# called from the MAIN thread only, so output integrity doesn't depend
# on worker count.
N_WORKERS   = 1


# --------- streaming simplifier ---------

def _build_geo_metadata(primary_col, crs_dict, geometry_types, global_bbox):
    """Build a GeoParquet 1.1 'geo' file-metadata block.

    Includes both the global covering bbox (for whole-file skip checks)
    and the per-row 'bbox' column pointer (for row-group pushdown).
    """
    return {
        'version': '1.1.0',
        'primary_column': primary_col,
        'columns': {
            primary_col: {
                'encoding': 'WKB',
                'geometry_types': geometry_types,
                'crs': crs_dict,
                'bbox': [float(v) for v in global_bbox],
                'covering': {
                    'bbox': {
                        'xmin': ['bbox', 'xmin'],
                        'ymin': ['bbox', 'ymin'],
                        'xmax': ['bbox', 'xmax'],
                        'ymax': ['bbox', 'ymax'],
                    }
                }
            }
        }
    }


def _process_batch_wkb(wkb_arr, crs_dict):
    """Worker function: WKB-bytes -> simplified WKB-bytes + bounds frame.

    Pure CPU work, no I/O, no global state. Returns None when no polygons
    survive the area filter. The heavy ops (from_wkb, simplify, area,
    bounds) release the GIL in Shapely 2.x, so this is safe to run in
    multiple threads concurrently.
    """
    geoms = gpd.GeoSeries(from_wkb(wkb_arr), crs=crs_dict)
    geoms = geoms.simplify(tolerance=TOLERANCE, preserve_topology=True)

    mask = geoms.area >= MIN_AREA
    if not mask.any():
        return None
    geoms = geoms[mask].reset_index(drop=True)
    if len(geoms) == 0:
        return None

    bounds = geoms.bounds  # DataFrame with minx, miny, maxx, maxy
    wkb_bytes = [g.wkb for g in geoms]
    return wkb_bytes, bounds


def simplify_tile_streaming(input_path, output_path):
    """Streamed simplify+filter for one tile.

    Sequential when N_WORKERS<=1; otherwise uses a bounded ThreadPoolExecutor
    over batches with at most ~2*N_WORKERS futures in flight.

    Returns (n_in, n_out, mb_in, mb_out) for reduction reporting.
    """
    pf = pq.ParquetFile(input_path)

    # Pull GeoParquet metadata from the INPUT so we can preserve CRS and
    # discover the primary geometry column name.
    file_kv = pf.metadata.metadata or {}
    if b'geo' not in file_kv:
        raise ValueError(f"{input_path}: input lacks GeoParquet metadata")
    in_geo = json.loads(file_kv[b'geo'])
    primary_col   = in_geo.get('primary_column', 'geometry')
    in_col_meta   = in_geo['columns'][primary_col]
    crs_dict      = in_col_meta.get('crs')
    # Be liberal about what we declare we may emit: simplify can introduce
    # MultiPolygons from polygons that become disconnected.
    geometry_types = ['Polygon', 'MultiPolygon']

    n_in  = pf.metadata.num_rows
    n_out_counter = [0]  # boxed so the nested writer fn can mutate

    # Global bbox accumulator (in input CRS units).
    gx = [math.inf, math.inf, -math.inf, -math.inf]  # [xmin, ymin, xmax, ymax]

    writer = [None]  # boxed for nested function

    # Total batches for the inner progress bar (last batch may be partial).
    n_batches_expected = max(1, math.ceil(n_in / BATCH_SIZE))

    def _write_result(result):
        """Called from main thread only. Writes one processed batch to the
        output parquet and updates the running global bbox + count."""
        if result is None:
            return
        wkb_bytes, bounds = result
        n_out_counter[0] += len(wkb_bytes)

        gx[0] = min(gx[0], bounds.minx.min())
        gx[1] = min(gx[1], bounds.miny.min())
        gx[2] = max(gx[2], bounds.maxx.max())
        gx[3] = max(gx[3], bounds.maxy.max())

        geom_pa = pa.array(wkb_bytes, type=pa.binary())
        bbox_pa = pa.StructArray.from_arrays(
            [
                pa.array(bounds.minx.values, type=pa.float64()),
                pa.array(bounds.miny.values, type=pa.float64()),
                pa.array(bounds.maxx.values, type=pa.float64()),
                pa.array(bounds.maxy.values, type=pa.float64()),
            ],
            names=['xmin', 'ymin', 'xmax', 'ymax'],
        )
        chunk_table = pa.table({primary_col: geom_pa, 'bbox': bbox_pa})
        if writer[0] is None:
            writer[0] = pq.ParquetWriter(
                output_path, chunk_table.schema, compression=COMPRESSION
            )
        writer[0].write_table(chunk_table)

    try:
        # Read only the geometry column. The cropland tiles are
        # geometry-only; if you ever point this at a tile with attribute
        # columns, the attributes will be dropped - simplified output is
        # geometry only.
        batch_iter = pf.iter_batches(batch_size=BATCH_SIZE, columns=[primary_col])
        desc = f"  {os.path.basename(input_path):<28}"

        if N_WORKERS <= 1:
            # ---- Sequential path ----
            for batch in tqdm(batch_iter, total=n_batches_expected,
                              desc=desc, unit="batch", leave=False):
                wkb_arr = batch.column(primary_col).to_pylist()
                if not wkb_arr:
                    continue
                _write_result(_process_batch_wkb(wkb_arr, crs_dict))
        else:
            # ---- Threaded path: bounded queue, FIFO submit, finish-as-completed ----
            # At most max_in_flight futures alive at once; the I/O-bound main
            # thread does the writes, workers do the CPU work in parallel.
            max_in_flight = N_WORKERS * 2
            in_flight = set()
            pbar = tqdm(total=n_batches_expected, desc=desc,
                        unit="batch", leave=False)
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                exhausted = False
                while True:
                    # Top up the queue
                    while not exhausted and len(in_flight) < max_in_flight:
                        try:
                            batch = next(batch_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        wkb_arr = batch.column(primary_col).to_pylist()
                        if not wkb_arr:
                            continue
                        in_flight.add(pool.submit(_process_batch_wkb, wkb_arr, crs_dict))

                    if not in_flight:
                        break

                    # Wait for at least one to complete, then write all that did.
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        _write_result(fut.result())
                        pbar.update(1)
            pbar.close()
    except Exception:
        if writer[0] is not None:
            try: writer[0].close()
            except Exception: pass
        raise

    if writer[0] is not None:
        # Attach the GeoParquet 1.1 'geo' metadata. The global bbox we
        # collected lets downstream code do whole-file skip checks; the
        # covering.bbox pointer lets pyarrow do row-group pushdown.
        global_bbox = tuple(gx)
        geo_md = _build_geo_metadata(primary_col, crs_dict, geometry_types, global_bbox)
        writer[0].add_key_value_metadata({'geo': json.dumps(geo_md)})
        writer[0].close()

    mb_in  = os.path.getsize(input_path)  / (1024 * 1024)
    mb_out = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0
    return n_in, n_out_counter[0], mb_in, mb_out


# --------- driver ---------

def main():
    in_folder  = Path(INPUT_FOLDER)
    out_folder = Path(OUTPUT_FOLDER)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Only top-level *.parquet, NOT recursive - don't pick up files we
    # already wrote into bbox/.
    tiles = sorted(glob.glob(str(in_folder / '*.parquet')))
    tiles = [t for t in tiles if Path(t).parent.resolve() != out_folder.resolve()]

    if not tiles:
        print(f"No .parquet tiles found in {in_folder}")
        return

    print(f"Found {len(tiles)} cropland tiles in {in_folder}")
    print(f"Writing simplified output to {out_folder}")
    print(f"Tolerance: {TOLERANCE} m   Min area: {MIN_AREA} m^2   "
          f"Compression: {COMPRESSION}   Batch size: {BATCH_SIZE:,} rows")
    print("=" * 78)

    overall_start = time.time()
    total_in_polys = total_out_polys = 0
    total_in_mb = total_out_mb = 0.0
    n_processed = n_skipped = n_failed = 0

    # Outer bar: ticks once per tile, description shows current filename.
    outer = tqdm(tiles, desc="Tiles", unit="tile")
    for i, input_path in enumerate(outer, 1):
        name = os.path.basename(input_path)
        outer.set_postfix_str(name)
        output_path = str(out_folder / name)

        if os.path.exists(output_path):
            tqdm.write(f"  [{i:>2}/{len(tiles)}] {name} -- already exists, skipping")
            n_skipped += 1
            continue

        tqdm.write(f"  [{i:>2}/{len(tiles)}] {name} -- processing...")
        t0 = time.time()
        try:
            n_in, n_out, mb_in, mb_out = simplify_tile_streaming(input_path, output_path)
        except Exception as e:
            tqdm.write(f"      FAILED: {e}")
            try: os.remove(output_path)
            except OSError: pass
            n_failed += 1
            continue

        elapsed = time.time() - t0
        poly_drop = 100.0 * (1 - n_out / max(n_in, 1))
        size_drop = 100.0 * (1 - mb_out / max(mb_in, 1e-9))
        tqdm.write(
            f"      {n_in:>10,} -> {n_out:>10,} polys ({poly_drop:5.1f}% dropped) | "
            f"{mb_in:7.1f} MB -> {mb_out:7.1f} MB ({size_drop:5.1f}% smaller) | "
            f"{elapsed:.1f}s"
        )
        total_in_polys  += n_in
        total_out_polys += n_out
        total_in_mb     += mb_in
        total_out_mb    += mb_out
        n_processed     += 1
    outer.close()

    elapsed = time.time() - overall_start
    print("=" * 78)
    print(f"Processed: {n_processed}   Skipped (already done): {n_skipped}   Failed: {n_failed}")
    if n_processed:
        poly_drop = 100.0 * (1 - total_out_polys / max(total_in_polys, 1))
        size_drop = 100.0 * (1 - total_out_mb / max(total_in_mb, 1e-9))
        print(
            f"Newly-written tiles: "
            f"{total_in_polys:,} -> {total_out_polys:,} polys ({poly_drop:.1f}% reduction) | "
            f"{total_in_mb:.0f} MB -> {total_out_mb:.0f} MB ({size_drop:.1f}% reduction)"
        )
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
