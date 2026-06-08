"""
Simplify_Cropland.py

Convert ONE large cropland parquet into a simplified, area-filtered
EPSG:4326 parquet with a per-row covering bbox so downstream readers
(QGIS, geopandas, tile_loader.read_tiles_in_bbox) can do row-group
pushdown on bbox filters.

Per-batch streamed pipeline:
  1. Read a row group's WKB geometries in the source CRS
     (ESA cropland tiles are stored in an equal-area meter projection
     like ESRI:54034).
  2. Simplify with TOLERANCE in source-CRS units (meters).
  3. Drop polygons whose area is below MIN_AREA m^2.
  4. Reproject survivors to EPSG:4326.
  5. Append to the output parquet as a row group, carrying a per-row
     bbox column for the covering metadata.

Why simplify + filter BEFORE reprojecting:
  - The source is an equal-area projection - area in m^2 is physically
    meaningful and constant across the tile.
  - EPSG:4326 is degrees - area in 4326 has no physical meaning and
    varies with latitude by orders of magnitude. Filtering by "area
    >= 500" in 4326 would mean very different things at the equator
    vs near the poles.
  - Simplify tolerance has the same issue: 10 m in a CEA projection
    means 10 m everywhere; ~0.0001 degrees in 4326 means different
    ground distances at different latitudes.

Memory:
  - Peak is bounded by BATCH_SIZE rows in flight (sequential) or
    BATCH_SIZE * 2 * N_WORKERS (threaded).
  - Threading uses Shapely 2.x GIL-releasing ops, so no pickling tax.
"""
import json
import math
import os
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ---- Paths (single file, single file out) ----
INPUT_PATH  = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\n30e000cropland.parquet"
OUTPUT_PATH = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\bbox\n30e000cropland_bbox.parquet"

# ---- Simplification parameters ----
# Both thresholds are interpreted in the SOURCE CRS (the cropland tiles'
# native equal-area meter projection). Iterative values - tune as needed.
TOLERANCE   = 10       # max distance (m) a vertex may move during simplification
MIN_AREA    = 500      # drop polygons smaller than this (m^2)

# ---- Output ----
OUTPUT_CRS  = 'EPSG:4326'
COMPRESSION = 'brotli'

# ---- Streaming / concurrency ----
# Each "batch" is BATCH_SIZE rows from the input read via pyarrow.
# Lower BATCH_SIZE for less peak memory per batch.
BATCH_SIZE  = 50000
# 1 = fully sequential. Raise to use a bounded ThreadPoolExecutor over
# batches. Memory peak scales with N_WORKERS - keep it conservative.
N_WORKERS   = 18


# --------- GeoParquet metadata helper ---------

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


# --------- per-batch worker ---------

def _process_batch(wkb_arr, src_crs, dst_crs):
    """Pure CPU work, no I/O, no shared state.

    Deserialize WKB -> simplify (in source CRS) -> area filter (m^2 in
    source CRS) -> reproject to dst_crs -> serialize back to WKB +
    return per-row bounds (in dst_crs units, i.e. degrees for 4326).

    Returns (wkb_bytes_list, bounds_df) or None if nothing survives.
    The heavy ops (from_wkb, simplify, area, to_crs, bounds) release the
    GIL in Shapely 2.x, so this is safe to run in multiple threads.
    """
    geoms = gpd.GeoSeries(from_wkb(wkb_arr), crs=src_crs)
    geoms = geoms.simplify(tolerance=TOLERANCE, preserve_topology=True)

    # Area filter in source CRS (equal-area meters).
    mask = geoms.area >= MIN_AREA
    if not mask.any():
        return None
    geoms = geoms[mask].reset_index(drop=True)
    if len(geoms) == 0:
        return None

    # Reproject to the output CRS. Per-row bounds for the covering bbox
    # column are computed AFTER reprojection so the bbox column units
    # match the output CRS (degrees for 4326).
    geoms = geoms.to_crs(dst_crs)
    bounds = geoms.bounds
    wkb_bytes = [g.wkb for g in geoms]
    return wkb_bytes, bounds


# --------- streamed driver ---------

def simplify_tile(input_path, output_path):
    """Stream-convert one cropland tile.

    Returns (n_in, n_out, mb_in, mb_out) for reduction reporting.
    """
    pf = pq.ParquetFile(input_path)

    # Recover the source CRS from the input GeoParquet metadata so we
    # don't have to hard-code ESRI:54034 here. Different tiles could in
    # principle live in different CRSes - this stays correct.
    file_kv = pf.metadata.metadata or {}
    if b'geo' not in file_kv:
        raise ValueError(f"{input_path}: input lacks GeoParquet metadata")
    in_geo = json.loads(file_kv[b'geo'])
    primary_col   = in_geo.get('primary_column', 'geometry')
    src_crs_dict  = in_geo['columns'][primary_col].get('crs')

    # Output declares both Polygon and MultiPolygon because simplify can
    # disconnect thin polygons into multipart geometries.
    out_geometry_types = ['Polygon', 'MultiPolygon']

    # Build the output CRS dict as JSON-serializable PROJJSON for the
    # GeoParquet metadata block. CRS.from_user_input(OUTPUT_CRS) handles
    # 'EPSG:4326' / dict / WKT uniformly.
    from pyproj import CRS as _CRS
    out_crs_dict = json.loads(_CRS.from_user_input(OUTPUT_CRS).to_json())

    n_in  = pf.metadata.num_rows
    n_out_counter = [0]                          # boxed for nested fn
    gx = [math.inf, math.inf, -math.inf, -math.inf]  # global bbox accum
    writer = [None]                              # boxed for nested fn

    n_batches_expected = max(1, math.ceil(n_in / BATCH_SIZE))

    def _write_result(result):
        """Main-thread-only. Append one processed batch as a row group
        and update the running global bbox + count."""
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
        batch_iter = pf.iter_batches(batch_size=BATCH_SIZE, columns=[primary_col])
        desc = f"  {os.path.basename(input_path):<28}"

        if N_WORKERS <= 1:
            # ---- Sequential ----
            for batch in tqdm(batch_iter, total=n_batches_expected,
                              desc=desc, unit="batch"):
                wkb_arr = batch.column(primary_col).to_pylist()
                if not wkb_arr:
                    continue
                _write_result(_process_batch(wkb_arr, src_crs_dict, OUTPUT_CRS))
        else:
            # ---- Bounded thread pool ----
            max_in_flight = N_WORKERS * 2
            in_flight = set()
            pbar = tqdm(total=n_batches_expected, desc=desc, unit="batch")
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                exhausted = False
                while True:
                    while not exhausted and len(in_flight) < max_in_flight:
                        try:
                            batch = next(batch_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        wkb_arr = batch.column(primary_col).to_pylist()
                        if not wkb_arr:
                            continue
                        in_flight.add(
                            pool.submit(_process_batch, wkb_arr, src_crs_dict, OUTPUT_CRS)
                        )
                    if not in_flight:
                        break
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
        # Attach GeoParquet metadata with OUTPUT CRS (EPSG:4326) and the
        # global bbox we accumulated in 4326 degrees.
        geo_md = _build_geo_metadata(
            primary_col, out_crs_dict, out_geometry_types, tuple(gx)
        )
        writer[0].add_key_value_metadata({'geo': json.dumps(geo_md)})
        writer[0].close()

    mb_in  = os.path.getsize(input_path)  / (1024 * 1024)
    mb_out = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0
    return n_in, n_out_counter[0], mb_in, mb_out


# --------- main ---------

def main():
    in_path  = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)

    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:       {in_path}")
    print(f"Output:      {out_path}")
    print(f"Tolerance:   {TOLERANCE} m (source CRS)   "
          f"Min area: {MIN_AREA} m^2 (source CRS)")
    print(f"Output CRS:  {OUTPUT_CRS}   "
          f"Compression: {COMPRESSION}   "
          f"Batch: {BATCH_SIZE:,} rows   Workers: {N_WORKERS}")
    print("=" * 78)

    t0 = time.time()
    try:
        n_in, n_out, mb_in, mb_out = simplify_tile(str(in_path), str(out_path))
    except Exception as e:
        print(f"FAILED: {e}")
        try: os.remove(out_path)
        except OSError: pass
        return

    elapsed = time.time() - t0
    poly_drop = 100.0 * (1 - n_out / max(n_in, 1))
    size_drop = 100.0 * (1 - mb_out / max(mb_in, 1e-9))
    print("=" * 78)
    print(
        f"{n_in:,} -> {n_out:,} polys ({poly_drop:.1f}% dropped) | "
        f"{mb_in:.1f} MB -> {mb_out:.1f} MB ({size_drop:.1f}% smaller) | "
        f"{elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
