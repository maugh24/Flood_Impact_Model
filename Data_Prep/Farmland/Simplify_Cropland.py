"""
Simplify_Cropland.py

Convert ONE large cropland parquet into a FOLDER of simplified,
area-filtered EPSG:4326 chunk parquets. Each chunk is a complete
GeoParquet 1.1 with its own per-row covering bbox, so downstream
readers (tile_loader.read_tiles_in_bbox, QGIS, etc.) can do bbox-skip
on a per-chunk basis.

Path A architecture (per-worker chunk writes):
  - Each worker takes a batch (~BATCH_SIZE rows) from the input.
  - Each worker does the full pipeline AND writes its own chunk file.
  - No single-threaded main-thread writer = no compression bottleneck.
  - Output is a folder of chunk_NNNNNN.parquet files.

Per-batch pipeline (inside each worker):
  1. Read WKB geometries in the source CRS.
  2. Simplify with TOLERANCE in source-CRS units (meters).
  3. Drop polygons whose area is below MIN_AREA m^2.
  4. Reproject survivors to EPSG:4326.
  5. Write a chunk parquet with brotli compression, covering bbox
     metadata, and CRS=EPSG:4326.

Spatial coherence of chunks:
  The input was vectorized in row-major raster order, so consecutive
  rows = a horizontal strip of the source tile. Chunk N covers a strip
  at a specific latitude band - which gives downstream bbox queries
  very effective per-chunk skipping.

Resumability:
  Chunks are numbered by their batch index. A chunk that already exists
  on disk is skipped on rerun, so a crashed run can be picked up exactly
  where it left off without re-processing completed chunks.
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


# ---- Paths ----
# INPUT_PATH: a single source parquet (cropland tile in equal-area meters)
# OUTPUT_FOLDER: the destination directory for chunk_NNNNNN.parquet files
INPUT_PATH    = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\n30e000cropland.parquet"
OUTPUT_FOLDER = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\bbox\n30e000cropland_bbox"

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
BATCH_SIZE  = 50000
# Number of worker threads. Each one does its own full process+write,
# including brotli compression. Brotli releases the GIL during compress
# so these run truly in parallel.
#
# Past ~8, returns diminish because disk I/O bandwidth becomes the
# limiting factor on most systems. 6-8 is the sweet spot.
N_WORKERS   = 8


# --------- GeoParquet metadata helper ---------

def _build_geo_metadata(primary_col, crs_dict, geometry_types, global_bbox):
    """Build a GeoParquet 1.1 'geo' file-metadata block.

    Includes both the chunk's covering bbox (for whole-file skip checks)
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


# --------- per-batch worker (process + write) ---------

def _process_and_write_chunk(batch_idx, wkb_arr, src_crs, dst_crs,
                              dst_crs_json, primary_col, chunk_dir):
    """Worker function: process one batch AND write it to its own chunk file.

    Returns a dict with:
      n_in      : input polygon count for this batch
      n_out     : output polygon count (after simplify + area filter)
      chunk_path: Path to the written chunk, or None if no polygons survived

    Each chunk file is a complete GeoParquet 1.1 with its own covering bbox
    metadata so downstream readers can skip non-overlapping chunks cheaply.
    Brotli compression inside ParquetWriter.write_table releases the GIL,
    so multiple workers can compress in true parallel.
    """
    n_in = len(wkb_arr)

    geoms = gpd.GeoSeries(from_wkb(wkb_arr), crs=src_crs)
    geoms = geoms.simplify(tolerance=TOLERANCE, preserve_topology=True)

    # Area filter in source CRS (equal-area meters).
    mask = geoms.area >= MIN_AREA
    if not mask.any():
        return {'n_in': n_in, 'n_out': 0, 'chunk_path': None}
    geoms = geoms[mask].reset_index(drop=True)
    if len(geoms) == 0:
        return {'n_in': n_in, 'n_out': 0, 'chunk_path': None}

    # Reproject AFTER filtering so the bbox column ends up in dst_crs units.
    geoms = geoms.to_crs(dst_crs)
    bounds = geoms.bounds
    wkb_bytes = [g.wkb for g in geoms]

    chunk_path = chunk_dir / f"chunk_{batch_idx:06d}.parquet"

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

    chunk_bbox = (
        float(bounds.minx.min()),
        float(bounds.miny.min()),
        float(bounds.maxx.max()),
        float(bounds.maxy.max()),
    )
    geo_md = _build_geo_metadata(
        primary_col, dst_crs_json, ['Polygon', 'MultiPolygon'], chunk_bbox
    )

    # Each worker manages its own ParquetWriter. open -> single write_table
    # -> attach geo metadata -> close. The brotli compression inside
    # write_table is where this gets its parallelism.
    writer = pq.ParquetWriter(str(chunk_path), chunk_table.schema, compression=COMPRESSION)
    writer.write_table(chunk_table)
    writer.add_key_value_metadata({'geo': json.dumps(geo_md)})
    writer.close()

    return {'n_in': n_in, 'n_out': len(wkb_bytes), 'chunk_path': chunk_path}


# --------- driver ---------

def simplify_tile(input_path, output_folder):
    """Stream-convert one cropland tile to a folder of chunk parquets.

    Returns (n_in, n_out, chunks_written, mb_in, mb_out_folder_total).
    """
    pf = pq.ParquetFile(input_path)

    # Recover the source CRS from the input's GeoParquet metadata.
    file_kv = pf.metadata.metadata or {}
    if b'geo' not in file_kv:
        raise ValueError(f"{input_path}: input lacks GeoParquet metadata")
    in_geo = json.loads(file_kv[b'geo'])
    primary_col  = in_geo.get('primary_column', 'geometry')
    src_crs_dict = in_geo['columns'][primary_col].get('crs')

    # Build the output CRS as JSON-serializable PROJJSON once. Each worker
    # uses it when emitting its chunk's geo metadata - no per-batch CRS
    # construction.
    from pyproj import CRS as _CRS
    out_crs_dict = json.loads(_CRS.from_user_input(OUTPUT_CRS).to_json())

    chunk_dir = Path(output_folder)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n_in_total = pf.metadata.num_rows
    n_out_total    = [0]            # boxed for nested fn
    chunks_written = [0]

    n_batches_expected = max(1, math.ceil(n_in_total / BATCH_SIZE))
    desc = f"  {os.path.basename(input_path):<28}"

    def _record_result(result):
        n_out_total[0] += result['n_out']
        if result['chunk_path'] is not None:
            chunks_written[0] += 1

    def _resume_existing(batch_idx):
        """If chunk_NNNNNN.parquet already exists, count its rows toward
        the totals and return True so the main loop can skip processing
        that batch."""
        chunk_path = chunk_dir / f"chunk_{batch_idx:06d}.parquet"
        if not chunk_path.exists():
            return False
        try:
            n_out_total[0]    += pq.ParquetFile(chunk_path).metadata.num_rows
            chunks_written[0] += 1
        except Exception:
            # Corrupt chunk from a crashed run - delete it and reprocess.
            try: chunk_path.unlink()
            except OSError: pass
            return False
        return True

    batch_iter = enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=[primary_col]))

    if N_WORKERS <= 1:
        # ---- Sequential: still useful for debugging ----
        for batch_idx, batch in tqdm(batch_iter, total=n_batches_expected,
                                     desc=desc, unit="batch"):
            if _resume_existing(batch_idx):
                continue
            wkb_arr = batch.column(primary_col).to_pylist()
            if not wkb_arr:
                continue
            _record_result(_process_and_write_chunk(
                batch_idx, wkb_arr, src_crs_dict, OUTPUT_CRS,
                out_crs_dict, primary_col, chunk_dir,
            ))
    else:
        # ---- Bounded thread pool: workers write their own chunks ----
        # The brotli compression that used to bottleneck on the main thread
        # is now per-worker, releasing the GIL during compress.
        max_in_flight = N_WORKERS * 2
        in_flight = set()
        pbar = tqdm(total=n_batches_expected, desc=desc, unit="batch")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            exhausted = False
            while True:
                while not exhausted and len(in_flight) < max_in_flight:
                    try:
                        batch_idx, batch = next(batch_iter)
                    except StopIteration:
                        exhausted = True
                        break

                    if _resume_existing(batch_idx):
                        pbar.update(1)
                        continue

                    wkb_arr = batch.column(primary_col).to_pylist()
                    if not wkb_arr:
                        pbar.update(1)
                        continue

                    in_flight.add(pool.submit(
                        _process_and_write_chunk,
                        batch_idx, wkb_arr, src_crs_dict, OUTPUT_CRS,
                        out_crs_dict, primary_col, chunk_dir,
                    ))

                if not in_flight:
                    break

                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    _record_result(fut.result())
                    pbar.update(1)
        pbar.close()

    mb_in  = os.path.getsize(input_path) / (1024 * 1024)
    mb_out = sum(p.stat().st_size for p in chunk_dir.glob('chunk_*.parquet')) / (1024 * 1024)
    return n_in_total, n_out_total[0], chunks_written[0], mb_in, mb_out


# --------- main ---------

def main():
    in_path    = Path(INPUT_PATH)
    out_folder = Path(OUTPUT_FOLDER)

    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    out_folder.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:        {in_path}")
    print(f"Output:       {out_folder}{os.sep}   (folder of chunk parquets)")
    print(f"Tolerance:    {TOLERANCE} m (source CRS)   "
          f"Min area: {MIN_AREA} m^2 (source CRS)")
    print(f"Output CRS:   {OUTPUT_CRS}   Compression: {COMPRESSION}   "
          f"Batch: {BATCH_SIZE:,} rows   Workers: {N_WORKERS}")
    print("=" * 78)

    t0 = time.time()
    try:
        n_in, n_out, chunks_written, mb_in, mb_out = simplify_tile(
            str(in_path), str(out_folder)
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return

    elapsed = time.time() - t0
    poly_drop = 100.0 * (1 - n_out / max(n_in, 1))
    size_drop = 100.0 * (1 - mb_out / max(mb_in, 1e-9))
    print("=" * 78)
    print(
        f"{n_in:,} -> {n_out:,} polys ({poly_drop:.1f}% dropped) | "
        f"{mb_in:.1f} MB -> {mb_out:.1f} MB ({size_drop:.1f}% smaller) | "
        f"{chunks_written} chunks written | "
        f"{elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
