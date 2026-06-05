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


def simplify_tile_streaming(input_path, output_path):
    """Streamed simplify+filter for one tile.

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
    n_out = 0

    # Global bbox accumulator (in input CRS units).
    gxmin = gymin =  math.inf
    gxmax = gymax = -math.inf

    writer = None
    # Total batches for the inner progress bar (last batch may be partial).
    n_batches_expected = max(1, math.ceil(n_in / BATCH_SIZE))
    try:
        # Read only the geometry column. The cropland tiles are
        # geometry-only, so this is also "all" of the data; if you ever
        # point this at a tile with attribute columns, the attributes
        # will be dropped - the simplified output is geometry only.
        batch_iter = pf.iter_batches(batch_size=BATCH_SIZE, columns=[primary_col])
        for batch in tqdm(
            batch_iter,
            total=n_batches_expected,
            desc=f"  {os.path.basename(input_path):<28}",
            unit="batch",
            leave=False,
        ):
            wkb_arr = batch.column(primary_col).to_pylist()
            if not wkb_arr:
                continue

            # WKB -> shapely -> GeoSeries (so we get vectorized simplify/area).
            geoms = gpd.GeoSeries(from_wkb(wkb_arr), crs=crs_dict)

            # Simplify. preserve_topology stops thin polygons from collapsing.
            geoms = geoms.simplify(tolerance=TOLERANCE, preserve_topology=True)

            # Area filter AFTER simplify (simplify can shrink polygons
            # slightly, so doing it after catches a few more sub-threshold
            # cases than doing it before).
            mask = geoms.area >= MIN_AREA
            if not mask.any():
                continue
            geoms = geoms[mask].reset_index(drop=True)
            if len(geoms) == 0:
                continue

            # Per-row bounds for the output bbox column + global bbox update.
            bounds = geoms.bounds  # DataFrame with minx, miny, maxx, maxy
            gxmin = min(gxmin, bounds.minx.min())
            gymin = min(gymin, bounds.miny.min())
            gxmax = max(gxmax, bounds.maxx.max())
            gymax = max(gymax, bounds.maxy.max())

            # Build a pyarrow Table with two columns:
            #   primary_col  : binary  (WKB)
            #   'bbox'       : struct of four float64 (xmin, ymin, xmax, ymax)
            wkb_bytes = [g.wkb for g in geoms]
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

            # Lazily open the writer on the first non-empty chunk so we
            # can pick up the schema from the actual data.
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    chunk_table.schema,
                    compression=COMPRESSION,
                )

            writer.write_table(chunk_table)
            n_out += len(geoms)
    except Exception:
        if writer is not None:
            try: writer.close()
            except Exception: pass
        raise

    if writer is not None:
        # Attach the GeoParquet 1.1 'geo' metadata. The global bbox we
        # collected lets downstream code do whole-file skip checks; the
        # covering.bbox pointer lets pyarrow do row-group pushdown.
        global_bbox = (gxmin, gymin, gxmax, gymax)
        geo_md = _build_geo_metadata(primary_col, crs_dict, geometry_types, global_bbox)
        writer.add_key_value_metadata({'geo': json.dumps(geo_md)})
        writer.close()

    mb_in  = os.path.getsize(input_path)  / (1024 * 1024)
    mb_out = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0
    return n_in, n_out, mb_in, mb_out


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
