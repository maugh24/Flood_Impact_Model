"""
Simplify_Cropland.py

Preprocessing pass that walks every ESA cropland tile in INPUT_FOLDER,
simplifies polygon geometries and drops tiny noise polygons, then writes
each tile to OUTPUT_FOLDER with brotli compression and a per-row covering
bbox column. The simplified tiles are typically a small fraction of the
input size, which dramatically lowers peak memory for the farmland step
of the impact model.

Memory strategy:
  - One tile is loaded, processed, and written before moving on. Peak
    memory is bounded by a single tile - not the whole folder. We do NOT
    concat tiles together because that would inflate memory and lose the
    per-tile spatial locality the impact model relies on for bbox reads.
  - Tiles are processed sequentially. Parallelizing wouldn't help much
    here: each worker would multiply peak memory by N, which is exactly
    the bottleneck we're trying to fix.
  - Tiles whose output already exists are skipped, so a crashed run can
    be resumed by just rerunning the script.
"""
import glob
import os
import time
import warnings
from pathlib import Path

import geopandas as gpd

warnings.filterwarnings('ignore')


# ---- Paths ----
INPUT_FOLDER  = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet"
OUTPUT_FOLDER = r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet\bbox"

# ---- Simplification parameters (in CEA meters, since the cropland
# tiles were saved in cylindrical equal-area projection by
# cropland_raster_to_parquet.py) ----
TOLERANCE   = 10       # max distance (m) a vertex may be moved during simplification
MIN_AREA    = 500      # drop polygons smaller than this (m^2)
COMPRESSION = 'brotli'


def simplify_tile(input_path, output_path):
    """Read one tile, simplify + filter, write to output_path.

    Returns (n_in, n_out, mb_in, mb_out) for reduction reporting.
    """
    gdf = gpd.read_parquet(input_path)
    n_in = len(gdf)

    # Simplify first. preserve_topology stops polygons collapsing to
    # invalid shapes around narrow features.
    gdf['geometry'] = gdf['geometry'].simplify(
        tolerance=TOLERANCE, preserve_topology=True
    )

    # Then filter by area. Simplify can shrink polygons slightly, so
    # doing the area check AFTER catches a few more sub-threshold
    # geometries than doing it before.
    gdf = gdf[gdf.geometry.area >= MIN_AREA].reset_index(drop=True)
    n_out = len(gdf)

    # write_covering_bbox is critical: the impact model's tile_loader
    # uses it to do row-group bbox pushdown during reads. Without it
    # every read would have to scan the whole tile.
    gdf.to_parquet(
        output_path,
        index=False,
        compression=COMPRESSION,
        write_covering_bbox=True,
    )

    mb_in  = os.path.getsize(input_path)  / (1024 * 1024)
    mb_out = os.path.getsize(output_path) / (1024 * 1024)
    return n_in, n_out, mb_in, mb_out


def main():
    in_folder  = Path(INPUT_FOLDER)
    out_folder = Path(OUTPUT_FOLDER)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Only top-level *.parquet, NOT recursive - we don't want to re-process
    # files we already wrote into the bbox/ subfolder.
    tiles = sorted(glob.glob(str(in_folder / '*.parquet')))
    # Defensive: drop any tile path that lives under the output folder,
    # in case INPUT and OUTPUT ever overlap.
    tiles = [t for t in tiles if Path(t).parent.resolve() != out_folder.resolve()]

    if not tiles:
        print(f"No .parquet tiles found in {in_folder}")
        return

    print(f"Found {len(tiles)} cropland tiles in {in_folder}")
    print(f"Writing simplified output to {out_folder}")
    print(f"Tolerance: {TOLERANCE} m   Min area: {MIN_AREA} m^2   Compression: {COMPRESSION}")
    print("=" * 78)

    overall_start = time.time()
    total_in_mb = total_out_mb = 0.0
    total_in_polys = total_out_polys = 0
    n_processed = n_skipped = n_failed = 0

    for i, input_path in enumerate(tiles, 1):
        name = os.path.basename(input_path)
        output_path = str(out_folder / name)

        if os.path.exists(output_path):
            print(f"  [{i:>2}/{len(tiles)}] {name} -- already exists, skipping")
            n_skipped += 1
            continue

        print(f"  [{i:>2}/{len(tiles)}] {name} -- processing...")
        t0 = time.time()
        try:
            n_in, n_out, mb_in, mb_out = simplify_tile(input_path, output_path)
        except Exception as e:
            print(f"      FAILED: {e}")
            # Remove partial output so the next run reprocesses cleanly.
            try:
                os.remove(output_path)
            except OSError:
                pass
            n_failed += 1
            continue

        elapsed = time.time() - t0
        poly_drop = 100.0 * (1 - n_out / max(n_in, 1))
        size_drop = 100.0 * (1 - mb_out / max(mb_in, 1e-9))
        print(
            f"      {n_in:>10,} -> {n_out:>10,} polys ({poly_drop:5.1f}% dropped) | "
            f"{mb_in:7.1f} MB -> {mb_out:7.1f} MB ({size_drop:5.1f}% smaller) | "
            f"{elapsed:.1f}s"
        )
        total_in_polys  += n_in
        total_out_polys += n_out
        total_in_mb     += mb_in
        total_out_mb    += mb_out
        n_processed     += 1

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
