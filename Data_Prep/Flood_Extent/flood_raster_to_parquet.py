"""
flood_raster_to_parquet.py

Reclassify a flood-extent raster into binary flood/dry, then vectorize the
flood pixels into a single parquet file that matches the schema used by
the rest of the impact-analysis pipeline.

Reclassification rules:
  - pixel value >  0           -> FLOOD (1)
  - pixel value == 0           -> dry  (excluded from parquet)
  - pixel value == NoData      -> dry  (excluded from parquet)

The dry class is implicit - only flood polygons are stored. This matches
the cropland and population parquet conventions.
"""
import numpy as np
import rasterio
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape
from pathlib import Path


def reclassify_and_vectorize(flood_raster, output_parquet):
    """
    Read a flood-extent raster, build a binary flood mask (cells > 0 and
    not NoData), and vectorize the flood mask into polygons written to
    parquet in EPSG:4326.
    """
    with rasterio.open(flood_raster) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Build the flood mask: > 0 AND not NoData. Anything else is dry.
    flood_mask = data > 0
    if nodata is not None:
        flood_mask &= (data != nodata)

    if not flood_mask.any():
        print(f"No flood pixels found in {flood_raster}. Nothing to write.")
        return

    # Reclassify to a uniform value before vectorizing. rasterio.features.shapes
    # traces connected runs of EQUAL values, so if we hand it the raw raster
    # it would split each connected flood region into one polygon per distinct
    # pixel value (e.g., depths of 50 and 75 become two polygons even when
    # they touch). Collapsing the mask to a single value first guarantees
    # adjacent flood pixels merge into one polygon regardless of original value.
    reclassified = flood_mask.astype(np.uint8)

    shapes_gen = features.shapes(reclassified, mask=flood_mask, transform=transform)
    geoms = [{"geometry": shape(geom), "flood": 1} for geom, _ in shapes_gen]

    gdf = gpd.GeoDataFrame(geoms, crs=crs)

    # Reproject to EPSG:4326 to match the rest of the pipeline (basins, OSM,
    # cropland, population all live in WGS84).
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # write_covering_bbox lets the impact modules read with a bbox filter
    # for fast per-chunk loads, matching how population.parquet is written.
    gdf.to_parquet(output_parquet, index=False, write_covering_bbox=True)
    print(f"Wrote {len(gdf):,} flood polygons to {output_parquet}")


if __name__ == "__main__":
    # ---- UPDATE THESE TWO PATHS TO YOUR FILES ----
    input_tif = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\sula_valley_honduras_100_year_rp.tif"
    output_pq = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\sula_valley_honduras_100_year_rp.parquet"
    # ----------------------------------------------

    if Path(input_tif).exists():
        reclassify_and_vectorize(input_tif, output_pq)
    else:
        print(f"Input not found: {input_tif}")
