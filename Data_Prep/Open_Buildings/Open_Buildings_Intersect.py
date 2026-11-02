"""
Intersect a Google Open Buildings footprint parquet with a single flood
extent (Chiang Rai).

The buildings parquet carries a GeoParquet ``bbox`` covering column, so the
read is spatially pre-filtered to the flood's bounding box before any
geometry work happens. This keeps memory low even though the source file is
large.

Counting rule: centroid-in-polygon containment. A building is counted at
most once - a footprint that straddles the flood boundary is included only
if its centroid lies inside the flood extent (same correctness rule as the
SFE / basin pipelines).

Outputs (written to the configured output folder):
  - open_buildings_flooded.parquet : the intersected building footprints
  - open_buildings_flooded.csv     : one-row summary with the building count
"""
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd


def calculate_flooded_buildings(flood_parquet, building_parquet):
    """Return a GeoDataFrame of Open Buildings whose centroid falls inside
    the flood extent. Original footprint geometry and attributes are kept."""
    flood = gpd.read_parquet(flood_parquet)
    if len(flood) == 0:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))

    # Bounding box of the flood extent -> drives the spatial pre-filter.
    flood_bounds = flood.total_bounds  # [minx, miny, maxx, maxy]

    # The buildings file has a bbox covering column, so this only pulls
    # footprints whose bbox overlaps the flood's bounding box.
    buildings = gpd.read_parquet(building_parquet, bbox=tuple(flood_bounds))
    if len(buildings) == 0:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=flood.crs))

    # Align CRS before any spatial predicate.
    if buildings.crs != flood.crs:
        buildings = buildings.to_crs(flood.crs)

    # Centroid-in-polygon: assign each building to the flood polygon that
    # contains its centroid (dropped if no polygon contains it). Prevents
    # double-counting of footprints that cross the flood boundary.
    building_centroids = buildings.copy()
    building_centroids["geometry"] = buildings.geometry.centroid

    joined = gpd.sjoin(
        building_centroids,
        flood,
        how="inner",
        predicate="within",
    )

    # Defensive dedup: a centroid landing exactly on an interior flood
    # boundary can match two polygons.
    joined = joined[~joined.index.duplicated(keep="first")]

    # Restore the original footprint geometry (sjoin left us with centroids)
    # and drop the join bookkeeping column.
    result = buildings.loc[joined.index].copy()
    result = result.drop(columns=["index_right"], errors="ignore")
    return result


def summarize(result):
    """One-row summary DataFrame with the flooded building count."""
    return pd.DataFrame({"building_count": [int(len(result))]})


# ===== USAGE =====
if __name__ == "__main__":
    building_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Napal_Floods\30d_buildings_bbox.parquet"
    flood_parquet = r"C:\C_Drive_Brians_Stuff\Python_Projects\Napal_Floods\Chian_Rai\GEOGLOWS_dem_99.70939_19.79092_99.97134_19.99271_ARC_Flood_rp100extent.parquet"

    output_folder = Path(r"C:\C_Drive_Brians_Stuff\Python_Projects\Napal_Floods\Chian_Rai")
    output_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPEN BUILDINGS x FLOOD EXTENT INTERSECT")
    print("=" * 80)
    print(f"Buildings: {building_parquet}")
    print(f"Flood:     {flood_parquet}")
    start = time.time()

    flooded = calculate_flooded_buildings(flood_parquet, building_parquet)

    parquet_out = output_folder / "open_buildings_flooded.parquet"
    csv_out = output_folder / "open_buildings_flooded.csv"

    flooded.to_parquet(parquet_out, index=False)
    summarize(flooded).to_csv(csv_out, index=False)

    elapsed = time.time() - start
    print("-" * 80)
    print(f"Flooded buildings: {len(flooded)}")
    print(f"Wrote: {parquet_out}")
    print(f"Wrote: {csv_out}")
    print(f"Done in {elapsed:.1f}s")
    print("=" * 80)
