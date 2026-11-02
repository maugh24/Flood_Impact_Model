"""
Farmland impact within a single flood extent (no basins, no linknos).

Mirrors the basin pipeline: sjoin pre-filter (with dedup) + overlay to clip
farmland polygons to the flood extent, then sum area_m2.
"""
import geopandas as gpd
import pandas as pd

_FARM_GDF: gpd.GeoDataFrame = None


def _empty_result():
    """Schema-stable empty return."""
    return gpd.GeoDataFrame({
        'area_m2': [],
        'geometry': []
    }, geometry='geometry', crs='EPSG:4326')


def get_farmland_gdf(farmland_parquet):
    global _FARM_GDF
    if _FARM_GDF is None:
        _FARM_GDF = gpd.read_parquet(farmland_parquet)
    return _FARM_GDF


def calculate_flood_farmland(flood_parquet, farmland_parquet):
    flood = gpd.read_parquet(flood_parquet)
    if len(flood) == 0:
        return _empty_result()

    farm_gdf = get_farmland_gdf(farmland_parquet)
    if len(farm_gdf) == 0 or 'geometry' not in farm_gdf.columns:
        return _empty_result()

    if farm_gdf.crs != flood.crs:
        farm_gdf = farm_gdf.to_crs(flood.crs)

    # sjoin pre-filter to drop farm polygons that can't touch the flood,
    # then dedup by original farm index so overlay doesn't double-count
    # farms that span multiple flood polygons (same correctness fix as
    # the basin pipeline).
    farm_in_flood = gpd.sjoin(
        farm_gdf,
        flood[['geometry']],
        how='inner',
        predicate='intersects'
    )
    farm_in_flood = farm_in_flood[~farm_in_flood.index.duplicated(keep='first')]
    farm_in_flood = farm_in_flood.drop(columns='index_right')

    if len(farm_in_flood) == 0:
        return _empty_result()

    # Equal-area projection for accurate area calc
    flood_cea = flood.to_crs({'proj': 'cea'})
    farm_in_flood_cea = farm_in_flood.to_crs({'proj': 'cea'})

    intersections = gpd.overlay(flood_cea, farm_in_flood_cea, how="intersection")
    if len(intersections) == 0:
        return _empty_result()

    intersections['area_m2'] = intersections.geometry.area
    intersections_wgs84 = intersections.to_crs('EPSG:4326')
    return intersections_wgs84[['area_m2', 'geometry']].copy()


def aggregate_farmland_for_csv(farmland_result):
    """Single-row CSV: total flooded farmland area in m^2."""
    df = pd.DataFrame(farmland_result.drop(columns='geometry', errors='ignore'))
    total = df['area_m2'].sum(min_count=1) if len(df) else None
    return pd.DataFrame({'area_m2': [total]})
