import os

import geopandas as gpd
import pandas as pd

from tile_loader import read_tiles_in_bbox, resolve_source

# Basin id column. HydroBASINS uses HYBAS_ID (TDX used LINKNO).
BASIN_ID = "HYBAS_ID"


def _empty_result(ids):
    """Schema-stable empty return for early-exit paths."""
    return gpd.GeoDataFrame({
        BASIN_ID: ids,
        'area_m2': [None] * len(ids),
        'geometry': [None] * len(ids)
    }, crs='EPSG:4326')


def _load_farmland(farmland_source, bbox):
    """Read cropland polygons inside bbox.

    farmland_source may be a directory of tiled cropland parquets (possibly
    nested one level deep) - the '**/*.parquet' pattern picks up both - or a
    single pre-merged parquet (legacy case).
    """
    folder, fallback_path = resolve_source(farmland_source, '**/*.parquet')
    if folder is not None:
        return read_tiles_in_bbox(folder, '**/*.parquet', bbox=tuple(bbox))
    return gpd.read_parquet(fallback_path, bbox=tuple(bbox))


def calculate_basin_farmland(basins, farmland_source):
    """`basins` is an already-loaded GeoDataFrame subset (HYBAS_ID + geometry)
    in EPSG:4326, handed in by the orchestrator (no per-chunk file read)."""
    if len(basins) == 0:
        return _empty_result([])

    basins = basins[[BASIN_ID, 'geometry']]
    ids = basins[BASIN_ID].tolist()

    # Bbox-filtered farmland load. For tile folders this skips non-overlapping
    # ESA tiles entirely (footer check) and uses row-group bbox pushdown on
    # the rest.
    farm_gdf = _load_farmland(farmland_source, basins.total_bounds)

    if len(farm_gdf) == 0 or 'geometry' not in farm_gdf.columns:
        return _empty_result(ids)

    # Ensure same CRS
    if farm_gdf.crs != basins.crs:
        farm_gdf = farm_gdf.to_crs(basins.crs)

    # Spatial join for quick filtering. sjoin duplicates the LEFT (farm) row
    # once per matching basin, so a farm polygon spanning N basins shows up N
    # times. Deduplicate by the original farm index before overlay so the
    # intersection geometries aren't double-counted.
    farm_in_basins = gpd.sjoin(
        farm_gdf,
        basins[['geometry']],
        how='inner',
        predicate='intersects'
    )
    farm_in_basins = farm_in_basins[~farm_in_basins.index.duplicated(keep='first')]
    farm_in_basins = farm_in_basins.drop(columns='index_right')

    if len(farm_in_basins) == 0:
        return _empty_result(ids)

    # Project to equal area CRS for accurate area calculation
    basins_cea = basins.to_crs({'proj': 'cea'})
    farm_in_basins_cea = farm_in_basins.to_crs({'proj': 'cea'})

    # Overlay creates intersection polygons (clips farmland to basin boundaries)
    intersections = gpd.overlay(basins_cea, farm_in_basins_cea, how="intersection")

    if len(intersections) == 0:
        return _empty_result(ids)

    # Calculate area in m^2 (CEA projection units).
    intersections['area_m2'] = intersections.geometry.area

    # Reproject polygons back to WGS84 for visualization
    intersections_wgs84 = intersections.to_crs('EPSG:4326')

    result = intersections_wgs84[[BASIN_ID, 'area_m2', 'geometry']].copy()

    # Add missing basins (no farmland)
    missing = set(ids) - set(result[BASIN_ID])
    if missing:
        missing_gdf = gpd.GeoDataFrame({
            BASIN_ID: list(missing),
            'area_m2': [None] * len(missing),
            'geometry': [None] * len(missing)
        }, crs='EPSG:4326')
        result = pd.concat([result, missing_gdf], ignore_index=True)

    return result


def calculate_basin_farmland_wrapper(args):
    return calculate_basin_farmland(*args)


def aggregate_farmland_for_csv(farmland_result):
    """Collapse per-intersection rows into one row per basin (HYBAS_ID, area_m2)
    with a prepended TOTAL row. Basins with no farmland keep a NaN area row
    (min_count=1 prevents the sum from collapsing missing data to 0)."""
    df = pd.DataFrame(farmland_result.drop(columns='geometry', errors='ignore'))

    per_basin = df.groupby(BASIN_ID, dropna=False, as_index=False)['area_m2'].sum(min_count=1)

    total = per_basin['area_m2'].sum(min_count=1)
    total_row = pd.DataFrame({BASIN_ID: ['TOTAL'], 'area_m2': [total]})
    return pd.concat([total_row, per_basin], ignore_index=True)
