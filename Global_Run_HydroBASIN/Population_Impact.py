import os
import geopandas as gpd
import pandas as pd

from Global_Run_TDX.tile_loader import read_tiles_in_bbox, resolve_source

# Basin id column. HydroBASINS uses HYBAS_ID (TDX used LINKNO).
BASIN_ID = "HYBAS_ID"


def _empty_result(ids):
    return pd.DataFrame(data={
        BASIN_ID: ids,
        'pop_value': [None] * len(ids),
        'x': [None] * len(ids),
        'y': [None] * len(ids)
    })


def _load_population(population_source, bbox):
    """Read population tiles inside bbox.

    population_source may be a folder of tiled population parquets or a
    single pre-merged parquet (current case).
    """
    folder, fallback_path = resolve_source(population_source, '*.parquet')
    if folder is not None:
        return read_tiles_in_bbox(folder, '*.parquet', bbox=tuple(bbox))
    return gpd.read_parquet(fallback_path, bbox=tuple(bbox))


def calculate_basin_population(basins, population_source):
    """`basins` is an already-loaded GeoDataFrame subset (HYBAS_ID + geometry)
    in EPSG:4326. The orchestrator loads the global basin file once and hands
    each worker its chunk, so this no longer reads the basin file itself."""
    if len(basins) == 0:
        return _empty_result([])

    basins = basins[[BASIN_ID, 'geometry']]
    ids = basins[BASIN_ID].tolist()
    bbox = basins.total_bounds

    pop_gdf = _load_population(population_source, bbox)

    if len(pop_gdf) == 0:
        return _empty_result(ids)

    if pop_gdf.crs != basins.crs:
        pop_gdf = pop_gdf.to_crs(basins.crs)

    # Project to equal-area CRS so area ratios are accurate
    basins_cea = basins.to_crs({'proj': 'cea'})
    pop_cea = pop_gdf.to_crs({'proj': 'cea'}).copy()

    # Record each tile's full area and full pop BEFORE overlay so we can
    # weight by the area fraction that lands in each basin.
    pop_cea['_tile_area'] = pop_cea.geometry.area
    pop_cea['_tile_pop'] = pop_cea['pop_value']

    intersections = gpd.overlay(
        basins_cea[[BASIN_ID, 'geometry']],
        pop_cea[['_tile_area', '_tile_pop', 'geometry']],
        how='intersection'
    )

    if len(intersections) == 0:
        return _empty_result(ids)

    # Area-weight: pop assigned to basin = tile_pop * (clipped_area / tile_area).
    intersections['_clip_area'] = intersections.geometry.area
    intersections['pop_value'] = (
        intersections['_tile_pop'] * (intersections['_clip_area'] / intersections['_tile_area'])
    )

    # Centroid of the clipped piece, reported in WGS84 for downstream plotting.
    intersections_wgs84 = intersections.to_crs('EPSG:4326')
    centroid = intersections_wgs84.geometry.centroid
    intersections_wgs84['x'] = centroid.x
    intersections_wgs84['y'] = centroid.y

    result = intersections_wgs84[[BASIN_ID, 'pop_value', 'x', 'y']].copy()

    # Add missing basins (no population overlap)
    missing = set(ids) - set(result[BASIN_ID])
    if missing:
        missing_df = pd.DataFrame(data={
            BASIN_ID: list(missing),
            'pop_value': [None] * len(missing),
            'x': [None] * len(missing),
            'y': [None] * len(missing)
        })
        result = pd.concat([result, missing_df], ignore_index=True)

    return result


def calculate_basin_population_wrapper(args):
    return calculate_basin_population(*args)


def aggregate_population_for_csv(population_result):
    """Per-basin pop_value sums with a TOTAL row."""
    df = pd.DataFrame(population_result).drop(
        columns=['x', 'y', 'geometry'], errors='ignore'
    )

    per_basin = df.groupby(BASIN_ID, dropna=False, as_index=False)['pop_value'].sum(min_count=1)

    total = per_basin['pop_value'].sum(min_count=1)
    total_row = pd.DataFrame({BASIN_ID: ['TOTAL'], 'pop_value': [total]})
    return pd.concat([total_row, per_basin], ignore_index=True)
