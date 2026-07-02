import geopandas as gpd
import pandas as pd

from tile_loader import read_tiles_in_bbox, resolve_source

BASIN_ID = "HYBAS_ID"


def _empty_result(ids):
    return gpd.GeoDataFrame({
        BASIN_ID: ids,
        'area_m2': [None] * len(ids),
        'geometry': [None] * len(ids)
    }, crs='EPSG:4326')


def _load_farmland(farmland_source, bbox):
    folder, fallback_path = resolve_source(farmland_source, '**/*.parquet')
    if folder is not None:
        return read_tiles_in_bbox(folder, '**/*.parquet', bbox=tuple(bbox))
    return gpd.read_parquet(fallback_path, bbox=tuple(bbox))


def calculate_basin_farmland(basins, farmland_source):
    if len(basins) == 0:
        return _empty_result([])

    basins = basins[[BASIN_ID, 'geometry']]
    ids = basins[BASIN_ID].tolist()

    farm_gdf = _load_farmland(farmland_source, basins.total_bounds)
    if len(farm_gdf) == 0 or 'geometry' not in farm_gdf.columns:
        return _empty_result(ids)

    if farm_gdf.crs != basins.crs:
        farm_gdf = farm_gdf.to_crs(basins.crs)

    farm_in_basins = gpd.sjoin(farm_gdf, basins[['geometry']], how='inner', predicate='intersects')
    farm_in_basins = farm_in_basins[~farm_in_basins.index.duplicated(keep='first')]
    farm_in_basins = farm_in_basins.drop(columns='index_right')
    if len(farm_in_basins) == 0:
        return _empty_result(ids)

    basins_cea = basins.to_crs({'proj': 'cea'})
    farm_in_basins_cea = farm_in_basins.to_crs({'proj': 'cea'})

    intersections = gpd.overlay(basins_cea, farm_in_basins_cea, how="intersection")
    if len(intersections) == 0:
        return _empty_result(ids)

    intersections['area_m2'] = intersections.geometry.area
    intersections_wgs84 = intersections.to_crs('EPSG:4326')
    result = intersections_wgs84[[BASIN_ID, 'area_m2', 'geometry']].copy()

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
    df = pd.DataFrame(farmland_result.drop(columns='geometry', errors='ignore'))
    per_basin = df.groupby(BASIN_ID, dropna=False, as_index=False)['area_m2'].sum(min_count=1)
    total = per_basin['area_m2'].sum(min_count=1)
    total_row = pd.DataFrame({BASIN_ID: ['TOTAL'], 'area_m2': [total]})
    return pd.concat([total_row, per_basin], ignore_index=True)
