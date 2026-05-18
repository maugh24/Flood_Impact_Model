import geopandas as gpd
import pandas as pd

_FARM_GDF: gpd.GeoDataFrame = None


def _empty_result(rivers):
    """Schema-stable empty return for early-exit paths."""
    return gpd.GeoDataFrame({
        'linkno': rivers,
        'area_m2': [None] * len(rivers),
        'geometry': [None] * len(rivers)
    }, crs='EPSG:4326')


def get_farmland_gdf(farmland_parquet):
    global _FARM_GDF
    if _FARM_GDF is None:
        _FARM_GDF = gpd.read_parquet(farmland_parquet)
    return _FARM_GDF


def calculate_basin_farmland(basin_file, rivers, farmland_parquet):
    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)])

    if len(basins) == 0:
        return _empty_result(rivers)

    # Get cached farmland data
    farm_gdf = get_farmland_gdf(farmland_parquet)

    if len(farm_gdf) == 0 or 'geometry' not in farm_gdf.columns:
        return _empty_result(rivers)

    # Ensure same CRS
    if farm_gdf.crs != basins.crs:
        farm_gdf = farm_gdf.to_crs(basins.crs)

    # Spatial join for quick filtering. sjoin duplicates the LEFT (farm) row
    # once per matching basin, so a farm polygon spanning N basins shows up N
    # times. If we hand that to overlay, every basin gets intersected with
    # each duplicate copy and the resulting intersection geometries are
    # double-counted. Deduplicate by the original farm index before overlay.
    farm_in_basins = gpd.sjoin(
        farm_gdf,
        basins[['geometry']],
        how='inner',
        predicate='intersects'
    )
    farm_in_basins = farm_in_basins[~farm_in_basins.index.duplicated(keep='first')]
    farm_in_basins = farm_in_basins.drop(columns='index_right')

    if len(farm_in_basins) == 0:
        return _empty_result(rivers)

    # Project to equal area CRS for accurate area calculation
    basins_cea = basins.to_crs({'proj': 'cea'})
    farm_in_basins_cea = farm_in_basins.to_crs({'proj': 'cea'})

    # Overlay creates intersection polygons (clips farmland to basin boundaries)
    intersections = gpd.overlay(basins_cea, farm_in_basins_cea, how="intersection")

    if len(intersections) == 0:
        return _empty_result(rivers)

    # Calculate area in m^2 (CEA projection units). Single source of truth -
    # convert to other units at the output layer if needed.
    intersections['area_m2'] = intersections.geometry.area

    # Reproject polygons back to WGS84 for visualization
    intersections_wgs84 = intersections.to_crs('EPSG:4326')

    # Select final columns (keep geometry as polygons)
    result = intersections_wgs84[['linkno', 'area_m2', 'geometry']].copy()

    # Add missing linknos (basins with no farmland)
    missing_linknos = set(rivers) - set(result['linkno'])

    if missing_linknos:
        missing_gdf = gpd.GeoDataFrame({
            'linkno': list(missing_linknos),
            'area_m2': [None] * len(missing_linknos),
            'geometry': [None] * len(missing_linknos)
        }, crs='EPSG:4326')
        result = pd.concat([result, missing_gdf], ignore_index=True)

    return result


def calculate_basin_farmland_wrapper(args):
    return calculate_basin_farmland(*args)


def aggregate_farmland_for_csv(farmland_result):
    """Collapse the per-intersection farmland_result into one row per basin
    for CSV export. Output has exactly two columns: linkno and area_m2.
    A TOTAL row is prepended. Basins with no farmland keep a NaN area row
    (min_count=1 prevents the sum from collapsing missing data to 0)."""
    df = pd.DataFrame(farmland_result.drop(columns='geometry', errors='ignore'))

    per_basin = df.groupby('linkno', dropna=False, as_index=False)['area_m2'].sum(min_count=1)

    total = per_basin['area_m2'].sum(min_count=1)
    total_row = pd.DataFrame({'linkno': ['TOTAL'], 'area_m2': [total]})
    return pd.concat([total_row, per_basin], ignore_index=True)
