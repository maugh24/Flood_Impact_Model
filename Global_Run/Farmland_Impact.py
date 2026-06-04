import os
import geopandas as gpd
import pandas as pd

from tile_loader import read_tiles_in_bbox, resolve_source


def _empty_result(rivers):
    """Schema-stable empty return for early-exit paths."""
    return gpd.GeoDataFrame({
        'LINKNO': rivers,
        'area_m2': [None] * len(rivers),
        'geometry': [None] * len(rivers)
    }, crs='EPSG:4326')


def _load_farmland(farmland_source, bbox):
    """Read cropland polygons inside bbox.

    farmland_source may be:
      - a directory containing tiled cropland parquets (Global_Run case);
        only tiles overlapping bbox are read.
      - a single pre-merged parquet (legacy case).
    """
    folder, fallback_path = resolve_source(farmland_source, '*.parquet')
    if folder is not None:
        return read_tiles_in_bbox(folder, '*.parquet', bbox=tuple(bbox))
    return gpd.read_parquet(fallback_path, bbox=tuple(bbox))


def calculate_basin_farmland(basin_file, rivers, farmland_source):
    basins = gpd.read_parquet(basin_file, filters=[('LINKNO', 'in', rivers)])

    if len(basins) == 0:
        return _empty_result(rivers)

    # Bbox-filtered farmland load. For tile folders this skips non-overlapping
    # ESA tiles entirely (footer check) and uses row-group bbox pushdown on
    # the rest.
    farm_gdf = _load_farmland(farmland_source, basins.total_bounds)

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

    # Calculate area in m^2 (CEA projection units).
    intersections['area_m2'] = intersections.geometry.area

    # Reproject polygons back to WGS84 for visualization
    intersections_wgs84 = intersections.to_crs('EPSG:4326')

    # Select final columns (keep geometry as polygons)
    result = intersections_wgs84[['LINKNO', 'area_m2', 'geometry']].copy()

    # Add missing LINKNOs (basins with no farmland)
    missing_linknos = set(rivers) - set(result['LINKNO'])

    if missing_linknos:
        missing_gdf = gpd.GeoDataFrame({
            'LINKNO': list(missing_linknos),
            'area_m2': [None] * len(missing_linknos),
            'geometry': [None] * len(missing_linknos)
        }, crs='EPSG:4326')
        result = pd.concat([result, missing_gdf], ignore_index=True)

    return result


def calculate_basin_farmland_wrapper(args):
    return calculate_basin_farmland(*args)


def aggregate_farmland_for_csv(farmland_result):
    """Collapse the per-intersection farmland_result into one row per basin
    for CSV export. Output has exactly two columns: LINKNO and area_m2.
    A TOTAL row is prepended. Basins with no farmland keep a NaN area row
    (min_count=1 prevents the sum from collapsing missing data to 0)."""
    df = pd.DataFrame(farmland_result.drop(columns='geometry', errors='ignore'))

    per_basin = df.groupby('LINKNO', dropna=False, as_index=False)['area_m2'].sum(min_count=1)

    total = per_basin['area_m2'].sum(min_count=1)
    total_row = pd.DataFrame({'LINKNO': ['TOTAL'], 'area_m2': [total]})
    return pd.concat([total_row, per_basin], ignore_index=True)
