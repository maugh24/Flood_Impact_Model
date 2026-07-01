import os
import geopandas as gpd
import pandas as pd
import numpy as np

from Global_Run_TDX.tile_loader import read_tiles_in_bbox, resolve_source


_HIGHWAY_VALUES = [
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'busway', 'footway', 'cycleway'
]
_RAILWAY_VALUES = ['light_rail', 'monorail', 'rail', 'subway', 'tram']


def _empty_result():
    return gpd.GeoDataFrame(
        columns=['LINKNO', 'infrastructure_type', 'feature_value',
                 'length_m', 'length_km', 'geometry'],
        geometry='geometry', crs='EPSG:4326'
    )


def _load_transportation(transport_source, bbox):
    """Read OSM line features inside bbox.

    transport_source may be a folder of per-continent line parquets
    (Global_Run case) or a single pre-merged parquet (legacy case).
    """
    read_kwargs = dict(columns=['highway', 'railway', 'name', 'geometry'])
    folder, fallback_path = resolve_source(transport_source, '*lines*.parquet')
    if folder is not None:
        return read_tiles_in_bbox(folder, '*lines*.parquet', bbox=tuple(bbox), **read_kwargs)
    return gpd.read_parquet(fallback_path, bbox=tuple(bbox), **read_kwargs)


def calculate_basin_transportation(basin_file, rivers, transport_source):
    basins = gpd.read_parquet(basin_file, filters=[('LINKNO', 'in', rivers)])

    if len(basins) == 0:
        return _empty_result()

    basin_bounds = basins.total_bounds  # [minx, miny, maxx, maxy]

    transportation = _load_transportation(transport_source, basin_bounds)

    if len(transportation) == 0:
        return _empty_result()

    # ===== VECTORIZED FILTERING + TYPE/VALUE ASSIGNMENT =====
    has_highway_col = 'highway' in transportation.columns
    has_railway_col = 'railway' in transportation.columns

    is_highway = (
        transportation['highway'].isin(_HIGHWAY_VALUES)
        if has_highway_col else
        pd.Series(False, index=transportation.index)
    )
    is_railway = (
        transportation['railway'].isin(_RAILWAY_VALUES)
        if has_railway_col else
        pd.Series(False, index=transportation.index)
    )

    transportation = transportation[is_highway | is_railway].copy()
    is_highway = is_highway.loc[transportation.index]
    is_railway = is_railway.loc[transportation.index]

    if len(transportation) == 0:
        return _empty_result()

    hw_vals = transportation['highway'] if has_highway_col else pd.Series(None, index=transportation.index, dtype=object)
    rw_vals = transportation['railway'] if has_railway_col else pd.Series(None, index=transportation.index, dtype=object)
    transportation['infrastructure_type'] = np.where(is_highway, 'highway', 'railway')
    transportation['feature_value'] = np.where(is_highway, hw_vals, rw_vals)

    # Ensure same CRS as basins
    if transportation.crs != basins.crs:
        transportation = transportation.to_crs(basins.crs)

    # Reproject to equal-area for accurate length/clip
    transportation_cea = transportation.to_crs({'proj': 'cea'})
    basins_cea = basins.to_crs(transportation_cea.crs)

    transportation_in_basins = gpd.overlay(
        transportation_cea,
        basins_cea,
        how='intersection'
    )

    if len(transportation_in_basins) == 0:
        return _empty_result()

    # Explode MultiLineStrings into individual LineStrings (so road segments
    # crossing a basin boundary multiple times aren't dropped).
    transportation_in_basins = transportation_in_basins.explode(index_parts=False, ignore_index=True)
    transportation_in_basins = transportation_in_basins[transportation_in_basins.geometry.geom_type == 'LineString']

    if len(transportation_in_basins) == 0:
        return _empty_result()

    transportation_in_basins['length_m'] = transportation_in_basins.geometry.length
    transportation_in_basins['length_km'] = transportation_in_basins['length_m'] / 1000.0

    out_gdf = transportation_in_basins[['LINKNO', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry']].copy()
    out_gdf = out_gdf.set_geometry('geometry').to_crs('EPSG:4326')
    return out_gdf


def calculate_basin_transportation_wrapper(args):
    return calculate_basin_transportation(*args)


def aggregate_transportation_for_csv(transportation_result):
    """Per-basin highway_km + railway_km columns with a TOTAL row."""
    df = pd.DataFrame(transportation_result.drop(columns='geometry', errors='ignore'))

    pivot = (
        df.groupby(['LINKNO', 'infrastructure_type'])['length_km']
          .sum(min_count=1)
          .unstack('infrastructure_type')
    )

    pivot = pivot.rename(columns={'highway': 'highway_km', 'railway': 'railway_km'})
    for col in ('highway_km', 'railway_km'):
        if col not in pivot.columns:
            pivot[col] = None

    per_basin = pivot.reset_index()[['LINKNO', 'highway_km', 'railway_km']]

    total_row = pd.DataFrame({
        'LINKNO': ['TOTAL'],
        'highway_km': [per_basin['highway_km'].sum(min_count=1)],
        'railway_km': [per_basin['railway_km'].sum(min_count=1)],
    })
    return pd.concat([total_row, per_basin], ignore_index=True)
