"""
Transportation impact within a single flood extent (no basins, no linknos).

Mirrors the basin pipeline: vectorized highway/railway filter and type
assignment, then overlay with the flood polygon and explode any
MultiLineStrings so segments aren't silently dropped.
"""
import geopandas as gpd
import pandas as pd
import numpy as np


def _empty_result():
    return gpd.GeoDataFrame(
        columns=['infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'],
        geometry='geometry',
        crs='EPSG:4326'
    )


def calculate_flood_transportation(flood_parquet, transportation_parquet):
    flood = gpd.read_parquet(flood_parquet)
    if len(flood) == 0:
        return _empty_result()

    flood_bounds = flood.total_bounds
    transportation = gpd.read_parquet(
        transportation_parquet,
        columns=['highway', 'railway', 'name', 'geometry'],
        bbox=flood_bounds
    )
    if len(transportation) == 0:
        return _empty_result()

    highway_values = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'living_street', 'busway', 'footway', 'cycleway'
    ]
    railway_values = ['light_rail', 'monorail', 'rail', 'subway', 'tram']

    # Vectorized filter + type assignment (no per-row .apply).
    has_highway_col = 'highway' in transportation.columns
    has_railway_col = 'railway' in transportation.columns

    is_highway = (
        transportation['highway'].isin(highway_values)
        if has_highway_col else
        pd.Series(False, index=transportation.index)
    )
    is_railway = (
        transportation['railway'].isin(railway_values)
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

    if transportation.crs != flood.crs:
        transportation = transportation.to_crs(flood.crs)

    # Equal-area projection for accurate length / clip
    transportation_cea = transportation.to_crs({'proj': 'cea'})
    flood_cea = flood.to_crs(transportation_cea.crs)

    transportation_in_flood = gpd.overlay(transportation_cea, flood_cea, how='intersection')
    if len(transportation_in_flood) == 0:
        return _empty_result()

    # Explode MultiLineStrings into individual LineStrings so road segments
    # crossing the flood boundary multiple times aren't dropped.
    transportation_in_flood = transportation_in_flood.explode(index_parts=False, ignore_index=True)
    transportation_in_flood = transportation_in_flood[transportation_in_flood.geometry.geom_type == 'LineString']

    if len(transportation_in_flood) == 0:
        return _empty_result()

    transportation_in_flood['length_m'] = transportation_in_flood.geometry.length
    transportation_in_flood['length_km'] = transportation_in_flood['length_m'] / 1000.0

    out = transportation_in_flood[['infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry']].copy()
    out = out.set_geometry('geometry').to_crs('EPSG:4326')
    return out


def aggregate_transportation_for_csv(transportation_result):
    """Single-row CSV: highway_km and railway_km totals inside the flood extent."""
    df = pd.DataFrame(transportation_result.drop(columns='geometry', errors='ignore'))
    if len(df) == 0:
        return pd.DataFrame({'highway_km': [None], 'railway_km': [None]})

    by_type = df.groupby('infrastructure_type')['length_km'].sum(min_count=1)
    return pd.DataFrame({
        'highway_km': [by_type.get('highway', None)],
        'railway_km': [by_type.get('railway', None)]
    })
