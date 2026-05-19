import geopandas as gpd
import pandas as pd
import numpy as np

def calculate_basin_transportation(basin_file, rivers, transportation_parquet):
    # Read basins for this chunk
    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)])

    if len(basins) == 0:
        # Return empty GeoDataFrame with expected columns
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

    # Bounding box to limit reading of transportation parquet
    basin_bounds = basins.total_bounds  # [minx, miny, maxx, maxy]

    transportation = gpd.read_parquet(
        transportation_parquet,
        columns=['highway', 'railway', 'name', 'geometry'],
        bbox=basin_bounds
    )

    if len(transportation) == 0:
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

    # ===== VECTORIZED FILTERING + TYPE/VALUE ASSIGNMENT =====
    highway_values = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'living_street', 'busway', 'footway', 'cycleway'
    ]
    railway_values = ['light_rail', 'monorail', 'rail', 'subway', 'tram']

    # Build the highway/railway masks once and reuse them for both the row
    # filter and the type/value assignment. Avoids the per-row .apply() call,
    # which was O(N) Python overhead on every OSM feature.
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

    # Keep only rows that matched at least one tag set, then re-align the
    # masks to the filtered DataFrame.
    transportation = transportation[is_highway | is_railway].copy()
    is_highway = is_highway.loc[transportation.index]
    is_railway = is_railway.loc[transportation.index]

    if len(transportation) == 0:
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

    # Vectorized type/value assignment. Highway takes precedence when a row
    # matches both tag sets (matches the ordering of the old per-row logic).
    # Since every surviving row is is_highway OR is_railway, the "else"
    # branch of the np.where always corresponds to a railway match.
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

    # Clip (intersection) - this returns the geometry segments inside basins and brings basin attributes in
    transportation_in_basins = gpd.overlay(
        transportation_cea,
        basins_cea,
        how='intersection'
    )

    if len(transportation_in_basins) == 0:
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

    # Overlay can return MultiLineStrings when a road's intersection with a
    # basin isn't connected - e.g., a highway that crosses a wiggly basin
    # boundary multiple times, or a basin shape that touches the same road
    # in two separate places. Previously we filtered to LineString only,
    # which silently DROPPED those MultiLineString rows and erased real
    # road segments from the output. Explode them into individual
    # LineStrings instead so every piece is preserved.
    transportation_in_basins = transportation_in_basins.explode(index_parts=False, ignore_index=True)

    # After exploding, drop anything that isn't a LineString (e.g., Points
    # from a road that only touched a basin corner, or empty geometries).
    transportation_in_basins = transportation_in_basins[transportation_in_basins.geometry.geom_type == 'LineString']

    if len(transportation_in_basins) == 0:
        # no line geometries remain
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

    # Compute length in meters (CEA units) and km
    transportation_in_basins['length_m'] = transportation_in_basins.geometry.length
    transportation_in_basins['length_km'] = transportation_in_basins['length_m'] / 1000.0

    # Reduce to required columns and reproject to EPSG:4326 for output
    out_gdf = transportation_in_basins[['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry']].copy()
    out_gdf = out_gdf.set_geometry('geometry').to_crs('EPSG:4326')

    return out_gdf


def calculate_basin_transportation_wrapper(args):
    return calculate_basin_transportation(*args)


def aggregate_transportation_for_csv(transportation_result):
    """Collapse the per-segment transportation_result into one row per basin
    with separate columns for highway and railway cumulative length (km).
    A TOTAL row sums each column independently. Geometry and length_m are
    dropped - the parquet retains those for spatial use."""
    df = pd.DataFrame(transportation_result.drop(columns='geometry', errors='ignore'))

    # Pivot: one row per linkno, one column per infrastructure_type, summed length_km.
    pivot = (
        df.groupby(['linkno', 'infrastructure_type'])['length_km']
          .sum(min_count=1)
          .unstack('infrastructure_type')
    )

    # Rename columns and guarantee both exist even if one type is absent.
    pivot = pivot.rename(columns={'highway': 'highway_km', 'railway': 'railway_km'})
    for col in ('highway_km', 'railway_km'):
        if col not in pivot.columns:
            pivot[col] = None

    per_basin = pivot.reset_index()[['linkno', 'highway_km', 'railway_km']]

    # TOTAL row sums each column independently (so highway and railway totals
    # are reported separately, not merged).
    total_row = pd.DataFrame({
        'linkno': ['TOTAL'],
        'highway_km': [per_basin['highway_km'].sum(min_count=1)],
        'railway_km': [per_basin['railway_km'].sum(min_count=1)],
    })
    return pd.concat([total_row, per_basin], ignore_index=True)