import geopandas as gpd
import pandas as pd

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

    # ===== VECTORIZED FILTERING (same filters you already use) =====
    highway_values = [
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'living_street', 'busway', 'footway', 'cycleway'
    ]
    railway_values = ['light_rail', 'monorail', 'rail', 'subway', 'tram']

    filter_criteria = pd.Series(False, index=transportation.index)
    if 'highway' in transportation.columns:
        filter_criteria |= transportation['highway'].isin(highway_values)
    if 'railway' in transportation.columns:
        filter_criteria |= transportation['railway'].isin(railway_values)

    transportation = transportation[filter_criteria]

    if len(transportation) == 0:
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

    # Assign type and value
    def assign_type(row):
        if 'highway' in row and pd.notna(row.get('highway')) and row['highway'] in highway_values:
            return 'highway', row['highway']
        if 'railway' in row and pd.notna(row.get('railway')) and row['railway'] in railway_values:
            return 'railway', row['railway']
        return None, None

    transportation[['infrastructure_type', 'feature_value']] = transportation.apply(assign_type, axis=1, result_type='expand')
    transportation = transportation[transportation['infrastructure_type'].notna()]

    if len(transportation) == 0:
        return gpd.GeoDataFrame(columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'], geometry='geometry', crs='EPSG:4326')

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

    # Keep only pure LineString geometries (drop MultiLineString); if you later want to explode MultiLineStrings, replace this filter
    transportation_in_basins = transportation_in_basins[transportation_in_basins.geometry.geom_type == 'LineString']

    if len(transportation_in_basins) == 0:
        # no pure LineStrings remain
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