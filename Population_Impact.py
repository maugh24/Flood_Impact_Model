import geopandas as gpd
import pandas as pd


def calculate_basin_population(basin_file, rivers, population_parquet):

    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)]).to_crs(4326)

    if len(basins) == 0:
        return pd.DataFrame(data={
            'linkno': rivers,
            'pop_value': [None] * len(rivers),
            'x': [None] * len(rivers),
            'y': [None] * len(rivers)
        })

    bbox = basins.total_bounds

    pop_gdf = gpd.read_parquet(population_parquet, bbox=bbox)

    if len(pop_gdf) == 0:
        return pd.DataFrame(data={
            'linkno': rivers,
            'pop_value': [None] * len(rivers),
            'x': [None] * len(rivers),
            'y': [None] * len(rivers)
        })

    if pop_gdf.crs != basins.crs:
        pop_gdf = pop_gdf.to_crs(basins.crs)

    # Project to equal-area CRS so area ratios are accurate
    basins_cea = basins.to_crs({'proj': 'cea'})
    pop_cea = pop_gdf.to_crs({'proj': 'cea'}).copy()

    # Record each tile's full area and full pop BEFORE overlay so we can
    # weight by the area fraction that lands in each basin.
    pop_cea['_tile_area'] = pop_cea.geometry.area
    pop_cea['_tile_pop'] = pop_cea['pop_value']

    # Overlay produces one row per (basin, pop_tile) intersection. A tile
    # spanning basins A and B yields one row in A (with A's geometry) and
    # one row in B (with B's geometry).
    intersections = gpd.overlay(
        basins_cea[['linkno', 'geometry']],
        pop_cea[['_tile_area', '_tile_pop', 'geometry']],
        how='intersection'
    )

    if len(intersections) == 0:
        return pd.DataFrame(data={
            'linkno': rivers,
            'pop_value': [None] * len(rivers),
            'x': [None] * len(rivers),
            'y': [None] * len(rivers)
        })

    # Area-weight: pop assigned to basin = tile_pop * (clipped_area / tile_area).
    # If 40% of the tile falls in basin A, basin A gets 40% of its population.
    intersections['_clip_area'] = intersections.geometry.area
    intersections['pop_value'] = (
        intersections['_tile_pop'] * (intersections['_clip_area'] / intersections['_tile_area'])
    )

    # Centroid of the clipped piece, reported in WGS84 for downstream plotting.
    intersections_wgs84 = intersections.to_crs('EPSG:4326')
    centroid = intersections_wgs84.geometry.centroid
    intersections_wgs84['x'] = centroid.x
    intersections_wgs84['y'] = centroid.y

    result = intersections_wgs84[['linkno', 'pop_value', 'x', 'y']].copy()

    # Add missing linknos (basins with no population overlap)
    missing_linknos = set(rivers) - set(result['linkno'])
    if missing_linknos:
        missing_df = pd.DataFrame(data={
            'linkno': list(missing_linknos),
            'pop_value': [None] * len(missing_linknos),
            'x': [None] * len(missing_linknos),
            'y': [None] * len(missing_linknos)
        })
        result = pd.concat([result, missing_df], ignore_index=True)

    return result


def calculate_basin_population_wrapper(args):
    return calculate_basin_population(*args)


def aggregate_population_for_csv(population_result):
    """Collapse the per-intersection population_result into a clean CSV layout:
    one row per basin with pop_value summed, plus a TOTAL row at the top.
    Drops the x/y/geometry clutter used for spatial outputs."""
    df = pd.DataFrame(population_result).drop(
        columns=['x', 'y', 'geometry'], errors='ignore'
    )

    # Sum pop_value per basin. min_count=1 keeps basins with no population as NaN
    # rather than coercing them to 0.
    per_basin = df.groupby('linkno', dropna=False, as_index=False)['pop_value'].sum(min_count=1)

    total = per_basin['pop_value'].sum(min_count=1)
    total_row = pd.DataFrame({'linkno': ['TOTAL'], 'pop_value': [total]})
    return pd.concat([total_row, per_basin], ignore_index=True)
