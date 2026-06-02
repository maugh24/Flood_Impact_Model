"""
Population impact within a single flood extent (no basins, no linknos).

Mirrors the area-weighted overlay logic from the basin-based pipeline:
each population tile contributes a share of its pop_value proportional to
the fraction of its area that falls inside the flood polygon.
"""
import geopandas as gpd
import pandas as pd


def calculate_flood_population(flood_parquet, population_parquet):
    flood = gpd.read_parquet(flood_parquet)
    if len(flood) == 0:
        return pd.DataFrame(data={'pop_value': [], 'x': [], 'y': []})

    # Make sure we're in EPSG:4326 to match the population parquet convention.
    if flood.crs is None or flood.crs.to_epsg() != 4326:
        flood = flood.to_crs(4326)

    bbox = flood.total_bounds
    pop_gdf = gpd.read_parquet(population_parquet, bbox=bbox)

    if len(pop_gdf) == 0:
        return pd.DataFrame(data={'pop_value': [], 'x': [], 'y': []})

    if pop_gdf.crs != flood.crs:
        pop_gdf = pop_gdf.to_crs(flood.crs)

    # Equal-area projection so area ratios are accurate
    flood_cea = flood.to_crs({'proj': 'cea'})
    pop_cea = pop_gdf.to_crs({'proj': 'cea'}).copy()

    # Record each tile's full area and full pop BEFORE overlay so we can
    # weight by the area fraction that lands inside the flood polygon.
    pop_cea['_tile_area'] = pop_cea.geometry.area
    pop_cea['_tile_pop'] = pop_cea['pop_value']

    intersections = gpd.overlay(
        flood_cea[['geometry']],
        pop_cea[['_tile_area', '_tile_pop', 'geometry']],
        how='intersection'
    )

    if len(intersections) == 0:
        return pd.DataFrame(data={'pop_value': [], 'x': [], 'y': []})

    # Area-weight: pop assigned to flood = tile_pop * (clipped_area / tile_area)
    intersections['_clip_area'] = intersections.geometry.area
    intersections['pop_value'] = (
        intersections['_tile_pop'] * (intersections['_clip_area'] / intersections['_tile_area'])
    )

    # Centroid of each clipped piece, reported in WGS84 for downstream plotting.
    intersections_wgs84 = intersections.to_crs('EPSG:4326')
    centroid = intersections_wgs84.geometry.centroid
    intersections_wgs84['x'] = centroid.x
    intersections_wgs84['y'] = centroid.y

    return intersections_wgs84[['pop_value', 'x', 'y']].copy()


def aggregate_population_for_csv(population_result):
    """Single-row CSV: total flooded population."""
    df = pd.DataFrame(population_result).drop(columns=['x', 'y', 'geometry'], errors='ignore')
    total = df['pop_value'].sum(min_count=1) if len(df) else None
    return pd.DataFrame({'pop_value': [total]})
