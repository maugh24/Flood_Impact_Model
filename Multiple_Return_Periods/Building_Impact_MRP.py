"""
Building impact within a single flood extent (no basins, no linknos).

Uses centroid-in-polygon containment so each building is counted at most
once: a building whose polygon straddles the flood boundary is included
only if its centroid lies inside the flood extent.
"""
import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds


def _empty_result():
    return pd.DataFrame(data={
        'building': [],
        'name': [],
        'amenity': [],
        'x': [],
        'y': []
    })


def calculate_flood_buildings(flood_parquet, building_parquet):
    flood = gpd.read_parquet(flood_parquet)
    if len(flood) == 0:
        return _empty_result()

    flood_bounds = flood.total_bounds
    buildings = gpd.read_parquet(
        building_parquet,
        columns=['building', 'name', 'amenity', 'geometry'],
        bbox=flood_bounds,
        filters=ds.field("building").is_valid()
    )
    if len(buildings) == 0:
        return _empty_result()

    building_values = [
        'yes', 'apartments', 'industrial', 'commercial', 'retail', 'residential',
        'civic', 'house', 'policlinic', 'hotel', 'stadium', 'church', 'government',
        'hospital', 'school', 'college', 'fire_station', 'university', 'monastery',
        'public', 'office', 'terminal', 'castle', 'ruins', 'garage', 'garages',
        'shed', 'barracks', 'bungalow', 'cabin', 'detached', 'annexe', 'dormitory',
        'farm', 'ger', 'boathouse', 'semidetached_house', 'static_caravan',
        'stilt_house', 'terrace', 'tree_house', 'trullo', 'kiosk', 'supermarket',
        'warehouse', 'religious', 'cathedral', 'chapel', 'military', 'houseboat',
        'kingdom_hall', 'mosque', 'presbytery', 'shrine', 'synagogue', 'temple',
        'bakehouse', 'bridge', 'clock_tower', 'gatehouse', 'kindergarten', 'museum',
        'toilets', 'train_station', 'barn', 'conservatory', 'cowshed',
        'farm_auxiliary', 'greenhouse', 'slurry_tank', 'stable', 'sty', 'livestock',
        'grandstand', 'pavilion', 'riding_hall', 'sports_hall', 'sports_centre',
        'allotment_house', 'hangar', 'hut', 'carport', 'parking', 'digester',
        'service', 'tech_cab', 'transformer_tower', 'water_tower', 'storage_tank',
        'silo', 'beach_hut', 'bunker', 'construction', 'container', 'guardhouse',
        'outbuilding', 'pagoda', 'quonset_hut', 'roof', 'ship', 'tent', 'tower',
        'triumphal_arch', 'windmill'
    ]

    if 'building' not in buildings.columns:
        return _empty_result()

    buildings = buildings[buildings['building'].isin(building_values)]
    if len(buildings) == 0:
        return _empty_result()

    if buildings.crs != flood.crs:
        buildings = buildings.to_crs(flood.crs)

    # Centroid-in-polygon: each building gets assigned to the flood polygon
    # that contains its centroid (or dropped if no flood polygon contains it).
    # Prevents double-counting of buildings whose polygons straddle the
    # flood boundary - same correctness fix as the basin pipeline.
    buildings_centroids = buildings.copy()
    buildings_centroids['geometry'] = buildings.geometry.centroid

    buildings_in_flood = gpd.sjoin(
        buildings_centroids,
        flood,
        how='inner',
        predicate='within'
    )
    buildings_in_flood = buildings_in_flood[~buildings_in_flood.index.duplicated(keep='first')]

    buildings_in_flood['x'] = buildings_in_flood.geometry.x
    buildings_in_flood['y'] = buildings_in_flood.geometry.y
    buildings_in_flood = buildings_in_flood.drop(columns=['geometry', 'index_right'])
    return buildings_in_flood


def aggregate_buildings_for_csv(building_result):
    """Single-row CSV: count of flooded buildings."""
    if len(building_result) == 0 or 'building' not in building_result.columns:
        count = 0
    else:
        count = building_result['building'].notna().sum()
    return pd.DataFrame({'building_count': [int(count)]})
