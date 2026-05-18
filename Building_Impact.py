import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds

def calculate_basin_buildings(basin_file, rivers, building_parquet):

    basins = gpd.read_parquet(basin_file, filters=[('linkno', 'in', rivers)])

    if len(basins) == 0:
        return pd.DataFrame(data={'building': [None]*len(rivers),
                                  'name': [None]*len(rivers),
                                  'amenity': [None]*len(rivers),
                                  'linkno': rivers,
                                   'x': [None]*len(rivers),
                                    'y': [None]*len(rivers)})

    # Get bounding box of basins for spatial filtering
    basin_bounds = basins.total_bounds  # [minx, miny, maxx, maxy]

    buildings = gpd.read_parquet(building_parquet,
                                 columns=['building','name','amenity','geometry'],
                                 bbox=basin_bounds,
                                 filters=ds.field("building").is_valid()
                                 )
    # buildings.to_parquet(bbox_building_parquet,write_covering_bbox=True)

    if len(buildings) == 0:
        return pd.DataFrame(data={'building': [None]*len(rivers),
                                  'name': [None]*len(rivers),
                                  'amenity': [None]*len(rivers),
                                  'linkno': rivers,
                                   'x': [None]*len(rivers),
                                    'y': [None]*len(rivers)})

    # ===== VECTORIZED FILTERING =====
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

    # Check building column exists
    if 'building' not in buildings.columns:
        return pd.DataFrame(data={'building': [None]*len(rivers),
                                  'name': [None]*len(rivers),
                                  'amenity': [None]*len(rivers),
                                  'linkno': rivers,
                                   'x': [None]*len(rivers),
                                    'y': [None]*len(rivers)})

    # Filter by building type
    buildings = buildings[buildings['building'].isin(building_values)]

    if len(buildings) == 0:
        return pd.DataFrame(data={'building': [None]*len(rivers),
                                  'name': [None]*len(rivers),
                                  'amenity': [None]*len(rivers),
                                  'linkno': rivers,
                                   'x': [None]*len(rivers),
                                    'y': [None]*len(rivers)})

    # ===== SPATIAL JOIN WITH BASINS =====
    # Ensure same CRS
    if buildings.crs != basins.crs:
        buildings = buildings.to_crs(basins.crs)

    # Vectorized spatial join
    buildings_in_basins = gpd.sjoin(
        buildings,
        basins,
        how='inner',
        predicate='intersects'
    )

    centroid = buildings_in_basins.centroid
    buildings_in_basins['x'] = centroid.x
    buildings_in_basins['y'] = centroid.y
    buildings_in_basins = buildings_in_basins.drop(columns=['geometry', 'index_right'])
    missing_linknos = set(rivers) - set(buildings_in_basins['linkno'])
    if missing_linknos:
        buildings_in_basins = pd.concat([buildings_in_basins, pd.DataFrame(data={'building': [None]*len(missing_linknos),
                                  'name': [None]*len(missing_linknos),
                                  'amenity': [None]*len(missing_linknos),
                                  'linkno': list(missing_linknos),
                                   'x': [None]*len(missing_linknos),
                                    'y': [None]*len(missing_linknos)})], ignore_index=True)
    return buildings_in_basins

def calculate_basin_building_wrapper(args):
    return calculate_basin_buildings(*args)