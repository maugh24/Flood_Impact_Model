import os
import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds

from tile_loader import read_tiles_in_bbox, resolve_source

BASIN_ID = "HYBAS_ID"

_BUILDING_VALUES = [
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


def _empty_result(ids):
    return pd.DataFrame(data={
        'building': [None] * len(ids),
        'name': [None] * len(ids),
        'amenity': [None] * len(ids),
        BASIN_ID: ids,
        'x': [None] * len(ids),
        'y': [None] * len(ids)
    })


def _load_buildings(building_source, bbox):
    read_kwargs = dict(
        columns=['building', 'name', 'amenity', 'geometry'],
        filters=ds.field("building").is_valid()
    )
    if os.path.isdir(building_source):
        return read_tiles_in_bbox(building_source, '*polygons*.parquet', bbox=tuple(bbox), **read_kwargs)

    import pyarrow.parquet as _pq
    try:
        cols = _pq.read_schema(building_source).names
    except Exception as e:
        raise FileNotFoundError(
            f"building_source is neither a folder nor a readable parquet: {building_source}"
        ) from e
    if 'building' not in cols:
        raise ValueError(
            f"building_source '{building_source}' has no 'building' column "
            f"(looks like a lines file). Point it at the folder of *-polygons-*.parquet files."
        )
    return gpd.read_parquet(building_source, bbox=tuple(bbox), **read_kwargs)


def calculate_basin_buildings(basins, building_source):
    if len(basins) == 0:
        return _empty_result([])

    basins = basins[[BASIN_ID, 'geometry']]
    ids = basins[BASIN_ID].tolist()

    buildings = _load_buildings(building_source, basins.total_bounds)
    if len(buildings) == 0 or 'building' not in buildings.columns:
        return _empty_result(ids)

    buildings = buildings[buildings['building'].isin(_BUILDING_VALUES)]
    if len(buildings) == 0:
        return _empty_result(ids)

    if buildings.crs != basins.crs:
        buildings = buildings.to_crs(basins.crs)

    # Assign each building to the basin its centroid falls in.
    buildings = buildings.set_geometry(buildings.geometry.centroid)

    buildings_in_basins = gpd.sjoin(buildings, basins[[BASIN_ID, 'geometry']],
                                    how='inner', predicate='within')
    buildings_in_basins = buildings_in_basins[~buildings_in_basins.index.duplicated(keep='first')]

    buildings_in_basins['x'] = buildings_in_basins.geometry.x
    buildings_in_basins['y'] = buildings_in_basins.geometry.y
    buildings_in_basins = buildings_in_basins.drop(columns=['geometry', 'index_right'])

    missing = set(ids) - set(buildings_in_basins[BASIN_ID])
    if missing:
        buildings_in_basins = pd.concat([
            buildings_in_basins,
            pd.DataFrame(data={
                'building': [None] * len(missing),
                'name': [None] * len(missing),
                'amenity': [None] * len(missing),
                BASIN_ID: list(missing),
                'x': [None] * len(missing),
                'y': [None] * len(missing)
            })
        ], ignore_index=True)
    return buildings_in_basins


def calculate_basin_building_wrapper(args):
    return calculate_basin_buildings(*args)
