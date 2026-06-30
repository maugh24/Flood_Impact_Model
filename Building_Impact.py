import os
import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds

from tile_loader import read_tiles_in_bbox, resolve_source


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


def _empty_result(rivers):
    return pd.DataFrame(data={
        'building': [None] * len(rivers),
        'name': [None] * len(rivers),
        'amenity': [None] * len(rivers),
        'LINKNO': rivers,
        'x': [None] * len(rivers),
        'y': [None] * len(rivers)
    })


def _load_buildings(building_source, bbox):
    """Read OSM buildings inside bbox.

    building_source may be a folder of per-continent polygon parquets
    (Global_Run case) or a single pre-merged parquet (legacy case).
    """
    read_kwargs = dict(
        columns=['building', 'name', 'amenity', 'geometry'],
        filters=ds.field("building").is_valid()
    )
    folder, fallback_path = resolve_source(building_source, '*polygons*.parquet')
    if folder is not None:
        return read_tiles_in_bbox(folder, '*polygons*.parquet', bbox=tuple(bbox), **read_kwargs)
    return gpd.read_parquet(fallback_path, bbox=tuple(bbox), **read_kwargs)


def calculate_basin_buildings(basin_file, rivers, building_source):
    basins = gpd.read_parquet(basin_file, filters=[('LINKNO', 'in', rivers)])

    if len(basins) == 0:
        return _empty_result(rivers)

    basin_bounds = basins.total_bounds  # [minx, miny, maxx, maxy]

    buildings = _load_buildings(building_source, basin_bounds)

    if len(buildings) == 0:
        return _empty_result(rivers)

    if 'building' not in buildings.columns:
        return _empty_result(rivers)

    buildings = buildings[buildings['building'].isin(_BUILDING_VALUES)]

    if len(buildings) == 0:
        return _empty_result(rivers)

    # ===== SPATIAL JOIN WITH BASINS =====
    if buildings.crs != basins.crs:
        buildings = buildings.to_crs(basins.crs)

    # Centroid-in-basin assignment so a building straddling a basin boundary
    # only gets counted once (the basin its centroid falls in). Replace the
    # polygon geometry with its centroid in place instead of .copy()-ing the
    # whole frame first: the copy held the heavy building polygons in memory
    # twice, which was a major source of the building-pass memory spike.
    # Swapping polygons for points also makes the sjoin below far lighter.
    buildings = buildings.set_geometry(buildings.geometry.centroid)

    # Only the basin id is needed from the join. Restricting basins to
    # [LINKNO, geometry] keeps the joined frame small (the catchment file
    # carries many other attribute columns that sjoin would otherwise copy
    # onto every building row).
    basins_join = basins[['LINKNO', 'geometry']]

    buildings_in_basins = gpd.sjoin(
        buildings,
        basins_join,
        how='inner',
        predicate='within'
    )

    # Defensive dedup (centroid lands exactly on an interior basin boundary).
    buildings_in_basins = buildings_in_basins[~buildings_in_basins.index.duplicated(keep='first')]

    buildings_in_basins['x'] = buildings_in_basins.geometry.x
    buildings_in_basins['y'] = buildings_in_basins.geometry.y
    buildings_in_basins = buildings_in_basins.drop(columns=['geometry', 'index_right'])

    missing_linknos = set(rivers) - set(buildings_in_basins['LINKNO'])
    if missing_linknos:
        buildings_in_basins = pd.concat([
            buildings_in_basins,
            pd.DataFrame(data={
                'building': [None] * len(missing_linknos),
                'name': [None] * len(missing_linknos),
                'amenity': [None] * len(missing_linknos),
                'LINKNO': list(missing_linknos),
                'x': [None] * len(missing_linknos),
                'y': [None] * len(missing_linknos)
            })
        ], ignore_index=True)
    return buildings_in_basins


def calculate_basin_building_wrapper(args):
    return calculate_basin_buildings(*args)
