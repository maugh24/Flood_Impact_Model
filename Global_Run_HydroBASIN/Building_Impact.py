import os
import json
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

from tile_loader import _glob_files, _file_bbox, _bboxes_overlap

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

_EMPTY_CENTERS = gpd.GeoDataFrame({'building': [], 'name': [], 'amenity': []},
                                  geometry=[], crs='EPSG:4326')


def _empty_result(ids):
    return pd.DataFrame(data={
        'building': [None] * len(ids),
        'name': [None] * len(ids),
        'amenity': [None] * len(ids),
        BASIN_ID: ids,
        'x': [None] * len(ids),
        'y': [None] * len(ids)
    })


def _covering_col(path):
    """Name of the GeoParquet covering bbox column ('bbox' or 'geometry_bbox')."""
    kv = pq.read_metadata(path).metadata or {}
    if b'geo' not in kv:
        return None
    geo = json.loads(kv[b'geo'])
    primary = geo.get('primary_column', 'geometry')
    cov = geo.get('columns', {}).get(primary, {}).get('covering', {}).get('bbox')
    return cov['xmin'][0] if cov else None


def _read_centers(path, bbox):
    """Read building centroids inside bbox from one polygon tile using ONLY the
    covering bbox column (center of each building's bbox) - the heavy polygon
    geometry column is never read. Falls back to true centroids if a file has
    no covering bbox."""
    names = pq.read_schema(path).names
    if 'building' not in names:
        return _EMPTY_CENTERS
    col = _covering_col(path)
    if col is None:
        return _read_geometry_centroids(path, bbox)

    minx, miny, maxx, maxy = bbox
    cols = [c for c in ('building', 'name', 'amenity') if c in names] + [col]
    filt = ((pc.field(col, 'xmax') >= minx) & (pc.field(col, 'xmin') <= maxx) &
            (pc.field(col, 'ymax') >= miny) & (pc.field(col, 'ymin') <= maxy) &
            pc.field('building').isin(_BUILDING_VALUES))
    t = pq.read_table(path, columns=cols, filters=filt)
    if t.num_rows == 0:
        return _EMPTY_CENTERS

    bbc = t.column(col)
    cx = (pc.struct_field(bbc, 'xmin').to_numpy(zero_copy_only=False) +
          pc.struct_field(bbc, 'xmax').to_numpy(zero_copy_only=False)) / 2.0
    cy = (pc.struct_field(bbc, 'ymin').to_numpy(zero_copy_only=False) +
          pc.struct_field(bbc, 'ymax').to_numpy(zero_copy_only=False)) / 2.0
    data = {c: (t.column(c).to_pandas() if c in names else [None] * t.num_rows)
            for c in ('building', 'name', 'amenity')}
    return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(cx, cy), crs='EPSG:4326')


def _read_geometry_centroids(path, bbox):
    """Fallback for a tile with no covering bbox: read the polygons and use true
    centroids."""
    g = gpd.read_parquet(path, bbox=tuple(bbox), columns=['building', 'name', 'amenity', 'geometry'])
    if len(g) == 0 or 'building' not in g.columns:
        return _EMPTY_CENTERS
    g = g[g['building'].isin(_BUILDING_VALUES)]
    if len(g) == 0:
        return _EMPTY_CENTERS
    return g.set_geometry(g.geometry.centroid)[['building', 'name', 'amenity', 'geometry']]


def _load_buildings(building_source, bbox):
    bbox = tuple(bbox)
    if os.path.isdir(building_source):
        files = _glob_files(building_source, '*polygons*.parquet')
    else:
        if 'building' not in pq.read_schema(building_source).names:
            raise ValueError(
                f"building_source '{building_source}' has no 'building' column "
                f"(looks like a lines file). Point it at the *-polygons-*.parquet folder/file."
            )
        files = [building_source]

    frames = []
    for f in files:
        fb = _file_bbox(f)
        if fb is not None and not _bboxes_overlap(fb, bbox):
            continue
        g = _read_centers(f, bbox)
        if len(g):
            frames.append(g)

    if not frames:
        return _EMPTY_CENTERS
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs='EPSG:4326')


def calculate_basin_buildings(basins, building_source):
    if len(basins) == 0:
        return _empty_result([])

    basins = basins[[BASIN_ID, 'geometry']]
    ids = basins[BASIN_ID].tolist()

    # Buildings arrive as centroid points (from the covering bbox center).
    buildings = _load_buildings(building_source, basins.total_bounds)
    if len(buildings) == 0:
        return _empty_result(ids)

    if buildings.crs != basins.crs:
        buildings = buildings.to_crs(basins.crs)

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
