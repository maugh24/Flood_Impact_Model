"""
Layer processors: read one batch of row groups from a prepped GeoParquet,
assign every feature (or piece of a feature) to an area of interest (AOI),
and return a small per-AOI aggregate table.

The AOI (H3 cells or a polygon set, see aoi.py) is passed in as `a`; this
file only decides what point or piece of each feature is assigned:
  population  pixel polygon bbox centre -> AOI, sum pop_value
  buildings   footprint bbox centre     -> AOI, count
  farmland    cropland blocks (Data_Prep/Farmland/farmland_prep.py): block
              centre -> AOI, sum area_m2
              older polygon files: small polygons by bbox centre, large ones
              clipped exactly to the overlapping AOIs
  roads       lines densified, each segment's midpoint -> AOI, sum length
              (roads + railways combined into transportation_km); road/rail
              segments tagged bridge=* also counted (bbox centre -> AOI)
  dams        waterway=dam / weir / check_dam from the Lines, Polygons and
              Points files, bbox centre -> AOI, count

All functions are top-level so they pickle for multiprocessing on Windows.
"""
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely

from aoi import lonlat_area_m2

# Authalic sphere radius (m), used for road-segment lengths.
EARTH_R = 6371007.180918475

# ---------------------------------------------------------------------------
# Feature classes (kept consistent with the catchment pipeline)
# ---------------------------------------------------------------------------
BUILDING_VALUES = [
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
MAJOR_ROADS = ['motorway', 'trunk', 'primary',
               'motorway_link', 'trunk_link', 'primary_link']
MINOR_ROADS = ['secondary', 'tertiary', 'residential', 'secondary_link',
               'tertiary_link', 'living_street', 'busway', 'footway', 'cycleway']
RAILWAYS = ['light_rail', 'monorail', 'rail', 'subway', 'tram']
DAMS = ['dam']                      # waterway=* values counted as dams
WEIRS = ['weir', 'check_dam']       # smaller barriers, counted separately

# Output columns produced by each layer.
LAYER_COLUMNS = {
    "population": ["population"],
    "buildings":  ["buildings"],
    "farmland":   ["farmland_km2"],
    "roads":      ["transportation_km", "bridges"],
    "dams":       ["dams", "weirs"],
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def sum_by_aoi(ids, **values):
    """Group equal AOI ids and sum each value array -> DataFrame(aoi, *values).
    Ids of -1 (not inside any AOI) are dropped."""
    ids = np.asarray(ids, dtype=np.int64)
    keep = ids >= 0
    if not keep.all():
        ids = ids[keep]
        values = {k: np.asarray(v, dtype=float)[keep] for k, v in values.items()}
    if len(ids) == 0:
        return pd.DataFrame({'aoi': np.array([], dtype=np.int64),
                             **{k: np.array([], dtype=float) for k in values}})
    uniq, inv = np.unique(ids, return_inverse=True)
    out = {'aoi': uniq}
    for k, v in values.items():
        out[k] = np.bincount(inv, weights=np.asarray(v, dtype=float), minlength=len(uniq))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# GeoParquet reading helpers
# ---------------------------------------------------------------------------
def file_geo_info(path):
    """Return (geometry_column, crs_projjson_or_None, covering_bbox_column_or_None)."""
    schema = pq.read_schema(path)
    md = schema.metadata or {}
    geom_col, crs, cover = None, None, None
    if b'geo' in md:
        geo = json.loads(md[b'geo'])
        geom_col = geo.get('primary_column', 'geometry')
        cmeta = geo.get('columns', {}).get(geom_col, {})
        crs = cmeta.get('crs')
        cov = (cmeta.get('covering') or {}).get('bbox')
        if cov and cov.get('xmin'):
            cover = cov['xmin'][0]
    if geom_col is None or geom_col not in schema.names:
        # File written without GeoParquet metadata: find the WKB column.
        if 'geometry' in schema.names:
            geom_col = 'geometry'
        else:
            geom_col = next((f.name for f in schema
                             if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type)), None)
    if cover is None and 'bbox' in schema.names and pa.types.is_struct(schema.field('bbox').type):
        cover = 'bbox'
    return geom_col, crs, cover


@lru_cache(maxsize=32)
def _transformer(crs_json):
    """pyproj transformer to lon/lat, or None if the CRS is already geographic."""
    if crs_json is None:
        return None
    from pyproj import CRS, Transformer
    src = CRS.from_json_dict(json.loads(crs_json))
    if src.is_geographic:
        return None
    return Transformer.from_crs(src, "EPSG:4326", always_xy=True)


def _crs_key(crs):
    return None if crs is None else json.dumps(crs, sort_keys=True)


def points_to_lonlat(x, y, crs):
    tr = _transformer(_crs_key(crs))
    if tr is None:
        return x, y
    return tr.transform(x, y)


def geoms_to_lonlat(geoms, crs):
    tr = _transformer(_crs_key(crs))
    if tr is not None:
        geoms = shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    return geoms


def _check_degrees(lng, lat, path):
    if len(lng) and (np.nanmax(np.abs(lng)) > 180.001 or np.nanmax(np.abs(lat)) > 90.001):
        raise ValueError(f"{path}: coordinates are not lon/lat degrees and the file has "
                         f"no CRS metadata to reproject from.")


def read_table(path, row_groups, columns):
    pf = pq.ParquetFile(path)
    cols = [c for c in dict.fromkeys(columns) if c is not None and c in pf.schema_arrow.names]
    return pf.read_row_groups(row_groups, columns=cols)


def decode_geoms(table, geom_col):
    # on_invalid='ignore': malformed geometries (e.g. 1-point OSM lines) become
    # None and are skipped downstream instead of crashing the task.
    return shapely.from_wkb(table.column(geom_col).to_numpy(zero_copy_only=False),
                            on_invalid='ignore')


def feature_centres(table, geom_col, cover, crs, path):
    """lng, lat of each feature's bbox centre (uses the covering column when
    present, so geometry is never decoded)."""
    if cover is not None and cover in table.column_names:
        b = table.column(cover).combine_chunks()
        x = (pc.struct_field(b, 'xmin').to_numpy(zero_copy_only=False) +
             pc.struct_field(b, 'xmax').to_numpy(zero_copy_only=False)) / 2
        y = (pc.struct_field(b, 'ymin').to_numpy(zero_copy_only=False) +
             pc.struct_field(b, 'ymax').to_numpy(zero_copy_only=False)) / 2
    else:
        bnd = shapely.bounds(decode_geoms(table, geom_col))
        x = (bnd[:, 0] + bnd[:, 2]) / 2
        y = (bnd[:, 1] + bnd[:, 3]) / 2
    lng, lat = points_to_lonlat(np.asarray(x, float), np.asarray(y, float), crs)
    lng, lat = np.asarray(lng), np.asarray(lat)
    _check_degrees(lng, lat, path)
    return lng, lat


def _isin(table, col, values):
    if col not in table.column_names:
        return np.zeros(table.num_rows, dtype=bool)
    arr = table.column(col)
    mask = pc.is_in(arr, value_set=pa.array(values, type=arr.type))
    return pc.fill_null(mask, False).to_numpy(zero_copy_only=False)


# ---------------------------------------------------------------------------
# Layer processors: (path, row_groups, settings, aoi) -> DataFrame(aoi, <metrics>)
# ---------------------------------------------------------------------------
def process_population(path, row_groups, s, a):
    geom_col, crs, cover = file_geo_info(path)
    t = read_table(path, row_groups, ['pop_value', cover if cover else geom_col])
    pop = t.column('pop_value').to_numpy(zero_copy_only=False).astype(float)
    keep = np.isfinite(pop) & (pop > 0)
    if not keep.any():
        return sum_by_aoi(np.array([], np.int64), population=[])
    t = t.filter(pa.array(keep))
    lng, lat = feature_centres(t, geom_col, cover, crs, path)
    cells = a.assign(lat, lng)
    return sum_by_aoi(cells, population=pop[keep])


def process_buildings(path, row_groups, s, a):
    geom_col, crs, cover = file_geo_info(path)
    t = read_table(path, row_groups, ['building', cover if cover else geom_col])
    keep = _isin(t, 'building', BUILDING_VALUES)
    if not keep.any():
        return sum_by_aoi(np.array([], np.int64), buildings=[])
    t = t.filter(pa.array(keep))
    lng, lat = feature_centres(t, geom_col, cover, crs, path)
    cells = a.assign(lat, lng)
    return sum_by_aoi(cells, buildings=np.ones(len(cells)))


ROAD_SUB_BATCH = 100_000   # lines decoded/densified at a time (caps worker memory)


def process_roads(path, row_groups, s, a):
    geom_col, crs, _ = file_geo_info(path)
    t = read_table(path, row_groups, ['highway', 'railway', 'bridge', geom_col])
    parts = [_roads_table(t.slice(i, ROAD_SUB_BATCH), geom_col, crs, path, s, a)
             for i in range(0, max(t.num_rows, 1), ROAD_SUB_BATCH)]
    df = pd.concat(parts, ignore_index=True)
    return df.groupby('aoi', as_index=False, sort=True).sum() if len(df) else df


def _roads_table(t, geom_col, crs, path, s, aoi):
    major = _isin(t, 'highway', MAJOR_ROADS)
    minor = _isin(t, 'highway', MINOR_ROADS)
    rail = _isin(t, 'railway', RAILWAYS) & ~(major | minor)   # highway tag wins, as before
    klass = np.select([major, minor, rail], [0, 1, 2], -1)
    keep = klass >= 0
    empty = sum_by_aoi(np.array([], np.int64), transportation_km=[], bridges=[])
    if not keep.any():
        return empty
    t = t.filter(pa.array(keep))
    klass = klass[keep]

    geoms = geoms_to_lonlat(decode_geoms(t, geom_col), crs)

    # Bridges: road/rail segments with a bridge=* tag other than "no". Each is
    # counted once, in the cell holding the centre of its bounding box.
    # (Files made before the bridge column existed simply give 0.)
    if 'bridge' in t.column_names:
        bcol = t.column('bridge')
        is_bridge = pc.fill_null(pc.and_(pc.is_valid(bcol), pc.not_equal(bcol, 'no')), False)
        is_bridge = is_bridge.to_numpy(zero_copy_only=False) & ~pd.isna(geoms)
    else:
        is_bridge = np.zeros(len(geoms), dtype=bool)
    if is_bridge.any():
        bb = shapely.bounds(geoms[is_bridge])
        bridge_cells = aoi.assign((bb[:, 1] + bb[:, 3]) / 2, (bb[:, 0] + bb[:, 2]) / 2)
    else:
        bridge_cells = np.array([], np.int64)
    parts, idx = shapely.get_parts(geoms, return_index=True)
    is_line = np.isin(shapely.get_type_id(parts), [1, 2])     # LineString / LinearRing
    # Drop degenerate lines with fewer than 2 points (occur in raw OSM data);
    # they have no length and make shapely.segmentize raise.
    is_line &= shapely.get_num_coordinates(parts) >= 2
    parts, idx = parts[is_line], idx[is_line]
    if len(parts) == 0:
        return sum_by_aoi(bridge_cells, transportation_km=np.zeros(len(bridge_cells)),
                           bridges=np.ones(len(bridge_cells)))
    part_class = klass[idx]
    parts = shapely.segmentize(parts, s['road_seg_deg'])
    coords, ci = shapely.get_coordinates(parts, return_index=True)
    _check_degrees(coords[:, 0], coords[:, 1], path)

    same = ci[:-1] == ci[1:]
    p1, p2, seg_part = coords[:-1][same], coords[1:][same], ci[:-1][same]
    lng1, lat1, lng2, lat2 = (np.radians(p1[:, 0]), np.radians(p1[:, 1]),
                              np.radians(p2[:, 0]), np.radians(p2[:, 1]))
    h = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    length_km = 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(h, 0, 1))) / 1000.0

    cells = aoi.assign((p1[:, 1] + p2[:, 1]) / 2, (p1[:, 0] + p2[:, 0]) / 2)
    # Classes are kept above only to decide which lines count; for now all
    # roads and railways are reported as one total.
    n_seg, n_br = len(cells), len(bridge_cells)
    return sum_by_aoi(np.concatenate([cells, bridge_cells]),
                       transportation_km=np.concatenate([length_km, np.zeros(n_br)]),
                       bridges=np.concatenate([np.zeros(n_seg), np.ones(n_br)]))


def process_dams(path, row_groups, s, a):
    """Count dams and weirs. Reads only the waterway tag and the bbox column,
    so it is cheap even on the big Lines/Polygons files."""
    geom_col, crs, cover = file_geo_info(path)
    empty = sum_by_aoi(np.array([], np.int64), dams=[], weirs=[])
    if 'waterway' not in pq.read_schema(path).names:
        return empty                     # older files without the waterway column
    t = read_table(path, row_groups, ['waterway', cover if cover else geom_col])
    is_dam = _isin(t, 'waterway', DAMS)
    is_weir = _isin(t, 'waterway', WEIRS)
    keep = is_dam | is_weir
    if not keep.any():
        return empty
    t = t.filter(pa.array(keep))
    lng, lat = feature_centres(t, geom_col, cover, crs, path)
    cells = a.assign(lat, lng)
    return sum_by_aoi(cells, dams=is_dam[keep], weirs=is_weir[keep])


FARM_SUB_BATCH = 50_000    # polygons decoded/reprojected at a time (caps worker memory)


def process_farmland(path, row_groups, s, a):
    geom_col, crs, cover = file_geo_info(path)
    if 'area_m2' in pq.read_schema(path).names:
        # Cropland blocks: one point per block carrying its exact cropland area.
        t = read_table(path, row_groups, ['area_m2', cover if cover else geom_col])
        if t.num_rows == 0:
            return sum_by_aoi(np.array([], np.int64), farmland_km2=[])
        lng, lat = feature_centres(t, geom_col, cover, crs, path)
        area_km2 = t.column('area_m2').to_numpy(zero_copy_only=False) / 1e6
        return sum_by_aoi(a.assign(lat, lng), farmland_km2=area_km2)

    # Older polygon files (traced + simplified cropland polygons).
    t = read_table(path, row_groups, [geom_col])
    parts = [_farmland_table(t.slice(i, FARM_SUB_BATCH), geom_col, crs, path, s, a)
             for i in range(0, max(t.num_rows, 1), FARM_SUB_BATCH)]
    df = pd.concat(parts, ignore_index=True)
    return df.groupby('aoi', as_index=False, sort=True).sum() if len(df) else df


def _farmland_table(t, geom_col, crs, path, s, a):
    geoms = geoms_to_lonlat(decode_geoms(t, geom_col), crs)
    geoms = geoms[~(shapely.is_empty(geoms) | pd.isna(geoms))]
    if len(geoms) == 0:
        return sum_by_aoi(np.array([], np.int64), farmland_km2=[])
    bnd = shapely.bounds(geoms)
    _check_degrees(bnd[:, [0, 2]].ravel(), bnd[:, [1, 3]].ravel(), path)
    extent = np.maximum(bnd[:, 2] - bnd[:, 0], bnd[:, 3] - bnd[:, 1])
    small = extent <= s['farm_split_deg']

    cx = (bnd[small, 0] + bnd[small, 2]) / 2
    cy = (bnd[small, 1] + bnd[small, 3]) / 2
    cells = [a.assign(cy, cx)]
    area = [lonlat_area_m2(shapely.area(geoms[small]), cy)]

    for poly in geoms[~small]:
        c, ar = a.split(poly)
        cells.append(c)
        area.append(ar)

    cells = np.concatenate(cells)
    area_km2 = np.concatenate(area) / 1e6
    return sum_by_aoi(cells, farmland_km2=area_km2)


PROCESSORS = {
    "population": process_population,
    "buildings":  process_buildings,
    "farmland":   process_farmland,
    "roads":      process_roads,
    "dams":       process_dams,
}


# ---------------------------------------------------------------------------
# Task extent (lets a polygon AOI load only the polygons a task needs)
# ---------------------------------------------------------------------------
def units_bbox(units, pad_deg=0.01):
    """lon/lat (minx, miny, maxx, maxy) covering every feature in a task's
    row groups, read from the covering-bbox column statistics (no data read).
    None if any file lacks them - the AOI then loads all its polygons."""
    from aoi import _rg_bbox
    box = [np.inf, np.inf, -np.inf, -np.inf]
    for path, row_groups in units:
        _, crs, cover = file_geo_info(path)
        if cover is None:
            return None
        md = pq.read_metadata(path)
        for rg in row_groups:
            rb = _rg_bbox(md, rg, cover)
            if rb is None:
                return None
            xs = np.array([rb[0], rb[2], rb[0], rb[2], (rb[0] + rb[2]) / 2, (rb[0] + rb[2]) / 2])
            ys = np.array([rb[1], rb[1], rb[3], rb[3], rb[1], rb[3]])
            xs, ys = points_to_lonlat(xs, ys, crs)     # no-op for lon/lat files
            box = [min(box[0], np.min(xs)), min(box[1], np.min(ys)),
                   max(box[2], np.max(xs)), max(box[3], np.max(ys))]
    if box[0] > box[2]:
        return (0.0, 0.0, 0.0, 0.0)
    return (box[0] - pad_deg, box[1] - pad_deg, box[2] + pad_deg, box[3] + pad_deg)
