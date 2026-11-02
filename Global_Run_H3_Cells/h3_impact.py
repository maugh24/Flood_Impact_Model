"""
Core per-task logic: read one batch of row groups from a prepped GeoParquet,
assign every feature (or piece of a feature) to an H3 cell, and return a small
per-cell aggregate table.

No polygon overlays against AOIs are done. H3 is used as an index:
  population  pixel polygon bbox centre -> cell, sum pop_value
  buildings   footprint bbox centre     -> cell, count
  farmland    small polygons: bbox centre -> cell, sum geodesic-ish area
              large polygons: clipped exactly to the overlapping H3 cells
  roads       lines densified, each segment's midpoint -> cell, sum length
              (roads + railways combined into transportation_km)

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
from h3.api import basic_int as h3i

# Authalic sphere radius (m) - same as H3's own area calculations.
EARTH_R = 6371007.180918475
DEG2M = EARTH_R * math.pi / 180.0

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

# Output columns produced by each layer.
LAYER_COLUMNS = {
    "population": ["population"],
    "buildings":  ["buildings"],
    "farmland":   ["farmland_km2"],
    "roads":      ["transportation_km"],
}


# ---------------------------------------------------------------------------
# H3 helpers
# ---------------------------------------------------------------------------
def latlng_to_cells(lat, lng, res):
    """Vector of uint64 H3 cells for arrays of lat/lng (degrees)."""
    f = h3i.latlng_to_cell
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    out = np.empty(len(lat), dtype=np.uint64)
    step = 500_000                                 # bounded Python-object memory
    for i in range(0, len(lat), step):
        la, ln = lat[i:i + step].tolist(), lng[i:i + step].tolist()
        out[i:i + step] = np.fromiter((f(a, b, res) for a, b in zip(la, ln)),
                                      dtype=np.uint64, count=len(la))
    return out


def sum_by_cell(cells, **values):
    """Group equal cells and sum each value array -> DataFrame(h3, *values)."""
    if len(cells) == 0:
        return pd.DataFrame({'h3': np.array([], dtype=np.uint64),
                             **{k: np.array([], dtype=float) for k in values}})
    uniq, inv = np.unique(cells, return_inverse=True)
    out = {'h3': uniq}
    for k, v in values.items():
        out[k] = np.bincount(inv, weights=np.asarray(v, dtype=float), minlength=len(uniq))
    return pd.DataFrame(out)


@lru_cache(maxsize=200_000)
def _cell_polygon(cell):
    """Shapely polygon (lng, lat) of an H3 cell with antimeridian unwrapped
    so its longitudes are continuous (may extend past +/-180)."""
    ring = np.array([(lng, lat) for lat, lng in h3i.cell_to_boundary(cell)])
    lngs = ring[:, 0]
    if lngs.max() - lngs.min() > 180:
        lngs[lngs < 0] += 360.0
    return shapely.Polygon(ring)


def _cell_polygon_near(cell, ref_lng):
    poly = _cell_polygon(cell)
    cx = (poly.bounds[0] + poly.bounds[2]) / 2
    shift = 360.0 * round((ref_lng - cx) / 360.0)
    return shapely.transform(poly, lambda xy: xy + [shift, 0]) if shift else poly


WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


def lonlat_area_m2(area_deg2, lat):
    """Area in m^2 of small lon/lat shapes on the WGS84 ellipsoid:
    deg^2 * (pi/180)^2 * M(lat) * N(lat) * cos(lat)."""
    phi = np.radians(lat)
    w2 = 1.0 - WGS84_E2 * np.sin(phi) ** 2
    m_n = (WGS84_A ** 2) * (1.0 - WGS84_E2) / (w2 ** 2)   # M * N
    return area_deg2 * (math.pi / 180.0) ** 2 * m_n * np.cos(phi)


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
# Layer processors: (path, row_groups, settings) -> DataFrame(h3, <metrics>)
# ---------------------------------------------------------------------------
def process_population(path, row_groups, s):
    geom_col, crs, cover = file_geo_info(path)
    t = read_table(path, row_groups, ['pop_value', cover if cover else geom_col])
    pop = t.column('pop_value').to_numpy(zero_copy_only=False).astype(float)
    keep = np.isfinite(pop) & (pop > 0)
    if not keep.any():
        return sum_by_cell(np.array([], np.uint64), population=[])
    t = t.filter(pa.array(keep))
    lng, lat = feature_centres(t, geom_col, cover, crs, path)
    cells = latlng_to_cells(lat, lng, s['res'])
    return sum_by_cell(cells, population=pop[keep])


def process_buildings(path, row_groups, s):
    geom_col, crs, cover = file_geo_info(path)
    t = read_table(path, row_groups, ['building', cover if cover else geom_col])
    keep = _isin(t, 'building', BUILDING_VALUES)
    if not keep.any():
        return sum_by_cell(np.array([], np.uint64), buildings=[])
    t = t.filter(pa.array(keep))
    lng, lat = feature_centres(t, geom_col, cover, crs, path)
    cells = latlng_to_cells(lat, lng, s['res'])
    return sum_by_cell(cells, buildings=np.ones(len(cells)))


ROAD_SUB_BATCH = 100_000   # lines decoded/densified at a time (caps worker memory)


def process_roads(path, row_groups, s):
    geom_col, crs, _ = file_geo_info(path)
    t = read_table(path, row_groups, ['highway', 'railway', geom_col])
    parts = [_roads_table(t.slice(i, ROAD_SUB_BATCH), geom_col, crs, path, s)
             for i in range(0, max(t.num_rows, 1), ROAD_SUB_BATCH)]
    df = pd.concat(parts, ignore_index=True)
    return df.groupby('h3', as_index=False, sort=True).sum() if len(df) else df


def _roads_table(t, geom_col, crs, path, s):
    major = _isin(t, 'highway', MAJOR_ROADS)
    minor = _isin(t, 'highway', MINOR_ROADS)
    rail = _isin(t, 'railway', RAILWAYS) & ~(major | minor)   # highway tag wins, as before
    klass = np.select([major, minor, rail], [0, 1, 2], -1)
    keep = klass >= 0
    empty = sum_by_cell(np.array([], np.uint64), transportation_km=[])
    if not keep.any():
        return empty
    t = t.filter(pa.array(keep))
    klass = klass[keep]

    geoms = geoms_to_lonlat(decode_geoms(t, geom_col), crs)
    parts, idx = shapely.get_parts(geoms, return_index=True)
    is_line = np.isin(shapely.get_type_id(parts), [1, 2])     # LineString / LinearRing
    # Drop degenerate lines with fewer than 2 points (occur in raw OSM data);
    # they have no length and make shapely.segmentize raise.
    is_line &= shapely.get_num_coordinates(parts) >= 2
    parts, idx = parts[is_line], idx[is_line]
    if len(parts) == 0:
        return empty
    part_class = klass[idx]
    parts = shapely.segmentize(parts, s['road_seg_deg'])
    coords, ci = shapely.get_coordinates(parts, return_index=True)
    _check_degrees(coords[:, 0], coords[:, 1], path)

    same = ci[:-1] == ci[1:]
    a, b, seg_part = coords[:-1][same], coords[1:][same], ci[:-1][same]
    lng1, lat1, lng2, lat2 = (np.radians(a[:, 0]), np.radians(a[:, 1]),
                              np.radians(b[:, 0]), np.radians(b[:, 1]))
    h = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    length_km = 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(h, 0, 1))) / 1000.0

    cells = latlng_to_cells((a[:, 1] + b[:, 1]) / 2, (a[:, 0] + b[:, 0]) / 2, s['res'])
    # Classes are kept above only to decide which lines count; for now all
    # roads and railways are reported as one total.
    return sum_by_cell(cells, transportation_km=length_km)


def _split_polygon(poly, res):
    """Exact split of one (large) lon/lat polygon across H3 cells.
    Returns (cells, area_m2)."""
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    minx, miny, maxx, maxy = poly.bounds
    box = h3i.LatLngPoly([(miny, minx), (miny, maxx), (maxy, maxx), (maxy, minx)])
    cand = list(h3i.h3shape_to_cells_experimental(box, res, contain='overlap'))
    if not cand:
        cand = [h3i.latlng_to_cell((miny + maxy) / 2, (minx + maxx) / 2, res)]
    ref = (minx + maxx) / 2
    cell_polys = np.array([_cell_polygon_near(c, ref) for c in cand], dtype=object)
    pieces = shapely.intersection(poly, cell_polys)
    area_deg = shapely.area(pieces)
    nz = area_deg > 0
    if not nz.any():
        return np.array([], np.uint64), np.array([])
    bnd = shapely.bounds(pieces[nz])
    lat_c = (bnd[:, 1] + bnd[:, 3]) / 2
    return np.array(cand, dtype=np.uint64)[nz], lonlat_area_m2(area_deg[nz], lat_c)


FARM_SUB_BATCH = 50_000    # polygons decoded/reprojected at a time (caps worker memory)


def process_farmland(path, row_groups, s):
    geom_col, crs, _ = file_geo_info(path)
    t = read_table(path, row_groups, [geom_col])
    parts = [_farmland_table(t.slice(i, FARM_SUB_BATCH), geom_col, crs, path, s)
             for i in range(0, max(t.num_rows, 1), FARM_SUB_BATCH)]
    df = pd.concat(parts, ignore_index=True)
    return df.groupby('h3', as_index=False, sort=True).sum() if len(df) else df


def _farmland_table(t, geom_col, crs, path, s):
    geoms = geoms_to_lonlat(decode_geoms(t, geom_col), crs)
    geoms = geoms[~(shapely.is_empty(geoms) | pd.isna(geoms))]
    if len(geoms) == 0:
        return sum_by_cell(np.array([], np.uint64), farmland_km2=[])
    bnd = shapely.bounds(geoms)
    _check_degrees(bnd[:, [0, 2]].ravel(), bnd[:, [1, 3]].ravel(), path)
    extent = np.maximum(bnd[:, 2] - bnd[:, 0], bnd[:, 3] - bnd[:, 1])
    small = extent <= s['farm_split_deg']

    cx = (bnd[small, 0] + bnd[small, 2]) / 2
    cy = (bnd[small, 1] + bnd[small, 3]) / 2
    cells = [latlng_to_cells(cy, cx, s['res'])]
    area = [lonlat_area_m2(shapely.area(geoms[small]), cy)]

    for poly in geoms[~small]:
        c, a = _split_polygon(poly, s['res'])
        cells.append(c)
        area.append(a)

    cells = np.concatenate(cells)
    area_km2 = np.concatenate(area) / 1e6
    return sum_by_cell(cells, farmland_km2=area_km2)


PROCESSORS = {
    "population": process_population,
    "buildings":  process_buildings,
    "farmland":   process_farmland,
    "roads":      process_roads,
}
