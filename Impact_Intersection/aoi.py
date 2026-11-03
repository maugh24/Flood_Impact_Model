"""
Areas of interest (AOIs): turn coordinates into AOI ids.

Every layer processor works the same way - it reduces features to points
(pixel centres, building centres, road-segment midpoints, ...) or, for large
farmland polygons, to pieces - and asks the AOI object which area each one
belongs to. Two kinds of AOI exist:

  H3AOI       H3 cells at a fixed resolution. Pure index lookup, no geometry.
  PolygonAOI  any polygon set (GEOGLOWS catchments, HydroBASINS, ...) stored
              as a prepared GeoParquet (Data_Prep/AOI/prepare_aoi.py). For
              each task only the polygons inside that task's bounding box are
              loaded, so a worker never holds the whole global set.

Both return int64 ids; -1 means "not in any AOI" (e.g. a point in the ocean
for a catchment set), and those are dropped.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely
from h3.api import basic_int as h3i

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


def lonlat_area_m2(area_deg2, lat):
    """Area in m^2 of small lon/lat shapes on the WGS84 ellipsoid:
    deg^2 * (pi/180)^2 * M(lat) * N(lat) * cos(lat)."""
    phi = np.radians(lat)
    w2 = 1.0 - WGS84_E2 * np.sin(phi) ** 2
    m_n = (WGS84_A ** 2) * (1.0 - WGS84_E2) / (w2 ** 2)   # M * N
    return area_deg2 * (math.pi / 180.0) ** 2 * m_n * np.cos(phi)


def _pieces_area(poly, others):
    """Intersect one polygon with candidate AOI polygons -> (mask, area_m2)."""
    pieces = shapely.intersection(poly, others)
    area_deg = shapely.area(pieces)
    nz = area_deg > 0
    if not nz.any():
        return nz, np.array([])
    bnd = shapely.bounds(pieces[nz])
    return nz, lonlat_area_m2(area_deg[nz], (bnd[:, 1] + bnd[:, 3]) / 2)


# ---------------------------------------------------------------------------
# H3 cells
# ---------------------------------------------------------------------------
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


class H3AOI:
    def __init__(self, resolution):
        self.res = int(resolution)

    def assign(self, lat, lng):
        f = h3i.latlng_to_cell
        lat, lng = np.asarray(lat, float), np.asarray(lng, float)
        out = np.empty(len(lat), dtype=np.int64)    # res-6 ids fit easily in int64
        step = 500_000                               # bounded Python-object memory
        for i in range(0, len(lat), step):
            la, ln = lat[i:i + step].tolist(), lng[i:i + step].tolist()
            out[i:i + step] = np.fromiter((f(a, b, self.res) for a, b in zip(la, ln)),
                                          dtype=np.int64, count=len(la))
        return out

    def split(self, poly):
        """Exact split of one (large) lon/lat polygon across H3 cells."""
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
        minx, miny, maxx, maxy = poly.bounds
        box = h3i.LatLngPoly([(miny, minx), (miny, maxx), (maxy, maxx), (maxy, minx)])
        cand = list(h3i.h3shape_to_cells_experimental(box, self.res, contain='overlap'))
        if not cand:
            cand = [h3i.latlng_to_cell((miny + maxy) / 2, (minx + maxx) / 2, self.res)]
        ref = (minx + maxx) / 2
        cells = np.array([_cell_polygon_near(c, ref) for c in cand], dtype=object)
        nz, area = _pieces_area(poly, cells)
        return np.array(cand, dtype=np.int64)[nz], area


# ---------------------------------------------------------------------------
# Polygon sets
# ---------------------------------------------------------------------------
def aoi_file_info(path):
    """(geometry column, covering bbox column) of a prepared AOI file."""
    md = pq.read_schema(path).metadata or {}
    geo = json.loads(md[b'geo']) if b'geo' in md else {}
    geom_col = geo.get('primary_column', 'geometry')
    cov = (geo.get('columns', {}).get(geom_col, {}).get('covering') or {}).get('bbox')
    return geom_col, (cov['xmin'][0] if cov else None)


def _rg_bbox(md, rg, cover):
    """(minx, miny, maxx, maxy) of a row group from its column statistics."""
    vals = {}
    col_index = {md.schema.column(j).path: j for j in range(md.num_columns)}
    for key, agg in (('xmin', 'min'), ('ymin', 'min'), ('xmax', 'max'), ('ymax', 'max')):
        j = col_index.get(f"{cover}.{key}")
        st = md.row_group(rg).column(j).statistics if j is not None else None
        if st is None or not st.has_min_max:
            return None
        vals[key] = getattr(st, agg)
    return vals['xmin'], vals['ymin'], vals['xmax'], vals['ymax']


def _overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def aoi_parts(path):
    """Prepared AOI files: the file itself, or every *.parquet in a folder
    (e.g. one per GEOGLOWS VPU or HydroBASINS region)."""
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.glob("*.parquet") if not f.name.startswith(("_", ".")))
    return [p]


def _file_bbox(path):
    md = pq.read_schema(path).metadata or {}
    geo = json.loads(md[b'geo']) if b'geo' in md else {}
    b = geo.get('columns', {}).get(geo.get('primary_column', 'geometry'), {}).get('bbox')
    return tuple(b) if b and len(b) == 4 else None


def _load_part(path, id_column, bbox):
    """(ids, geoms) of one prepared file, limited to `bbox` (None = all)."""
    if bbox is not None:
        fb = _file_bbox(path)
        if fb is not None and not _overlaps(fb, bbox):
            return None
    geom_col, cover = aoi_file_info(path)
    pf = pq.ParquetFile(path)
    md = pf.metadata
    rgs = []
    for rg in range(md.num_row_groups):
        rb = _rg_bbox(md, rg, cover) if (cover and bbox) else None
        if rb is None or _overlaps(rb, bbox):
            rgs.append(rg)
    if not rgs:
        return None
    t = pf.read_row_groups(rgs, columns=[id_column, geom_col])
    if t.num_rows == 0:
        return None
    geoms = shapely.from_wkb(t.column(geom_col).to_numpy(zero_copy_only=False))
    ids = t.column(id_column).to_numpy(zero_copy_only=False).astype(np.int64)
    if bbox is not None:
        gb = shapely.bounds(geoms)
        keep = ~((gb[:, 2] < bbox[0]) | (gb[:, 0] > bbox[2]) |
                 (gb[:, 3] < bbox[1]) | (gb[:, 1] > bbox[3]))
        geoms, ids = geoms[keep], ids[keep]
    return ids, geoms


class PolygonAOI:
    def __init__(self, path, id_column, bbox=None):
        """Load the AOI polygons (from one prepared file or a folder of them)
        whose bbox overlaps `bbox` (None = all)."""
        loaded = [x for x in (_load_part(f, id_column, bbox) for f in aoi_parts(path)) if x]
        if loaded:
            self.ids = np.concatenate([x[0] for x in loaded])
            self.geoms = np.concatenate([x[1] for x in loaded])
        else:
            self.ids, self.geoms = np.array([], np.int64), np.array([], dtype=object)
        shapely.prepare(self.geoms)          # makes repeated point tests fast
        self._tree = None

    def assign(self, lat, lng):
        out = np.full(len(lat), -1, dtype=np.int64)
        if len(lat) == 0 or len(self.geoms) == 0:
            return out
        pts = shapely.points(np.asarray(lng, float), np.asarray(lat, float))
        # Index the POINTS and query with each (prepared) polygon: much faster
        # than testing every point against a tree of complex polygons.
        poly_i, pt_i = shapely.STRtree(pts).query(self.geoms, predicate='intersects')
        if len(pt_i):
            # A point exactly on a shared border matches two polygons: keep one.
            first = np.unique(pt_i, return_index=True)[1]
            out[pt_i[first]] = self.ids[poly_i[first]]
        return out

    def split(self, poly):
        if len(self.geoms) == 0:
            return np.array([], np.int64), np.array([])
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
        if self._tree is None:
            self._tree = shapely.STRtree(self.geoms)
        cand = self._tree.query(poly, predicate='intersects')
        if len(cand) == 0:
            return np.array([], np.int64), np.array([])
        nz, area = _pieces_area(poly, self.geoms[cand])
        return self.ids[cand][nz], area


# ---------------------------------------------------------------------------
# Build from a config preset
# ---------------------------------------------------------------------------
def make_aoi(spec, bbox=None):
    if spec["type"] == "h3":
        return H3AOI(spec["level"])
    return PolygonAOI(spec["path"], spec["id_column"], bbox)
