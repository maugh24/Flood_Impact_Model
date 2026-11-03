"""
Helpers for reading folders of tiled parquets (OSM continents, ESA cropland
tiles) with a bbox-filtered concat. Two-stage filter: skip files whose footer
covering bbox doesn't overlap, then bbox-filter the rest via pyarrow pushdown.
"""
import glob
import json
import os
import warnings

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq


# Per-worker caches (each pool process gets its own copy).
_BBOX_CACHE = {}   # parquet_path -> (minx, miny, maxx, maxy) or None
_GLOB_CACHE = {}   # (folder_or_pattern, pattern) -> sorted list of file paths


def _file_bbox(parquet_path):
    if parquet_path in _BBOX_CACHE:
        return _BBOX_CACHE[parquet_path]

    result = None
    try:
        meta = pq.read_metadata(parquet_path)
        kv = meta.metadata or {}
        if b'geo' in kv:
            geo = json.loads(kv[b'geo'])
            primary = geo.get('primary_column', 'geometry')
            col_meta = geo.get('columns', {}).get(primary, {})
            bbox = col_meta.get('bbox')
            if bbox and len(bbox) == 4:
                result = tuple(bbox)
    except Exception:
        result = None

    _BBOX_CACHE[parquet_path] = result
    return result


def _glob_files(folder_or_pattern, pattern):
    key = (folder_or_pattern, pattern)
    cached = _GLOB_CACHE.get(key)
    if cached is not None:
        return cached

    if pattern is None:
        files = sorted(glob.glob(folder_or_pattern, recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(folder_or_pattern, pattern), recursive=True))

    _GLOB_CACHE[key] = files
    return files


def _read_wkb_fallback(parquet_path, bbox, read_kwargs):
    """Read a parquet lacking GeoParquet 'geo' metadata by rebuilding geometry
    from its WKB column (assuming EPSG:4326) and filtering in memory."""
    cols = read_kwargs.get('columns')
    df = pd.read_parquet(parquet_path, columns=cols) if cols else pd.read_parquet(parquet_path)

    geom_col = 'geometry' if 'geometry' in df.columns else None
    if geom_col is None:
        for c in df.columns:
            if len(df) and isinstance(df[c].iloc[0], (bytes, bytearray)):
                geom_col = c
                break
    if geom_col is None:
        raise ValueError("no geometry/WKB column found")

    geom = gpd.GeoSeries.from_wkb(df[geom_col])
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs='EPSG:4326')

    if bbox is not None and len(gdf):
        from shapely.geometry import box
        gdf = gdf[gdf.geometry.intersects(box(*bbox))]
    return gdf


def _bboxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def read_tiles_in_bbox(folder_or_pattern, pattern=None, bbox=None, **read_kwargs):
    files = _glob_files(folder_or_pattern, pattern)
    if not files:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    gdfs = []
    for f in files:
        if bbox is not None:
            file_bb = _file_bbox(f)
            if file_bb is not None and not _bboxes_overlap(file_bb, bbox):
                continue

        try:
            gdf = gpd.read_parquet(f, bbox=bbox, **read_kwargs)
        except Exception:
            try:
                gdf = gpd.read_parquet(f, **read_kwargs)
                if bbox is not None and len(gdf):
                    from shapely.geometry import box
                    gdf = gdf[gdf.geometry.intersects(box(*bbox))]
            except Exception:
                try:
                    gdf = _read_wkb_fallback(f, bbox, read_kwargs)
                except Exception as e:
                    warnings.warn(f"Skipping {f}: {e}")
                    continue

        if len(gdf):
            gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def iter_vpu_files(catchments_folder, pattern='vpu=*/catchments_*.parquet'):
    files = sorted(glob.glob(os.path.join(catchments_folder, pattern)))
    for f in files:
        vpu_dir = os.path.basename(os.path.dirname(f))
        vpu = vpu_dir.split('=', 1)[1] if '=' in vpu_dir else vpu_dir
        yield vpu, f


def resolve_source(path, default_pattern='*.parquet'):
    if os.path.isdir(path):
        return path, default_pattern
    return None, path
