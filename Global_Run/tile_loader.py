"""
tile_loader.py

Helpers for reading multi-file parquet datasets with bbox-filtered concat.
Used by the Global_Run impact modules so they can transparently work over
folders of tiled parquets (OSM continents, ESA cropland tiles, etc.)
instead of pre-merged global files.

Two-stage filter for speed:
  1. Skip any parquet whose top-level GeoParquet covering bbox does not
     overlap the requested bbox. This only reads the parquet footer (<1 KB).
  2. On overlapping files, call gpd.read_parquet(file, bbox=...) so PyArrow
     pushes the filter down to row groups and only the relevant pages are
     loaded.
"""
import glob
import json
import os
import warnings

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq


def _file_bbox(parquet_path):
    """Return (minx, miny, maxx, maxy) of the file's GeoParquet covering bbox,
    or None if the file lacks geo metadata. Reads the footer only - never
    touches the data pages, so this is essentially free even on huge files."""
    try:
        meta = pq.read_metadata(parquet_path)
        kv = meta.metadata or {}
        if b'geo' not in kv:
            return None
        geo = json.loads(kv[b'geo'])
        primary = geo.get('primary_column', 'geometry')
        col_meta = geo.get('columns', {}).get(primary, {})
        bbox = col_meta.get('bbox')
        if bbox and len(bbox) == 4:
            return tuple(bbox)
        return None
    except Exception:
        return None


def _bboxes_overlap(a, b):
    """Test whether two (minx, miny, maxx, maxy) tuples overlap (inclusive)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def read_tiles_in_bbox(folder_or_pattern, pattern=None, bbox=None, **read_kwargs):
    """Read every parquet matching pattern under folder, bbox-filtered.

    Calling conventions:
        read_tiles_in_bbox('/path/to/folder', '*.parquet', bbox=(...))
        read_tiles_in_bbox('/path/to/folder/*.parquet', bbox=(...))   # single arg

    Parameters
    ----------
    folder_or_pattern : str
        Either a folder path (then `pattern` is required) or a glob pattern.
    pattern : str, optional
        Glob pattern relative to folder, e.g. '*.parquet' or '**/*.parquet'.
    bbox : tuple of float
        (minx, miny, maxx, maxy) in the same CRS as the files.
    **read_kwargs
        Forwarded to gpd.read_parquet (columns, filters, etc.).

    Returns
    -------
    GeoDataFrame
        Concatenation of all matching rows. Empty (CRS=EPSG:4326) if no tile
        overlaps the bbox.
    """
    if pattern is None:
        # Single-arg form: treat folder_or_pattern as a full glob pattern.
        files = sorted(glob.glob(folder_or_pattern, recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(folder_or_pattern, pattern), recursive=True))

    if not files:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    gdfs = []
    for f in files:
        # Stage 1: cheap footer check
        if bbox is not None:
            file_bb = _file_bbox(f)
            if file_bb is not None and not _bboxes_overlap(file_bb, bbox):
                continue  # file's covering bbox doesn't touch our area

        # Stage 2: bbox-filtered read (row groups pushed down by pyarrow)
        try:
            gdf = gpd.read_parquet(f, bbox=bbox, **read_kwargs)
        except Exception:
            # File may not have row-group bbox covering; fall back to full
            # read then in-memory filter.
            try:
                gdf = gpd.read_parquet(f, **read_kwargs)
                if bbox is not None and len(gdf):
                    from shapely.geometry import box
                    gdf = gdf[gdf.geometry.intersects(box(*bbox))]
            except Exception as e:
                warnings.warn(f"Skipping {f}: {e}")
                continue

        if len(gdf):
            gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def iter_vpu_files(catchments_folder, pattern='vpu=*/catchments_*.parquet'):
    """Yield (vpu_id, file_path) tuples for every VPU partition under
    catchments_folder. Sorted by VPU id for reproducibility.

    Works with the Hive-style layout:
        catchments_folder/
            vpu=101/catchments_101.parquet
            vpu=102/catchments_102.parquet
            ...
    """
    files = sorted(glob.glob(os.path.join(catchments_folder, pattern)))
    for f in files:
        # parent dir is vpu=NNN
        vpu_dir = os.path.basename(os.path.dirname(f))
        vpu = vpu_dir.split('=', 1)[1] if '=' in vpu_dir else vpu_dir
        yield vpu, f


def resolve_source(path, default_pattern='*.parquet'):
    """If `path` is a directory, return (path, default_pattern) for use with
    read_tiles_in_bbox. If it's a single file, return (None, path) so the
    caller can just call gpd.read_parquet(path, ...) directly.

    Lets impact modules transparently accept either a single-file path
    (legacy) or a tile folder (global run)."""
    if os.path.isdir(path):
        return path, default_pattern
    return None, path
