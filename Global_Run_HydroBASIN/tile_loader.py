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
import sys
import time
import warnings

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq


def call_with_io_retry(func, args, attempts=4, base_delay=2.0):
    """Call func(*args), retrying on OSError (which covers PermissionError /
    WinError 5 / FileNotFound and other transient filesystem hiccups). External
    USB drives under many concurrent workers occasionally return a momentary
    'access denied' or drop out for a beat; a short backoff-and-retry rides
    over that instead of letting one blip kill a multi-hour run. If it still
    fails after `attempts`, the last error is re-raised so genuine problems
    (bad path, drive truly gone) still surface."""
    last = None
    for i in range(attempts):
        try:
            return func(*args)
        except OSError as e:
            last = e
            if i < attempts - 1:
                print(f"[io-retry] {type(e).__name__} on attempt {i+1}/{attempts}: "
                      f"{str(e)[:120]} - retrying in {base_delay*(i+1):.0f}s",
                      file=sys.stderr, flush=True)
                time.sleep(base_delay * (i + 1))
    raise last


# Module-level caches. Under multiprocessing each pool worker is a separate
# process with its own copy of these dicts, so they act as per-worker caches
# that persist across the many basin-chunk calls that worker handles. The set
# of tile files and their footer bboxes are static during a run, so caching is
# safe and turns the per-call cost from "glob 50k files + read 50k footers"
# into in-memory dict lookups after the first call.
_BBOX_CACHE = {}   # parquet_path -> (minx, miny, maxx, maxy) or None
_GLOB_CACHE = {}   # (folder_or_pattern, pattern) -> sorted list of file paths


def _file_bbox(parquet_path):
    """Return (minx, miny, maxx, maxy) of the file's GeoParquet covering bbox,
    or None if the file lacks geo metadata. Reads the footer only - never
    touches the data pages, so this is essentially free even on huge files.

    The result is memoized in _BBOX_CACHE so each file's footer is read at most
    once per worker process, even though the file is bbox-tested on every call.
    """
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
    """Resolve and sort the parquet file list for a folder/pattern, memoized in
    _GLOB_CACHE. The directory contents are static during a run, so the
    (potentially expensive) recursive glob over tens of thousands of tile
    chunks runs once per worker instead of on every basin-chunk call."""
    key = (folder_or_pattern, pattern)
    cached = _GLOB_CACHE.get(key)
    if cached is not None:
        return cached

    if pattern is None:
        # Single-arg form: treat folder_or_pattern as a full glob pattern.
        files = sorted(glob.glob(folder_or_pattern, recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(folder_or_pattern, pattern), recursive=True))

    _GLOB_CACHE[key] = files
    return files


def _read_wkb_fallback(parquet_path, bbox, read_kwargs):
    """Read a parquet that lacks GeoParquet 'geo' footer metadata.

    Some simplified ESA cropland chunks were written without the 'geo'
    metadata key, so gpd.read_parquet refuses them even though they contain a
    valid WKB geometry column. We read with pandas, rebuild the geometry from
    WKB (assuming EPSG:4326, which is what every other tile uses), then apply
    the bbox filter in memory. This keeps those chunks' data instead of
    skipping them.
    """
    cols = read_kwargs.get('columns')
    df = pd.read_parquet(parquet_path, columns=cols) if cols else pd.read_parquet(parquet_path)

    # Locate the geometry column: prefer one literally named 'geometry',
    # otherwise the first column holding WKB byte values.
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
    files = _glob_files(folder_or_pattern, pattern)

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
            except Exception:
                # File may lack GeoParquet 'geo' metadata entirely. Rebuild
                # geometry from the WKB column so its data isn't dropped.
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
