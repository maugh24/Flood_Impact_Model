"""
Flood-impact exposure per area of interest (AOI), for FEWS4All.

Normally started through run_impact.sh:
    ./run_impact.sh h3              # AOI preset name from config.AOIS
    ./run_impact.sh geoglows --check

or directly:
    python run_impact.py --aoi h3 [--check] [--layers ...] [--workers N]

Outputs (in OUTPUT_FOLDER/<aoi>/):
    impact_<aoi>_<label>.csv           one row per AOI with any exposure
    impact_<aoi>_<label>_summary.csv   totals (one row)
    impact_<aoi>_<label>_metadata.csv  column units, descriptions, sources

Resumable: every task writes a small CSV under OUTPUT_FOLDER/<aoi>/_partials_<id>/.
<id> changes whenever the AOI definition (H3 resolution, or the polygon file)
changes, so results for an old AOI version are never mixed in. Rerunning skips
finished tasks; only partials belonging to the CURRENT inputs are merged.
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                                              # noqa: E402
from aoi import make_aoi, aoi_file_info                    # noqa: E402
from layers import PROCESSORS, LAYER_COLUMNS, units_bbox   # noqa: E402

try:
    from tqdm import tqdm
except ImportError:                                        # tqdm is optional
    tqdm = None

COLUMN_ORDER = ["population", "buildings", "farmland_km2", "transportation_km",
                "bridges", "dams", "weirs"]
COLUMN_INFO = {
    "population":    ("people", "Population in the cell"),
    "buildings":     ("count", "OSM buildings whose footprint centre is in the cell"),
    "farmland_km2":  ("km2", "Cropland area in the cell"),
    "transportation_km": ("km", "Roads (motorway, trunk, primary, secondary, tertiary, residential, "
                                "links, living street, busway, footway, cycleway) plus railways "
                                "(rail, light rail, subway, monorail, tram)"),
    "bridges":       ("count", "OSM road/rail segments tagged bridge=* (not 'no'), by segment centre"),
    "dams":          ("count", "OSM waterway=dam features (lines, areas, points), by feature centre"),
    "weirs":         ("count", "OSM waterway=weir and check_dam features, by feature centre"),
}
DECIMALS = {"population": 0, "buildings": 0, "farmland_km2": 2, "transportation_km": 2,
            "bridges": 0, "dams": 0, "weirs": 0}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def resolve_path(data_root, raw):
    # Accept Windows-style backslashes on any OS.
    p = Path(str(raw).replace("\\", "/"))
    if not p.is_absolute():
        p = Path(str(data_root).replace("\\", "/")) / p
    return p


def layer_files(data_root, spec):
    """(list of resolved paths, [(file, root, rel_name), ...]) for one layer.
    spec["path"] may be one path or a list of paths (e.g. the dams layer reads
    the Lines, Polygons and Points folders)."""
    raw_paths = spec["path"] if isinstance(spec["path"], (list, tuple)) else [spec["path"]]
    pattern = spec.get("pattern", "*.parquet")
    paths, found = [], []
    for raw in raw_paths:
        p = resolve_path(data_root, raw)
        paths.append(p)
        files, root = list_files(p, pattern)
        for f in files:
            rel = f.relative_to(root).as_posix() if f != root else f.name
            if len(raw_paths) > 1:
                rel = f"{root.name}/{rel}"      # keep ids unique across folders
            found.append((f, rel))
    return paths, found


def list_files(path, pattern):
    """Parquet files for a layer. Hidden files (e.g. macOS '._' resource-fork
    files on external drives) are skipped."""
    if path.is_file():
        return [path], path.parent
    if path.is_dir():
        return sorted(f for f in path.glob(pattern)
                      if f.is_file() and not f.name.startswith(".")), path
    return [], path


def _footer(f):
    try:
        md = pq.read_metadata(f)
        return f, [md.row_group(i).num_rows for i in range(md.num_row_groups)], None
    except Exception as e:
        return f, None, e


def plan_layer(layer, data_root, target_rows):
    """Split a layer into tasks of ~target_rows. A task is a list of
    (file, [row groups]) units, so many small files (e.g. ESA chunks) can share
    one task. Task ids are deterministic, which is what makes resume work."""
    paths, found = layer_files(data_root, config.INPUTS[layer])
    files = [f for f, _ in found]
    rel_of = {f: rel for f, rel in found}
    with ThreadPoolExecutor(max_workers=16) as ex:          # footer reads are I/O bound
        footers = list(ex.map(_footer, files))

    tasks, total_rows = [], 0
    units, desc, rows = [], [], 0

    def flush():
        nonlocal units, desc, rows, total_rows
        if units:
            first = Path(units[0][0])
            tag = hashlib.md5("|".join(desc).encode()).hexdigest()[:10]
            tasks.append((layer, units, f"{first.stem}__rg{units[0][1][0]:05d}__{tag}"))
            total_rows += rows
        units, desc, rows = [], [], 0

    for f, rg_rows, err in footers:
        if err is not None:
            print(f"  [{layer}] cannot read {f}: {err}")
            continue
        rel = rel_of[f]
        cur = []
        for rg, n in enumerate(rg_rows):
            cur.append(rg)
            rows += n
            if rows >= target_rows:
                units.append((str(f), cur))
                desc.append(f"{rel}:{cur[0]}-{cur[-1]}")
                flush()
                cur = []
        if cur:
            units.append((str(f), cur))
            desc.append(f"{rel}:{cur[0]}-{cur[-1]}")
    flush()
    return paths, files, tasks, total_rows


def finished_tasks(part_dir, columns):
    """Task ids with a partial CSV that has every column this code writes.
    A partial from an older version of the code (missing a column) is redone."""
    done = set()
    if part_dir.exists():
        for p in part_dir.glob("*.csv"):
            with open(p) as fh:
                header = fh.readline().strip().split(",")
            if set(columns) <= set(header):
                done.add(p.stem)
    return done


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_H3_AOI = {}      # per-worker cache: an H3 AOI is the same for every task


def task_aoi(spec, units):
    if spec["type"] == "h3":
        key = spec["resolution"]
        if key not in _H3_AOI:
            _H3_AOI[key] = make_aoi(spec)
        return _H3_AOI[key]
    # Polygon set: load only the polygons overlapping this task's features.
    return make_aoi(spec, units_bbox(units))


def run_task(args):
    (layer, units, task_id), settings, out_dir = args
    out = Path(out_dir) / layer / f"{task_id}.csv"
    for attempt in range(3):
        try:
            aoi = task_aoi(settings["aoi"], units)
            dfs = [PROCESSORS[layer](path, row_groups, settings, aoi) for path, row_groups in units]
            df = pd.concat(dfs, ignore_index=True)
            if len(units) > 1 and len(df):
                df = df.groupby("aoi", as_index=False, sort=True).sum()
            tmp = out.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, out)                 # atomic: never a half-written partial
            return layer, task_id, None
        except OSError:
            # Read errors from an external drive are often transient
            # (e.g. "Input/output error" when it spins down): retry twice.
            if attempt < 2:
                time.sleep(10)
                continue
            return layer, task_id, traceback.format_exc()
        except Exception:
            return layer, task_id, traceback.format_exc()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def _reduce(frames):
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("aoi", sort=False).sum()


def merge_layer(part_dir, columns, task_ids, batch_rows=5_000_000):
    files = [part_dir / f"{t}.csv" for t in sorted(task_ids)]
    dtypes = {"aoi": np.int64, **{c: float for c in columns}}
    total, batch, n = None, [], 0
    for f in files:
        df = pd.read_csv(f, dtype=dtypes, usecols=["aoi", *columns])
        if len(df):
            batch.append(df)
            n += len(df)
        if n >= batch_rows:
            part = _reduce(batch)
            total = part if total is None else _reduce([total.reset_index(), part.reset_index()])
            batch, n = [], 0
    if batch:
        part = _reduce(batch)
        total = part if total is None else _reduce([total.reset_index(), part.reset_index()])
    if total is None:
        total = pd.DataFrame(columns=columns, index=pd.Index([], dtype=np.int64, name="aoi"))
    return total[columns]


def write_outputs(out_dir, part_root, aoi_name, spec, layers, label, sources, planned):
    stem = f"impact_{aoi_name}_{label}"
    frames = []
    for layer in layers:
        print(f"  merging {layer} ...")
        frames.append(merge_layer(part_root / layer, LAYER_COLUMNS[layer], planned[layer]))
    combined = pd.concat(frames, axis=1).fillna(0.0)
    cols = [c for c in COLUMN_ORDER if c in combined.columns]
    combined = combined[cols]

    totals = combined.sum()
    rounded = pd.DataFrame(index=combined.index)
    for c in cols:
        d = DECIMALS[c]
        rounded[c] = combined[c].round(d).astype("int64") if d == 0 else combined[c].round(d)
    rounded = rounded[(rounded != 0).any(axis=1)].sort_index()
    if spec["type"] == "h3":
        id_col = "h3_id"
        id_desc = f"H3 cell index (resolution {spec['resolution']}), lowercase hex string"
        id_src = "H3 (Uber) v4"
        rounded.insert(0, id_col, [format(int(h), "x") for h in rounded.index])
    else:
        id_col = spec["id_column"]
        id_desc = spec.get("description", f"{id_col} of the AOI polygon")
        id_src = spec.get("source", str(spec["path"]))
        rounded.insert(0, id_col, rounded.index.astype(np.int64))

    main_csv = out_dir / f"{stem}.csv"
    rounded.to_csv(main_csv, index=False)

    summary = {"aoi_count": [len(rounded)]}      # AOIs (cells/basins) with any exposure
    for c in cols:
        summary[c] = [round(float(totals[c]), DECIMALS[c]) if DECIMALS[c] else int(round(totals[c]))]
    pd.DataFrame(summary).to_csv(out_dir / f"{stem}_summary.csv", index=False)

    run_date = datetime.now().strftime("%Y-%m-%d")
    meta = []
    meta.append({"column": id_col, "unit": "", "description": id_desc, "source": id_src,
                 "aoi": aoi_name, "run_date": run_date})
    for c in cols:
        unit, desc = COLUMN_INFO[c]
        meta.append({"column": c, "unit": unit, "description": desc,
                     "source": sources.get(c, ""), "aoi": aoi_name, "run_date": run_date})
    pd.DataFrame(meta).to_csv(out_dir / f"{stem}_metadata.csv", index=False)

    print(f"  wrote {main_csv} ({len(rounded):,} cells)")
    print(pd.DataFrame(summary).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", required=True, choices=list(config.AOIS),
                    help="area-of-interest preset from config.AOIS")
    ap.add_argument("--data-root", default=config.DATA_ROOT)
    ap.add_argument("--output", default=config.OUTPUT_FOLDER)
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    ap.add_argument("--layers", nargs="+", default=config.LAYERS, choices=list(PROCESSORS))
    ap.add_argument("--label", default=config.RUN_LABEL)
    ap.add_argument("--check", action="store_true", help="plan only; show inputs and task counts")
    ap.add_argument("--merge-only", action="store_true", help="skip processing; build CSVs from partials")
    args = ap.parse_args()

    label = args.label or datetime.now().strftime("%Y-%m")
    spec = dict(config.AOIS[args.aoi])
    if spec["type"] == "h3":
        fingerprint = f"h3r{spec['resolution']}"
        aoi_desc = f"H3 cells, resolution {spec['resolution']}"
    else:
        spec["path"] = str(resolve_path(args.data_root, spec["path"]))
        p = Path(spec["path"])
        if not p.is_file():
            print(f"AOI file not found: {p}\nPrepare it with Data_Prep/AOI/prepare_aoi.py first.")
            return 1
        geom_col, cover = aoi_file_info(p)
        names = pq.read_schema(p).names
        if spec["id_column"] not in names or cover is None:
            print(f"{p} is not a prepared AOI file (needs column '{spec['id_column']}' and a "
                  f"covering bbox). Run Data_Prep/AOI/prepare_aoi.py on it.")
            return 1
        st = p.stat()
        fingerprint = "poly_" + hashlib.md5(
            f"{p.name}|{st.st_size}|{int(st.st_mtime)}|{spec['id_column']}".encode()).hexdigest()[:8]
        aoi_desc = f"{pq.read_metadata(p).num_rows:,} polygons, id {spec['id_column']}  <- {p}"

    out_dir = Path(args.output) / args.aoi
    part_root = out_dir / f"_partials_{fingerprint}"
    settings = {"aoi": spec,
                "farm_split_deg": config.FARMLAND_SPLIT_EXTENT_DEG,
                "road_seg_deg": config.ROAD_MAX_SEGMENT_DEG}

    print("=" * 78)
    print(f"FLOOD IMPACT  |  AOI {args.aoi}  |  label {label}  |  workers {args.workers}")
    print(f"AOI      : {aoi_desc}")
    print(f"data root: {args.data_root}")
    print(f"output   : {out_dir}")
    print("=" * 78)

    # Farmland needs an h3 recent enough to split polygons along cell edges.
    from h3.api import basic_int as _h3i
    if (spec["type"] == "h3" and "farmland" in args.layers
            and not hasattr(_h3i, "h3shape_to_cells_experimental")):
        import h3
        print(f"h3 {h3.__version__} is too old (need >= 4.1). Run: pip install -U \"h3>=4.1\"")
        return 1

    all_tasks, missing, planned = [], [], {}
    for layer in args.layers:
        paths, files, tasks, rows = plan_layer(layer, args.data_root, config.TARGET_ROWS_PER_TASK[layer])
        planned[layer] = {t[2] for t in tasks}
        done = finished_tasks(part_root / layer, LAYER_COLUMNS[layer])
        todo = [t for t in tasks if t[2] not in done]
        shown = "  +  ".join(str(p) for p in paths)
        print(f"{layer:<11} {len(files):>6} files  {rows:>14,} rows  "
              f"{len(tasks):>6} tasks  ({len(tasks) - len(todo)} done)  <- {shown}")
        if not files:
            missing.append(layer)
        all_tasks.extend(todo)
    if missing:
        print(f"\nNo input files found for: {', '.join(missing)}. Check DATA_ROOT / INPUTS in config.py.")
        return 1
    if args.check:
        return 0
    if args.merge_only and all_tasks:
        print(f"\n{len(all_tasks)} task(s) not finished yet; run without --merge-only first.")
        return 1

    for layer in args.layers:
        (part_root / layer).mkdir(parents=True, exist_ok=True)

    failed = []
    if all_tasks and not args.merge_only:
        # Interleave layers so heavy and light tasks mix across workers.
        all_tasks.sort(key=lambda t: (t[2], t[0]))
        jobs = [(t, settings, str(part_root)) for t in all_tasks]
        start = time.time()
        with mp.Pool(processes=args.workers, maxtasksperchild=25) as pool:
            it = pool.imap_unordered(run_task, jobs)
            if tqdm:
                it = tqdm(it, total=len(jobs), desc="tasks", smoothing=0.05)
            for i, (layer, task_id, err) in enumerate(it, 1):
                if err:
                    failed.append({"layer": layer, "task_id": task_id, "error": err.strip().splitlines()[-1]})
                    print(f"\n  FAILED [{layer}] {task_id}\n{err}")
                elif not tqdm and i % 50 == 0:
                    print(f"  {i}/{len(jobs)} tasks  {time.time() - start:,.0f}s")
        print(f"processing done in {time.time() - start:,.0f}s")

    if failed:
        pd.DataFrame(failed).to_csv(out_dir / "failed_tasks.csv", index=False)
        print(f"\n{len(failed)} task(s) failed (see failed_tasks.csv). Rerun to retry them; "
              f"final CSVs are not written until every task succeeds.")
        return 1
    if (out_dir / "failed_tasks.csv").exists():
        try:
            (out_dir / "failed_tasks.csv").unlink()
        except OSError:
            pass

    print("Building output CSVs ...")
    write_outputs(out_dir, part_root, args.aoi, spec, args.layers, label, config.SOURCES, planned)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
