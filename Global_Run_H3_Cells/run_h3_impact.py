"""
Global flood-impact (exposure) per H3 cell, for FEWS4All.

    python run_h3_impact.py            # full run with the settings in config.py
    python run_h3_impact.py --check    # list inputs/tasks only, process nothing

Outputs (in OUTPUT_FOLDER):
    impact_h3r6_<label>.csv           one row per cell with any exposure
    impact_h3r6_<label>_summary.csv   global totals (one row)
    impact_h3r6_<label>_metadata.csv  column units, descriptions, sources

Resumable: every task writes a small CSV under OUTPUT_FOLDER/_partials/<layer>/.
Rerunning skips tasks that already have one, so a crash only loses in-flight
tasks. The final CSVs are only written once every task of every layer is done.
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
from h3_impact import PROCESSORS, LAYER_COLUMNS            # noqa: E402

try:
    from tqdm import tqdm
except ImportError:                                        # tqdm is optional
    tqdm = None

COLUMN_ORDER = ["population", "buildings", "farmland_km2", "transportation_km"]
COLUMN_INFO = {
    "h3_id":         ("", "H3 cell index (resolution {res}), lowercase hex string"),
    "population":    ("people", "Population in the cell"),
    "buildings":     ("count", "OSM buildings whose footprint centre is in the cell"),
    "farmland_km2":  ("km2", "Cropland area in the cell"),
    "transportation_km": ("km", "Roads (motorway, trunk, primary, secondary, tertiary, residential, "
                                "links, living street, busway, footway, cycleway) plus railways "
                                "(rail, light rail, subway, monorail, tram)"),
}
DECIMALS = {"population": 0, "buildings": 0, "farmland_km2": 2, "transportation_km": 2}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def resolve_input(data_root, spec):
    # Accept Windows-style backslashes on any OS.
    p = Path(str(spec["path"]).replace("\\", "/"))
    if not p.is_absolute():
        p = Path(str(data_root).replace("\\", "/")) / p
    return p, spec.get("pattern", "*.parquet")


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
    path, pattern = resolve_input(data_root, config.INPUTS[layer])
    files, root = list_files(path, pattern)
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
        rel = f.relative_to(root).as_posix() if f != root else f.name
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
    return path, files, tasks, total_rows


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_task(args):
    (layer, units, task_id), settings, out_dir = args
    out = Path(out_dir) / layer / f"{task_id}.csv"
    try:
        dfs = [PROCESSORS[layer](path, row_groups, settings) for path, row_groups in units]
        df = pd.concat(dfs, ignore_index=True)
        if len(units) > 1 and len(df):
            df = df.groupby("h3", as_index=False, sort=True).sum()
        tmp = out.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, out)                     # atomic: never a half-written partial
        return layer, task_id, None
    except Exception:
        return layer, task_id, traceback.format_exc()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def _reduce(frames):
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("h3", sort=False).sum()


def merge_layer(part_dir, columns, batch_rows=5_000_000):
    files = sorted(part_dir.glob("*.csv"))
    dtypes = {"h3": np.uint64, **{c: float for c in columns}}
    total, batch, n = None, [], 0
    for f in files:
        df = pd.read_csv(f, dtype=dtypes)
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
        total = pd.DataFrame(columns=columns, index=pd.Index([], dtype=np.uint64, name="h3"))
    return total[columns]


def write_outputs(out_dir, layers, res, label, sources):
    stem = f"impact_h3r{res}_{label}"
    frames = []
    for layer in layers:
        print(f"  merging {layer} ...")
        frames.append(merge_layer(out_dir / "_partials" / layer, LAYER_COLUMNS[layer]))
    combined = pd.concat(frames, axis=1).fillna(0.0)
    cols = [c for c in COLUMN_ORDER if c in combined.columns]
    combined = combined[cols]

    totals = combined.sum()
    rounded = pd.DataFrame(index=combined.index)
    for c in cols:
        d = DECIMALS[c]
        rounded[c] = combined[c].round(d).astype("int64") if d == 0 else combined[c].round(d)
    rounded = rounded[(rounded != 0).any(axis=1)].sort_index()
    rounded.insert(0, "h3_id", [format(int(h), "x") for h in rounded.index])

    main_csv = out_dir / f"{stem}.csv"
    rounded.to_csv(main_csv, index=False)

    summary = {"cells": [len(rounded)]}
    for c in cols:
        summary[c] = [round(float(totals[c]), DECIMALS[c]) if DECIMALS[c] else int(round(totals[c]))]
    pd.DataFrame(summary).to_csv(out_dir / f"{stem}_summary.csv", index=False)

    run_date = datetime.now().strftime("%Y-%m-%d")
    meta = []
    for c in ["h3_id"] + cols:
        unit, desc = COLUMN_INFO[c]
        meta.append({"column": c, "unit": unit, "description": desc.format(res=res),
                     "source": sources.get(c, "H3 (Uber) v4" if c == "h3_id" else ""),
                     "h3_resolution": res, "run_date": run_date})
    pd.DataFrame(meta).to_csv(out_dir / f"{stem}_metadata.csv", index=False)

    print(f"  wrote {main_csv} ({len(rounded):,} cells)")
    print(pd.DataFrame(summary).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=config.DATA_ROOT)
    ap.add_argument("--output", default=config.OUTPUT_FOLDER)
    ap.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    ap.add_argument("--layers", nargs="+", default=config.LAYERS, choices=list(PROCESSORS))
    ap.add_argument("--label", default=config.RUN_LABEL)
    ap.add_argument("--check", action="store_true", help="plan only; show inputs and task counts")
    ap.add_argument("--merge-only", action="store_true", help="skip processing; build CSVs from partials")
    args = ap.parse_args()

    res = config.H3_RESOLUTION
    label = args.label or datetime.now().strftime("%Y-%m")
    out_dir = Path(args.output)
    settings = {"res": res,
                "farm_split_deg": config.FARMLAND_SPLIT_EXTENT_DEG,
                "road_seg_deg": config.ROAD_MAX_SEGMENT_DEG}

    print("=" * 78)
    print(f"H3 FLOOD IMPACT  |  res {res}  |  label {label}  |  workers {args.workers}")
    print(f"data root: {args.data_root}")
    print(f"output   : {out_dir}")
    print("=" * 78)

    all_tasks, missing = [], []
    for layer in args.layers:
        path, files, tasks, rows = plan_layer(layer, args.data_root, config.TARGET_ROWS_PER_TASK[layer])
        part_dir = out_dir / "_partials" / layer
        done = {p.stem for p in part_dir.glob("*.csv")} if part_dir.exists() else set()
        todo = [t for t in tasks if t[2] not in done]
        print(f"{layer:<11} {len(files):>6} files  {rows:>14,} rows  "
              f"{len(tasks):>6} tasks  ({len(tasks) - len(todo)} done)  <- {path}")
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
        (out_dir / "_partials" / layer).mkdir(parents=True, exist_ok=True)

    failed = []
    if all_tasks and not args.merge_only:
        # Interleave layers so heavy and light tasks mix across workers.
        all_tasks.sort(key=lambda t: (t[2], t[0]))
        jobs = [(t, settings, str(out_dir / "_partials")) for t in all_tasks]
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
    write_outputs(out_dir, args.layers, res, label, config.SOURCES)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
