"""
Orchestrator for multiple-return-period flood impact analysis.

Each return period (e.g. 10yr, 50yr, 100yr, 500yr) is represented by its own
flood-extent parquet file.  The pipeline:

  1. Loops over a dict of  {label: flood_parquet_path}  entries.
  2. Runs the four impact analyses (population, farmland, buildings, roads)
     for each return period.
  3. Writes results into a dedicated subfolder per return period:
       <master_output>/<label>/Statistics/

Parallelism
-----------
Each return period is dispatched to a separate OS process via
ProcessPoolExecutor.  This is worthwhile because:

  * The analyses are CPU- and I/O-heavy (geopandas overlays, parquet reads).
  * Return periods are fully independent — no data is shared between them.
  * ProcessPoolExecutor sidesteps Python's GIL, so all CPU cores are used.

Set MAX_WORKERS to the number of return periods you want to run simultaneously.
A safe default is min(len(RETURN_PERIODS), os.cpu_count()).  If you are memory-
constrained (each process loads the same large parquet files), lower MAX_WORKERS.

NOTE: all four impact-module imports must be importable by the worker processes,
so this script must be run from (or have on sys.path) the Multiple_Return_Periods
directory, and the if __name__ == "__main__": guard is required for Windows.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Worker function (runs in a child process)
# ---------------------------------------------------------------------------

def run_single_return_period(label: str, flood_file: str, config: dict, master_output: str):
    """
    Analyse one return-period flood extent.  Called in a worker process.

    Parameters
    ----------
    label        : Human-readable return period name, e.g. "100yr".
    flood_file   : Path to the flood-extent parquet for this return period.
    config       : Dict of input data paths (population, farmland, buildings, roads).
    master_output: Root output directory; results go to  master_output/label/Statistics/.

    Returns
    -------
    (label, elapsed_seconds, output_dir)  on success.
    Raises on failure so the parent process can report it.
    """
    # Imports inside the function guarantee they are resolved in the worker
    # process, which is important on Windows (spawn start method).
    from Population_Impact_MRP import calculate_flood_population, aggregate_population_for_csv
    from Farmland_Impact_MRP  import calculate_flood_farmland,   aggregate_farmland_for_csv
    from Building_Impact_MRP  import calculate_flood_buildings,  aggregate_buildings_for_csv
    from Road_Impact_MRP      import calculate_flood_transportation, aggregate_transportation_for_csv

    out_dir = Path(master_output) / label / "Statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{label}] Starting  —  {datetime.now().strftime('%H:%M:%S')}")
    t0 = time.time()

    # ----- Population --------------------------------------------------------
    print(f"[{label}] Population...")
    pop_result = calculate_flood_population(flood_file, config["population_parquet"])
    if len(pop_result) > 0:
        pop_gdf = gpd.GeoDataFrame(
            pop_result,
            geometry=gpd.points_from_xy(pop_result["x"], pop_result["y"]),
            crs="EPSG:4326",
        )
    else:
        pop_gdf = gpd.GeoDataFrame(pop_result, geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    pop_gdf.to_parquet(out_dir / "population_statistics.parquet", index=False)
    aggregate_population_for_csv(pop_result).to_csv(
        out_dir / "population_statistics.csv", index=False
    )

    # ----- Farmland ----------------------------------------------------------
    print(f"[{label}] Farmland...")
    farm_result = calculate_flood_farmland(flood_file, config["farmland_parquet"])
    farm_result.to_parquet(out_dir / "farmland_statistics.parquet", index=False)
    aggregate_farmland_for_csv(farm_result).to_csv(
        out_dir / "farmland_statistics.csv", index=False
    )

    # ----- Buildings ---------------------------------------------------------
    print(f"[{label}] Buildings...")
    building_result = calculate_flood_buildings(flood_file, config["building_parquet"])
    if len(building_result) > 0:
        building_gdf = gpd.GeoDataFrame(
            building_result,
            geometry=gpd.points_from_xy(building_result["x"], building_result["y"]),
            crs="EPSG:4326",
        )
    else:
        building_gdf = gpd.GeoDataFrame(
            building_result, geometry=gpd.GeoSeries([], crs="EPSG:4326")
        )
    building_gdf.to_parquet(out_dir / "building_statistics.parquet", index=False)
    aggregate_buildings_for_csv(building_result).to_csv(
        out_dir / "building_statistics.csv", index=False
    )

    # ----- Transportation ----------------------------------------------------
    print(f"[{label}] Transportation...")
    trans_result = calculate_flood_transportation(
        flood_file, config["transportation_parquet"]
    )
    trans_result.to_parquet(out_dir / "transportation_statistics.parquet", index=False)
    aggregate_transportation_for_csv(trans_result).to_csv(
        out_dir / "transportation_statistics.csv", index=False
    )

    elapsed = time.time() - t0
    print(f"[{label}] Done in {elapsed:.1f}s  —  outputs in {out_dir}")
    return label, elapsed, str(out_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # CONFIGURATION — edit these paths for your study area
    # ------------------------------------------------------------------

    # One entry per return period: label → flood-extent parquet path.
    # Labels are used as output subfolder names, so keep them filesystem-safe.
    RETURN_PERIODS = {
        "10yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\flood_10yr.parquet",
        "50yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\flood_50yr.parquet",
        "100yr": r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\flood_100yr.parquet",
        "500yr": r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\flood_500yr.parquet",
    }

    osm_buildings      = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-polygons_bbox.parquet"
    osm_transportation = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines_bbox.parquet"

    CONFIG = {
        "population_parquet":     r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\population.parquet",
        "farmland_parquet":       r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\ESA_Parquet\s30w120cropland.parquet",
        "building_parquet":       osm_buildings,
        "transportation_parquet": osm_transportation,
    }

    MASTER_OUTPUT = r"C:\C_Drive_Brians_Stuff\Python_Projects\Multiple_Return_Period_Results"

    # Number of return periods to process simultaneously.
    # Lower this if you run into memory pressure (each worker loads its own
    # copy of the large input parquets).
    MAX_WORKERS = min(len(RETURN_PERIODS), os.cpu_count())

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print("=" * 80)
    print("MULTIPLE RETURN PERIOD FLOOD IMPACT ANALYSIS")
    print("=" * 80)
    print(f"Start time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Return periods: {list(RETURN_PERIODS.keys())}")
    print(f"Worker processes: {MAX_WORKERS}")
    print(f"Master output : {MASTER_OUTPUT}")
    print("=" * 80)

    overall_start = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all return periods up front so they can run in parallel
        futures = {
            executor.submit(
                run_single_return_period,
                label,
                flood_file,
                CONFIG,
                MASTER_OUTPUT,
            ): label
            for label, flood_file in RETURN_PERIODS.items()
        }

        for future in as_completed(futures):
            label = futures[future]
            try:
                rp_label, elapsed, out_dir = future.result()
                results[rp_label] = {"status": "OK", "elapsed_s": round(elapsed, 1), "output": out_dir}
            except Exception as exc:
                results[label] = {"status": f"FAILED — {exc}", "elapsed_s": None, "output": None}
                print(f"[{label}] ERROR: {exc}")

    total_elapsed = time.time() - overall_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for rp, info in sorted(results.items()):
        status = info["status"]
        t      = f"{info['elapsed_s']}s" if info["elapsed_s"] is not None else "—"
        print(f"  {rp:<8}  {status:<8}  {t}")
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")
    print("=" * 80)
