"""
Orchestrator for multiple-return-period, multiple-area flood impact analysis.

Output structure (mirrors Impact_Analysis_Results exactly):

    <master_output>/
      <Area>_Results/
        <return_period>/
          population_statistics.parquet
          population_statistics.csv
          population_statistics.geojson
          farmland_statistics.parquet
          farmland_statistics.csv
          farmland_statistics.geojson
          building_statistics.parquet
          building_statistics.csv
          building_statistics.geojson
          transportation_statistics.parquet
          transportation_statistics.csv
          transportation_statistics.geojson
        <Area>_compiled.csv          ← all return periods in one table

Each (area, return_period) pair is one parallel job. The GeoJSON and compiled
CSV are produced automatically — no separate conversion scripts needed.

Parallelism
-----------
Uses ProcessPoolExecutor so each job runs in its own OS process, bypassing
the GIL and making full use of available CPU cores. Tune MAX_WORKERS down
if you run into memory pressure (each worker loads its own copy of the large
input parquets).

NOTE: run this script from (or with sys.path pointing to) the
Multiple_Return_Periods directory so worker processes can import the
impact modules. The if __name__ == "__main__": guard is required on Windows.
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
# Helpers shared by worker and main process
# ---------------------------------------------------------------------------

def _to_geojson(gdf: gpd.GeoDataFrame, path: Path, return_period: str):
    """Write a GeoDataFrame to GeoJSON with return_period as the first property."""
    gdf = gdf.drop(columns=["x", "y"], errors="ignore").copy()
    gdf.insert(0, "return_period", return_period)
    gdf.to_file(path, driver="GeoJSON")


def _read_scalar(csv_path: Path, column: str):
    """Read a single-value CSV and return the first value, or None."""
    try:
        df = pd.read_csv(csv_path)
        if column in df.columns and len(df) > 0:
            val = df[column].iloc[0]
            return None if pd.isna(val) else val
    except Exception:
        pass
    return None


def _compile_area_csv(area_results_dir: Path, area_label: str, return_periods: list):
    """
    Scan completed return-period subfolders and write a single compiled CSV
    (<area_label>_compiled.csv) inside area_results_dir.
    """
    rows = []
    for rp in return_periods:
        rp_dir = area_results_dir / rp
        if not rp_dir.exists():
            continue
        rows.append({
            "return_period":    rp,
            "pop_value":        _read_scalar(rp_dir / "population_statistics.csv",    "pop_value"),
            "building_count":   _read_scalar(rp_dir / "building_statistics.csv",      "building_count"),
            "farmland_area_m2": _read_scalar(rp_dir / "farmland_statistics.csv",      "area_m2"),
            "highway_km":       _read_scalar(rp_dir / "transportation_statistics.csv", "highway_km"),
            "railway_km":       _read_scalar(rp_dir / "transportation_statistics.csv", "railway_km"),
        })

    df = pd.DataFrame(rows)
    out_path = area_results_dir / f"{area_label}_compiled.csv"
    df.to_csv(out_path, index=False)
    print(f"  Compiled CSV → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Worker function (runs in a child process)
# ---------------------------------------------------------------------------

def run_single_job(area_label: str, rp_label: str, flood_file: str,
                   config: dict, master_output: str):
    """
    Analyse one (area, return_period) flood extent and write all three output
    formats (parquet, CSV, GeoJSON) into the correct subfolder.

    Returns (area_label, rp_label, elapsed_seconds) on success.
    Raises on failure so the parent process can report it.
    """
    # Deferred imports — required for Windows spawn start method
    from Population_Impact_MRP import calculate_flood_population, aggregate_population_for_csv
    from Farmland_Impact_MRP   import calculate_flood_farmland,   aggregate_farmland_for_csv
    from Building_Impact_MRP   import calculate_flood_buildings,  aggregate_buildings_for_csv
    from Road_Impact_MRP       import calculate_flood_transportation, aggregate_transportation_for_csv

    tag     = f"[{area_label} | {rp_label}]"
    out_dir = Path(master_output) / f"{area_label}_Results" / rp_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{tag} Starting — {datetime.now().strftime('%H:%M:%S')}")
    t0 = time.time()

    # ----- Population --------------------------------------------------------
    print(f"{tag} Population...")
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
    aggregate_population_for_csv(pop_result).to_csv(out_dir / "population_statistics.csv", index=False)
    _to_geojson(pop_gdf, out_dir / "population_statistics.geojson", rp_label)

    # ----- Farmland ----------------------------------------------------------
    print(f"{tag} Farmland...")
    farm_result = calculate_flood_farmland(flood_file, config["farmland_parquet"])
    farm_result.to_parquet(out_dir / "farmland_statistics.parquet", index=False)
    aggregate_farmland_for_csv(farm_result).to_csv(out_dir / "farmland_statistics.csv", index=False)
    _to_geojson(farm_result, out_dir / "farmland_statistics.geojson", rp_label)

    # ----- Buildings ---------------------------------------------------------
    print(f"{tag} Buildings...")
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
    aggregate_buildings_for_csv(building_result).to_csv(out_dir / "building_statistics.csv", index=False)
    _to_geojson(building_gdf, out_dir / "building_statistics.geojson", rp_label)

    # ----- Transportation ----------------------------------------------------
    print(f"{tag} Transportation...")
    trans_result = calculate_flood_transportation(flood_file, config["transportation_parquet"])
    trans_result.to_parquet(out_dir / "transportation_statistics.parquet", index=False)
    aggregate_transportation_for_csv(trans_result).to_csv(out_dir / "transportation_statistics.csv", index=False)
    _to_geojson(trans_result, out_dir / "transportation_statistics.geojson", rp_label)

    elapsed = time.time() - t0
    print(f"{tag} Done in {elapsed:.1f}s")
    return area_label, rp_label, elapsed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # CONFIGURATION — edit these for your study area
    # ------------------------------------------------------------------

    # Nested dict: area name → {return_period_label: flood_parquet_path}
    # Area names become the top-level output folder (<name>_Results/).
    # Return period labels become the subfolders inside.
    # Add or remove areas and return periods freely — the pipeline adapts.
    AREAS = {
        "Chian_Rai": {
            "2yr":   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Chian_Rai\flood_2yr.parquet",
            "5yr":   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Chian_Rai\flood_5yr.parquet",
            "10yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Chian_Rai\flood_10yr.parquet",
            "25yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Chian_Rai\flood_25yr.parquet",
            "50yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Chian_Rai\flood_50yr.parquet",
            "100yr": r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Chian_Rai\flood_100yr.parquet",
        },
        "Kamala": {
            "2yr":   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Kamala\flood_2yr.parquet",
            "5yr":   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Kamala\flood_5yr.parquet",
            "10yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Kamala\flood_10yr.parquet",
            "25yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Kamala\flood_25yr.parquet",
            "50yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Kamala\flood_50yr.parquet",
            "100yr": r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Kamala\flood_100yr.parquet",
        },
        "Karnali": {
            "2yr":   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Karnali\flood_2yr.parquet",
            "5yr":   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Karnali\flood_5yr.parquet",
            "10yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Karnali\flood_10yr.parquet",
            "25yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Karnali\flood_25yr.parquet",
            "50yr":  r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Karnali\flood_50yr.parquet",
            "100yr": r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\Karnali\flood_100yr.parquet",
        },
    }

    osm_buildings      = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-polygons_bbox.parquet"
    osm_transportation = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines_bbox.parquet"

    CONFIG = {
        "population_parquet":     r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\population.parquet",
        "farmland_parquet":       r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\ESA_Parquet\s30w120cropland.parquet",
        "building_parquet":       osm_buildings,
        "transportation_parquet": osm_transportation,
    }

    MASTER_OUTPUT = r"C:\C_Drive_Brians_Stuff\Python_Projects\Napal_Floods\Impact_Analysis_Results"

    # Total jobs = areas × return periods. Lower MAX_WORKERS if memory is tight.
    all_jobs      = [(a, rp, f) for a, rps in AREAS.items() for rp, f in rps.items()]
    MAX_WORKERS   = min(len(all_jobs), os.cpu_count())

    # ------------------------------------------------------------------
    # Run parallel jobs
    # ------------------------------------------------------------------
    print("=" * 80)
    print("MULTIPLE RETURN PERIOD FLOOD IMPACT ANALYSIS")
    print("=" * 80)
    print(f"Start time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Areas         : {list(AREAS.keys())}")
    print(f"Total jobs    : {len(all_jobs)}  ({MAX_WORKERS} workers)")
    print(f"Master output : {MASTER_OUTPUT}")
    print("=" * 80)

    overall_start = time.time()
    results       = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single_job, area, rp, flood_file, CONFIG, MASTER_OUTPUT): (area, rp)
            for area, rp, flood_file in all_jobs
        }
        for future in as_completed(futures):
            area, rp = futures[future]
            key = f"{area} | {rp}"
            try:
                a, r, elapsed = future.result()
                results[key] = {"status": "OK", "elapsed_s": round(elapsed, 1)}
            except Exception as exc:
                results[key] = {"status": f"FAILED — {exc}", "elapsed_s": None}
                print(f"[{area} | {rp}] ERROR: {exc}")

    # ------------------------------------------------------------------
    # Compile per-area summary CSVs (runs in main process after all jobs)
    # ------------------------------------------------------------------
    print("\nCompiling per-area summary CSVs...")
    for area_label, rps in AREAS.items():
        area_dir        = Path(MASTER_OUTPUT) / f"{area_label}_Results"
        return_periods  = list(rps.keys())
        _compile_area_csv(area_dir, area_label, return_periods)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, info in sorted(results.items()):
        t = f"{info['elapsed_s']}s" if info["elapsed_s"] is not None else "—"
        print(f"  {key:<25}  {info['status']:<8}  {t}")
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")
    print("=" * 80)
