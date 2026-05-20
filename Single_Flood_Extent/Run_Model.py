"""
Orchestrator for the single-flood-extent impact analysis.

Mirrors the original Run_Model.py but without basin chunking or linknos.
Each impact module is called once against the single flood polygon, and
each produces:
  - <module>_statistics.parquet : the per-feature spatial result
  - <module>_statistics.csv     : a one-row summary
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from Population_Impact import calculate_flood_population, aggregate_population_for_csv
from Farmland_Impact import calculate_flood_farmland, aggregate_farmland_for_csv
from Building_Impact import calculate_flood_buildings, aggregate_buildings_for_csv
from Road_Impact import calculate_flood_transportation, aggregate_transportation_for_csv


class FloodImpactWorkflow:
    """
    Single-extent flood impact pipeline. Reads one flood polygon parquet
    and computes population, farmland, building, and road impacts inside
    that extent.
    """

    def __init__(self, flood_file, config, master_output_folder):
        self.flood_file = flood_file
        self.config = config
        self.master_output = Path(master_output_folder)
        self.master_output.mkdir(parents=True, exist_ok=True)
        self.consolidated_stats = self.master_output / "Statistics"
        self.consolidated_stats.mkdir(exist_ok=True)

    def run_all_analyses(self):
        print("=" * 80)
        print("SINGLE FLOOD EXTENT IMPACT ANALYSIS")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Flood file:    {self.flood_file}")
        print(f"Master output: {self.master_output}")
        print("=" * 80)
        start = time.time()

        # ----- Population -----
        print("Population...")
        pop_result = calculate_flood_population(self.flood_file, self.config['population_parquet'])
        if len(pop_result) > 0:
            pop_gdf = gpd.GeoDataFrame(
                pop_result,
                geometry=gpd.points_from_xy(pop_result['x'], pop_result['y']),
                crs='EPSG:4326'
            )
        else:
            pop_gdf = gpd.GeoDataFrame(pop_result, geometry=gpd.GeoSeries([], crs='EPSG:4326'))
        pop_gdf.to_parquet(self.consolidated_stats / "population_statistics.parquet", index=False)
        aggregate_population_for_csv(pop_result).to_csv(
            self.consolidated_stats / "population_statistics.csv", index=False
        )

        # ----- Farmland -----
        print("Farmland...")
        farm_result = calculate_flood_farmland(self.flood_file, self.config['farmland_parquet'])
        farm_result.to_parquet(self.consolidated_stats / "farmland_statistics.parquet", index=False)
        aggregate_farmland_for_csv(farm_result).to_csv(
            self.consolidated_stats / "farmland_statistics.csv", index=False
        )

        # ----- Buildings -----
        print("Buildings...")
        building_result = calculate_flood_buildings(self.flood_file, self.config['building_parquet'])
        if len(building_result) > 0:
            building_gdf = gpd.GeoDataFrame(
                building_result,
                geometry=gpd.points_from_xy(building_result['x'], building_result['y']),
                crs='EPSG:4326'
            )
        else:
            building_gdf = gpd.GeoDataFrame(building_result, geometry=gpd.GeoSeries([], crs='EPSG:4326'))
        building_gdf.to_parquet(self.consolidated_stats / "building_statistics.parquet", index=False)
        aggregate_buildings_for_csv(building_result).to_csv(
            self.consolidated_stats / "building_statistics.csv", index=False
        )

        # ----- Transportation -----
        print("Transportation...")
        trans_result = calculate_flood_transportation(self.flood_file, self.config['transportation_parquet'])
        trans_result.to_parquet(self.consolidated_stats / "transportation_statistics.parquet", index=False)
        aggregate_transportation_for_csv(trans_result).to_csv(
            self.consolidated_stats / "transportation_statistics.csv", index=False
        )

        elapsed = time.time() - start
        print("=" * 80)
        print(f"Done in {elapsed:.1f}s. Outputs in {self.consolidated_stats}")
        print("=" * 80)


# ===== USAGE =====
if __name__ == "__main__":
    # The single flood extent to analyze
    flood_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\sula_valley_honduras_100_year_rp.parquet"

    # Same input data paths as the basin pipeline
    osm_buildings = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-polygons_bbox.parquet"
    osm_transportation = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines_bbox.parquet"
    config = {
        'population_parquet': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\population.parquet",
        'farmland_parquet':   r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\ESA_Parquet\s30w120cropland.parquet",
        'building_parquet':   osm_buildings,
        'transportation_parquet': osm_transportation,
    }

    # Outputs go to a new folder so they don't collide with the basin pipeline's results
    master_output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Single_Flood_Impact_Results"

    workflow = FloodImpactWorkflow(flood_file, config, master_output_folder)
    workflow.run_all_analyses()
