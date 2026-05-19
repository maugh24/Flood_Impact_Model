from operator import truediv

import geopandas as gpd
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import warnings
import multiprocessing as mp
import numpy as np
import tqdm
warnings.filterwarnings('ignore')

# Import your four impact models
from Population_Impact import calculate_basin_population_wrapper, aggregate_population_for_csv
from Farmland_Impact import calculate_basin_farmland_wrapper, aggregate_farmland_for_csv
from Building_Impact import calculate_basin_building_wrapper
from Road_Impact import calculate_basin_transportation_wrapper, aggregate_transportation_for_csv


class ImpactAnalysisWorkflow:
    def __init__(self, basin_file, config, master_output_folder, max_workers=4):

        self.basin_file = basin_file
        self.config = config
        self.master_output = Path(master_output_folder)
        self.max_workers = max_workers

        # Create master output folder structure
        self.master_output.mkdir(parents=True, exist_ok=True)
        self.consolidated_stats = self.master_output / "Statistics"
        self.consolidated_stats.mkdir(exist_ok=True)

        # Tracking
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def get_sorted_rivids(self):
        # By sorting, this helps computation because we get basins that are close to each other
        gdf = gpd.read_parquet(self.basin_file)
        centroid = gdf.geometry.centroid
        distance_from_0_0 = np.sqrt(centroid.x**2 + centroid.y**2)
        gdf['distance_from_0_0'] = distance_from_0_0
        gdf = gdf.sort_values('distance_from_0_0')
        gdf.to_parquet(self.master_output / "sorted_basins.parquet") # optional, for debugging
        return gdf['linkno'].tolist()

    # @profile
    def run_all_analyses(self):
        """Run all four impact analyses in parallel."""

        print("=" * 80)
        print("GLOBAL IMPACT ANALYSIS WORKFLOW (PARALLEL EXECUTION)")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Basin file: {self.basin_file}")
        print(f"Master output: {self.master_output}")
        print(f"Max parallel workers: {self.max_workers}")
        print("=" * 80)
        # self.basins = gpd.read_parquet(self.basin_file)
        self.start_time = time.time()

        # Run analyses in parallel
        # Sort so that basins are processed in a consistent order (top to bottom, left to right)
        n = 100 # Sweet spot?
        rivers = self.get_sorted_rivids() # 6:32 kmeans, 5:55 hilbert, 4:44 distance from 0,0
        rivers_split = [rivers[i:i+n] for i in range(0, len(rivers), n)]
        with mp.Pool(processes=self.max_workers) as pool:
             #Population
             args = [(basin_file, rivs, self.config['population_parquet']) for rivs in rivers_split]
             population_dfs = list(tqdm.tqdm(pool.imap_unordered(calculate_basin_population_wrapper, args), total=len(args),desc="Population    "))
             population_result = pd.concat(population_dfs, ignore_index=True)
             population_gdf = gpd.GeoDataFrame(
                 population_result,
                 geometry=gpd.points_from_xy(population_result['x'], population_result['y']),
                 crs='EPSG:4326'
             )
             population_gdf.to_parquet(self.consolidated_stats / "population_statistics.parquet", index=False)
             population_csv = aggregate_population_for_csv(population_result)
             population_csv.to_csv(self.consolidated_stats / "population_statistics.csv", index=False)


             #Farmland
             args = [(basin_file, rivs, self.config['farmland_parquet']) for rivs in rivers_split]
             farmland_gdfs = list(tqdm.tqdm(pool.imap_unordered(calculate_basin_farmland_wrapper, args), total=len(args),desc="Farmland      "))
             farmland_result = gpd.GeoDataFrame(pd.concat(farmland_gdfs, ignore_index=True), crs='EPSG:4326')
             farmland_result.to_parquet(self.consolidated_stats / "farmland_statistics.parquet", index=False)
             farmland_csv = aggregate_farmland_for_csv(farmland_result)
             farmland_csv.to_csv(self.consolidated_stats / "farmland_statistics.csv", index=False)


             # Buildings
             args = [(basin_file, rivs, self.config['building_parquet']) for rivs in rivers_split]
             building_dfs = list(tqdm.tqdm(pool.imap_unordered(calculate_basin_building_wrapper, args), total=len(args),desc="Buildings     "))
             building_result = pd.concat(building_dfs, ignore_index=True)

             building_gdf = gpd.GeoDataFrame(
                 building_result,
                 geometry=gpd.points_from_xy(building_result['x'], building_result['y']),
                 crs='EPSG:4326')

             building_count = building_result[building_result['linkno'] != 'TOTAL'].groupby('linkno').size().reset_index(name='building_count')
             total_buildings = building_result[building_result['linkno'] != 'TOTAL'].shape[0]
             total_buildings_row = pd.DataFrame({'linkno': ['TOTAL'], 'building_count': [total_buildings]})
             building_count_with_total = pd.concat([total_buildings_row, building_count], ignore_index=True)
             building_gdf.to_parquet(self.consolidated_stats / "building_statistics.parquet", index=False)
             building_count_with_total.to_csv(self.consolidated_stats / "building_statistics.csv", index=False)


             # Transportation
             args = [(basin_file, rivs, self.config['transportation_parquet']) for rivs in rivers_split]
             transportation_dfs = list(tqdm.tqdm(pool.imap_unordered(calculate_basin_transportation_wrapper, args), total=len(args),desc="Transportation"))
             if transportation_dfs:
                 transportation_result = gpd.GeoDataFrame(pd.concat(transportation_dfs, ignore_index=True),
                                                          geometry='geometry', crs='EPSG:4326')
             else:
                 transportation_result = gpd.GeoDataFrame(
                     columns=['linkno', 'infrastructure_type', 'feature_value', 'length_m', 'length_km', 'geometry'],
                     geometry='geometry', crs='EPSG:4326')
             transportation_result.to_parquet(self.consolidated_stats / "transportation_statistics.parquet", index=False)
             transport_csv = aggregate_transportation_for_csv(transportation_result)
             transport_csv.to_csv(self.consolidated_stats / "transportation_statistics.csv", index=False)


def run_worfklow(basin_file, config, master_output_folder):
    workflow = ImpactAnalysisWorkflow(
        basin_file,
        config,
        master_output_folder
    )
    workflow.run_all_analyses()

# ===== USAGE =====
if __name__ == "__main__":
    # Configuration
    basin_file = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\catchments_718.parquet"
    osm_buildings = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-polygons_bbox.parquet"
    osm_transportation = r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines_bbox.parquet"
    config = {
        'population_parquet': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\population.parquet",
        'farmland_parquet': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\ESA_Parquet\cropland.parquet",
        'building_parquet': osm_buildings,
        'transportation_parquet': osm_transportation,
    }

    master_output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Impact_Analysis_Results"

    # Run workflow with parallel execution (adjust max_workers based on your RAM)
    workflow = ImpactAnalysisWorkflow(
        basin_file,
        config,
        master_output_folder,
        max_workers=mp.cpu_count()//3 # Adjust based on your system (4 is good for 32GB+ RAM)
    )
    workflow.run_all_analyses()
