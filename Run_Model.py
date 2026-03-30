import geopandas as gpd
import pandas as pd
from pathlib import Path
import shutil
import time
from datetime import datetime
import warnings
import rasterio
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Import your four impact models
from Population_Impact import calculate_basin_population
from Farmland_Impact import calculate_basin_farmland
from Building_Impact import calculate_basin_buildings
from Road_Impact import calculate_basin_transportation


class ImpactAnalysisWorkflow:
    """
    Workflow manager for running all impact analyses in parallel and consolidating outputs.
    Includes rasters embedded in GeoPackage.
    """

    def __init__(self, basin_file, config, master_output_folder, max_workers=4):
        """
        Initialize the workflow.

        Parameters:
        -----------
        basin_file : str
            Path to basin parquet file
        config : dict
            Configuration dictionary with paths to input data
        master_output_folder : str
            Path to master output folder for all results
        max_workers : int
            Number of parallel processes (default: 4)
        """
        self.basin_file = basin_file
        self.config = config
        self.master_output = Path(master_output_folder)
        self.max_workers = max_workers

        # Create master output folder structure
        self.master_output.mkdir(parents=True, exist_ok=True)
        self.consolidated_gpkg = self.master_output / "consolidated_impacts.gpkg"
        self.consolidated_stats = self.master_output / "consolidated_statistics"
        self.consolidated_stats.mkdir(exist_ok=True)

        # Tracking
        self.results = {}
        self.start_time = None
        self.end_time = None

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

        # Define analyses to run
        # analyses = {
        #     "Population": self._run_population_analysis,
        #     "Farmland": self._run_farmland_analysis,
        #     "Buildings": self._run_building_analysis,
        #     "Transportation": self._run_transportation_analysis
        # }

        # Run analyses in parallel
        # print(f"\nStarting {len(analyses)} analyses in parallel...")
        rivers = pd.read_parquet(basin_file, columns=['linkno']).values[:, 0]
        n = 1000
        rivers_split = [rivers[i:i+n] for i in range(0, len(rivers), n)]
      # with mp.Pool(processes=self.max_workers) as pool:
      #     temp_output = str(self.master_output / "_temp_population")
      #     args = [(basin_file, rivs, self.config['population_raster'], temp_output, index) for index, rivs in enumerate(rivers_split)]
      #     pool.starmap(calculate_basin_population, args)
      #   with mp.Pool(processes=self.max_workers) as pool:
        temp_output = str(self.master_output / "_temp_farmland")
        args = [(basin_file, rivs, self.config['farmland_raster_folder'], temp_output, index) for index, rivs in enumerate(rivers_split)]
        calculate_basin_farmland(basin_file, rivers_split[0], self.config['farmland_raster_folder'], temp_output, 0)
      #       pool.starmap(calculate_basin_farmland, args)
      # with mp.Pool(processes=self.max_workers) as pool:
      #     temp_output = str(self.master_output / "_temp_buildings")
      #     args = [(basin_file, rivs, self.config['building_parquet'], temp_output, index) for index, rivs in enumerate(rivers_split)]
      #     pool.starmap(calculate_basin_buildings, args)
      # with mp.Pool(processes=self.max_workers) as pool:
      #     temp_output = str(self.master_output / "_temp_transportation")
      #     args = [(basin_file, rivs, self.config['transportation_parquet'], temp_output, index) for index, rivs in enumerate(rivers_split)]
      #     pool.starmap(calculate_basin_transportation, args)
        # self._run_population_analysis()
        # self._run_farmland_analysis()
        # self._run_building_analysis()
        # self._run_transportation_analysis()


        self.end_time = time.time()

        # Consolidate outputs
        print(f"\n{'=' * 80}")
        print("CONSOLIDATING OUTPUTS")
        print(f"{'=' * 80}")
        self._consolidate_outputs()

        # Print final summary
        self._print_summary()

    def _run_population_analysis(self):
        """Run population impact analysis."""
        print(f"[Population] Starting analysis...")
        temp_output = self.master_output / "_temp_population"

        result = calculate_basin_population(
            self.basins,
            self.config['population_raster'],
            str(temp_output)
        )

        print(f"[Population] Analysis complete!")
        return {'temp_folder': temp_output, 'stats': result}

    def _run_farmland_analysis(self):
        """Run farmland impact analysis."""
        print(f"[Farmland] Starting analysis...")
        temp_output = self.master_output / "_temp_farmland"

        result = calculate_basin_farmland(
            self.basins,
            self.config['farmland_raster_folder'],
            str(temp_output),
            farmland_value=40
        )

        print(f"[Farmland] Analysis complete!")
        return {'temp_folder': temp_output, 'stats': result}

    def _run_building_analysis(self):
        """Run building impact analysis."""
        print(f"[Buildings] Starting analysis...")
        temp_output = self.master_output / "_temp_buildings"

        result = calculate_basin_buildings(
            self.basins,
            self.config['building_parquet'],
            str(temp_output)
        )

        print(f"[Buildings] Analysis complete!")
        return {'temp_folder': temp_output, 'stats': result}

    def _run_transportation_analysis(self):
        """Run transportation impact analysis."""
        print(f"[Transportation] Starting analysis...")
        temp_output = self.master_output / "_temp_transportation"

        result = calculate_basin_transportation(
            self.basins,
            self.config['transportation_parquet'],
            str(temp_output)
        )

        print(f"[Transportation] Analysis complete!")
        return {'temp_folder': temp_output, 'stats': result}

    def _consolidate_outputs(self):
        """Consolidate all outputs into master GeoPackage (including rasters) and statistics folder."""

        print("\n1. Consolidating vector outputs into master GeoPackage...")

        # Track what we're adding
        layers_added = []

        for analysis_name, result_data in self.results.items():
            if result_data['status'] != 'SUCCESS':
                continue

            temp_folder = result_data['result']['temp_folder']

            # Find GeoPackage files
            gpkg_files = list(temp_folder.glob("*.gpkg"))

            for gpkg_file in gpkg_files:
                print(f"\n   Processing: {analysis_name} - {gpkg_file.name}")

                # Read all layers from this GeoPackage
                import fiona
                try:
                    layers = fiona.listlayers(str(gpkg_file))

                    for layer in layers:
                        gdf = gpd.read_file(gpkg_file, layer=layer)

                        # Create descriptive layer name
                        layer_name = f"{analysis_name.lower()}_{layer}"

                        # Write to consolidated GeoPackage
                        gdf.to_file(
                            self.consolidated_gpkg,
                            driver='GPKG',
                            layer=layer_name
                        )

                        layers_added.append({
                            'analysis': analysis_name,
                            'layer': layer_name,
                            'type': 'vector',
                            'features': len(gdf)
                        })

                        print(f"     ✓ Added vector layer: {layer_name} ({len(gdf):,} features)")

                except Exception as e:
                    print(f"     ✗ Error reading {gpkg_file.name}: {e}")

        # Add raster files to GeoPackage as raster layers
        print("\n2. Adding raster outputs to GeoPackage...")

        for analysis_name, result_data in self.results.items():
            if result_data['status'] != 'SUCCESS':
                continue

            temp_folder = result_data['result']['temp_folder']

            # Find TIF files
            tif_files = list(temp_folder.glob("*.tif"))

            for tif_file in tif_files:
                try:
                    # Read the raster
                    with rasterio.open(tif_file) as src:
                        raster_data = src.read()
                        profile = src.profile

                        # Convert raster to GeoDataFrame of raster pixels (for GeoPackage compatibility)
                        # Note: This creates a gridded point layer representing the raster
                        layer_name = f"{analysis_name.lower()}_raster_{tif_file.stem}"

                        # Alternative 1: Store raster metadata as a table
                        self._add_raster_metadata_to_gpkg(
                            tif_file,
                            layer_name,
                            profile,
                            analysis_name
                        )

                        layers_added.append({
                            'analysis': analysis_name,
                            'layer': layer_name + '_metadata',
                            'type': 'raster_metadata',
                            'features': 1
                        })

                        print(f"     ✓ Added raster metadata: {layer_name}_metadata")

                        # Also keep a copy of the actual raster file alongside the GeoPackage
                        dest_name = f"{analysis_name.lower()}_{tif_file.name}"
                        dest_path = self.master_output / dest_name
                        shutil.copy2(tif_file, dest_path)
                        print(f"     ✓ Copied raster file: {dest_name}")

                except Exception as e:
                    print(f"     ✗ Error processing raster {tif_file.name}: {e}")

        # Consolidate statistics CSVs
        print("\n3. Consolidating statistics CSVs...")

        for analysis_name, result_data in self.results.items():
            if result_data['status'] != 'SUCCESS':
                continue

            temp_folder = result_data['result']['temp_folder']

            # Find CSV files
            csv_files = list(temp_folder.glob("*.csv"))

            for csv_file in csv_files:
                dest_name = f"{analysis_name.lower()}_{csv_file.name}"
                dest_path = self.consolidated_stats / dest_name

                shutil.copy2(csv_file, dest_path)
                print(f"   ✓ Copied: {dest_name}")

        # Create consolidated summary CSV
        print("\n4. Creating master summary CSV...")
        self._create_master_summary(layers_added)

        # Clean up temp folders
        print("\n5. Cleaning up temporary folders...")
        for analysis_name, result_data in self.results.items():
            if result_data['status'] == 'SUCCESS':
                temp_folder = result_data['result']['temp_folder']
                if temp_folder.exists():
                    shutil.rmtree(temp_folder)
                    print(f"   ✓ Removed: {temp_folder.name}")

    def _add_raster_metadata_to_gpkg(self, raster_file, layer_name, profile, analysis_name):
        """Add raster metadata as a table in the GeoPackage."""

        with rasterio.open(raster_file) as src:
            # Create metadata dictionary
            metadata = {
                'analysis': [analysis_name],
                'layer_name': [layer_name],
                'raster_file': [raster_file.name],
                'width': [src.width],
                'height': [src.height],
                'crs': [str(src.crs)],
                # 'transform': [str(src.transform)],
                'bounds_minx': [src.bounds.left],
                'bounds_miny': [src.bounds.bottom],
                'bounds_maxx': [src.bounds.right],
                'bounds_maxy': [src.bounds.top],
                'nodata': [src.nodata],
                'dtype': [str(src.dtypes[0])],
                'count': [src.count]
            }

            # Create DataFrame
            metadata_df = pd.DataFrame(metadata)

            # Create a simple point geometry at raster center for GeoPackage compatibility
            center_x = (src.bounds.left + src.bounds.right) / 2
            center_y = (src.bounds.bottom + src.bounds.top) / 2

            from shapely.geometry import Point
            metadata_df['geometry'] = [Point(center_x, center_y)]

            # Convert to GeoDataFrame
            metadata_gdf = gpd.GeoDataFrame(metadata_df, geometry='geometry', crs=src.crs)

            # Write to GeoPackage
            metadata_gdf.to_file(
                self.consolidated_gpkg,
                driver='GPKG',
                layer=f"{layer_name}_metadata"
            )

    def _create_master_summary(self, layers_added):
        """Create a master summary CSV with all results."""

        summary_data = []

        # Add analysis summaries
        for analysis_name, result_data in self.results.items():
            if result_data['status'] == 'SUCCESS':
                stats = result_data['result']['stats']
                duration = result_data['duration']

                summary_data.append({
                    'analysis': analysis_name,
                    'status': 'SUCCESS',
                    'duration_seconds': round(duration, 2),
                    'duration_formatted': self._format_duration(duration)
                })
            else:
                summary_data.append({
                    'analysis': analysis_name,
                    'status': 'FAILED',
                    'error': result_data.get('error', 'Unknown error'),
                    'duration_seconds': 0,
                    'duration_formatted': 'N/A'
                })

        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.master_output / "workflow_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        print(f"   ✓ Created: {summary_csv.name}")

        # Create layer inventory
        if layers_added:
            layers_df = pd.DataFrame(layers_added)
            layers_csv = self.master_output / "layer_inventory.csv"
            layers_df.to_csv(layers_csv, index=False)
            print(f"   ✓ Created: {layers_csv.name}")

    def _print_summary(self):
        """Print final workflow summary."""

        total_duration = self.end_time - self.start_time

        print("\n" + "=" * 80)
        print("WORKFLOW SUMMARY")
        print("=" * 80)

        print(f"\nTotal duration: {self._format_duration(total_duration)}")
        print(f"  (Parallel execution with {self.max_workers} workers)")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nAnalysis Results:")
        successful = 0
        failed = 0

        for name, result in self.results.items():
            status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
            duration_str = self._format_duration(result['duration']) if result['status'] == 'SUCCESS' else "N/A"
            print(f"  {status_symbol} {name:15} - {result['status']:8} ({duration_str})")

            if result['status'] == 'SUCCESS':
                successful += 1
            else:
                failed += 1

        print(f"\nSuccess: {successful}/{len(self.results)}")
        if failed > 0:
            print(f"Failed:  {failed}/{len(self.results)}")

        print(f"\nOutputs consolidated in: {self.master_output}")
        print("\nFiles created:")
        print(f"  1. consolidated_impacts.gpkg (all spatial outputs + raster metadata)")
        print(f"  2. consolidated_statistics/ (all CSV files)")
        print(f"  3. *_affected.tif (raster files - referenced in GPKG)")
        print(f"  4. workflow_summary.csv (analysis metadata)")
        print(f"  5. layer_inventory.csv (GeoPackage layer details)")

        # Calculate speedup estimate
        if all(r['status'] == 'SUCCESS' for r in self.results.values()):
            sequential_time = sum(r['duration'] for r in self.results.values())
            speedup = sequential_time / total_duration
            print(f"\nEstimated speedup: {speedup:.1f}x faster than sequential execution")

        print("=" * 80)

    @staticmethod
    def _format_duration(seconds):
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


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

    config = {
        'population_raster': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\global_pop_2025_CN_1km_R2025A_UA_v1.tif",
        'farmland_raster_folder': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\ESA_Caribbean",
        'building_parquet': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-polygons.parquet",
        'transportation_parquet': r"C:\C_Drive_Brians_Stuff\Python_Projects\Files\OSM_Parquet\central-america-QGIS-lines.parquet"
    }

    master_output_folder = r"C:\C_Drive_Brians_Stuff\Python_Projects\Global_Impact_Analysis"

    # Run workflow with parallel execution (adjust max_workers based on your RAM)
    workflow = ImpactAnalysisWorkflow(
        basin_file,
        config,
        master_output_folder,
        max_workers=4  # Adjust based on your system (4 is good for 32GB+ RAM)
    )
    workflow.run_all_analyses()


    # basins = [...]
    # args = [(basin_file, config, master_output_folder) for basin_file in basins]
    # with mp.Pool(processes=4) as pool:
    #     pool.starmap(run_worfklow, args)
