import geopandas as gpd
import pandas as pd
from pathlib import Path
import shutil
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import your four impact models
from Population_Impact import calculate_basin_population
from Farmland_Impact import calculate_basin_farmland
from Building_Impact import calculate_basin_buildings_from_parquet
from Road_Impact import calculate_basin_transportation_from_parquet


class ImpactAnalysisWorkflow:
    """
    Workflow manager for running all impact analyses individually and consolidating outputs.
    Combines into one geopackage
    Throws all csv files in one results folder
    If one analysis fails, it continues with the others and reports the failure in the final summary.
    Tracks the time taken for each analysis and the overall workflow, and prints a final summary at the end.
    """

    def __init__(self, basin_file, config, master_output_folder):
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
        """
        self.basin_file = basin_file
        self.config = config
        self.master_output = Path(master_output_folder)

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
        """Run all four impact analyses in sequence."""

        print("=" * 80)
        print("GLOBAL IMPACT ANALYSIS WORKFLOW")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Basin file: {self.basin_file}")
        print(f"Master output: {self.master_output}")
        print("=" * 80)

        self.start_time = time.time()

        # Run each analysis
        analyses = [
            ("Population", self._run_population_analysis),
            ("Farmland", self._run_farmland_analysis),
            ("Buildings", self._run_building_analysis),
            ("Transportation", self._run_transportation_analysis)
        ]

        for name, analysis_func in analyses:
            print(f"\n{'=' * 80}")
            print(f"RUNNING: {name} Impact Analysis")
            print(f"{'=' * 80}")

            try:
                analysis_start = time.time()
                result = analysis_func()
                analysis_end = time.time()

                self.results[name] = {
                    'status': 'SUCCESS',
                    'result': result,
                    'duration': analysis_end - analysis_start
                }

                print(f"\n✓ {name} analysis completed in {self._format_duration(analysis_end - analysis_start)}")

            except Exception as e:
                print(f"\n✗ {name} analysis FAILED: {str(e)}")
                self.results[name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'duration': 0
                }

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
        temp_output = self.master_output / "_temp_population"

        result = calculate_basin_population(
            self.basin_file,
            self.config['population_raster'],
            str(temp_output)
        )

        return {'temp_folder': temp_output, 'stats': result}

    def _run_farmland_analysis(self):
        """Run farmland impact analysis."""
        temp_output = self.master_output / "_temp_farmland"

        result = calculate_basin_farmland(
            self.basin_file,
            self.config['farmland_raster_folder'],
            str(temp_output),
            farmland_value=40
        )

        return {'temp_folder': temp_output, 'stats': result}

    def _run_building_analysis(self):
        """Run building impact analysis."""
        temp_output = self.master_output / "_temp_buildings"

        result = calculate_basin_buildings_from_parquet(
            self.basin_file,
            self.config['building_parquet'],
            str(temp_output)
        )

        return {'temp_folder': temp_output, 'stats': result}

    def _run_transportation_analysis(self):
        """Run transportation impact analysis."""
        temp_output = self.master_output / "_temp_transportation"

        result = calculate_basin_transportation_from_parquet(
            self.basin_file,
            self.config['transportation_parquet'],
            str(temp_output)
        )

        return {'temp_folder': temp_output, 'stats': result}

    def _consolidate_outputs(self):
        """Consolidate all outputs into master GeoPackage and statistics folder."""

        print("\n1. Consolidating spatial outputs into master GeoPackage...")

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
                            'features': len(gdf)
                        })

                        print(f"     ✓ Added layer: {layer_name} ({len(gdf):,} features)")

                except Exception as e:
                    print(f"     ✗ Error reading {gpkg_file.name}: {e}")

        # Copy raster files to master output
        print("\n2. Copying raster outputs...")

        for analysis_name, result_data in self.results.items():
            if result_data['status'] != 'SUCCESS':
                continue

            temp_folder = result_data['result']['temp_folder']

            # Find TIF files
            tif_files = list(temp_folder.glob("*.tif"))

            for tif_file in tif_files:
                dest_name = f"{analysis_name.lower()}_{tif_file.name}"
                dest_path = self.master_output / dest_name

                shutil.copy2(tif_file, dest_path)
                print(f"   ✓ Copied: {dest_name}")

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
                    'error': result_data.get('error', 'Unknown error')
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
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nAnalysis Results:")
        for name, result in self.results.items():
            status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
            duration_str = self._format_duration(result['duration']) if result['status'] == 'SUCCESS' else "N/A"
            print(f"  {status_symbol} {name:15} - {result['status']:8} ({duration_str})")

        print(f"\nOutputs consolidated in: {self.master_output}")
        print("\nFiles created:")
        print(f"  1. consolidated_impacts.gpkg (all spatial outputs)")
        print(f"  2. consolidated_statistics/ (all CSV files)")
        print(f"  3. *_affected.tif (raster outputs)")
        print(f"  4. workflow_summary.csv (analysis metadata)")
        print(f"  5. layer_inventory.csv (GeoPackage layer details)")

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

    # Run workflow
    workflow = ImpactAnalysisWorkflow(basin_file, config, master_output_folder)
    workflow.run_all_analyses()
