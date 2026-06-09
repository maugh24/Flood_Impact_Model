"""
Global impact analysis orchestrator.

Iterates over every VPU partition in a Hive-laid catchments folder
(vpu=NNN/catchments_NNN.parquet) and runs the four impact analyses on
each VPU's basins. OSM and ESA inputs are folders of tiled parquets;
each impact module bbox-filters the tiles relevant to each basin chunk.

Output layout:
    master_output/
        Statistics/
            vpu=101/
                population_statistics.parquet
                population_statistics.csv
                farmland_statistics.parquet
                farmland_statistics.csv
                building_statistics.parquet
                building_statistics.csv
                transportation_statistics.parquet
                transportation_statistics.csv
            vpu=102/
                ...

A VPU whose output folder already contains all 4 CSVs is skipped, so
the run is resumable after a crash.
"""
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

from Population_Impact import calculate_basin_population_wrapper, aggregate_population_for_csv
from Farmland_Impact import calculate_basin_farmland_wrapper, aggregate_farmland_for_csv
from Building_Impact import calculate_basin_building_wrapper
from Road_Impact import calculate_basin_transportation_wrapper, aggregate_transportation_for_csv
from tile_loader import iter_vpu_files


# Filenames a completed VPU should have. Used by the skip check.
_REQUIRED_CSVS = (
    "population_statistics.csv",
    "farmland_statistics.csv",
    "building_statistics.csv",
    "transportation_statistics.csv",
)


class GlobalImpactWorkflow:
    """Outer driver over all VPUs. One mp.Pool is reused across VPUs."""

    def __init__(self, catchments_folder, config, master_output_folder,
                 max_workers=4, chunk_size=100):
        self.catchments_folder = catchments_folder
        self.config = config
        self.master_output = Path(master_output_folder)
        self.max_workers = max_workers
        self.chunk_size = chunk_size

        self.master_output.mkdir(parents=True, exist_ok=True)
        self.stats_root = self.master_output / "Statistics"
        self.stats_root.mkdir(exist_ok=True)

    # ----- per-VPU helpers -----

    @staticmethod
    def _sort_rivids(basin_file):
        """Sort by distance from (0,0) so consecutive chunks tend to be
        geographically close - improves tile-cache locality."""
        gdf = gpd.read_parquet(basin_file)
        centroid = gdf.geometry.centroid
        gdf['_d'] = np.sqrt(centroid.x ** 2 + centroid.y ** 2)
        gdf = gdf.sort_values('_d')
        return gdf['LINKNO'].tolist()

    def _vpu_already_done(self, vpu_stats_dir):
        return all((vpu_stats_dir / name).exists() for name in _REQUIRED_CSVS)

    def _run_vpu(self, vpu_id, vpu_file, pool):
        vpu_stats_dir = self.stats_root / f"vpu={vpu_id}"
        if self._vpu_already_done(vpu_stats_dir):
            print(f"  [vpu={vpu_id}] already complete, skipping")
            return
        vpu_stats_dir.mkdir(parents=True, exist_ok=True)

        rivers = self._sort_rivids(vpu_file)
        if not rivers:
            print(f"  [vpu={vpu_id}] no basins, skipping")
            return

        n = self.chunk_size
        rivers_split = [rivers[i:i+n] for i in range(0, len(rivers), n)]

        # ----- Population -----
      #  args = [(vpu_file, rivs, self.config['population_source']) for rivs in rivers_split]
      #  pop_dfs = list(tqdm.tqdm(
      #      pool.imap_unordered(calculate_basin_population_wrapper, args),
      #      total=len(args), desc=f"[vpu={vpu_id}] Population    "
      #  ))
      #  pop_result = pd.concat(pop_dfs, ignore_index=True)
      #  pop_gdf = gpd.GeoDataFrame(
      #      pop_result,
      #      geometry=gpd.points_from_xy(pop_result['x'], pop_result['y']),
      #      crs='EPSG:4326'
      #  )
      #  pop_gdf.to_parquet(vpu_stats_dir / "population_statistics.parquet", index=False)
      #  aggregate_population_for_csv(pop_result).to_csv(
      #      vpu_stats_dir / "population_statistics.csv", index=False
      #  )
#
      #  # ----- Farmland -----
        args = [(vpu_file, rivs, self.config['farmland_source']) for rivs in rivers_split]
        farm_gdfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_farmland_wrapper, args),
            total=len(args), desc=f"[vpu={vpu_id}] Farmland      "
        ))
        farm_result = gpd.GeoDataFrame(pd.concat(farm_gdfs, ignore_index=True), crs='EPSG:4326')
        farm_result.to_parquet(vpu_stats_dir / "farmland_statistics.parquet", index=False)
        aggregate_farmland_for_csv(farm_result).to_csv(
            vpu_stats_dir / "farmland_statistics.csv", index=False
        )

        # ----- Buildings -----
        args = [(vpu_file, rivs, self.config['building_source']) for rivs in rivers_split]
        building_dfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_building_wrapper, args),
            total=len(args), desc=f"[vpu={vpu_id}] Buildings     "
        ))
        building_result = pd.concat(building_dfs, ignore_index=True)
        building_gdf = gpd.GeoDataFrame(
            building_result,
            geometry=gpd.points_from_xy(building_result['x'], building_result['y']),
            crs='EPSG:4326'
        )
        building_count = (
            building_result[building_result['LINKNO'] != 'TOTAL']
                .groupby('LINKNO').size().reset_index(name='building_count')
        )
        total_buildings = building_result[building_result['LINKNO'] != 'TOTAL'].shape[0]
        building_count_with_total = pd.concat([
            pd.DataFrame({'LINKNO': ['TOTAL'], 'building_count': [total_buildings]}),
            building_count
        ], ignore_index=True)
        building_gdf.to_parquet(vpu_stats_dir / "building_statistics.parquet", index=False)
        building_count_with_total.to_csv(vpu_stats_dir / "building_statistics.csv", index=False)

        # ----- Transportation -----
        args = [(vpu_file, rivs, self.config['transportation_source']) for rivs in rivers_split]
        trans_dfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_transportation_wrapper, args),
            total=len(args), desc=f"[vpu={vpu_id}] Transportation"
        ))
        if trans_dfs:
            trans_result = gpd.GeoDataFrame(pd.concat(trans_dfs, ignore_index=True),
                                            geometry='geometry', crs='EPSG:4326')
        else:
            trans_result = gpd.GeoDataFrame(
                columns=['LINKNO', 'infrastructure_type', 'feature_value',
                         'length_m', 'length_km', 'geometry'],
                geometry='geometry', crs='EPSG:4326'
            )
        trans_result.to_parquet(vpu_stats_dir / "transportation_statistics.parquet", index=False)
        aggregate_transportation_for_csv(trans_result).to_csv(
            vpu_stats_dir / "transportation_statistics.csv", index=False
        )

    # ----- global summary -----

    def _consolidate_global_summary(self):
        """Walk every vpu=*/ folder under Statistics/, read the TOTAL row of
        each impact CSV, sum across all VPUs, and write Global_Summary.csv.

        Layout (long format, one row per metric):
            metric,value
            total_population,X
            total_farmland_m2,Y
            total_building_count,Z
            total_highway_km,A
            total_railway_km,B

        Missing CSVs (e.g., VPUs that failed mid-run) are skipped with a
        warning rather than aborting. NaN totals from VPUs with no data
        in that category contribute 0 to the sum, which is the right
        thing for a global rollup.
        """
        # (csv filename, column-to-sum, metric label)
        spec = [
            ("population_statistics.csv",    "pop_value",      "total_population"),
            ("farmland_statistics.csv",      "area_m2",        "total_farmland_m2"),
            ("building_statistics.csv",      "building_count", "total_building_count"),
            ("transportation_statistics.csv","highway_km",     "total_highway_km"),
            ("transportation_statistics.csv","railway_km",     "total_railway_km"),
        ]

        vpu_dirs = sorted(p for p in self.stats_root.glob("vpu=*") if p.is_dir())
        if not vpu_dirs:
            print("  [global summary] no vpu=*/ folders found, skipping")
            return

        totals = {label: 0.0 for _, _, label in spec}
        vpus_seen = {label: 0 for _, _, label in spec}

        for vdir in vpu_dirs:
            for csv_name, col, label in spec:
                csv_path = vdir / csv_name
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"  [global summary] could not read {csv_path}: {e}")
                    continue
                if col not in df.columns:
                    continue
                # Prefer the TOTAL row if present (cheap, already computed
                # per-VPU); fall back to summing the column otherwise.
                if 'LINKNO' in df.columns and (df['LINKNO'].astype(str) == 'TOTAL').any():
                    val = df.loc[df['LINKNO'].astype(str) == 'TOTAL', col].iloc[0]
                else:
                    val = df[col].sum(skipna=True)
                # Coerce to float; treat NaN as 0 for global rollup.
                try:
                    val = float(val)
                    if pd.isna(val):
                        val = 0.0
                except (TypeError, ValueError):
                    val = 0.0
                totals[label] += val
                vpus_seen[label] += 1

        summary = pd.DataFrame({
            'metric': [label for _, _, label in spec],
            'value':  [totals[label] for _, _, label in spec],
            'vpus_contributing': [vpus_seen[label] for _, _, label in spec],
        })
        out_path = self.stats_root / "Global_Summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"  [global summary] wrote {out_path}")
        # Echo it to stdout so it's visible at the end of a long run.
        print(summary.to_string(index=False))

    # ----- public entry point -----

    def run_all_analyses(self):
        print("=" * 80)
        print("GLOBAL IMPACT ANALYSIS WORKFLOW (PARALLEL EXECUTION)")
        print("=" * 80)
        print(f"Start time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Catchments folder: {self.catchments_folder}")
        print(f"Master output:     {self.master_output}")
        print(f"Workers:           {self.max_workers}")
        print(f"Chunk size:        {self.chunk_size}")
        print("=" * 80)

        vpu_list = list(iter_vpu_files(self.catchments_folder))
        print(f"Found {len(vpu_list)} VPU partitions")
        print("=" * 80)

        start = time.time()
        with mp.Pool(processes=self.max_workers) as pool:
            for vpu_id, vpu_file in vpu_list:
                print(f"\n--- VPU {vpu_id} ({vpu_file}) ---")
                t0 = time.time()
                try:
                    self._run_vpu(vpu_id, vpu_file, pool)
                except Exception as e:
                    # Don't let one bad VPU kill the whole run.
                    print(f"  [vpu={vpu_id}] FAILED: {e}")
                    continue
                print(f"  [vpu={vpu_id}] done in {time.time()-t0:.1f}s")

        print("=" * 80)
        print("Consolidating global summary...")
        self._consolidate_global_summary()

        print("=" * 80)
        print(f"Total elapsed: {time.time()-start:.1f}s")
        print(f"Outputs in:    {self.stats_root}")
        print("=" * 80)


# ===== USAGE =====
if __name__ == "__main__":
    # Folder containing vpu=NNN/catchments_NNN.parquet (Hive partitioned)
    catchments_folder = r"D:\Brian\Flood_Impact_Model\Files\Catchment\GEOGLOW_Catchments"

    # Folders of tiled inputs. Each impact module bbox-filters per basin chunk.
    config = {
        'population_source':     r"D:\Brian\Flood_Impact_Model\Files\Population\population.parquet",
        'farmland_source':       r"D:\Brian\Flood_Impact_Model\Files\ESA\Parquet",
        'building_source':       r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet",
        'transportation_source': r"D:\Brian\Flood_Impact_Model\Files\OSM\Parquet",
    }

    master_output_folder = r"D:\Brian\Flood_Impact_Model\Global_Impact_Results"

    workflow = GlobalImpactWorkflow(
        catchments_folder=catchments_folder,
        config=config,
        master_output_folder=master_output_folder,
        max_workers=mp.cpu_count() // 4,  # tune to your RAM (~3-4 GB per worker on big VPUs)
        chunk_size=100
    )
    workflow.run_all_analyses()
