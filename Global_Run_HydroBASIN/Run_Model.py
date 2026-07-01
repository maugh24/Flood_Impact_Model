"""
Global impact analysis orchestrator - HydroBASINS (single-file, in-memory) variant.

Instead of iterating Hive-partitioned TDX VPU files, this loads ONE global
HUC12 HydroBASINS parquet once, spatially sorts the basins, splits them into
fixed-size chunks, and runs the four impact analyses over those chunks in a
worker pool. Each worker is handed its chunk's basins GeoDataFrame directly, so
the 1M-row basin file is never re-read per chunk. The basin id column is
HYBAS_ID.

OSM and ESA inputs are still folders of tiled parquets; each impact module
bbox-filters the tiles relevant to each chunk.

Output layout (a single set of files - there are no VPUs here):
    master_output/
        Statistics/
            population_statistics.{parquet,csv}
            farmland_statistics.{parquet,csv}
            building_statistics.{parquet,csv}
            transportation_statistics.{parquet,csv}
            Global_Summary.csv

Per-statistic resume: a statistic whose CSV already exists is skipped, so a
re-run only recomputes the unfinished statistics.
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

from Global_Run_TDX.Population_Impact import calculate_basin_population_wrapper, aggregate_population_for_csv
from Global_Run_TDX.Farmland_Impact import calculate_basin_farmland_wrapper, aggregate_farmland_for_csv
from Global_Run_TDX.Building_Impact import calculate_basin_building_wrapper
from Global_Run_TDX.Road_Impact import calculate_basin_transportation_wrapper, aggregate_transportation_for_csv

# Basin id column for the HydroBASINS file.
BASIN_ID = "HYBAS_ID"


class GlobalImpactWorkflow:
    """Loads the global basin file once and streams basin chunks to a pool."""

    def __init__(self, basin_file, config, master_output_folder,
                 max_workers=4, chunk_size=100):
        self.basin_file = basin_file
        self.config = config
        self.master_output = Path(master_output_folder)
        self.max_workers = max_workers
        self.chunk_size = chunk_size

        self.master_output.mkdir(parents=True, exist_ok=True)
        self.stats_root = self.master_output / "Statistics"
        self.stats_root.mkdir(exist_ok=True)

        self.basins = None      # sorted GeoDataFrame [HYBAS_ID, geometry]
        self.n_chunks = 0

    # ----- basin loading / chunking -----

    def _load_basins(self):
        """Read the global basin file once, ensure EPSG:4326, spatially sort
        for tile-cache locality, and keep only [HYBAS_ID, geometry]."""
        cols = [BASIN_ID, 'PFAF_ID', 'geometry']
        try:
            gdf = gpd.read_parquet(self.basin_file, columns=cols)
        except Exception:
            # PFAF_ID missing for some reason - fall back to id + geometry.
            gdf = gpd.read_parquet(self.basin_file, columns=[BASIN_ID, 'geometry'])

        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs('EPSG:4326')

        # Spatial-ish ordering so consecutive chunks are geographically close,
        # which keeps each chunk's bbox tight and reuses cached OSM/ESA tiles.
        # PFAF_ID (Pfafstetter) is hierarchical/spatially coherent and free to
        # sort on; fall back to centroid distance from (0,0) if it's absent.
        if 'PFAF_ID' in gdf.columns:
            gdf = gdf.sort_values('PFAF_ID')
        else:
            c = gdf.geometry.centroid
            gdf = gdf.assign(_d=np.sqrt(c.x ** 2 + c.y ** 2)).sort_values('_d')

        gdf = gdf[[BASIN_ID, 'geometry']].reset_index(drop=True)
        return gdf

    def _iter_chunks(self):
        """Yield successive [HYBAS_ID, geometry] chunks of chunk_size basins.
        A fresh generator is produced on each call so every statistic can
        iterate the full basin set."""
        n = self.chunk_size
        gdf = self.basins
        for i in range(0, len(gdf), n):
            yield gdf.iloc[i:i + n]

    def _stat_done(self, csv_name):
        """True if this statistic's CSV already exists, so it can be skipped."""
        return (self.stats_root / csv_name).exists()

    # ----- the four statistics -----

    def _run_population(self, pool):
        if self._stat_done("population_statistics.csv"):
            print("  population already done, skipping")
            return
        args = ((chunk, self.config['population_source']) for chunk in self._iter_chunks())
        pop_dfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_population_wrapper, args),
            total=self.n_chunks, desc="Population    "
        ))
        pop_result = pd.concat(pop_dfs, ignore_index=True)
        pop_gdf = gpd.GeoDataFrame(
            pop_result,
            geometry=gpd.points_from_xy(pop_result['x'], pop_result['y']),
            crs='EPSG:4326'
        )
        pop_gdf.to_parquet(self.stats_root / "population_statistics.parquet", index=False)
        aggregate_population_for_csv(pop_result).to_csv(
            self.stats_root / "population_statistics.csv", index=False
        )

    def _run_farmland(self, pool):
        if self._stat_done("farmland_statistics.csv"):
            print("  farmland already done, skipping")
            return
        args = ((chunk, self.config['farmland_source']) for chunk in self._iter_chunks())
        farm_gdfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_farmland_wrapper, args),
            total=self.n_chunks, desc="Farmland      "
        ))
        farm_result = gpd.GeoDataFrame(pd.concat(farm_gdfs, ignore_index=True), crs='EPSG:4326')
        farm_result.to_parquet(self.stats_root / "farmland_statistics.parquet", index=False)
        aggregate_farmland_for_csv(farm_result).to_csv(
            self.stats_root / "farmland_statistics.csv", index=False
        )

    def _run_buildings(self, pool):
        if self._stat_done("building_statistics.csv"):
            print("  buildings already done, skipping")
            return
        args = ((chunk, self.config['building_source']) for chunk in self._iter_chunks())
        building_dfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_building_wrapper, args),
            total=self.n_chunks, desc="Buildings     "
        ))
        building_result = pd.concat(building_dfs, ignore_index=True)
        building_gdf = gpd.GeoDataFrame(
            building_result,
            geometry=gpd.points_from_xy(building_result['x'], building_result['y']),
            crs='EPSG:4326'
        )
        real = building_result[building_result[BASIN_ID] != 'TOTAL']
        building_count = real.groupby(BASIN_ID).size().reset_index(name='building_count')
        total_buildings = real.shape[0]
        building_count_with_total = pd.concat([
            pd.DataFrame({BASIN_ID: ['TOTAL'], 'building_count': [total_buildings]}),
            building_count
        ], ignore_index=True)
        building_gdf.to_parquet(self.stats_root / "building_statistics.parquet", index=False)
        building_count_with_total.to_csv(self.stats_root / "building_statistics.csv", index=False)

    def _run_transportation(self, pool):
        if self._stat_done("transportation_statistics.csv"):
            print("  transportation already done, skipping")
            return
        args = ((chunk, self.config['transportation_source']) for chunk in self._iter_chunks())
        trans_dfs = list(tqdm.tqdm(
            pool.imap_unordered(calculate_basin_transportation_wrapper, args),
            total=self.n_chunks, desc="Transportation"
        ))
        if trans_dfs:
            trans_result = gpd.GeoDataFrame(pd.concat(trans_dfs, ignore_index=True),
                                            geometry='geometry', crs='EPSG:4326')
        else:
            trans_result = gpd.GeoDataFrame(
                columns=[BASIN_ID, 'infrastructure_type', 'feature_value',
                         'length_m', 'length_km', 'geometry'],
                geometry='geometry', crs='EPSG:4326'
            )
        trans_result.to_parquet(self.stats_root / "transportation_statistics.parquet", index=False)
        aggregate_transportation_for_csv(trans_result).to_csv(
            self.stats_root / "transportation_statistics.csv", index=False
        )

    # ----- global summary -----

    def _consolidate_global_summary(self):
        """Read the TOTAL row of each statistic's CSV and write Global_Summary.csv."""
        spec = [
            ("population_statistics.csv",     "pop_value",      "total_population"),
            ("farmland_statistics.csv",       "area_m2",        "total_farmland_m2"),
            ("building_statistics.csv",       "building_count", "total_building_count"),
            ("transportation_statistics.csv", "highway_km",     "total_highway_km"),
            ("transportation_statistics.csv", "railway_km",     "total_railway_km"),
        ]
        rows = []
        for csv_name, col, label in spec:
            csv_path = self.stats_root / csv_name
            value = float('nan')
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if col in df.columns:
                        if BASIN_ID in df.columns and (df[BASIN_ID].astype(str) == 'TOTAL').any():
                            value = df.loc[df[BASIN_ID].astype(str) == 'TOTAL', col].iloc[0]
                        else:
                            value = df[col].sum(skipna=True)
                except Exception as e:
                    print(f"  [global summary] could not read {csv_path}: {e}")
            rows.append({'metric': label, 'value': value})

        summary = pd.DataFrame(rows)
        out_path = self.stats_root / "Global_Summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"  [global summary] wrote {out_path}")
        print(summary.to_string(index=False))

    # ----- public entry point -----

    def run_all_analyses(self):
        print("=" * 80)
        print("GLOBAL IMPACT ANALYSIS WORKFLOW - HydroBASINS (in-memory)")
        print("=" * 80)
        print(f"Start time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Basin file:    {self.basin_file}")
        print(f"Master output: {self.master_output}")
        print(f"Workers:       {self.max_workers}")
        print(f"Chunk size:    {self.chunk_size}")
        print("=" * 80)

        t_load = time.time()
        self.basins = self._load_basins()
        self.n_chunks = (len(self.basins) + self.chunk_size - 1) // self.chunk_size
        print(f"Loaded {len(self.basins):,} basins -> {self.n_chunks:,} chunks "
              f"in {time.time()-t_load:.1f}s")
        print("=" * 80)

        start = time.time()
        with mp.Pool(processes=self.max_workers) as pool:
            # Population is disabled here (as in the TDX run). Uncomment to enable.
            # self._run_population(pool)
            self._run_farmland(pool)
            self._run_buildings(pool)
            self._run_transportation(pool)

        print("=" * 80)
        print("Consolidating global summary...")
        self._consolidate_global_summary()

        print("=" * 80)
        print(f"Total elapsed: {time.time()-start:.1f}s")
        print(f"Outputs in:    {self.stats_root}")
        print("=" * 80)


# ===== USAGE =====
if __name__ == "__main__":
    # Single global HUC12 HydroBASINS parquet (id column = HYBAS_ID).
    basin_file = "/Users/maugh24/Flood_Impact_Model/HUC12.parquet"

    # Folders of tiled inputs. Each impact module bbox-filters per chunk.
    config = {
        'population_source':     "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/Population/population.parquet",
        'farmland_source':       "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/ESA/bbox",
        'building_source':       "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/OSM/Parquet_sorted",
        'transportation_source': "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/OSM/Parquet_sorted",
    }

    master_output_folder = "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Global_Impact_Results_HydroBASIN"

    workflow = GlobalImpactWorkflow(
        basin_file=basin_file,
        config=config,
        master_output_folder=master_output_folder,
        max_workers=mp.cpu_count(),  # tune to your RAM (~3-4 GB per worker on dense chunks)
        chunk_size=100
    )
    workflow.run_all_analyses()
