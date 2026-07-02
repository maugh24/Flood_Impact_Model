"""
Global impact analysis orchestrator - HydroBASINS (single-file, in-memory).
Loads one global HUC12 parquet, chunks the basins, and runs the impact
analyses over the chunks in a worker pool. Basin id column is HYBAS_ID.

Checkpointed: each chunk writes a geometry parquet fragment (in <stat>_by_chunk/)
and a small marker/aggregate CSV (in _partial/<stat>/). On restart, chunks that
already have a marker are skipped, so a crash only ever costs the in-flight
chunk. Marker CSVs are combined into the single <stat>_statistics.csv at the end.
"""
import time
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import tqdm

warnings.filterwarnings('ignore')

from Population_Impact import calculate_basin_population_wrapper
from Farmland_Impact import calculate_basin_farmland_wrapper
from Building_Impact import calculate_basin_building_wrapper
from Road_Impact import calculate_basin_transportation_wrapper

BASIN_ID = "HYBAS_ID"


# Attach the chunk index to each result so the driver can checkpoint per chunk
# and skip finished chunks on resume. Top-level so they pickle for the pool.
def _keyed_population(args):
    key, chunk, source = args
    return key, calculate_basin_population_wrapper((chunk, source))


def _keyed_farmland(args):
    key, chunk, source = args
    return key, calculate_basin_farmland_wrapper((chunk, source))


def _keyed_building(args):
    key, chunk, source = args
    return key, calculate_basin_building_wrapper((chunk, source))


def _keyed_transportation(args):
    key, chunk, source = args
    return key, calculate_basin_transportation_wrapper((chunk, source))


class GlobalImpactWorkflow:

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

        self.basins = None
        self.n_chunks = 0

    def _load_basins(self):
        cols = [BASIN_ID, 'PFAF_ID', 'geometry']
        try:
            gdf = gpd.read_parquet(self.basin_file, columns=cols)
        except Exception:
            gdf = gpd.read_parquet(self.basin_file, columns=[BASIN_ID, 'geometry'])

        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs('EPSG:4326')

        if 'PFAF_ID' in gdf.columns:
            gdf = gdf.sort_values('PFAF_ID')
        else:
            c = gdf.geometry.centroid
            gdf = gdf.assign(_d=np.sqrt(c.x ** 2 + c.y ** 2)).sort_values('_d')

        return gdf[[BASIN_ID, 'geometry']].reset_index(drop=True)

    def _stat_done(self, csv_name):
        return (self.stats_root / csv_name).exists()

    def _run_checkpointed(self, pool, keyed_wrapper, source_key, stat_name, desc,
                          process_fn, geom_folder, final_csv, value_cols, min_count):
        partial = self.stats_root / "_partial" / stat_name
        partial.mkdir(parents=True, exist_ok=True)
        geom_dir = self.stats_root / geom_folder
        geom_dir.mkdir(exist_ok=True)

        def marker(i):
            return partial / f"chunk_{i:06d}.csv"

        n = self.chunk_size
        pending = sum(1 for i in range(self.n_chunks) if not marker(i).exists())
        if pending:
            print(f"  {stat_name}: {self.n_chunks - pending} chunks done, {pending} to compute")

            def gen():
                src = self.config[source_key]
                for i in range(self.n_chunks):
                    if not marker(i).exists():
                        yield (i, self.basins.iloc[i * n:(i + 1) * n], src)

            for key, res in tqdm.tqdm(pool.imap_unordered(keyed_wrapper, gen()),
                                      total=pending, desc=desc,
                                      mininterval=0.5, dynamic_ncols=True):
                geom, agg = process_fn(res)
                if geom is not None and len(geom):
                    geom.to_parquet(geom_dir / f"chunk_{key:06d}.parquet",
                                    index=False, write_covering_bbox=True)
                # Marker written last: its existence means the chunk is fully done.
                agg.to_csv(marker(key), index=False)
        else:
            print(f"  {stat_name}: all {self.n_chunks} chunks already done")

        self._combine_csv(partial, final_csv, value_cols, min_count)

    def _combine_csv(self, partial, final_csv, value_cols, min_count):
        frames = []
        for c in sorted(partial.glob("chunk_*.csv")):
            try:
                d = pd.read_csv(c)
            except Exception:
                continue
            if len(d):
                frames.append(d)
        per_basin = (pd.concat(frames, ignore_index=True) if frames
                     else pd.DataFrame({BASIN_ID: [], **{c: [] for c in value_cols}}))
        total = {BASIN_ID: ['TOTAL']}
        for c in value_cols:
            if c not in per_basin.columns:
                per_basin[c] = pd.Series(dtype=float)
            total[c] = [per_basin[c].sum(min_count=1) if min_count else per_basin[c].sum()]
        out = pd.concat([pd.DataFrame(total), per_basin], ignore_index=True)
        out.to_csv(self.stats_root / final_csv, index=False)

    def _run_population(self, pool):
        if self._stat_done("population_statistics.csv"):
            print("  population already done, skipping")
            return

        def process(res):
            real = res[res['x'].notna()]
            geom = (gpd.GeoDataFrame(real, geometry=gpd.points_from_xy(real['x'], real['y']),
                                     crs='EPSG:4326') if len(real) else None)
            agg = pd.DataFrame(res).groupby(BASIN_ID, dropna=False, as_index=False)['pop_value'].sum(min_count=1)
            return geom, agg

        self._run_checkpointed(pool, _keyed_population, 'population_source', 'population',
                               "Population    ", process, "population_by_chunk",
                               "population_statistics.csv", ['pop_value'], True)

    def _run_farmland(self, pool):
        if self._stat_done("farmland_statistics.csv"):
            print("  farmland already done, skipping")
            return

        def process(res):
            geom = res[res.geometry.notna()]
            agg = (pd.DataFrame(res.drop(columns='geometry', errors='ignore'))
                   .groupby(BASIN_ID, dropna=False, as_index=False)['area_m2'].sum(min_count=1))
            return (geom if len(geom) else None), agg

        self._run_checkpointed(pool, _keyed_farmland, 'farmland_source', 'farmland',
                               "Farmland      ", process, "farmland_by_chunk",
                               "farmland_statistics.csv", ['area_m2'], True)

    def _run_buildings(self, pool):
        if self._stat_done("building_statistics.csv"):
            print("  buildings already done, skipping")
            return

        def process(res):
            real = res[res['building'].notna()]
            if len(real):
                geom = gpd.GeoDataFrame(real, geometry=gpd.points_from_xy(real['x'], real['y']),
                                        crs='EPSG:4326')
                agg = real.groupby(BASIN_ID).size().reset_index(name='building_count')
            else:
                geom, agg = None, pd.DataFrame({BASIN_ID: [], 'building_count': []})
            return geom, agg

        self._run_checkpointed(pool, _keyed_building, 'building_source', 'building',
                               "Buildings     ", process, "building_by_chunk",
                               "building_statistics.csv", ['building_count'], False)

    def _run_transportation(self, pool):
        if self._stat_done("transportation_statistics.csv"):
            print("  transportation already done, skipping")
            return

        def process(res):
            geom = res[res.geometry.notna()] if len(res) else res
            df = pd.DataFrame(res.drop(columns='geometry', errors='ignore'))
            if len(df):
                piv = (df.groupby([BASIN_ID, 'infrastructure_type'])['length_km']
                       .sum(min_count=1).unstack('infrastructure_type')
                       .rename(columns={'highway': 'highway_km', 'railway': 'railway_km'}))
                for c in ('highway_km', 'railway_km'):
                    if c not in piv.columns:
                        piv[c] = None
                agg = piv.reset_index()[[BASIN_ID, 'highway_km', 'railway_km']]
            else:
                agg = pd.DataFrame({BASIN_ID: [], 'highway_km': [], 'railway_km': []})
            return (geom if len(geom) else None), agg

        self._run_checkpointed(pool, _keyed_transportation, 'transportation_source', 'transportation',
                               "Transportation", process, "transportation_by_chunk",
                               "transportation_statistics.csv", ['highway_km', 'railway_km'], True)

    def _consolidate_global_summary(self):
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

    def run_all_analyses(self):
        print("=" * 80)
        print("GLOBAL IMPACT ANALYSIS WORKFLOW - HydroBASINS")
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
            # self._run_population(pool)   # disabled
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


if __name__ == "__main__":
    basin_file = "/Users/maugh24/Flood_Impact_Model/HUC12.parquet"

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
        max_workers=mp.cpu_count(),
        chunk_size=200
    )
    workflow.run_all_analyses()
