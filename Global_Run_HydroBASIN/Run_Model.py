"""
Global impact analysis orchestrator - HydroBASINS (single-file, in-memory) variant.

Loads ONE global HUC12 HydroBASINS parquet once, spatially sorts the basins,
splits them into fixed-size chunks, and runs the four impact analyses over
those chunks in a worker pool. Each worker is handed its chunk's basins
GeoDataFrame directly, so the 1M-row basin file is never re-read per chunk.
Basin id column is HYBAS_ID.

Each statistic writes a FOLDER of small per-basin GeoParquet files (one file per
basin, named '<HYBAS_ID>_<type>.parquet' with a covering bbox so they load
cleanly in QGIS) plus ONE combined summary CSV. Writing per basin as chunks
complete keeps memory low and sidesteps Arrow's 2 GB single-array limit, and it
means an interrupted run resumes without rewriting the basin files it already
produced.

Output layout (a single set of outputs - there are no VPUs here):
    master_output/
        Statistics/
            farmland_statistics.csv           (combined summary, with a TOTAL row)
            building_statistics.csv
            transportation_statistics.csv
            population_statistics.csv
            farmland_by_basin/<HYBAS_ID>_farmland.parquet
            building_by_basin/<HYBAS_ID>_buildings.parquet
            transportation_by_basin/<HYBAS_ID>_roads.parquet
            population_by_basin/<HYBAS_ID>_population.parquet
            Global_Summary.csv

Per-statistic resume: a statistic whose combined CSV already exists is skipped;
within a statistic, per-basin files that already exist are skipped too.
"""
import io
import time
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

warnings.filterwarnings('ignore')

from Population_Impact import calculate_basin_population_wrapper
from Farmland_Impact import calculate_basin_farmland_wrapper
from Building_Impact import calculate_basin_building_wrapper
from Road_Impact import calculate_basin_transportation_wrapper
from tile_loader import call_with_io_retry

# Basin id column for the HydroBASINS file.
BASIN_ID = "HYBAS_ID"


# Keyed wrappers (top-level so they pickle for the pool). Each attaches the
# chunk index to its result, so the driver can write a per-chunk checkpoint and,
# on resume, skip chunks that are already done.
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


class _GeoStreamWriter:
    """Append GeoDataFrames to a single GeoParquet file as successive row
    groups. Keeps peak memory at ~one chunk and keeps every WKB array well
    under Arrow's 2 GB per-array limit. Rows with null geometry are dropped
    (missing basins still reach the CSV via the per-basin aggregation).

    The first written chunk fixes the file schema; every later chunk is cast
    to it. Object columns that happen to be all-empty in a chunk (e.g. a chunk
    where no building carries an 'amenity', or no road a 'feature_value') get
    inferred by Arrow as the 'null' type; those are promoted to 'string' so
    they still match a schema where that column is 'string'."""

    def __init__(self, path):
        self.path = str(path)
        self._writer = None
        self._geo = None
        self._schema = None

    @staticmethod
    def _promote_null_columns(table):
        fields, changed = [], False
        for field in table.schema:
            if pa.types.is_null(field.type):
                fields.append(field.with_type(pa.string()))
                changed = True
            else:
                fields.append(field)
        if not changed:
            return table
        return table.cast(pa.schema(fields, metadata=table.schema.metadata))

    def write(self, gdf):
        if gdf is None or len(gdf) == 0:
            return
        gdf = gdf[gdf.geometry.notna()]
        if len(gdf) == 0:
            return
        gdf = gdf.reset_index(drop=True)  # avoid an index column in the arrow table
        if self._geo is None:
            buf = io.BytesIO()
            gdf.head(0).to_parquet(buf)
            buf.seek(0)
            self._geo = pq.read_schema(buf).metadata[b'geo']

        table = self._promote_null_columns(pa.table(gdf.to_arrow(geometry_encoding='WKB')))

        if self._writer is None:
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), b'geo': self._geo})
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema)
        else:
            # Cast to the file's schema so a chunk that inferred a column
            # slightly differently (e.g. null vs string) still matches.
            table = table.cast(self._schema)
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()


class GlobalImpactWorkflow:
    """Loads the global basin file once and streams basin chunks to a pool."""

    def __init__(self, basin_file, config, master_output_folder,
                 max_workers=4, chunk_size=100, max_tasks_per_child=100,
                 subfolder_digits=0):
        self.basin_file = basin_file
        self.config = config
        self.master_output = Path(master_output_folder)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        # Bucket per-basin output files into subfolders by the first N digits of
        # HYBAS_ID (0 = flat). At global scale a flat folder can hold hundreds
        # of thousands of files, which strains the filesystem; 4 keeps each
        # directory small.
        self.subfolder_digits = subfolder_digits
        # Recycle each pool worker after this many chunks so memory that
        # geopandas/GEOS/pyarrow don't return to the OS (C-library
        # fragmentation) is released instead of climbing until the machine runs
        # out. Lower = tighter memory ceiling but more worker restarts (each
        # restart rebuilds the per-worker tile caches).
        self.max_tasks_per_child = max_tasks_per_child

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
            gdf = gpd.read_parquet(self.basin_file, columns=[BASIN_ID, 'geometry'])

        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs('EPSG:4326')

        # PFAF_ID (Pfafstetter) is hierarchical / spatially coherent and free
        # to sort on; fall back to centroid distance from (0,0) if absent.
        if 'PFAF_ID' in gdf.columns:
            gdf = gdf.sort_values('PFAF_ID')
        else:
            c = gdf.geometry.centroid
            gdf = gdf.assign(_d=np.sqrt(c.x ** 2 + c.y ** 2)).sort_values('_d')

        return gdf[[BASIN_ID, 'geometry']].reset_index(drop=True)

    def _iter_chunks(self):
        """Yield successive [HYBAS_ID, geometry] chunks of chunk_size basins.
        A fresh generator is produced per call so every statistic can iterate
        the full basin set."""
        n = self.chunk_size
        gdf = self.basins
        for i in range(0, len(gdf), n):
            yield gdf.iloc[i:i + n]

    def _stat_done(self, csv_name):
        return (self.stats_root / csv_name).exists()

    def _args(self, source_key):
        return ((chunk, self.config[source_key]) for chunk in self._iter_chunks())

    def _write_per_basin(self, gdf, folder_name, suffix):
        """Write one small GeoParquet per basin into <Statistics>/<folder_name>/,
        named '<HYBAS_ID>_<suffix>.parquet' (with a covering bbox so it loads
        cleanly in QGIS). Rows with null geometry are dropped, and a basin whose
        file already exists is skipped - so an interrupted run resumes without
        rewriting the files it already produced.

        Each write is wrapped in call_with_io_retry so a transient filesystem
        resource error (e.g. Windows error 1450 when an external drive is
        swamped by many small writes) retries after a short pause instead of
        killing the run. If subfolder_digits > 0, files are bucketed into
        subfolders by the first N digits of HYBAS_ID so no single directory
        holds hundreds of thousands of files."""
        if gdf is None or len(gdf) == 0:
            return
        gdf = gdf[gdf.geometry.notna()]
        if len(gdf) == 0:
            return
        out_dir = self.stats_root / folder_name
        out_dir.mkdir(exist_ok=True)
        for hid, idx in gdf.groupby(BASIN_ID).groups.items():
            if self.subfolder_digits > 0:
                dst_dir = out_dir / str(hid)[:self.subfolder_digits]
                dst_dir.mkdir(exist_ok=True)
            else:
                dst_dir = out_dir
            dst = dst_dir / f"{hid}_{suffix}.parquet"
            if dst.exists():
                continue
            sub = gdf.loc[idx]
            call_with_io_retry(
                lambda s=sub, d=dst: s.to_parquet(d, write_covering_bbox=True), ())

    # ----- the statistics (checkpointed per chunk) -----
    # All statistics write a FOLDER of per-basin GeoParquet files
    # (<HYBAS_ID>_<type>.parquet) plus one combined summary CSV. Each chunk is
    # checkpointed with a small CSV fragment under Statistics/_partial/<stat>/;
    # a chunk whose fragment exists is skipped BEFORE it is sent to the pool, so
    # a resumed run never recomputes finished chunks. Fragments are stitched
    # into the combined CSV at the end.

    def _run_checkpointed(self, pool, keyed_wrapper, source_key, stat_name, desc, process_fn):
        """Compute one statistic, skipping chunks already checkpointed, and
        checkpoint each finished chunk. process_fn(res) writes that chunk's
        per-basin geometry files and returns a small per-basin summary
        DataFrame (the fragment). Returns the partial dir for the combine step."""
        partial_dir = self.stats_root / "_partial" / stat_name
        partial_dir.mkdir(parents=True, exist_ok=True)

        def frag(i):
            return partial_dir / f"chunk_{i:06d}.csv"

        n = self.chunk_size
        pending_count = sum(1 for i in range(self.n_chunks) if not frag(i).exists())
        done = self.n_chunks - pending_count
        if pending_count == 0:
            print(f"  {stat_name}: all {self.n_chunks} chunks already checkpointed")
            return partial_dir
        print(f"  {stat_name}: {done} chunks already done, {pending_count} to compute")

        def pending_args():
            src = self.config[source_key]
            for i in range(self.n_chunks):
                if frag(i).exists():
                    continue
                yield (i, self.basins.iloc[i * n:(i + 1) * n], src)

        for key, res in tqdm.tqdm(pool.imap_unordered(keyed_wrapper, pending_args()),
                                  total=pending_count, desc=desc,
                                  mininterval=0.5, dynamic_ncols=True):
            frag_df = process_fn(res)
            # Fragment written LAST (and via retry) so its existence guarantees
            # the chunk's geometry files are already on disk.
            call_with_io_retry(
                lambda df=frag_df, p=frag(key): df.to_csv(p, index=False), ())
        return partial_dir

    def _combine_fragments(self, partial_dir, final_csv, value_cols, min_count):
        """Concatenate every chunk fragment into the combined CSV with a TOTAL
        row. min_count=True keeps an all-NaN column NaN (area/length/pop);
        min_count=False uses a plain sum (counts)."""
        frames = []
        for f in sorted(partial_dir.glob("chunk_*.csv")):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if len(d):
                frames.append(d)
        if frames:
            per_basin = pd.concat(frames, ignore_index=True)
        else:
            per_basin = pd.DataFrame({BASIN_ID: [], **{c: [] for c in value_cols}})
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
            if len(real):
                self._write_per_basin(
                    gpd.GeoDataFrame(real, geometry=gpd.points_from_xy(real['x'], real['y']),
                                     crs='EPSG:4326'),
                    "population_by_basin", "population")
            return (pd.DataFrame(res).groupby(BASIN_ID, dropna=False, as_index=False)
                    ['pop_value'].sum(min_count=1))

        partial = self._run_checkpointed(pool, _keyed_population, 'population_source',
                                         'population', "Population    ", process)
        self._combine_fragments(partial, "population_statistics.csv", ['pop_value'], min_count=True)

    def _run_farmland(self, pool):
        if self._stat_done("farmland_statistics.csv"):
            print("  farmland already done, skipping")
            return

        def process(res):
            self._write_per_basin(res, "farmland_by_basin", "farmland")
            return (pd.DataFrame(res.drop(columns='geometry', errors='ignore'))
                    .groupby(BASIN_ID, dropna=False, as_index=False)['area_m2'].sum(min_count=1))

        partial = self._run_checkpointed(pool, _keyed_farmland, 'farmland_source',
                                         'farmland', "Farmland      ", process)
        self._combine_fragments(partial, "farmland_statistics.csv", ['area_m2'], min_count=True)

    def _run_buildings(self, pool):
        if self._stat_done("building_statistics.csv"):
            print("  buildings already done, skipping")
            return

        def process(res):
            real = res[res['building'].notna()]   # actual buildings (not empty-basin placeholders)
            if len(real):
                self._write_per_basin(
                    gpd.GeoDataFrame(real, geometry=gpd.points_from_xy(real['x'], real['y']),
                                     crs='EPSG:4326'),
                    "building_by_basin", "buildings")
                return real.groupby(BASIN_ID).size().reset_index(name='building_count')
            return pd.DataFrame({BASIN_ID: [], 'building_count': []})

        partial = self._run_checkpointed(pool, _keyed_building, 'building_source',
                                         'building', "Buildings     ", process)
        self._combine_fragments(partial, "building_statistics.csv", ['building_count'], min_count=False)

    def _run_transportation(self, pool):
        if self._stat_done("transportation_statistics.csv"):
            print("  transportation already done, skipping")
            return

        def process(res):
            self._write_per_basin(res, "transportation_by_basin", "roads")
            df = pd.DataFrame(res.drop(columns='geometry', errors='ignore'))
            if len(df):
                piv = (df.groupby([BASIN_ID, 'infrastructure_type'])['length_km']
                       .sum(min_count=1).unstack('infrastructure_type')
                       .rename(columns={'highway': 'highway_km', 'railway': 'railway_km'}))
                for c in ('highway_km', 'railway_km'):
                    if c not in piv.columns:
                        piv[c] = None
                return piv.reset_index()[[BASIN_ID, 'highway_km', 'railway_km']]
            return pd.DataFrame({BASIN_ID: [], 'highway_km': [], 'railway_km': []})

        partial = self._run_checkpointed(pool, _keyed_transportation, 'transportation_source',
                                         'transportation', "Transportation", process)
        self._combine_fragments(partial, "transportation_statistics.csv",
                                ['highway_km', 'railway_km'], min_count=True)

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
        print("GLOBAL IMPACT ANALYSIS WORKFLOW - HydroBASINS (in-memory, streamed)")
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
        with mp.Pool(processes=self.max_workers,
                     maxtasksperchild=self.max_tasks_per_child) as pool:
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
    basin_file = "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/Catchment/HydroBASIN/HUC12.parquet"

    # Folders of tiled inputs. Each impact module bbox-filters per chunk.
    config = {
        'population_source':     "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/Population/population.parquet",
        'farmland_source':       "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/ESA/bbox",
        'building_source':       "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/OSM/Parquet_sorted/Polygons",
        'transportation_source': "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Files/OSM/Parquet_sorted/Lines",
    }

    master_output_folder = "/Volumes/LAB_4TB/Brian/Flood_Impact_Model/Global_HUC12_Impact_Results"

    workflow = GlobalImpactWorkflow(
        basin_file=basin_file,
        config=config,
        master_output_folder=master_output_folder,
        max_workers=mp.cpu_count(),  # tune to your RAM (~3-4 GB per worker on dense chunks)
        chunk_size=200,  # smaller chunks: tighter per-chunk bbox + shorter overlay tail on dense regions
        max_tasks_per_child=100,  # recycle workers every N chunks so memory doesn't climb to full
        subfolder_digits=4  # bucket per-basin files into subfolders (fewer files per directory)
    )
    workflow.run_all_analyses()
