"""
aoi_prep.py - download and prepare the built-in polygon AOIs. run_impact.sh
calls this automatically when an AOI isn't prepared yet; you can also run it
yourself (e.g. to prepare ahead of time):

    python aoi_prep.py geoglows            # GEOGLOWS v2 catchments, all VPUs
    python aoi_prep.py hydrobasins 8       # HydroBASINS level 8 (1-12)

Output (one prepared file per VPU / region, see prepare_aoi.py):
    <work-dir>/AOI/geoglows_catchments/vpu_<NNN>.parquet
    <work-dir>/AOI/hydrobasins_lev<NN>/<region>.parquet
plus _complete.txt once every part is done; the intersection run only uses
complete folders.

Each download is deleted as soon as its part is prepared (--keep-download
keeps them). Note a HydroBASINS region zip holds all 12 levels, so preparing
a second level later downloads it again. Safe to rerun: finished parts are
skipped.
"""
import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_aoi import prepare                    # noqa: E402

# Name sent with every download (servers can refuse generic client names).
USER_AGENT = "FloodImpactModel/1.0"


def _request(url):
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

GEOGLOWS_BUCKET = "https://geoglows-v2.s3.us-west-2.amazonaws.com"
GEOGLOWS_PREFIX = "hydrography/"
HYDROBASINS_URL = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_{region}_lev01-12_v1c.zip"
HYDROBASINS_REGIONS = ["af", "ar", "as", "au", "eu", "gr", "na", "sa", "si"]
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

LEVELS = {"geoglows": None, "hydrobasins": range(1, 13)}


def aoi_folder(work_dir, name, level=None):
    if name == "geoglows":
        return Path(work_dir) / "AOI" / "geoglows_catchments"
    if name == "hydrobasins":
        return Path(work_dir) / "AOI" / f"hydrobasins_lev{level:02d}"
    raise ValueError(f"no automatic download for '{name}'")


def is_complete(folder):
    return (Path(folder) / "_complete.txt").exists()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, dest, tries=4):
    """Download url to dest (via dest.partial), checking the byte count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".partial")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as r, open(part, "wb") as f:
                expected = int(r.headers.get("Content-Length", -1))
                while chunk := r.read(8 << 20):
                    f.write(chunk)
            if expected >= 0 and part.stat().st_size != expected:
                raise IOError(f"got {part.stat().st_size} of {expected} bytes")
            part.replace(dest)
            return dest
        except Exception as e:
            if attempt == tries:
                raise RuntimeError(f"download failed: {url}: {e}") from e
            log(f"  download attempt {attempt} failed ({e}); retrying")
            time.sleep(10 * attempt)


def s3_folders(bucket, prefix):
    """Sub-folder names directly under prefix in a public S3 bucket."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(_request(f"{bucket}/?{urllib.parse.urlencode(q)}"), timeout=60) as r:
            root = ET.fromstring(r.read())
        out += [p.text for p in root.iter(f"{S3_NS}Prefix") if p.text and p.text != prefix]
        if root.findtext(f"{S3_NS}IsTruncated") != "true":
            return out
        token = root.findtext(f"{S3_NS}NextContinuationToken")


# ---------------------------------------------------------------------------
# GEOGLOWS catchments
# ---------------------------------------------------------------------------
def prep_geoglows(work_dir, keep=False, source_dir=None):
    folder = aoi_folder(work_dir, "geoglows")
    dl_dir = Path(work_dir) / "AOI" / "_download"
    if source_dir:      # catchment files already on disk (vpu=NNN/catchments_NNN.parquet)
        vpus = sorted(p.parent.name.split("=", 1)[1]
                      for p in Path(source_dir).glob("vpu=*/catchments_*.parquet"))
    else:
        vpus = sorted(p.rstrip("/").split("=", 1)[1]
                      for p in s3_folders(GEOGLOWS_BUCKET, GEOGLOWS_PREFIX) if "vpu=" in p)
    if not vpus:
        raise RuntimeError("no GEOGLOWS VPUs found")
    log(f"GEOGLOWS catchments: {len(vpus)} VPUs -> {folder}")
    total = 0
    for i, vpu in enumerate(vpus, 1):
        out = folder / f"vpu_{vpu}.parquet"
        if out.exists():
            continue
        if source_dir:
            src, downloaded = Path(source_dir) / f"vpu={vpu}" / f"catchments_{vpu}.parquet", False
        else:
            src = download(f"{GEOGLOWS_BUCKET}/{GEOGLOWS_PREFIX}vpu={vpu}/catchments_{vpu}.parquet",
                           dl_dir / f"catchments_{vpu}.parquet")
            downloaded = True
        n = prepare([src], "LINKNO", out, quiet=True)
        total += n
        log(f"  {i}/{len(vpus)} VPU {vpu}: {n:,} catchments")
        if downloaded and not keep:
            src.unlink()
    (folder / "_complete.txt").write_text(
        f"GEOGLOWS v2 catchments, {len(vpus)} VPUs, id LINKNO\nsource: {source_dir or GEOGLOWS_BUCKET}\n"
        f"prepared: {datetime.now():%Y-%m-%d %H:%M}\n")
    log("GEOGLOWS catchments ready")
    return folder


# ---------------------------------------------------------------------------
# HydroBASINS
# ---------------------------------------------------------------------------
def prep_hydrobasins(work_dir, level, keep=False, source_dir=None):
    if level not in LEVELS["hydrobasins"]:
        raise ValueError(f"HydroBASINS level must be 1-12, got {level}")
    folder = aoi_folder(work_dir, "hydrobasins", level)
    dl_dir = Path(work_dir) / "AOI" / "_download"
    log(f"HydroBASINS level {level}: {len(HYDROBASINS_REGIONS)} regions -> {folder}")
    for i, region in enumerate(HYDROBASINS_REGIONS, 1):
        out = folder / f"{region}.parquet"
        if out.exists():
            continue
        name = Path(HYDROBASINS_URL.format(region=region)).name
        local = Path(source_dir) / name if source_dir else None
        if local is not None and local.exists():
            zpath, downloaded = local, False
        else:
            zpath = dl_dir / name
            downloaded = not zpath.exists()
            if downloaded:
                log(f"  downloading {name}")
                download(HYDROBASINS_URL.format(region=region), zpath)
        want = f"_lev{level:02d}_v1c.shp"
        with zipfile.ZipFile(zpath) as z:
            member = next((m for m in z.namelist() if m.lower().endswith(want)), None)
        if member is None:
            raise RuntimeError(f"{zpath.name} has no level-{level} shapefile (*{want})")
        n = prepare([f"/vsizip/{zpath}/{member}"], "HYBAS_ID", out, quiet=True)
        log(f"  {i}/{len(HYDROBASINS_REGIONS)} {region}: {n:,} basins")
        if zpath.parent == dl_dir and not keep:
            zpath.unlink()
    (folder / "_complete.txt").write_text(
        f"HydroBASINS v1c standard, level {level}, id HYBAS_ID\nsource: {source_dir or HYDROBASINS_URL}\n"
        f"prepared: {datetime.now():%Y-%m-%d %H:%M}\n")
    log(f"HydroBASINS level {level} ready")
    return folder


def prepare_aoi(name, level, work_dir, keep=False, source_dir=None):
    if name == "geoglows":
        return prep_geoglows(work_dir, keep, source_dir)
    if name == "hydrobasins":
        return prep_hydrobasins(work_dir, level, keep, source_dir)
    raise ValueError(f"no automatic download for '{name}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=sorted(LEVELS))
    ap.add_argument("level", type=int, nargs="?", help="HydroBASINS level 1-12 (default 12)")
    ap.add_argument("--work-dir", default=str(Path.home() / "Flood_Impact_Model" / "Files"))
    ap.add_argument("--keep-download", action="store_true", help="keep downloaded files")
    ap.add_argument("--source-dir", help="use files already on disk instead of downloading "
                                         "(GEOGLOWS: folder of vpu=NNN/ folders; HydroBASINS: "
                                         "folder of hybas_<region>_lev01-12_v1c.zip); never deleted")
    args = ap.parse_args()
    level = args.level
    if args.name == "hydrobasins" and level is None:
        level = 12
    if args.name == "geoglows" and level is not None:
        print("GEOGLOWS catchments have a single level; leave the number off.")
        return 1
    prepare_aoi(args.name, level, args.work_dir, args.keep_download, args.source_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
