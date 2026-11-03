# Data_Prep/AOI

`prepare_aoi.py` turns any polygon set into an AOI file for `Impact_Intersection/run_impact.sh`:
id + geometry only, integer ids, lon/lat, repaired geometry, Hilbert-sorted, with a covering
bbox. See `Impact_Intersection/README.md` for usage.
