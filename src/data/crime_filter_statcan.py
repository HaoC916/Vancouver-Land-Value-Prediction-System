#!/usr/bin/env python3
"""Stream-filter the StatCan 35-10-0184 bulk CSV (inside its zip) down to the
Greater Vancouver + Fraser Valley policing jurisdictions, without extracting the
full ~GB CSV.

Keeps every row whose GEO contains one of the municipality names below
(this also picks up e.g. Metro Vancouver Transit Police — see README).
Writes data/raw/crime/statcan/35100184_greater_van_subset.csv and prints the
distinct GEO values kept, for verification.

Run from the repo root:  python -m src.data.crime_filter_statcan
"""
import csv
import io
import zipfile
from pathlib import Path

STATCAN = Path("data/raw/crime/statcan")
ZIP_PATH = STATCAN / "35100184-eng.zip"
OUT_PATH = STATCAN / "35100184_greater_van_subset.csv"

NAMES = [
    # Metro Vancouver
    "Vancouver", "Burnaby", "Richmond", "Surrey", "Langley", "Coquitlam",
    "North Vancouver", "West Vancouver", "Delta", "New Westminster",
    "Port Moody", "Maple Ridge", "White Rock", "Pitt Meadows",
    # Fraser Valley
    "Abbotsford", "Chilliwack", "Mission", "Kent", "Hope", "Agassiz", "Harrison",
]

kept = 0
total = 0
geos = {}
with zipfile.ZipFile(ZIP_PATH) as zf:
    with zf.open("35100184.csv") as raw, open(OUT_PATH, "w", newline="") as out:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        writer = csv.writer(out)
        header = next(reader)
        writer.writerow(header)
        geo_idx = header.index("GEO")
        for row in reader:
            total += 1
            geo = row[geo_idx]
            if any(n in geo for n in NAMES):
                writer.writerow(row)
                kept += 1
                geos[geo] = geos.get(geo, 0) + 1
    # also extract the metadata file as-is
    zf.extract("35100184_MetaData.csv", STATCAN)

print(f"total rows scanned: {total:,}; kept: {kept:,}")
print("\ndistinct GEO values kept:")
for g in sorted(geos):
    print(f"  {geos[g]:>8,}  {g}")
