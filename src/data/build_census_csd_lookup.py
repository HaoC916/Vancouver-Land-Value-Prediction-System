"""Build the small census lookup the AI assistant serves at runtime.

Reads data/processed/census_profile.csv (2021 census, Greater Vancouver +
Greater Toronto) and keeps only the CSD-level rows (municipalities) with the
clean numeric columns — the DA-level rows and the raw profile_json are dropped.
Output is a ~65-row JSON file small enough to ship with the API image.

Usage:
    python -m src.data.build_census_csd_lookup
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

SOURCE = Path("data/processed/census_profile.csv")
OUTPUT = Path("data/deploy/census_csd.json")

NUMERIC_COLUMNS = [
    "population_total",
    "population_density",
    "land_area",
    "average_age",
    "household_average_size",
    "median_household_income",
    "average_household_income",
    "low_income_pct",
    "not_in_labour_force_pct",
    "average_home_value",
    "condo_pct",
    "owner_pct",
    "renter_pct",
    "immigrant_pct",
    "postsecondary_15_plus_pct",
    "not_married_pct",
    "households_with_children_pct",
    "commute_public_transit_pct",
]


def split_geo_name(geo_name: str) -> tuple[str, str]:
    """'Langley, District municipality (DM)' -> ('Langley', 'District municipality')."""
    name = geo_name.split(",")[0].strip()
    kind_match = re.search(r",\s*([^,(]+?)\s*\(", geo_name)
    kind = kind_match.group(1).strip() if kind_match else ""
    return name, kind


def main() -> None:
    df = pd.read_csv(SOURCE, usecols=lambda c: c != "profile_json", low_memory=False)
    csd = df[df["geo_level"] == "CSD"].copy()

    records = []
    for _, row in csd.iterrows():
        name, kind = split_geo_name(str(row["geo_name"]))
        record = {
            "name": name,
            "kind": kind,
            "full_name": row["geo_name"],
            "region": "Greater Vancouver"
            if row["source_region"] == "greater_vancouver"
            else "Greater Toronto Area",
            "census_year": int(row["census_year"]),
        }
        for col in NUMERIC_COLUMNS:
            value = pd.to_numeric(row.get(col), errors="coerce")
            record[col] = None if pd.isna(value) else float(value)
        records.append(record)

    records.sort(key=lambda r: (r["region"], r["name"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=1))
    print(f"Wrote {len(records)} CSD profiles -> {OUTPUT}")


if __name__ == "__main__":
    main()
