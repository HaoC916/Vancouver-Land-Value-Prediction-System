from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd

from src.data._community import decode_point, load_subarea_trend


def test_decode_point_supports_refreshed_ewkt_and_legacy_ewkb() -> None:
    assert decode_point("SRID=4326;POINT(-122.9 49.2)") == (-122.9, 49.2)

    ewkb = (
        b"\x01"
        + struct.pack("<I", 0x20000001)
        + struct.pack("<I", 4326)
        + struct.pack("<dd", -123.1, 49.25)
    ).hex()
    assert decode_point(ewkb) == (-123.1, 49.25)


def test_load_subarea_trend_joins_names_and_excludes_municipality(tmp_path: Path) -> None:
    trend_path = tmp_path / "trend.csv"
    lookup_path = tmp_path / "lookup.csv"
    pd.DataFrame([
        {"geo_level": "Subarea", "region_id": 1, "period_start": "2026-06-01",
         "property_type": "APTU", "median_list_price": 700_000},
        {"geo_level": "Municipality", "region_id": 10, "period_start": "2026-06-01",
         "property_type": "APTU", "median_list_price": 710_000},
    ]).to_csv(trend_path, index=False)
    pd.DataFrame([
        {"region_id": 1, "subarea_name": "Metrotown", "region_type": "Subarea",
         "area_id": 195, "area_name": "Burnaby South", "area_region_type": "Area"},
    ]).to_csv(lookup_path, index=False)

    result = load_subarea_trend(trend_path, lookup_path, scoped=False)

    assert len(result) == 1
    assert result.iloc[0]["region_id"] == "1"
    assert result.iloc[0]["subarea_name"] == "Metrotown"
    assert result.iloc[0]["area_name"] == "Burnaby South"
