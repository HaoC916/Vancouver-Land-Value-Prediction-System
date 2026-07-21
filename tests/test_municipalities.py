from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.infer.municipalities import MunicipalityProfiles


def _write_profile(path: Path) -> None:
    pd.DataFrame([
        {"region_id": "10", "municipality_name": "Burnaby", "name_cn": "本拿比",
         "property_type": "APTU", "median_price": 660_000, "price_basis": "sold",
         "sold_12m": 1800, "median_dom": 40, "price_change_yoy_pct": -5.0,
         "latest_month": "2026-06"},
        {"region_id": "19", "municipality_name": "Chilliwack", "name_cn": "奇利瓦克",
         "property_type": "APTU", "median_price": 380_000, "price_basis": "sold",
         "sold_12m": 300, "median_dom": 50, "price_change_yoy_pct": 3.5,
         "latest_month": "2026-06"},
        {"region_id": "2", "municipality_name": "Abbotsford", "name_cn": "阿伯茨福德",
         "property_type": "APTU", "median_price": 390_000, "price_basis": "sold",
         "sold_12m": 550, "median_dom": 47, "price_change_yoy_pct": -10.0,
         "latest_month": "2026-06"},
    ]).to_parquet(path, index=False)


def test_recommend_municipalities_by_budget_and_appreciation(tmp_path: Path) -> None:
    path = tmp_path / "municipalities.parquet"
    _write_profile(path)
    profiles = MunicipalityProfiles(path)

    result = profiles.recommend("condo", max_price=500_000, sort_by="appreciation")

    assert result["status"] == "ok"
    assert [row["municipality"] for row in result["municipalities"]] == [
        "Chilliwack", "Abbotsford"
    ]


def test_recommend_municipalities_by_liquidity(tmp_path: Path) -> None:
    path = tmp_path / "municipalities.parquet"
    _write_profile(path)
    result = MunicipalityProfiles(path).recommend("condo", sort_by="liquidity")
    assert result["municipalities"][0]["municipality"] == "Burnaby"
