"""Municipality-level market recommendation lookup."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROFILE_PATH = Path("data/deploy/municipality_profile.parquet")

TYPE_MAP = {
    "house": "HOUSE", "detached": "HOUSE", "single family": "HOUSE",
    "condo": "APTU", "apartment": "APTU", "apt": "APTU", "aptu": "APTU",
    "townhouse": "TWIN", "townhome": "TWIN", "attached": "TWIN", "twin": "TWIN",
    "other": "OTHER",
}
TYPE_LABEL = {"HOUSE": "house", "APTU": "condo", "TWIN": "townhouse", "OTHER": "other"}
MIN_RELIABLE_SALES = 24


class MunicipalityProfiles:
    def __init__(self, path: Path = PROFILE_PATH) -> None:
        self.df = pd.read_parquet(path) if path.exists() else None

    def recommend(
        self,
        property_type: str,
        max_price: float | None = None,
        min_price: float | None = None,
        sort_by: str = "price",
        limit: int = 8,
    ) -> dict[str, Any]:
        if self.df is None:
            return {"status": "unavailable"}
        ptype = TYPE_MAP.get(str(property_type).strip().lower(), str(property_type).strip().upper())
        if ptype not in TYPE_LABEL:
            return {"status": "no_type", "note": "Use house, condo, townhouse, or other."}
        data = self.df[self.df["property_type"].eq(ptype)].copy()
        if max_price is not None:
            data = data[data["median_price"] <= float(max_price)]
        if min_price is not None:
            data = data[data["median_price"] >= float(min_price)]
        if data.empty:
            return {"status": "none_in_budget", "note": "No municipalities matched that budget."}

        reliable = data[data["sold_12m"] >= MIN_RELIABLE_SALES]
        used_thin_markets = len(reliable) < 3
        if not used_thin_markets:
            data = reliable

        key = str(sort_by).strip().lower()
        if key.startswith("app") or key.startswith("growth"):
            data = data[data["price_change_yoy_pct"].notna()].sort_values(
                ["price_change_yoy_pct", "sold_12m"], ascending=[False, False]
            )
            mode = "appreciation"
            note = "Ranked by observed year-over-year price change; this is historical, not a forecast."
        elif key.startswith("liq") or key.startswith("sales"):
            data = data.sort_values(["sold_12m", "median_dom"], ascending=[False, True])
            mode = "liquidity"
            note = "Ranked by recent sales volume, with days on market as a secondary signal."
        else:
            data = data.sort_values(["median_price", "sold_12m"], ascending=[True, False])
            mode = "price"
            note = "Ranked by typical recent municipality-level price; aggregate market data only."

        def optional(value: object, digits: int = 1):
            return None if pd.isna(value) else round(float(value), digits)

        cities = [{
            "municipality": row["municipality_name"],
            "municipality_name_cn": None if pd.isna(row["name_cn"]) else str(row["name_cn"]),
            "typical_price_cad": int(row["median_price"]),
            "price_basis": row["price_basis"],
            "sales_last_12m": int(row["sold_12m"]),
            "median_days_on_market": None if pd.isna(row["median_dom"]) else int(row["median_dom"]),
            "price_change_pct_year_over_year": optional(row["price_change_yoy_pct"]),
            "latest_month": row["latest_month"],
        } for _, row in data.head(max(1, min(int(limit), 12))).iterrows()]
        if used_thin_markets:
            note += " Some results have fewer than 24 sales in the last 12 months; treat them as thin markets."
        return {"status": "ok", "property_type": TYPE_LABEL[ptype], "sorted_by": mode,
                "count": len(cities), "municipalities": cities, "note": note}
