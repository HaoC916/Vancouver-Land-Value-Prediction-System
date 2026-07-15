"""Recommend real neighbourhoods that fit a budget, from the per-neighbourhood profile
lookup (recent typical price + market heat per subarea, built by
build_neighbourhood_deploy.py). Backs the agent's recommend_neighbourhoods tool so the
diagnostic-questioning flow can suggest concrete neighbourhoods when a user is unsure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROFILE_PATH = Path("data/deploy/neighbourhood_profile.parquet")

TYPE_MAP = {
    "house": "HOUSE", "detached": "HOUSE", "single family": "HOUSE",
    "condo": "APTU", "apartment": "APTU", "apt": "APTU", "aptu": "APTU",
    "townhouse": "TWIN", "townhome": "TWIN", "attached": "TWIN", "twin": "TWIN",
}
TYPE_LABEL = {"HOUSE": "house", "APTU": "condo", "TWIN": "townhouse"}
MIN_RELIABLE_SALES = 6  # a year's sales below this → price is thin, flag it


class NeighbourhoodProfiles:
    def __init__(self, path: Path = PROFILE_PATH) -> None:
        self.df = pd.read_parquet(path) if path.exists() else None

    def recommend(self, city: str, property_type: str | None = None,
                  max_price: float | None = None, min_price: float | None = None,
                  limit: int = 8) -> dict[str, Any]:
        if self.df is None:
            return {"status": "unavailable"}
        d = self.df
        needle = str(city or "").strip().lower()
        if needle:
            d = d[d["area_name"].str.lower().str.contains(needle, na=False)]
        if d.empty:
            return {"status": "no_city", "note": f"No neighbourhoods found for '{city}'."}

        ptype = TYPE_MAP.get(str(property_type or "").strip().lower())
        if ptype:
            d = d[d["property_type"] == ptype]
        if d.empty:
            return {"status": "no_type", "note": f"No {property_type} data in {city}."}

        # Prefer neighbourhoods with enough recent sales for a trustworthy price.
        reliable = d[d["sold_12m"] >= MIN_RELIABLE_SALES]
        d = reliable if len(reliable) >= 3 else d

        if max_price is not None:
            d = d[d["median_price"] <= float(max_price)]
        if min_price is not None:
            d = d[d["median_price"] >= float(min_price)]
        if d.empty:
            return {"status": "none_in_budget",
                    "note": "No neighbourhoods matched that budget — try a wider price range."}

        d = d.sort_values("median_price")
        out = [{
            "neighbourhood": r["subarea_name"], "area": r["area_name"],
            "typical_price_cad": int(r["median_price"]),
            "price_basis": r["price_basis"],
            "median_days_on_market": None if pd.isna(r["median_dom"]) else int(r["median_dom"]),
            "sales_last_12m": int(r["sold_12m"]),
        } for _, r in d.head(limit).iterrows()]
        return {
            "status": "ok",
            "city": city,
            "property_type": TYPE_LABEL.get(ptype, property_type),
            "count": len(out),
            "neighbourhoods": out,
            "note": "Typical recent price per neighbourhood (median). 'sales_last_12m' shows how "
                    "reliable it is; a low number means thin data.",
        }
