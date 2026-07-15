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
SCHOOL_PATH = Path("data/deploy/subarea_school.parquet")

TYPE_MAP = {
    "house": "HOUSE", "detached": "HOUSE", "single family": "HOUSE",
    "condo": "APTU", "apartment": "APTU", "apt": "APTU", "aptu": "APTU",
    "townhouse": "TWIN", "townhome": "TWIN", "attached": "TWIN", "twin": "TWIN",
}
TYPE_LABEL = {"HOUSE": "house", "APTU": "condo", "TWIN": "townhouse"}
MIN_RELIABLE_SALES = 6  # a year's sales below this → price is thin, flag it


class NeighbourhoodProfiles:
    def __init__(self, path: Path = PROFILE_PATH, school_path: Path = SCHOOL_PATH) -> None:
        if not path.exists():
            self.df = None
            return
        df = pd.read_parquet(path)
        if school_path.exists():
            df = df.merge(pd.read_parquet(school_path), on="region_id", how="left")
        for c in ("best_school_score", "avg_school_score", "school_count"):
            if c not in df.columns:
                df[c] = None
        self.df = df

    def recommend(self, city: str, property_type: str | None = None,
                  max_price: float | None = None, min_price: float | None = None,
                  sort_by: str = "price", limit: int = 8) -> dict[str, Any]:
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

        by_school = str(sort_by).lower().startswith("school")
        note = ("Typical recent price per neighbourhood (median). 'sales_last_12m' shows how "
                "reliable the price is; a low number means thin data.")
        if by_school:
            withschool = d[d["best_school_score"].notna()]
            if not withschool.empty:
                d = withschool.sort_values("best_school_score", ascending=False)
                note = ("Ranked by best nearby school (Fraser Institute score, 0-10). School "
                        "data covers most urban neighbourhoods but not all; price shown too.")
            else:
                d = d.sort_values("median_price")
                note = "No school scores for these neighbourhoods yet — showing by price instead."
        else:
            d = d.sort_values("median_price")

        def s(v):
            return None if pd.isna(v) else float(v)
        out = [{
            "neighbourhood": r["subarea_name"], "area": r["area_name"],
            "typical_price_cad": int(r["median_price"]),
            "best_school_score": s(r["best_school_score"]),
            "avg_school_score": s(r["avg_school_score"]),
            "median_days_on_market": None if pd.isna(r["median_dom"]) else int(r["median_dom"]),
            "sales_last_12m": int(r["sold_12m"]),
        } for _, r in d.head(limit).iterrows()]
        return {"status": "ok", "city": city,
                "property_type": TYPE_LABEL.get(ptype, property_type),
                "sorted_by": "schools" if by_school else "price",
                "count": len(out), "neighbourhoods": out, "note": note}
