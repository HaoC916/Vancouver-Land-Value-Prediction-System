"""Distill a small per-neighbourhood profile deploy lookup for the agent.

From the refreshed per-subarea monthly market data, compute
one row per (neighbourhood, property type) with a recent typical price and market-heat
(days on market, sold volume). Powers the agent's recommend_neighbourhoods tool — so when
a user is unsure which neighbourhood, it can suggest real ones that fit a budget.

Ships a tiny parquet (~a few thousand rows) — aggregate stats only, no individual listings.

Run:
    python -m src.data.build_neighbourhood_deploy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data._community import COMMUNITY_LOOKUP_PATH, COMMUNITY_TREND_PATH, load_subarea_trend

# Detached / condo(apartment) / townhouse.
TYPES = {"HOUSE", "APTU", "TWIN"}


def build(in_path: Path, out_path: Path, months: int,
          lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> pd.DataFrame:
    df = load_subarea_trend(in_path, lookup_path)
    df = df[df["property_type"].isin(TYPES)].copy()
    df = df[df["subarea_name"].notna() & df["area_name"].notna()].copy()
    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    for c in ["median_sold_price", "median_list_price", "median_dom", "sold_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cutoff = df["period_start"].max() - pd.DateOffset(months=months)
    recent = df[df["period_start"] > cutoff]

    rows = []
    for (rid, sub, area, ptype), g in recent.groupby(
            ["region_id", "subarea_name", "area_name", "property_type"]):
        sold = g.dropna(subset=["median_sold_price"])
        sold = sold[sold["sold_count"] > 0]
        if len(sold) >= 3:
            price, basis = float(sold["median_sold_price"].median()), "sold"
        else:
            lst = g.dropna(subset=["median_list_price"])
            if lst.empty:
                continue
            price, basis = float(lst["median_list_price"].median()), "list"
        rows.append({
            "region_id": str(rid), "subarea_name": sub, "area_name": area, "property_type": ptype,
            "median_price": round(price), "price_basis": basis,
            "sold_12m": int(g["sold_count"].fillna(0).sum()),
            "median_dom": round(float(g["median_dom"].median())) if g["median_dom"].notna().any() else None,
        })
    out = pd.DataFrame(rows).sort_values(["area_name", "property_type", "median_price"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[neighbourhood_profile] {len(out)} rows ({out['subarea_name'].nunique()} neighbourhoods) -> {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the per-neighbourhood profile deploy lookup.")
    p.add_argument("--in_path", default=str(COMMUNITY_TREND_PATH))
    p.add_argument("--lookup_path", default=str(COMMUNITY_LOOKUP_PATH))
    p.add_argument("--out_path", default="data/deploy/neighbourhood_profile.parquet")
    p.add_argument("--months", type=int, default=12)
    a = p.parse_args()
    build(Path(a.in_path), Path(a.out_path), a.months, Path(a.lookup_path))


if __name__ == "__main__":
    main()
