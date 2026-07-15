"""Build the compact market-trend table the AI assistant queries at runtime.

Reads data/processed/community_market_trend_enriched.csv (monthly community
market trends with subarea/area names joined in) and writes a slim parquet with
just the columns the assistant's tools need. Rows with property_type MISSING
are dropped; NULL strings become real nulls.

Usage:
    python -m src.data.build_market_trend_deploy
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data._scope import GREATER_VANCOUVER_AREAS

SOURCE = Path("data/processed/community_market_trend_enriched.csv")
OUTPUT = Path("data/deploy/market_trend.parquet")

KEEP = [
    "period_start",
    "area_name",
    "subarea_name",
    "property_type",
    "new_listing_count",
    "sold_count",
    "active_count",
    "median_list_price",
    "median_sold_price",
    "avg_sold_price",
    "median_dom",
    "absorption_rate",
]

NUMERIC = [
    "new_listing_count",
    "sold_count",
    "active_count",
    "median_list_price",
    "median_sold_price",
    "avg_sold_price",
    "median_dom",
    "absorption_rate",
]


def main() -> None:
    df = pd.read_csv(SOURCE, usecols=KEEP, na_values=["NULL"], low_memory=False)
    df = df[df["property_type"] != "MISSING"].copy()
    df = df[df["area_name"].isin(GREATER_VANCOUVER_AREAS)].copy()  # Greater Van + Fraser Valley only
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["period_start"] = pd.to_datetime(df["period_start"]).dt.strftime("%Y-%m")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(
        f"Wrote {len(df):,} rows, {df['area_name'].nunique()} areas, "
        f"{df['period_start'].min()}..{df['period_start'].max()} -> {OUTPUT} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
