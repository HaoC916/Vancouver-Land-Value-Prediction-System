"""Build the compact market-trend table the AI assistant queries at runtime.

Reads the refreshed monthly community market trends, joins names by region_id,
and writes a slim parquet with
just the columns the assistant's tools need. Rows with property_type MISSING
are dropped; NULL strings become real nulls.

Usage:
    python -m src.data.build_market_trend_deploy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data._community import COMMUNITY_LOOKUP_PATH, COMMUNITY_TREND_PATH, load_subarea_trend

SOURCE = COMMUNITY_TREND_PATH
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


def build(source: Path = SOURCE, output: Path = OUTPUT,
          lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> pd.DataFrame:
    df = load_subarea_trend(source, lookup_path)
    df = df[KEEP].copy()
    df = df[df["property_type"] != "MISSING"].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["period_start"] = pd.to_datetime(df["period_start"]).dt.strftime("%Y-%m")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    size_mb = output.stat().st_size / 1e6
    print(
        f"Wrote {len(df):,} rows, {df['area_name'].nunique()} areas, "
        f"{df['period_start'].min()}..{df['period_start'].max()} -> {output} ({size_mb:.1f} MB)"
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Build the compact Subarea market-trend lookup.")
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--lookup_path", default=str(COMMUNITY_LOOKUP_PATH))
    p.add_argument("--output", default=str(OUTPUT))
    a = p.parse_args()
    build(Path(a.source), Path(a.output), Path(a.lookup_path))


if __name__ == "__main__":
    main()
