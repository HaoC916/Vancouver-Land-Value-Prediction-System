"""Build compact Municipality-level trend and recommendation lookups."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data._community import COMMUNITY_TREND_PATH, REGION_PATH, load_municipality_trend

TYPES = {"HOUSE", "APTU", "TWIN", "OTHER"}
TREND_OUTPUT = Path("data/deploy/municipality_trend.parquet")
PROFILE_OUTPUT = Path("data/deploy/municipality_profile.parquet")

TREND_KEEP = [
    "region_id", "municipality_name", "name_cn", "period_start", "property_type",
    "new_listing_count", "sold_count", "active_count", "median_list_price",
    "median_sold_price", "avg_sold_price", "median_dom", "absorption_rate",
]
NUMERIC = [
    "new_listing_count", "sold_count", "active_count", "median_list_price",
    "median_sold_price", "avg_sold_price", "median_dom", "absorption_rate",
]


def _weighted_price(rows: pd.DataFrame) -> float | None:
    priced = rows.dropna(subset=["avg_sold_price"])
    priced = priced[priced["sold_count"] > 0]
    weight = priced["sold_count"].sum()
    if not weight:
        return None
    return float((priced["avg_sold_price"] * priced["sold_count"]).sum() / weight)


def _yoy_change(group: pd.DataFrame) -> float | None:
    monthly = group.sort_values("period_start")
    latest_month = monthly["period_start"].max()
    recent_start = latest_month - pd.DateOffset(months=2)
    prior_end = latest_month - pd.DateOffset(months=12)
    prior_start = prior_end - pd.DateOffset(months=2)
    latest = _weighted_price(monthly[monthly["period_start"].between(recent_start, latest_month)])
    prior = _weighted_price(monthly[monthly["period_start"].between(prior_start, prior_end)])
    return round((latest / prior - 1.0) * 100.0, 1) if latest and prior else None


def build(
    source: Path = COMMUNITY_TREND_PATH,
    region_path: Path = REGION_PATH,
    trend_output: Path = TREND_OUTPUT,
    profile_output: Path = PROFILE_OUTPUT,
    months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trend = load_municipality_trend(source, region_path)
    trend = trend[trend["property_type"].isin(TYPES)][TREND_KEEP].copy()
    trend["period_start"] = pd.to_datetime(trend["period_start"], errors="coerce")
    for col in NUMERIC:
        trend[col] = pd.to_numeric(trend[col], errors="coerce")
    trend = trend.dropna(subset=["period_start", "municipality_name"])

    latest_month = trend["period_start"].max()
    cutoff = latest_month - pd.DateOffset(months=months)
    recent = trend[trend["period_start"] > cutoff]
    rows = []
    for (rid, name, name_cn, ptype), group in recent.groupby(
        ["region_id", "municipality_name", "name_cn", "property_type"], dropna=False
    ):
        sold = group[(group["sold_count"] > 0) & group["median_sold_price"].notna()]
        if len(sold) >= 3:
            typical_price, basis = float(sold["median_sold_price"].median()), "sold"
        else:
            listed = group[group["median_list_price"].notna()]
            if listed.empty:
                continue
            typical_price, basis = float(listed["median_list_price"].median()), "list"
        all_history = trend[(trend["region_id"] == rid) & (trend["property_type"] == ptype)]
        rows.append({
            "region_id": str(rid), "municipality_name": name, "name_cn": name_cn,
            "property_type": ptype, "median_price": round(typical_price),
            "price_basis": basis, "sold_12m": int(group["sold_count"].fillna(0).sum()),
            "median_dom": round(float(group["median_dom"].median()))
            if group["median_dom"].notna().any() else None,
            "price_change_yoy_pct": _yoy_change(all_history),
            "latest_month": latest_month.strftime("%Y-%m"),
        })
    profile = pd.DataFrame(rows).sort_values(["property_type", "median_price"])

    trend["period_start"] = trend["period_start"].dt.strftime("%Y-%m")
    trend_output.parent.mkdir(parents=True, exist_ok=True)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    trend.to_parquet(trend_output, index=False)
    profile.to_parquet(profile_output, index=False)
    print(
        f"[municipality] trend={len(trend):,} rows / {trend['municipality_name'].nunique()} cities; "
        f"profile={len(profile)} rows; latest={latest_month:%Y-%m}"
    )
    return trend, profile


def main() -> None:
    p = argparse.ArgumentParser(description="Build Municipality trend and recommendation lookups.")
    p.add_argument("--source", default=str(COMMUNITY_TREND_PATH))
    p.add_argument("--region_path", default=str(REGION_PATH))
    p.add_argument("--trend_output", default=str(TREND_OUTPUT))
    p.add_argument("--profile_output", default=str(PROFILE_OUTPUT))
    p.add_argument("--months", type=int, default=12)
    a = p.parse_args()
    build(Path(a.source), Path(a.region_path), Path(a.trend_output), Path(a.profile_output), a.months)


if __name__ == "__main__":
    main()
