"""Phase 1 — turn the market fact table into a leakage-safe model table.

Input:  data/interim/market_fact_gv.parquet   (from build_market_fact.py)
Output: data/interim/market_model_table.parquet

Mirrors build_model_table.py: engineer features + add leakage-safe history via
``shift(1)`` (each property's own previous listing, and each area's prior-year price
level), with a row-count guard after every join. The prediction target is the list
price; outcome/label columns (sold price, days-on-market, original ask, status) are
dropped so they cannot leak into the features.

Run:
    python -m src.data.build_market_table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "listprice"
YEAR_COL = "list_year"
AREA_COL = "region_name"

# Numeric property features (coerced to numeric).
NUM_FEATURES = [
    "p_yearbuilt", "p_totalbed", "p_bedplus", "p_bedinbase", "p_bednobase",
    "p_fullbath", "p_halfbath", "p_totalbath", "p_totalfloorarea",
    "p_grandtotalfloorarea", "p_floorareamain", "p_totalparking",
]
# Categorical property features (ordinal-encoded at train time).
CAT_FEATURES = [
    "p_propertytype", "p_dwelltype", "p_dwellclass", "p_style", "p_view",
    "p_type", "p_constructiontype",
]
# Outcome / identifier / near-target columns: dropped so they cannot leak.
LEAKAGE_DROP = [
    "listing_id", "market_key", "property_id", "addressid", "p_addressid",
    "originalprice", "soldprice", "solddate", "closingdate", "status",
    "dom", "cdom", "transactiontype", "listpriceum",
    "p_category", "p_transactiontype", "p_totalareaum",
]


def _append(summary: list[dict], section: str, metric: str, value: object) -> None:
    summary.append({"section": section, "metric": metric, "value": value})


def _postal_fsa(postal: pd.Series) -> pd.Series:
    p = (postal.fillna("Unknown").astype(str).str.upper()
         .str.replace(r"\s+", "", regex=True).str.strip())
    p = p.replace({"": "Unknown", "NAN": "Unknown", "NONE": "Unknown"})
    valid = p.str.match(r"^[A-Z]\d[A-Z]")
    return pd.Series(np.where(valid, p.str[:3], "Unknown"), index=postal.index)


def _property_history(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Each property's own previous listing price. Leakage-safe: ordered by list
    date, shift(1), so a row only ever sees strictly earlier listings."""
    d = df.sort_values(["property_id", "listdate"]).copy()
    grp = d.groupby("property_id", sort=False)
    d["prop_prev_list_price"] = grp[TARGET_COL].shift(1)
    d["prop_prev_list_year"] = grp[YEAR_COL].shift(1)
    d["prop_years_since_prev_list"] = d[YEAR_COL] - d["prop_prev_list_year"]
    d["prop_has_prev_listing"] = d["prop_prev_list_price"].notna().astype(int)
    created = ["prop_prev_list_price", "prop_years_since_prev_list", "prop_has_prev_listing"]
    return d, created


def _area_history(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prior-year price level per area. Leakage-safe: shift(1) over years."""
    base = df[[YEAR_COL, AREA_COL, TARGET_COL]].copy()
    base[AREA_COL] = base[AREA_COL].fillna("Unknown").astype(str)
    agg = (base.groupby([YEAR_COL, AREA_COL], dropna=False)[TARGET_COL]
           .agg(["median", "count"]).reset_index()
           .sort_values([AREA_COL, YEAR_COL]).reset_index(drop=True))
    g = agg.groupby(AREA_COL, group_keys=False)
    agg["area_prev_year_median_list"] = g["median"].shift(1)
    agg["area_prev_year_listing_count"] = g["count"].shift(1)
    agg["area_prev_year_growth"] = g["median"].pct_change().shift(1)
    created = ["area_prev_year_median_list", "area_prev_year_listing_count", "area_prev_year_growth"]
    return agg[[YEAR_COL, AREA_COL] + created], created


def build_market_table(fact_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    summary: list[dict] = []
    df = pd.read_parquet(fact_path).copy()
    _append(summary, "table", "fact_rows_in", len(df))

    # Sales only (rentals already mostly removed by the price floor; drop explicit leases).
    df = df[df["transactiontype"].fillna("S") != "R"].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, YEAR_COL])
    df = df[df[TARGET_COL] > 0].copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    row_count = len(df)

    # Engineered features.
    for c in NUM_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "p_yearbuilt" in df.columns:
        df["property_age"] = df[YEAR_COL] - df["p_yearbuilt"]
        df.loc[(df["property_age"] < 0) | (df["property_age"] > 250), "property_age"] = np.nan
    df["postal_fsa"] = _postal_fsa(df["postal_code"])
    # Best-available floor area (coalesce, plausibility-clipped) — the dominant price
    # driver and the denominator for the price-per-sqft model.
    area_src = [c for c in ["p_grandtotalfloorarea", "p_totalfloorarea", "p_floorareamain"] if c in df.columns]
    df["sqft_best"] = df[area_src].bfill(axis=1).iloc[:, 0] if area_src else np.nan
    df.loc[(df["sqft_best"] < 200) | (df["sqft_best"] > 20000), "sqft_best"] = np.nan
    _append(summary, "engineered", "property_age", "created")
    _append(summary, "engineered", "postal_fsa", "created")
    _append(summary, "engineered", "sqft_best", "created")

    # Leakage-safe history.
    df, prop_cols = _property_history(df)
    for c in prop_cols:
        _append(summary, "history", c, "created")

    area_hist, area_cols = _area_history(df)
    df = df.merge(area_hist, on=[YEAR_COL, AREA_COL], how="left")
    if len(df) != row_count:
        raise ValueError(f"Row count changed after area merge: {row_count} -> {len(df)} (leakage risk).")
    for c in area_cols:
        _append(summary, "history", c, "created")

    # Assemble final model table: target + split key + features only.
    feature_cols = (
        [c for c in NUM_FEATURES if c in df.columns]
        + [c for c in CAT_FEATURES if c in df.columns]
        + ["property_age", "postal_fsa", "sqft_best", AREA_COL, "list_month"]
        + prop_cols + area_cols
    )
    keep = [TARGET_COL, YEAR_COL] + feature_cols
    out = df[keep].copy()
    out = out.drop(columns=[c for c in LEAKAGE_DROP if c in out.columns], errors="ignore")

    _append(summary, "table", "total_rows", len(out))
    _append(summary, "table", "total_columns", len(out.columns))
    _append(summary, "table", "feature_count", len(out.columns) - 2)
    _append(summary, "table", "year_coverage", f"{out[YEAR_COL].min()}-{out[YEAR_COL].max()}")
    _append(summary, "coverage", "prop_prev_list_price_present",
            float(out["prop_prev_list_price"].notna().mean()))
    for c in out.columns:
        _append(summary, "missing_rate", c, float(out[c].isna().mean()))
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe market model table (Phase 1).")
    parser.add_argument("--fact_path", default="data/interim/market_fact_gv.parquet")
    parser.add_argument("--out_path", default="data/interim/market_model_table.parquet")
    parser.add_argument("--summary_path", default="reports/figures/market_model_table_summary.csv")
    args = parser.parse_args()

    out, summary = build_market_table(Path(args.fact_path))
    out_path, summary_path = Path(args.out_path), Path(args.summary_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"[market_table] Saved: {out_path}  (rows={len(out):,}, cols={len(out.columns):,})")
    print(f"[market_table] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
