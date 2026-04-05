from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COL = "CURRENT_LAND_VALUE"
YEAR_COL = "REPORT_YEAR"
NEIGH_COL = "NEIGHBOURHOOD_CODE"
POSTAL_COL = "PROPERTY_POSTAL_CODE"
PLAN_COL = "PLAN"


def _append_summary(summary_rows: list[dict[str, object]], section: str, metric: str, value: object) -> None:
    summary_rows.append({"section": section, "metric": metric, "value": value})


def _load_fact_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fact table not found: {path}")

    df = pd.read_parquet(path).copy()
    required = [TARGET_COL, YEAR_COL, NEIGH_COL, POSTAL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fact table missing required columns: {missing}")

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, TARGET_COL]).copy()
    df = df[df[TARGET_COL] > 0].copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    return df


def _build_property_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created: list[str] = []

    out["REPORT_YEAR_NUM"] = pd.to_numeric(out[YEAR_COL], errors="coerce")
    out["REPORT_YEAR_CENTERED"] = out["REPORT_YEAR_NUM"] - 2023
    created.extend(["REPORT_YEAR_NUM", "REPORT_YEAR_CENTERED"])

    if "YEAR_BUILT" in out.columns:
        out["YEAR_BUILT"] = pd.to_numeric(out["YEAR_BUILT"], errors="coerce")
        out["PROPERTY_AGE"] = out[YEAR_COL] - out["YEAR_BUILT"]
        out.loc[(out["PROPERTY_AGE"] < 0) | (out["PROPERTY_AGE"] > 250), "PROPERTY_AGE"] = np.nan
        created.append("PROPERTY_AGE")

        out["BUILDING_AGE_BIN"] = pd.cut(
            out["YEAR_BUILT"],
            bins=[-np.inf, 1949, 1979, 1999, np.inf],
            labels=["pre_1950", "1950_1979", "1980_1999", "2000_plus"],
        ).astype("object")
        out["BUILDING_AGE_BIN"] = out["BUILDING_AGE_BIN"].fillna("unknown")
        created.append("BUILDING_AGE_BIN")

    if "BIG_IMPROVEMENT_YEAR" in out.columns:
        out["BIG_IMPROVEMENT_YEAR"] = pd.to_numeric(out["BIG_IMPROVEMENT_YEAR"], errors="coerce")
        out["YEARS_SINCE_IMPROVEMENT"] = out[YEAR_COL] - out["BIG_IMPROVEMENT_YEAR"]
        out.loc[
            (out["YEARS_SINCE_IMPROVEMENT"] < 0) | (out["YEARS_SINCE_IMPROVEMENT"] > 250),
            "YEARS_SINCE_IMPROVEMENT",
        ] = np.nan
        out["HAS_BIG_IMPROVEMENT"] = np.where(out["BIG_IMPROVEMENT_YEAR"].notna(), 1, 0)
        created.extend(["YEARS_SINCE_IMPROVEMENT", "HAS_BIG_IMPROVEMENT"])

        out["IMPROVEMENT_RECENCY_BIN"] = pd.cut(
            out["YEARS_SINCE_IMPROVEMENT"],
            bins=[-np.inf, 5, 15, 30, np.inf],
            labels=["0_5y", "6_15y", "16_30y", "31y_plus"],
        ).astype("object")
        out["IMPROVEMENT_RECENCY_BIN"] = out["IMPROVEMENT_RECENCY_BIN"].fillna("unknown")
        created.append("IMPROVEMENT_RECENCY_BIN")

    postal = (
        out[POSTAL_COL]
        .fillna("Unknown")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.strip()
    )
    postal = postal.replace({"": "Unknown", "NAN": "Unknown", "NONE": "Unknown", "NAT": "Unknown"})
    out[POSTAL_COL] = postal
    valid_fsa = postal.str.match(r"^[A-Z]\d[A-Z]")
    out["POSTAL_FSA"] = np.where(valid_fsa, postal.str[:3], "Unknown")
    created.append("POSTAL_FSA")

    # Normalize PLAN so later PLAN-level history features are stable.
    if PLAN_COL in out.columns:
        out[PLAN_COL] = (
            out[PLAN_COL]
            .fillna("Unknown")
            .astype(str)
            .str.upper()
            .str.strip()
            .replace("", "Unknown")
        )

    if "LEGAL_TYPE" in out.columns and "ZONING_CLASSIFICATION" in out.columns:
        legal = out["LEGAL_TYPE"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        zoning = (
            out["ZONING_CLASSIFICATION"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        out["LEGAL_ZONING_COMBO"] = legal + "__" + zoning
        created.append("LEGAL_ZONING_COMBO")

    return out, created


def _prepare_yearly_table(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, "missing_input"

    ext = pd.read_parquet(path).copy()
    if YEAR_COL not in ext.columns:
        return None, "missing_REPORT_YEAR"

    ext[YEAR_COL] = pd.to_numeric(ext[YEAR_COL], errors="coerce")
    ext = ext.dropna(subset=[YEAR_COL]).copy()
    ext[YEAR_COL] = ext[YEAR_COL].astype(int)

    feature_cols: list[str] = []
    for c in ext.columns:
        if c == YEAR_COL:
            continue
        ext[c] = pd.to_numeric(ext[c], errors="coerce")
        if ext[c].notna().any():
            feature_cols.append(c)
    if not feature_cols:
        return None, "no_numeric_features"

    ext = ext[[YEAR_COL] + feature_cols].sort_values(YEAR_COL).drop_duplicates(YEAR_COL, keep="last")

    # Keep yearly levels plus lag/yoy deltas.
    for c in feature_cols:
        ext[f"{c}_lag1"] = ext[c].shift(1)
        ext[f"{c}_yoy_change"] = ext[c] - ext[f"{c}_lag1"]

    return ext, "merged"


def _merge_yearly_macros(
    fact_df: pd.DataFrame, source_paths: dict[str, Path]
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    out = fact_df.copy()
    merged_cols: list[str] = []
    statuses: dict[str, str] = {}

    for source_name, path in source_paths.items():
        ext, status = _prepare_yearly_table(path)
        statuses[source_name] = status
        if ext is None:
            continue
        cols = [c for c in ext.columns if c != YEAR_COL]
        out = out.merge(ext, on=YEAR_COL, how="left")
        merged_cols.extend(cols)

    return out, sorted(set(merged_cols)), statuses


def _rolling_slope(window_values: np.ndarray) -> float:
    vals = np.asarray(window_values, dtype=float)
    mask = np.isfinite(vals)
    vals = vals[mask]
    if len(vals) < 3:
        return np.nan
    x = np.arange(len(vals), dtype=float)
    slope = np.polyfit(x, vals, 1)[0]
    return float(slope)


def _compute_group_history(
    fact_df: pd.DataFrame, group_col: str, prefix: str, include_extended: bool
) -> tuple[pd.DataFrame, list[str]]:
    base = fact_df[[YEAR_COL, group_col, TARGET_COL]].copy()
    base[group_col] = (
        base[group_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )

    agg = (
        base.groupby([YEAR_COL, group_col], dropna=False)[TARGET_COL]
        .agg(["median", "mean", "count"])
        .reset_index()
        .sort_values([group_col, YEAR_COL])
        .reset_index(drop=True)
    )

    grouped = agg.groupby(group_col, group_keys=False)
    agg[f"{prefix}_prev_year_median_land_value"] = grouped["median"].shift(1)
    agg[f"{prefix}_prev_year_mean_land_value"] = grouped["mean"].shift(1)
    agg[f"{prefix}_prev_year_property_count"] = grouped["count"].shift(1)
    agg[f"{prefix}_prev_year_growth_rate"] = grouped["median"].pct_change().shift(1)
    agg[f"{prefix}_prev_2yr_median_land_value"] = grouped["median"].shift(2)
    agg[f"{prefix}_prev_2yr_mean_land_value"] = grouped["mean"].shift(2)
    agg[f"{prefix}_prev_3yr_rolling_median_land_value"] = grouped["median"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).median()
    )
    agg[f"{prefix}_prev_3yr_rolling_mean_land_value"] = grouped["median"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    agg[f"{prefix}_prev_3yr_growth_trend"] = grouped["median"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).apply(_rolling_slope, raw=True)
    )

    created_cols = [
        f"{prefix}_prev_year_median_land_value",
        f"{prefix}_prev_year_mean_land_value",
        f"{prefix}_prev_year_property_count",
        f"{prefix}_prev_year_growth_rate",
        f"{prefix}_prev_2yr_median_land_value",
    ]

    if include_extended:
        agg[f"{prefix}_prev_3yr_value_std"] = grouped["median"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )
        created_cols.extend(
            [
                f"{prefix}_prev_2yr_mean_land_value",
                f"{prefix}_prev_3yr_rolling_median_land_value",
                f"{prefix}_prev_3yr_rolling_mean_land_value",
                f"{prefix}_prev_3yr_growth_trend",
                f"{prefix}_prev_3yr_value_std",
            ]
        )
    else:
        created_cols.extend(
            [
                f"{prefix}_prev_3yr_rolling_median_land_value",
                f"{prefix}_prev_3yr_growth_trend",
            ]
        )

    keep = [YEAR_COL, group_col] + created_cols
    return agg[keep], created_cols


def _merge_census_if_robust(
    df: pd.DataFrame, census_path: Path, merge_census: bool
) -> tuple[pd.DataFrame, str]:
    if not merge_census:
        return df, "deferred_default"
    if not census_path.exists():
        return df, "deferred_missing_input"

    census = pd.read_parquet(census_path).copy()

    # Robust merge requires shared geography keys. Current standardized census table is
    # Vancouver-level (CMA/CSD) and usually lacks neighbourhood/FSA keys.
    candidate_keys = ["NEIGHBOURHOOD_CODE", "POSTAL_FSA", "PROPERTY_POSTAL_CODE"]
    shared = [k for k in candidate_keys if k in census.columns and k in df.columns]
    if not shared:
        return df, "deferred_no_shared_geo_key"

    join_key = shared[0]
    census_cols = [c for c in census.columns if c.startswith("census_")]
    if not census_cols:
        return df, "deferred_no_census_features"

    small = census[[join_key] + census_cols].copy()
    if small[join_key].duplicated().any():
        return df, "deferred_non_unique_geo_key"

    merged = df.merge(small, on=join_key, how="left")
    return merged, f"merged_on_{join_key}"


def build_model_table(
    property_path: Path,
    mortgage_path: Path,
    ircc_pr_path: Path,
    ircc_study_path: Path,
    cmhc_path: Path,
    census_path: Path,
    merge_census: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    fact = _load_fact_table(property_path)
    row_count_before = len(fact)
    original_cols = set(fact.columns)
    _append_summary(summary_rows, "table", "fact_rows_before", row_count_before)

    fact, property_cols = _build_property_features(fact)
    for c in property_cols:
        _append_summary(summary_rows, "engineered_property_feature", c, "created")

    macro_paths = {
        "mortgage_rate": mortgage_path,
        "ircc_pr": ircc_pr_path,
        "ircc_study_permits": ircc_study_path,
        "cmhc_rental": cmhc_path,
    }
    fact, macro_cols, macro_statuses = _merge_yearly_macros(fact, macro_paths)
    for c in macro_cols:
        _append_summary(summary_rows, "engineered_macro_feature", c, "merged")
    for name, status in macro_statuses.items():
        _append_summary(summary_rows, "source_status", name, status)

    neigh_hist, neigh_cols = _compute_group_history(fact, NEIGH_COL, "neigh", include_extended=True)
    fact = fact.merge(neigh_hist, on=[YEAR_COL, NEIGH_COL], how="left")
    for c in neigh_cols:
        _append_summary(summary_rows, "engineered_group_feature", c, "created")

    fsa_hist, fsa_cols_full = _compute_group_history(fact, "POSTAL_FSA", "fsa", include_extended=False)
    fsa_cols_keep = [
        "fsa_prev_year_median_land_value",
        "fsa_prev_year_mean_land_value",
        "fsa_prev_year_property_count",
        "fsa_prev_2yr_median_land_value",
        "fsa_prev_3yr_rolling_median_land_value",
        "fsa_prev_3yr_growth_trend",
    ]
    fsa_keep = [YEAR_COL, "POSTAL_FSA"] + [c for c in fsa_cols_keep if c in fsa_cols_full]
    fact = fact.merge(fsa_hist[fsa_keep], on=[YEAR_COL, "POSTAL_FSA"], how="left")
    for c in fsa_keep[2:]:
        _append_summary(summary_rows, "engineered_group_feature", c, "created")
    
    # --------------------------------------------------------
    # New: PLAN-level history features
    #
    # For STRATA properties, PLAN is much closer to a building /
    # strata scheme context than neighbourhood or FSA alone.
    # --------------------------------------------------------
    plan_hist, plan_cols_full = _compute_group_history(
        fact,
        PLAN_COL,
        "plan",
        include_extended=False,
    )

    plan_cols_keep = [
        "plan_prev_year_median_land_value",
        "plan_prev_year_mean_land_value",
        "plan_prev_year_property_count",
        "plan_prev_2yr_median_land_value",
        "plan_prev_3yr_rolling_median_land_value",
        "plan_prev_3yr_growth_trend",
    ]

    plan_keep = [YEAR_COL, PLAN_COL] + [c for c in plan_cols_keep if c in plan_cols_full]
    fact = fact.merge(plan_hist[plan_keep], on=[YEAR_COL, PLAN_COL], how="left")

    for c in plan_keep[2:]:
        _append_summary(summary_rows, "engineered_group_feature", c, "created")

    fact, census_status = _merge_census_if_robust(fact, census_path, merge_census)
    _append_summary(summary_rows, "source_status", "census_profile", census_status)

    if len(fact) != row_count_before:
        raise ValueError(
            f"Row count changed after joins: before={row_count_before}, after={len(fact)}. "
            "This indicates a non-unique merge and potential leakage risk."
        )

    new_cols = sorted([c for c in fact.columns if c not in original_cols])
    years = sorted(pd.to_numeric(fact[YEAR_COL], errors="coerce").dropna().astype(int).unique().tolist())

    _append_summary(summary_rows, "table", "total_rows", len(fact))
    _append_summary(summary_rows, "table", "total_columns", len(fact.columns))
    _append_summary(summary_rows, "table", "new_columns_count", len(new_cols))
    _append_summary(summary_rows, "table", "new_columns", ",".join(new_cols))
    _append_summary(summary_rows, "table", "year_coverage", f"{years[0]}-{years[-1]}" if years else "none")
    _append_summary(summary_rows, "table", "sample_years_present", ",".join(map(str, years)))
    #_append_summary(summary_rows, "table", "group_feature_families", "neigh_history,fsa_history")
    _append_summary(summary_rows, "table", "group_feature_families", "neigh_history,fsa_history,plan_history")

    for c in new_cols:
        _append_summary(summary_rows, "missing_rate", c, float(fact[c].isna().mean()))

    return fact, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final merged model table with leakage-safe local history features."
    )
    parser.add_argument(
        "--property_path",
        type=str,
        default="data/interim/property_tax_clean.parquet",
        help="Property fact table parquet path",
    )
    parser.add_argument(
        "--mortgage_path",
        type=str,
        default="data/interim/mortgage_rate_yearly.parquet",
        help="Yearly mortgage features parquet path",
    )
    parser.add_argument(
        "--ircc_pr_path",
        type=str,
        default="data/interim/ircc_pr_yearly.parquet",
        help="Yearly IRCC PR features parquet path",
    )
    parser.add_argument(
        "--ircc_study_path",
        type=str,
        default="data/interim/ircc_study_permits_yearly.parquet",
        help="Yearly IRCC study permits features parquet path",
    )
    parser.add_argument(
        "--cmhc_path",
        type=str,
        default="data/interim/cmhc_rental_yearly.parquet",
        help="Yearly CMHC rental features parquet path",
    )
    parser.add_argument(
        "--census_path",
        type=str,
        default="data/interim/census_profile_standardized.parquet",
        help="Standardized census parquet path",
    )
    parser.add_argument(
        "--merge_census",
        action="store_true",
        help="Attempt census merge only if robust shared geography keys are available.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/model_table.parquet",
        help="Output merged model table parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/model_table_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    print(f"[model_table] Loading property fact table: {args.property_path}")
    model_table, summary_rows = build_model_table(
        property_path=Path(args.property_path),
        mortgage_path=Path(args.mortgage_path),
        ircc_pr_path=Path(args.ircc_pr_path),
        ircc_study_path=Path(args.ircc_study_path),
        cmhc_path=Path(args.cmhc_path),
        census_path=Path(args.census_path),
        merge_census=args.merge_census,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    model_table.to_parquet(out_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"[model_table] Saved: {out_path}")
    print(f"[model_table] Saved summary: {summary_path}")
    print(f"[model_table] Rows={len(model_table):,}, Cols={len(model_table.columns):,}")


if __name__ == "__main__":
    main()
