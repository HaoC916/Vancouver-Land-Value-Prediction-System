from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


USECOLS = [
    "CENSUS_YEAR",
    "DGUID",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "C1_COUNT_TOTAL",
    "C10_RATE_TOTAL",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": np.nan, "..": np.nan, "...": np.nan, "--": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _pick_target_geographies(geoindex_df: pd.DataFrame) -> pd.DataFrame:
    geo = geoindex_df.copy()
    geo["Geo Name"] = geo["Geo Name"].astype(str).str.strip()
    geo["Geo Code"] = geo["Geo Code"].astype(str).str.strip()

    cma = geo[
        geo["Geo Name"].str.lower().eq("vancouver")
        & geo["Geo Code"].str.startswith("2021S05", na=False)
    ].head(1)
    csd = geo[
        geo["Geo Name"].str.lower().eq("vancouver")
        & geo["Geo Code"].str.startswith("2021A", na=False)
    ].head(1)

    targets = pd.concat([cma, csd], ignore_index=True)
    if targets.empty:
        raise ValueError("Could not find Vancouver CMA/CSD rows in geoindex.")

    targets = targets.rename(columns={"Geo Code": "DGUID", "Geo Name": "GEO_NAME"})
    targets["geo_scope"] = np.where(
        targets["DGUID"].str.startswith("2021S05"), "vancouver_cma", "vancouver_csd"
    )
    return targets[["DGUID", "GEO_NAME", "geo_scope"]]


def _extract_features_for_geo(sub: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {
        "census_population_2021": np.nan,
        "census_avg_household_size": np.nan,
        "census_median_household_income_2020": np.nan,
        "census_owner_households": np.nan,
        "census_renter_households": np.nan,
        "census_age_65_plus_pct": np.nan,
    }

    def first_value(mask: pd.Series, value_col: str) -> float:
        match = sub.loc[mask, value_col].dropna()
        return float(match.iloc[0]) if not match.empty else np.nan

    name = sub["CHARACTERISTIC_NAME"].astype(str).str.strip()

    result["census_population_2021"] = first_value(
        name.eq("Population, 2021"), "C1_COUNT_TOTAL"
    )
    result["census_avg_household_size"] = first_value(
        name.eq("Average household size"), "C1_COUNT_TOTAL"
    )
    result["census_median_household_income_2020"] = first_value(
        name.str.startswith("Median total income of household in 2020"), "C1_COUNT_TOTAL"
    )
    result["census_owner_households"] = first_value(name.eq("Owner"), "C1_COUNT_TOTAL")
    result["census_renter_households"] = first_value(name.eq("Renter"), "C1_COUNT_TOTAL")

    age_mask = name.eq("65 years and over") & sub["C10_RATE_TOTAL"].between(0, 100)
    result["census_age_65_plus_pct"] = first_value(age_mask, "C10_RATE_TOTAL")
    return result


def build_census_standardized(
    data_path: Path, geoindex_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geoindex_df = pd.read_csv(geoindex_path, encoding="latin1")
    targets = _pick_target_geographies(geoindex_df)
    target_dguids = set(targets["DGUID"].tolist())

    chunks = []
    for chunk in pd.read_csv(
        data_path, encoding="latin1", usecols=USECOLS, low_memory=False, chunksize=200_000
    ):
        sub = chunk[chunk["DGUID"].astype(str).isin(target_dguids)]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        raise ValueError("No census rows found for target Vancouver geographies.")

    census = pd.concat(chunks, ignore_index=True)
    census["C1_COUNT_TOTAL"] = _to_numeric(census["C1_COUNT_TOTAL"])
    census["C10_RATE_TOTAL"] = _to_numeric(census["C10_RATE_TOTAL"])

    rows = []
    for _, target in targets.iterrows():
        dguid = target["DGUID"]
        sub = census[census["DGUID"].astype(str) == str(dguid)].copy()
        if sub.empty:
            continue
        features = _extract_features_for_geo(sub)
        census_year = int(pd.to_numeric(sub["CENSUS_YEAR"], errors="coerce").dropna().iloc[0])
        row = {
            "CENSUS_YEAR": census_year,
            "DGUID": str(dguid),
            "GEO_NAME": target["GEO_NAME"],
            "geo_scope": target["geo_scope"],
        }
        row.update(features)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("geo_scope").reset_index(drop=True)

    summary_rows = []
    summary_rows.append({"section": "meta", "metric": "target_geographies", "value": len(out_df)})
    for col in out_df.columns:
        if col.startswith("census_"):
            summary_rows.append(
                {
                    "section": "missing_rate",
                    "metric": col,
                    "value": float(out_df[col].isna().mean()) if len(out_df) else np.nan,
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    return out_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize Census Profile (future-ready Vancouver features)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/statcan_censusprofile2021_data_20260228.csv",
        help="Raw census data CSV path",
    )
    parser.add_argument(
        "--geoindex_path",
        type=str,
        default="data/raw/statcan_censusprofile2021_geoindex_20260228.csv",
        help="Raw census geoindex CSV path",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="data/raw/statcan_censusprofile2021_meta_20260228.txt",
        help="Raw census meta text path (validated for existence only)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/interim/census_profile_standardized.parquet",
        help="Output standardized census parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/census_profile_standardized_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    geoindex_path = Path(args.geoindex_path)
    meta_path = Path(args.meta_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    for p in [data_path, geoindex_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    print("[census] Reading raw census files and extracting Vancouver CMA/CSD features...")
    out_df, summary_df = build_census_standardized(data_path, geoindex_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[census] Saved standardized parquet: {out_path}")
    print(f"[census] Saved summary CSV: {summary_path}")
    print(
        "[census] Note: this is a future-ready, static 2021 table. "
        "Geography-level merge into model table is deferred by default."
    )


if __name__ == "__main__":
    main()
