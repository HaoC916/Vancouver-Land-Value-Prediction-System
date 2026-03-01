import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEEP_COLS = [
    "PID",
    "REPORT_YEAR",
    "NEIGHBOURHOOD_CODE",
    "CURRENT_LAND_VALUE",
    "LEGAL_TYPE",
    "ZONING_DISTRICT",
    "ZONING_CLASSIFICATION",
    "PROPERTY_POSTAL_CODE",
    "YEAR_BUILT",
    "BIG_IMPROVEMENT_YEAR",
    "LAND_COORDINATE",
]

CAT_COLS = [
    "LEGAL_TYPE",
    "ZONING_DISTRICT",
    "ZONING_CLASSIFICATION",
    "NEIGHBOURHOOD_CODE",
    "PROPERTY_POSTAL_CODE",
]

KEY_MISSING_COLS = [
    "CURRENT_LAND_VALUE",
    "YEAR_BUILT",
    "BIG_IMPROVEMENT_YEAR",
    "LAND_COORDINATE",
    "PROPERTY_POSTAL_CODE",
]


def clean_property_tax(in_path: Path, out_path: Path, summary_path: Path) -> None:
    df = pd.read_csv(in_path, sep=";", low_memory=False)
    total_rows_before = len(df)

    df = df[KEEP_COLS].copy()

    df["CURRENT_LAND_VALUE"] = pd.to_numeric(df["CURRENT_LAND_VALUE"], errors="coerce")
    df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
    df["LAND_COORDINATE"] = pd.to_numeric(df["LAND_COORDINATE"], errors="coerce")

    df = df.dropna(subset=["CURRENT_LAND_VALUE"])
    df = df[df["CURRENT_LAND_VALUE"] > 0]
    df = df.dropna(subset=["REPORT_YEAR"])
    df = df[(df["REPORT_YEAR"] >= 2020) & (df["REPORT_YEAR"] <= 2026)]

    for col in ["YEAR_BUILT", "BIG_IMPROVEMENT_YEAR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 1800, col] = np.nan
        df.loc[df[col] > df["REPORT_YEAR"], col] = np.nan

    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None", "NaT"]), col] = np.nan
        df[col] = df[col].fillna("Unknown")

    df["PROPERTY_AGE"] = df["REPORT_YEAR"] - df["YEAR_BUILT"]
    df.loc[df["YEAR_BUILT"].isna(), "PROPERTY_AGE"] = np.nan

    df["YEARS_SINCE_IMPROVEMENT"] = df["REPORT_YEAR"] - df["BIG_IMPROVEMENT_YEAR"]
    df.loc[df["BIG_IMPROVEMENT_YEAR"].isna(), "YEARS_SINCE_IMPROVEMENT"] = np.nan

    total_rows_after = len(df)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    summary_rows = []
    summary_rows.append(
        {"section": "rows", "metric": "total_rows_before", "value": total_rows_before}
    )
    summary_rows.append(
        {"section": "rows", "metric": "total_rows_after", "value": total_rows_after}
    )

    for col in KEY_MISSING_COLS:
        miss_rate = df[col].isna().mean()
        summary_rows.append(
            {"section": "missing_rate", "metric": col, "value": round(miss_rate, 6)}
        )

    year_counts = df["REPORT_YEAR"].value_counts(dropna=False).sort_index()
    for year, count in year_counts.items():
        summary_rows.append(
            {"section": "count_by_report_year", "metric": str(int(year)), "value": int(count)}
        )

    neigh_counts = df["NEIGHBOURHOOD_CODE"].value_counts(dropna=False).head(20)
    for neigh, count in neigh_counts.items():
        summary_rows.append(
            {
                "section": "count_by_neighbourhood_code_top20",
                "metric": str(neigh),
                "value": int(count),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean City of Vancouver property tax data")
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/raw/property-tax-report.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/interim/property_tax_clean.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    summary_path = Path("reports/figures/property_tax_clean_summary.csv")
    clean_property_tax(in_path, Path(args.out_path), summary_path)


if __name__ == "__main__":
    main()
