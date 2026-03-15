from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd


def _find_header_line(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if '"Geography"' in line and "195" in line:
            return idx
    raise ValueError("Could not find mortgage CSV header line with Geography/months.")


def build_mortgage_yearly(in_path: Path) -> pd.DataFrame:
    text = in_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = _find_header_line(lines)
    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))

    geography_col = df.columns[0]
    canada = df[df[geography_col].astype(str).str.strip().eq("Canada")].head(1)
    if canada.empty:
        raise ValueError("Could not find 'Canada' row in mortgage file.")

    long_df = canada.melt(
        id_vars=[geography_col], var_name="month_label", value_name="mortgage_rate"
    )
    long_df["month_label"] = (
        long_df["month_label"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    )
    long_df["date"] = pd.to_datetime(
        long_df["month_label"], format="%B %Y", errors="coerce"
    )
    long_df["mortgage_rate"] = pd.to_numeric(long_df["mortgage_rate"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "mortgage_rate"]).copy()
    long_df["REPORT_YEAR"] = long_df["date"].dt.year.astype(int)

    yearly_avg = (
        long_df.groupby("REPORT_YEAR", as_index=False)["mortgage_rate"]
        .mean()
        .rename(columns={"mortgage_rate": "mortgage_rate_5yr_avg"})
    )
    end_of_year = (
        long_df.sort_values("date")
        .groupby("REPORT_YEAR", as_index=False)
        .tail(1)[["REPORT_YEAR", "mortgage_rate"]]
        .rename(columns={"mortgage_rate": "mortgage_rate_5yr_end_of_year"})
    )
    out_df = yearly_avg.merge(end_of_year, on="REPORT_YEAR", how="left")
    out_df = out_df.sort_values("REPORT_YEAR").reset_index(drop=True)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize StatCan 5-year mortgage rate into yearly features."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/raw/statcan_mortgage_rate_5yr_20260228.csv",
        help="Raw mortgage CSV path",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/interim/mortgage_rate_yearly.parquet",
        help="Output yearly parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/mortgage_rate_yearly_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[mortgage] Reading raw file: {in_path}")
    out_df = build_mortgage_yearly(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    out_df.to_csv(summary_path, index=False)

    print(f"[mortgage] Saved yearly parquet: {out_path}")
    print(f"[mortgage] Saved summary CSV: {summary_path}")
    print(
        f"[mortgage] Year coverage: {int(out_df['REPORT_YEAR'].min())} - "
        f"{int(out_df['REPORT_YEAR'].max())}"
    )


if __name__ == "__main__":
    main()
