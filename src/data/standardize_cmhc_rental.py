from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_cmhc_yearly(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    required = {"Year", "net_change_total", "new_added", "existing_converted"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CMHC file missing required columns: {sorted(missing)}")

    out_df = df[
        ["Year", "net_change_total", "new_added", "existing_converted"]
    ].copy()
    out_df = out_df.rename(
        columns={
            "Year": "REPORT_YEAR",
            "net_change_total": "cmhc_rental_supply_net_change_total",
            "new_added": "cmhc_rental_supply_new_added",
            "existing_converted": "cmhc_rental_supply_existing_converted",
        }
    )

    out_df["REPORT_YEAR"] = pd.to_numeric(out_df["REPORT_YEAR"], errors="coerce")
    for col in [
        "cmhc_rental_supply_net_change_total",
        "cmhc_rental_supply_new_added",
        "cmhc_rental_supply_existing_converted",
    ]:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    out_df = out_df.dropna(subset=["REPORT_YEAR"]).copy()
    out_df["REPORT_YEAR"] = out_df["REPORT_YEAR"].astype(int)
    out_df = out_df.sort_values("REPORT_YEAR").reset_index(drop=True)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize CMHC Vancouver rental supply yearly features."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/raw/cmhc_vancouver_rental_supply_change_20260228.csv",
        help="Raw CMHC CSV path",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/interim/cmhc_rental_yearly.parquet",
        help="Output yearly parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/cmhc_rental_yearly_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[cmhc] Reading raw file: {in_path}")
    out_df = build_cmhc_yearly(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    out_df.to_csv(summary_path, index=False)

    print(f"[cmhc] Saved yearly parquet: {out_path}")
    print(f"[cmhc] Saved summary CSV: {summary_path}")
    print(
        f"[cmhc] Year coverage: {int(out_df['REPORT_YEAR'].min())} - "
        f"{int(out_df['REPORT_YEAR'].max())}"
    )


if __name__ == "__main__":
    main()
