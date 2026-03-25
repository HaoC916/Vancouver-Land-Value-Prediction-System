from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data._xlsx_xml import detect_year_total_columns, parse_number, read_xlsx_sheet_rows


def build_ircc_pr_yearly(in_path: Path, sheet_name: str) -> tuple[pd.DataFrame, str]:
    rows = read_xlsx_sheet_rows(in_path, sheet_name=sheet_name)
    year_cols = detect_year_total_columns(rows)
    years_sorted = sorted(year_cols)

    vancouver_row = None
    bc_total_row = None
    for row in rows:
        c0 = str(row[0]).strip() if len(row) > 0 else ""
        c1 = str(row[1]).strip() if len(row) > 1 else ""
        if c1.lower() == "vancouver":
            vancouver_row = row
        if c0.lower() == "british columbia total":
            bc_total_row = row

    if vancouver_row is None and bc_total_row is None:
        raise ValueError(
            "Could not find Vancouver CMA row or British Columbia Total row in IRCC PR file."
        )

    out_df = pd.DataFrame({"REPORT_YEAR": years_sorted})
    scope_note = ""

    # We prefer Vancouver CMA because it is closer to the project geography.
    if vancouver_row is not None:
        out_df["ircc_pr_vancouver_cma"] = [
            parse_number(vancouver_row[year_cols[y]] if year_cols[y] < len(vancouver_row) else "")
            for y in years_sorted
        ]
        scope_note = "Using Vancouver CMA series from PR - CMA sheet."
    else:
        scope_note = (
            "Vancouver CMA row not found; using BC total series as fallback."
        )

    if bc_total_row is not None:
        out_df["ircc_pr_bc_total"] = [
            parse_number(bc_total_row[year_cols[y]] if year_cols[y] < len(bc_total_row) else "")
            for y in years_sorted
        ]

    return out_df, scope_note


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize IRCC permanent residents yearly counts."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/raw/ircc_pr_cma_20260228.xlsx",
        help="Raw IRCC PR workbook path",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default="PR - CMA",
        help="Sheet name to parse",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/interim/ircc_pr_yearly.parquet",
        help="Output yearly parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/ircc_pr_yearly_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[ircc_pr] Reading workbook: {in_path} (sheet={args.sheet_name})")
    out_df, scope_note = build_ircc_pr_yearly(in_path, args.sheet_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    summary_df = out_df.copy()
    summary_df["note"] = scope_note
    summary_df.to_csv(summary_path, index=False)

    print(f"[ircc_pr] Saved yearly parquet: {out_path}")
    print(f"[ircc_pr] Saved summary CSV: {summary_path}")
    print(f"[ircc_pr] {scope_note}")


if __name__ == "__main__":
    main()
