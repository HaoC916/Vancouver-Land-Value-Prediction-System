from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data._xlsx_xml import detect_year_total_columns, parse_number, read_xlsx_sheet_rows


def build_ircc_study_permits_yearly(
    in_path: Path, sheet_name: str
) -> tuple[pd.DataFrame, str]:
    rows = read_xlsx_sheet_rows(in_path, sheet_name=sheet_name)
    year_cols = detect_year_total_columns(rows)
    years_sorted = sorted(year_cols)

    bc_total_idx = None
    for idx, row in enumerate(rows):
        c0 = str(row[0]).strip().lower() if len(row) > 0 else ""
        if c0 == "british columbia total":
            bc_total_idx = idx
            break

    if bc_total_idx is None:
        raise ValueError("Could not find 'British Columbia Total' row in study permits file.")

    bc_row = rows[bc_total_idx]
    out_df = pd.DataFrame(
        {
            "REPORT_YEAR": years_sorted,
            "study_permits_bc_total": [
                parse_number(bc_row[year_cols[y]] if year_cols[y] < len(bc_row) else "")
                for y in years_sorted
            ],
        }
    )

    # We keep only BC total in the current pipeline.
    # The category rows under BC ("Post Secondary", etc.) are present, but their totals
    # are ambiguous in this extract and do not reconcile clearly with BC totals in one pass.
    note = (
        "Using BC total yearly counts. Category-specific values are deferred pending "
        "deeper validation of workbook structure."
    )
    return out_df, note


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize IRCC study permits yearly counts."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/raw/ircc_studypermits_pt_studylevel_20260228.xlsx",
        help="Raw IRCC study permits workbook path",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default="TR - SP Study Level Calendar",
        help="Sheet name to parse",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/interim/ircc_study_permits_yearly.parquet",
        help="Output yearly parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/ircc_study_permits_yearly_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[ircc_study] Reading workbook: {in_path} (sheet={args.sheet_name})")
    out_df, note = build_ircc_study_permits_yearly(in_path, args.sheet_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    summary_df = out_df.copy()
    summary_df["note"] = note
    summary_df.to_csv(summary_path, index=False)

    print(f"[ircc_study] Saved yearly parquet: {out_path}")
    print(f"[ircc_study] Saved summary CSV: {summary_path}")
    print(f"[ircc_study] {note}")


if __name__ == "__main__":
    main()
