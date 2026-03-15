from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _merge_yearly_features(
    fact_df: pd.DataFrame, feature_path: Path, source_name: str
) -> tuple[pd.DataFrame, list[str], str]:
    if not feature_path.exists():
        return fact_df, [], "missing_input"

    ext = pd.read_parquet(feature_path)
    if "REPORT_YEAR" not in ext.columns:
        return fact_df, [], "missing_REPORT_YEAR"

    feature_cols = [c for c in ext.columns if c != "REPORT_YEAR"]
    if not feature_cols:
        return fact_df, [], "no_feature_columns"

    ext = ext.copy()
    ext["REPORT_YEAR"] = pd.to_numeric(ext["REPORT_YEAR"], errors="coerce")
    ext = ext.dropna(subset=["REPORT_YEAR"])
    ext["REPORT_YEAR"] = ext["REPORT_YEAR"].astype(int)
    ext = ext.sort_values("REPORT_YEAR").drop_duplicates(subset=["REPORT_YEAR"], keep="last")

    merged = fact_df.merge(ext, on="REPORT_YEAR", how="left")
    return merged, feature_cols, "merged"


def _append_summary_rows(summary_rows: list[dict[str, object]], section: str, metric: str, value: object) -> None:
    summary_rows.append({"section": section, "metric": metric, "value": value})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build merged model table v1 from property fact table + external yearly features."
    )
    parser.add_argument(
        "--property_path",
        type=str,
        default="data/interim/property_tax_clean.parquet",
        help="Clean property fact table parquet path",
    )
    parser.add_argument(
        "--mortgage_path",
        type=str,
        default="data/interim/mortgage_rate_yearly.parquet",
        help="Standardized mortgage yearly parquet path",
    )
    parser.add_argument(
        "--ircc_pr_path",
        type=str,
        default="data/interim/ircc_pr_yearly.parquet",
        help="Standardized IRCC PR yearly parquet path",
    )
    parser.add_argument(
        "--ircc_study_path",
        type=str,
        default="data/interim/ircc_study_permits_yearly.parquet",
        help="Standardized IRCC study permits yearly parquet path",
    )
    parser.add_argument(
        "--cmhc_path",
        type=str,
        default="data/interim/cmhc_rental_yearly.parquet",
        help="Standardized CMHC yearly parquet path",
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
        help="If set, attach Vancouver CMA census constants to all rows.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/model_table_v1.parquet",
        help="Output model table parquet path",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="reports/figures/model_table_v1_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    property_path = Path(args.property_path)
    if not property_path.exists():
        raise FileNotFoundError(f"Property fact table not found: {property_path}")

    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    print(f"[model_table] Loading fact table: {property_path}")
    fact_df = pd.read_parquet(property_path)
    original_cols = set(fact_df.columns)

    source_paths = {
        "mortgage_rate": Path(args.mortgage_path),
        "ircc_pr": Path(args.ircc_pr_path),
        "ircc_study_permits": Path(args.ircc_study_path),
        "cmhc_rental": Path(args.cmhc_path),
    }

    merged_sources: dict[str, str] = {}
    added_cols: list[str] = []

    for source_name, source_path in source_paths.items():
        print(f"[model_table] Merging source: {source_name} from {source_path}")
        fact_df, cols, status = _merge_yearly_features(fact_df, source_path, source_name)
        merged_sources[source_name] = status
        if status == "merged":
            added_cols.extend(cols)

    census_path = Path(args.census_path)
    if args.merge_census:
        if census_path.exists():
            census_df = pd.read_parquet(census_path)
            cma = census_df[census_df["geo_scope"].astype(str).str.lower().eq("vancouver_cma")]
            if cma.empty:
                merged_sources["census_profile"] = "merge_requested_but_no_vancouver_cma"
            else:
                row = cma.iloc[0]
                census_feature_cols = [
                    c for c in census_df.columns if c.startswith("census_")
                ]
                for col in census_feature_cols:
                    fact_df[col] = row[col]
                added_cols.extend(census_feature_cols)
                merged_sources["census_profile"] = "merged_vancouver_cma_constants"
        else:
            merged_sources["census_profile"] = "missing_input"
    else:
        merged_sources["census_profile"] = "deferred_v1"

    added_cols = sorted(set(added_cols))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fact_df.to_parquet(out_path, index=False)

    summary_rows: list[dict[str, object]] = []
    _append_summary_rows(summary_rows, "table", "total_rows", int(len(fact_df)))
    _append_summary_rows(summary_rows, "table", "total_columns", int(len(fact_df.columns)))
    _append_summary_rows(summary_rows, "table", "new_external_columns_count", int(len(added_cols)))
    _append_summary_rows(summary_rows, "table", "new_external_columns", ",".join(added_cols))
    years_present = sorted(pd.to_numeric(fact_df["REPORT_YEAR"], errors="coerce").dropna().astype(int).unique().tolist())
    _append_summary_rows(summary_rows, "table", "sample_years_present", ",".join(map(str, years_present)))

    for col in added_cols:
        _append_summary_rows(
            summary_rows,
            "missing_rate",
            col,
            float(fact_df[col].isna().mean()),
        )

    for source_name, status in merged_sources.items():
        _append_summary_rows(summary_rows, "source_status", source_name, status)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print(f"[model_table] Saved model table: {out_path}")
    print(f"[model_table] Saved summary: {summary_path}")
    print(f"[model_table] Added external columns: {len(added_cols)}")
    if merged_sources.get("census_profile", "").startswith("deferred"):
        print("[model_table] Census merge deferred in v1 (table standardized separately).")


if __name__ == "__main__":
    main()
