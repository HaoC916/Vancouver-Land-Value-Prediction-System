from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("artifacts/land_value_model.joblib")
SOURCE_PATH = Path("data/processed/model_table.parquet")
DEPLOY_DIR = Path("data/deploy")


def _safe_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    mode = s.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def _normalize_postal_code(value) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip().upper().replace(" ", "")
    if text in {"", "NAN", "NONE", "NAT", "UNKNOWN"}:
        return "Unknown"
    return text


def _build_group_lookup(
    df: pd.DataFrame,
    group_cols: list[str],
    numeric_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    numeric_value_cols = [c for c in numeric_cols if c in df.columns and c not in group_cols]
    cat_value_cols = [c for c in cat_cols if c in df.columns and c not in group_cols]

    if group_cols:
        num_df = (
            df[group_cols + numeric_value_cols]
            .groupby(group_cols, dropna=False)[numeric_value_cols]
            .median()
            .reset_index()
        )

        cat_df = (
            df[group_cols + cat_value_cols]
            .groupby(group_cols, dropna=False)[cat_value_cols]
            .agg(_safe_mode)
            .reset_index()
        )

        out_df = num_df.merge(cat_df, on=group_cols, how="outer")
    else:
        num_df = df[numeric_value_cols].median(numeric_only=True).to_frame().T
        cat_df = df[cat_value_cols].agg(_safe_mode).to_frame().T
        out_df = pd.concat([num_df, cat_df], axis=1)

    return out_df


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_PATH}")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source model table not found: {SOURCE_PATH}")

    bundle = joblib.load(MODEL_PATH)
    feature_cols = list(bundle["feature_cols"])
    cat_cols = list(bundle["cat_cols"])
    numeric_cols = list(bundle["numeric_cols"])

    option_cols = [
        "REPORT_YEAR",
        "PROPERTY_POSTAL_CODE",
        "POSTAL_FSA",
        "LEGAL_TYPE",
        "ZONING_DISTRICT",
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
    ]

    # PID is kept (not a feature) so we can build the per-unit value lookup below.
    keep_cols = sorted(set(feature_cols + option_cols + ["PID"]))

    df = pd.read_parquet(SOURCE_PATH)
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Keep only recent years for deploy demo
    df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
    df = df[(df["REPORT_YEAR"] >= 2024) & (df["REPORT_YEAR"] <= 2026)].copy()

    if "PROPERTY_POSTAL_CODE" in df.columns:
        df["PROPERTY_POSTAL_CODE"] = df["PROPERTY_POSTAL_CODE"].map(_normalize_postal_code)

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Options lookup
    options_df = df[[c for c in option_cols if c in df.columns]].drop_duplicates().copy()
    options_df.to_parquet(DEPLOY_DIR / "options_lookup.parquet", index=False)

    # 2. Year-level lookup
    year_lookup = _build_group_lookup(
        df=df,
        group_cols=["REPORT_YEAR"],
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
    )
    year_lookup.to_parquet(DEPLOY_DIR / "predict_year_lookup.parquet", index=False)

    # 3. Neighbourhood-level lookup
    if "NEIGHBOURHOOD_CODE" in df.columns:
        neigh_lookup = _build_group_lookup(
            df=df,
            group_cols=["REPORT_YEAR", "NEIGHBOURHOOD_CODE"],
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
        )
        neigh_lookup.to_parquet(DEPLOY_DIR / "predict_neigh_lookup.parquet", index=False)

    # 4. FSA-level lookup
    if "POSTAL_FSA" in df.columns:
        fsa_lookup = _build_group_lookup(
            df=df,
            group_cols=["REPORT_YEAR", "POSTAL_FSA"],
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
        )
        fsa_lookup.to_parquet(DEPLOY_DIR / "predict_fsa_lookup.parquet", index=False)

    # 5. Global defaults
    global_lookup = _build_group_lookup(
        df=df,
        group_cols=[],
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
    )
    global_lookup.to_parquet(DEPLOY_DIR / "predict_global_lookup.parquet", index=False)

    # 6. Per-PID previous-year value lookup. This is what makes per-unit estimates
    # work at inference: each selected unit's own prior assessment is fed to the
    # model's dominant feature (pid_prev_year_property_value).
    if "pid_prev_year_property_value" in df.columns and "PID" in df.columns:
        pid_lookup = (
            df[["PID", "REPORT_YEAR", "pid_prev_year_property_value"]]
            .dropna(subset=["pid_prev_year_property_value"])
            .drop_duplicates(["PID", "REPORT_YEAR"])
            .copy()
        )
        pid_lookup.to_parquet(DEPLOY_DIR / "predict_pid_lookup.parquet", index=False)

    print("Saved deploy lookup tables to data/deploy/")
    print("options_lookup.parquet")
    print("predict_year_lookup.parquet")
    print("predict_neigh_lookup.parquet")
    print("predict_fsa_lookup.parquet")
    print("predict_global_lookup.parquet")
    print("predict_pid_lookup.parquet")


if __name__ == "__main__":
    main()