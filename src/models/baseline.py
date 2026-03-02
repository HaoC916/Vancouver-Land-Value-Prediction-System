import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from src.eval.baseline_reports import write_baseline_reports
from src.eval.metrics import compute_metrics, prediction_sanity_stats, scale_warning
from src.viz.baseline_plots import save_model_plots

TARGET_COL = "CURRENT_LAND_VALUE"
REPORT_YEAR_COL = "REPORT_YEAR"

CAT_COLS = [
    "LEGAL_TYPE",
    "ZONING_DISTRICT",
    "ZONING_CLASSIFICATION",
    "NEIGHBOURHOOD_CODE",
    "PROPERTY_POSTAL_CODE",
]

NUM_COLS = [
    "LAND_COORDINATE",
    "YEAR_BUILT",
    "BIG_IMPROVEMENT_YEAR",
]

FEATURE_COLS = CAT_COLS + NUM_COLS


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = FEATURE_COLS + [TARGET_COL, REPORT_YEAR_COL]
    df = df[needed].copy()

    # Safety: ensure numeric columns are numeric (in case parquet came from mixed types)
    for c in NUM_COLS + [TARGET_COL, REPORT_YEAR_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic safety (should already be cleaned, but keep it cheap)
    df = df.dropna(subset=[TARGET_COL, REPORT_YEAR_COL])
    df = df[df[TARGET_COL] > 0]

    # Make year an int for clean split
    df[REPORT_YEAR_COL] = df[REPORT_YEAR_COL].astype(int)
    return df


def build_pipeline() -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Ridge is very sensitive to scale; MaxAbsScaler is safe with sparse stacking
            ("scale", MaxAbsScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ],
        remainder="drop",
    )

    model = Ridge(random_state=42)

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline model on cleaned parquet")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/interim/property_tax_clean.parquet",
        help="Path to cleaned parquet",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Optional fraction to sample from train/test for faster runs",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}")

    df = load_data(data_path)

    train_df = df[df[REPORT_YEAR_COL] < 2024]
    test_df = df[df[REPORT_YEAR_COL] >= 2024]

    if args.sample_frac < 1.0:
        train_df = train_df.sample(frac=args.sample_frac, random_state=42)
        test_df = test_df.sample(frac=args.sample_frac, random_state=42)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # ===== Train on log target to handle heavy tail =====
    y_train_log = np.log1p(y_train)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train_log)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # back to original dollar scale

    # Sanity stats
    y_true_stats, y_pred_stats = prediction_sanity_stats(y_test, y_pred)
    print(
        "Test y_true stats (p50/p90/p99/max): "
        f"{y_true_stats['p50']:.2f}, {y_true_stats['p90']:.2f}, "
        f"{y_true_stats['p99']:.2f}, {y_true_stats['max']:.2f}"
    )
    print(
        "Test y_pred stats (p50/p90/p99/max): "
        f"{y_pred_stats['p50']:.2f}, {y_pred_stats['p90']:.2f}, "
        f"{y_pred_stats['p99']:.2f}, {y_pred_stats['max']:.2f}"
    )
    warning = scale_warning(y_true_stats, y_pred_stats)
    if warning:
        print(warning)

    metrics = compute_metrics(y_test, y_pred, y_train)
    print(f"Test RMSE: {metrics['rmse']:,.2f}")
    print(f"Test MAE: {metrics['mae']:,.2f}")
    print(f"Test Median APE: {metrics['median_ape']:.4f}")
    print(f"Robust cap (p99.5 train): {metrics['robust_cap_p99_5']:,.2f}")
    print(f"Robust RMSE: {metrics['robust_rmse']:,.2f}")
    print(f"Robust MAE: {metrics['robust_mae']:,.2f}")

    report_path = Path("reports/figures/baseline_neighbourhood_error.csv")
    summary_df = write_baseline_reports(y_test, y_pred, test_df, report_path)
    save_model_plots(y_test, y_pred, summary_df, Path("reports/figures"), "baseline")


if __name__ == "__main__":
    main()