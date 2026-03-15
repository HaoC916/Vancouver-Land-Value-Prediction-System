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
ID_EXCLUDE_COLS = {"PID"}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    if REPORT_YEAR_COL not in df.columns:
        raise ValueError(f"Missing split column: {REPORT_YEAR_COL}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[REPORT_YEAR_COL] = pd.to_numeric(df[REPORT_YEAR_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, REPORT_YEAR_COL]).copy()
    df = df[df[TARGET_COL] > 0].copy()
    df[REPORT_YEAR_COL] = df[REPORT_YEAR_COL].astype(int)
    return df


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = {TARGET_COL, REPORT_YEAR_COL}.union(ID_EXCLUDE_COLS.intersection(df.columns))
    feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns available after exclusions.")

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, cat_cols, numeric_cols


def build_pipeline(cat_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", MaxAbsScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if not transformers:
        raise ValueError("Could not build preprocessors: no categorical or numeric features found.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Ridge(random_state=42)
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def run_baseline_merged(
    data_path: Path, sample_frac: float = 1.0, save_outputs: bool = True
) -> dict[str, float]:
    df = load_data(data_path)
    feature_cols, cat_cols, numeric_cols = infer_feature_columns(df)

    train_df = df[df[REPORT_YEAR_COL] < 2024].copy()
    test_df = df[df[REPORT_YEAR_COL] >= 2024].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Check REPORT_YEAR values in model table.")

    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    y_train_log = np.log1p(y_train)
    pipeline = build_pipeline(cat_cols, numeric_cols)
    pipeline.fit(X_train, y_train_log)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)

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

    metrics_with_meta: dict[str, float] = {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "median_ape": float(metrics["median_ape"]),
        "robust_rmse": float(metrics["robust_rmse"]),
        "robust_mae": float(metrics["robust_mae"]),
        "robust_cap_p99_5": float(metrics["robust_cap_p99_5"]),
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
        "n_features_total": float(len(feature_cols)),
        "n_features_cat": float(len(cat_cols)),
        "n_features_num": float(len(numeric_cols)),
    }

    if save_outputs:
        figures_dir = Path("reports/figures")
        if "NEIGHBOURHOOD_CODE" not in test_df.columns:
            test_df = test_df.copy()
            test_df["NEIGHBOURHOOD_CODE"] = "Unknown"
        report_path = figures_dir / "baseline_merged_neighbourhood_error.csv"
        summary_df = write_baseline_reports(y_test, y_pred, test_df, report_path)
        save_model_plots(y_test, y_pred, summary_df, figures_dir, "baseline_merged")

        metrics_path = figures_dir / "baseline_merged_metrics.csv"
        pd.DataFrame([{"model": "baseline_merged", **metrics_with_meta}]).to_csv(
            metrics_path, index=False
        )

    return metrics_with_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Merged-table baseline model on model_table_v1.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/model_table_v1.parquet",
        help="Path to merged model table parquet",
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

    run_baseline_merged(data_path=data_path, sample_frac=args.sample_frac, save_outputs=True)


if __name__ == "__main__":
    main()
