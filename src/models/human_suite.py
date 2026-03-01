import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    return df[needed].copy()


def build_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ],
        remainder="drop",
    )


def evaluate_model(
    name, model, X_train, y_train, X_test, y_test, test_df, out_dir, log_target=False
):
    start = datetime.now(timezone.utc)

    pre = build_preprocessor()
    pipeline = Pipeline(steps=[("preprocess", pre), ("model", model)])
    if log_target:
        y_train_fit = np.log1p(y_train)
    else:
        y_train_fit = y_train
    pipeline.fit(X_train, y_train_fit)

    y_pred = pipeline.predict(X_test)
    if log_target:
        y_pred = np.expm1(y_pred)

    y_true_stats, y_pred_stats = prediction_sanity_stats(y_test, y_pred)
    print(
        f"[{name}] Test y_true stats (p50/p90/p99/max): "
        f"{y_true_stats['p50']:.2f}, {y_true_stats['p90']:.2f}, "
        f"{y_true_stats['p99']:.2f}, {y_true_stats['max']:.2f}"
    )
    print(
        f"[{name}] Test y_pred stats (p50/p90/p99/max): "
        f"{y_pred_stats['p50']:.2f}, {y_pred_stats['p90']:.2f}, "
        f"{y_pred_stats['p99']:.2f}, {y_pred_stats['max']:.2f}"
    )
    warning = scale_warning(y_true_stats, y_pred_stats)
    if warning:
        print(f"[{name}] {warning}")

    metrics = compute_metrics(y_test, y_pred, y_train)

    out_dir = Path(out_dir)
    report_path = out_dir / f"{name}_neighbourhood_error.csv"
    summary_df = write_baseline_reports(y_test, y_pred, test_df, report_path)
    save_model_plots(y_test, y_pred, summary_df, out_dir, name)

    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds()

    return {
        "model": name,
        "status": "ok",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "median_ape": metrics["median_ape"],
        "robust_cap_p99_5": metrics["robust_cap_p99_5"],
        "robust_rmse": metrics["robust_rmse"],
        "robust_mae": metrics["robust_mae"],
        "start_time_utc": start.isoformat(),
        "end_time_utc": end.isoformat(),
        "duration_seconds": round(duration, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-track model suite")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/interim/property_tax_clean.parquet",
        help="Path to cleaned parquet",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}")

    df = load_data(data_path)
    train_df = df[df[REPORT_YEAR_COL] < 2024]
    test_df = df[df[REPORT_YEAR_COL] >= 2024]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    results = []

    ridge = Ridge(random_state=42)
    results.append(
        evaluate_model(
            "ridge",
            ridge,
            X_train,
            y_train,
            X_test,
            y_test,
            test_df,
            Path("reports/figures"),
        )
    )

    # Optional: XGBoost or LightGBM if installed
    tried_boost = False
    try:
        from xgboost import XGBRegressor  # type: ignore

        tried_boost = True
        xgb = XGBRegressor()
        results.append(
            evaluate_model(
                "xgboost",
                xgb,
                X_train,
                y_train,
                X_test,
                y_test,
                test_df,
                Path("reports/figures"),
            )
        )
    except Exception:
        pass

    if not tried_boost:
        try:
            from lightgbm import LGBMRegressor  # type: ignore

            tried_boost = True
            lgbm = LGBMRegressor()
            results.append(
                evaluate_model(
                    "lightgbm",
                    lgbm,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    test_df,
                    Path("reports/figures"),
                )
            )
        except Exception:
            print(
                "Boosted model not run: install xgboost or lightgbm to enable it."
            )
            results.append(
                {
                    "model": "xgboost_or_lightgbm",
                    "status": "missing_dependency",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "median_ape": np.nan,
                    "robust_cap_p99_5": np.nan,
                    "robust_rmse": np.nan,
                    "robust_mae": np.nan,
                    "start_time_utc": "",
                    "end_time_utc": "",
                    "duration_seconds": np.nan,
                }
            )

    results_df = pd.DataFrame(results)
    out_path = Path("reports/figures/human_model_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
