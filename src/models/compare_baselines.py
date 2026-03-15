import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import compute_metrics
from src.models import baseline
from src.models.baseline_merged import run_baseline_merged


def evaluate_property_only(data_path: Path, sample_frac: float = 1.0) -> dict[str, float]:
    df = baseline.load_data(data_path)

    train_df = df[df[baseline.REPORT_YEAR_COL] < 2024].copy()
    test_df = df[df[baseline.REPORT_YEAR_COL] >= 2024].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Property-only baseline has empty train/test split.")

    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)

    X_train = train_df[baseline.FEATURE_COLS]
    y_train = train_df[baseline.TARGET_COL]
    X_test = test_df[baseline.FEATURE_COLS]
    y_test = test_df[baseline.TARGET_COL]

    model = baseline.build_pipeline()
    model.fit(X_train, np.log1p(y_train))
    y_pred = np.expm1(model.predict(X_test))

    metrics = compute_metrics(y_test, y_pred, y_train)
    return {
        "model": "baseline_property_only",
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "median_ape": float(metrics["median_ape"]),
        "robust_rmse": float(metrics["robust_rmse"]),
        "robust_mae": float(metrics["robust_mae"]),
        "robust_cap_p99_5": float(metrics["robust_cap_p99_5"]),
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare property-only vs merged baseline.")
    parser.add_argument(
        "--property_data_path",
        type=str,
        default="data/interim/property_tax_clean.parquet",
        help="Path to property-only cleaned parquet",
    )
    parser.add_argument(
        "--merged_data_path",
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
    parser.add_argument(
        "--out_path",
        type=str,
        default="reports/figures/baseline_comparison.csv",
        help="Output comparison CSV path",
    )
    args = parser.parse_args()

    property_path = Path(args.property_data_path)
    merged_path = Path(args.merged_data_path)
    out_path = Path(args.out_path)

    if not property_path.exists():
        raise FileNotFoundError(f"Property-only parquet not found: {property_path}")
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged parquet not found: {merged_path}")

    print("[compare] Running property-only baseline evaluation...")
    property_metrics = evaluate_property_only(property_path, sample_frac=args.sample_frac)

    print("[compare] Running merged baseline evaluation...")
    merged_metrics = run_baseline_merged(
        data_path=merged_path, sample_frac=args.sample_frac, save_outputs=False
    )
    merged_metrics_row = {"model": "baseline_merged", **merged_metrics}

    comparison_df = pd.DataFrame([property_metrics, merged_metrics_row])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False)

    print(f"[compare] Saved baseline comparison table: {out_path}")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
