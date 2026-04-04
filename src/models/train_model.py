from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.eval.baseline_reports import write_baseline_reports
from src.eval.encoding_utils import apply_target_encoders, fit_target_encoders
from src.eval.feature_importance import save_feature_importance
from src.eval.metrics import compute_metrics, prediction_sanity_stats, scale_warning
from src.viz.baseline_plots import save_model_plots

TARGET_COL = "CURRENT_LAND_VALUE"
REPORT_YEAR_COL = "REPORT_YEAR"
ID_EXCLUDE_COLS = {"PID"}
HIGH_CARD_COLS = ["PROPERTY_POSTAL_CODE", "NEIGHBOURHOOD_CODE", "LEGAL_TYPE"]


def _choose_model():
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        return (
            "lightgbm",
            LGBMRegressor(
                random_state=42,
                n_estimators=400,
                learning_rate=0.05,
            ),
        )
    except Exception:
        pass

    try:
        from xgboost import XGBRegressor  # type: ignore

        return (
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_estimators=400,
                learning_rate=0.05,
            ),
        )
    except Exception:
        pass

    return (
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(random_state=42),
    )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")

    df = pd.read_parquet(path).copy()
    required = [TARGET_COL, REPORT_YEAR_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in model table: {missing}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[REPORT_YEAR_COL] = pd.to_numeric(df[REPORT_YEAR_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, REPORT_YEAR_COL]).copy()
    df = df[df[TARGET_COL] > 0].copy()
    df[REPORT_YEAR_COL] = df[REPORT_YEAR_COL].astype(int)
    return df


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COL, REPORT_YEAR_COL}.union(ID_EXCLUDE_COLS.intersection(df.columns))
    feature_cols = [c for c in df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError("No usable feature columns after exclusions.")
    return feature_cols


def infer_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return cat_cols, numeric_cols


def build_pipeline(cat_cols: list[str], numeric_cols: list[str], model) -> Pipeline:
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if numeric_cols:
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", num_pipe, numeric_cols))

    if not transformers:
        raise ValueError("Could not build preprocessors: no numeric/categorical columns found.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def _prepare_encoded_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    encoding_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    available_high_card = [c for c in HIGH_CARD_COLS if c in X_train.columns]
    y_train_log = np.log1p(y_train)

    # Train-only target encoding:
    # mappings are fitted ONLY on train data using log1p target for scale stability.
    encoders = fit_target_encoders(X_train, y_train_log, available_high_card)
    X_train_enc = apply_target_encoders(X_train, encoders)
    X_test_enc = apply_target_encoders(X_test, encoders)

    if encoding_mode == "b":
        drop_cols = [c for c in available_high_card if c in X_train_enc.columns]
        X_train_enc = X_train_enc.drop(columns=drop_cols)
        X_test_enc = X_test_enc.drop(columns=drop_cols)

    encoder_payload = {
        "enabled": True,
        "mode": encoding_mode,
        "target": "log1p_current_land_value",
        "cols": available_high_card,
        "encoders": {
            col: {
                "mapping": encoders[col].mapping,
                "global_mean": float(encoders[col].global_mean),
            }
            for col in available_high_card
            if col in encoders
        },
    }
    return X_train_enc, X_test_enc, available_high_card, encoder_payload


def train_and_evaluate(
    data_path: Path,
    artifact_path: Path,
    metadata_path: Path,
    sample_frac: float = 1.0,
    save_outputs: bool = True,
    encoding_mode: str = "b",
) -> dict[str, object]:
    if encoding_mode not in {"a", "b"}:
        raise ValueError("encoding_mode must be 'a' or 'b'")

    df = load_data(data_path)
    feature_cols = infer_feature_columns(df)

    train_df = df[df[REPORT_YEAR_COL] < 2024].copy()
    test_df = df[df[REPORT_YEAR_COL] >= 2024].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Check REPORT_YEAR values.")

    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)

    X_train_raw = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test_raw = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    X_train, X_test, encoded_cols, encoder_payload = _prepare_encoded_features(
        X_train=X_train_raw,
        X_test=X_test_raw,
        y_train=y_train,
        encoding_mode=encoding_mode,
    )
    cat_cols, numeric_cols = infer_types(X_train)

    model_backend, model = _choose_model()
    print(f"[train_model] Using model backend: {model_backend}")
    print(f"[train_model] Target encoding mode: {encoding_mode.upper()}")
    print(f"[train_model] Target encoded columns: {encoded_cols}")
    print("[train_model] Target encoding target: log1p(CURRENT_LAND_VALUE)")

    pipeline = build_pipeline(cat_cols, numeric_cols, model)
    y_train_log = np.log1p(y_train)
    pipeline.fit(X_train, y_train_log)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0.0)

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

    result: dict[str, object] = {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "median_ape": float(metrics["median_ape"]),
        "robust_rmse": float(metrics["robust_rmse"]),
        "robust_mae": float(metrics["robust_mae"]),
        "robust_cap_p99_5": float(metrics["robust_cap_p99_5"]),
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
        "n_features_total": float(X_train.shape[1]),
        "n_features_cat": float(len(cat_cols)),
        "n_features_num": float(len(numeric_cols)),
        "model_backend": model_backend,
        "encoding_mode": encoding_mode,
        "target_encoding_target": "log1p_current_land_value",
    }

    if save_outputs:
        figures_dir = Path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        report_df = test_df.copy()
        if "NEIGHBOURHOOD_CODE" not in report_df.columns:
            report_df["NEIGHBOURHOOD_CODE"] = "Unknown"

        summary_df = write_baseline_reports(
            y_test,
            y_pred,
            report_df,
            figures_dir / "model_neighbourhood_error.csv",
        )
        save_model_plots(
            y_test,
            y_pred,
            summary_df,
            figures_dir,
            "model",
        )

        metrics_row = {"model": f"model_{model_backend}", **result}
        pd.DataFrame([metrics_row]).to_csv(
            figures_dir / "model_metrics.csv",
            index=False,
        )

        save_feature_importance(
            pipeline=pipeline,
            X_test=X_test,
            y_test=np.log1p(y_test),
            out_csv_path=figures_dir / "model_feature_importance.csv",
            out_png_path=figures_dir / "model_feature_importance_top20.png",
            grouped_csv_path=figures_dir / "model_feature_importance_grouped.csv",
        )

        bundle = {
            "pipeline": pipeline,
            "feature_cols": list(X_train.columns),
            "cat_cols": cat_cols,
            "numeric_cols": numeric_cols,
            "target_col": TARGET_COL,
            "report_year_col": REPORT_YEAR_COL,
            "model_backend": model_backend,
            "target_encoding": encoder_payload,
        }
        joblib.dump(bundle, artifact_path)

        metadata = {
            "target": TARGET_COL,
            "prediction_note": "This model predicts assessed land value (CURRENT_LAND_VALUE), not guaranteed sale price.",
            "feature_names_used": list(X_train.columns),
            "n_features_total": X_train.shape[1],
            "train_year_rule": "REPORT_YEAR < 2024",
            "test_year_rule": "REPORT_YEAR >= 2024",
            "default_report_year": int(df[REPORT_YEAR_COL].max()),
            "model_backend": model_backend,
            "artifact_path": str(artifact_path),
            "target_encoding": {
                "enabled": True,
                "mode": encoding_mode,
                "target": "log1p_current_land_value",
                "columns": encoded_cols,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[train_model] Saved model artifact: {artifact_path}")
        print(f"[train_model] Saved model metadata: {metadata_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate final model on merged model table.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/model_table.parquet",
        help="Path to merged model table parquet",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Optional fraction to sample from train/test for faster runs",
    )
    parser.add_argument(
        "--artifact_path",
        type=str,
        default="artifacts/land_value_model.joblib",
        help="Path to save trained inference artifact",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="artifacts/model_metadata.json",
        help="Path to save model metadata JSON",
    )
    parser.add_argument(
        "--encoding_mode",
        type=str,
        default="b",
        choices=["a", "b"],
        help=(
            "Target encoding mode: 'b' (default) replaces high-cardinality raw columns "
            "with train-only target-encoded columns; 'a' keeps both."
        ),
    )
    args = parser.parse_args()

    train_and_evaluate(
        Path(args.data_path),
        artifact_path=Path(args.artifact_path),
        metadata_path=Path(args.metadata_path),
        sample_frac=args.sample_frac,
        save_outputs=True,
        encoding_mode=args.encoding_mode,
    )


if __name__ == "__main__":
    main()
