"""Phase 1 — train the market-price model (predicts list price).

Separate from train_model.py (the assessment-value model) so the deployed pipeline
stays untouched. Reuses the shared train-only target encoders and metrics.

Two design choices that materially cut error over a plain price model:
  * price-per-sqft target — model log(price / floor_area) where floor area is known
    (~97% of rows), which removes the size scale so the model can focus on location
    and quality; a direct log(price) model is the fallback for rows without area.
  * tuned LightGBM (falls back to HistGradientBoosting if LightGBM is unavailable).

Forward-in-time split (train earlier years, test the latest) so the reported error
reflects genuine out-of-time generalization.

Run:
    python -m src.models.train_market_model
"""
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

from src.eval.encoding_utils import apply_target_encoders, fit_target_encoders
from src.eval.feature_importance import save_feature_importance
from src.eval.metrics import (
    compute_metrics,
    neighbourhood_summary,
    prediction_sanity_stats,
    scale_warning,
)

TARGET_COL = "listprice"
YEAR_COL = "list_year"
SQFT_COL = "sqft_best"
HIGH_CARD_COLS = ["region_name", "postal_fsa"]

TUNED = dict(n_estimators=1500, learning_rate=0.03, num_leaves=127,
             min_child_samples=40, subsample=0.8, colsample_bytree=0.8)


def _make_model():
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        return "lightgbm", LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **TUNED)
    except Exception:
        return "hist_gradient_boosting", HistGradientBoostingRegressor(
            random_state=42, max_iter=TUNED["n_estimators"], learning_rate=TUNED["learning_rate"])


def _build_pipeline(cat_cols, numeric_cols) -> Pipeline:
    transformers = []
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    _, model = _make_model()
    return Pipeline([("preprocess", pre), ("model", model)])


def train(data_path: Path, split_year: int, save: bool) -> dict:
    df = pd.read_parquet(data_path).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, YEAR_COL])
    df = df[df[TARGET_COL] > 0].copy()

    feature_cols = [c for c in df.columns if c not in {TARGET_COL, YEAR_COL}]
    train_df = df[df[YEAR_COL] < split_year].copy()
    test_df = df[df[YEAR_COL] >= split_year].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty split at year {split_year}: train={len(train_df)}, test={len(test_df)}")
    backend, _ = _make_model()
    print(f"[market_model] train={len(train_df):,} (<{split_year})  test={len(test_df):,} (>={split_year})  backend={backend}")

    X_train_raw, y_train = train_df[feature_cols].copy(), train_df[TARGET_COL].copy()
    X_test_raw, y_test = test_df[feature_cols].copy(), test_df[TARGET_COL].copy()

    # Train-only target encoding on high-cardinality geography (log target for stability).
    y_train_log = np.log1p(y_train)
    high_card = [c for c in HIGH_CARD_COLS if c in X_train_raw.columns]
    encoders = fit_target_encoders(X_train_raw, y_train_log, high_card)
    X_train = apply_target_encoders(X_train_raw, encoders).drop(columns=high_card)
    X_test = apply_target_encoders(X_test_raw, encoders).drop(columns=high_card)

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    sqft_tr = pd.to_numeric(train_df[SQFT_COL], errors="coerce").values
    sqft_te = pd.to_numeric(test_df[SQFT_COL], errors="coerce").values
    m_tr, m_te = ~np.isnan(sqft_tr), ~np.isnan(sqft_te)

    # Direct log(price) model — trained on all rows, used as fallback.
    direct = _build_pipeline(cat_cols, numeric_cols)
    direct.fit(X_train, y_train_log)
    y_pred = np.expm1(direct.predict(X_test))

    # Price-per-sqft model — natural log(price / sqft) where floor area is known.
    ppsf = None
    if m_tr.sum() > 1000:
        ppsf = _build_pipeline(cat_cols, numeric_cols)
        ppsf.fit(X_train[m_tr], np.log(y_train.values[m_tr] / sqft_tr[m_tr]))
        y_pred[m_te] = np.exp(ppsf.predict(X_test[m_te])) * sqft_te[m_te]
    y_pred = np.maximum(y_pred, 0.0)

    _, y_pred_stats = prediction_sanity_stats(y_test, y_pred)
    w = scale_warning(*prediction_sanity_stats(y_test, y_pred))
    if w:
        print(w)
    metrics = compute_metrics(y_test, y_pred, y_train)
    print(f"[market_model] Test Median APE: {metrics['median_ape']:.4f}   (sqft coverage {m_te.mean()*100:.1f}%)")
    print(f"[market_model] Test MAE: {metrics['mae']:,.0f}   RMSE: {metrics['rmse']:,.0f}")
    print(f"[market_model] Robust MAE: {metrics['robust_mae']:,.0f}   Robust RMSE: {metrics['robust_rmse']:,.0f}")

    result = {
        "median_ape": float(metrics["median_ape"]),
        "mae": float(metrics["mae"]), "rmse": float(metrics["rmse"]),
        "robust_mae": float(metrics["robust_mae"]), "robust_rmse": float(metrics["robust_rmse"]),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_features": int(X_train.shape[1]), "model_backend": backend,
        "split_year": split_year, "sqft_coverage_test": float(m_te.mean()),
        "uses_price_per_sqft": ppsf is not None,
    }

    if save:
        figs = Path("reports/figures"); figs.mkdir(parents=True, exist_ok=True)
        art = Path("artifacts/market_price_model.joblib")
        meta = Path("artifacts/market_model_metadata.json")
        art.parent.mkdir(parents=True, exist_ok=True)

        neighbourhood_summary(y_test.values, y_pred, test_df["region_name"].values).rename(
            columns={"NEIGHBOURHOOD_CODE": "region_name"}).to_csv(figs / "market_region_error.csv", index=False)
        pd.DataFrame([{"model": f"market_{backend}", **result}]).to_csv(figs / "market_model_metrics.csv", index=False)
        try:
            imp_model = ppsf if ppsf is not None else direct
            save_feature_importance(
                pipeline=imp_model, X_test=X_test, y_test=np.log1p(y_test),
                out_csv_path=figs / "market_feature_importance.csv",
                out_png_path=figs / "market_feature_importance_top20.png",
                grouped_csv_path=figs / "market_feature_importance_grouped.csv")
        except Exception as e:  # noqa: BLE001
            print(f"[market_model] feature importance skipped: {e}")

        joblib.dump({
            "model_type": "ppsf_dual",
            "direct_pipeline": direct, "ppsf_pipeline": ppsf, "sqft_col": SQFT_COL,
            "ppsf_transform": "natural_log(price/sqft)", "direct_transform": "log1p(price)",
            "feature_cols": list(X_train.columns), "cat_cols": cat_cols, "numeric_cols": numeric_cols,
            "target_col": TARGET_COL, "year_col": YEAR_COL, "model_backend": backend,
            "target_encoding": {"cols": high_card,
                                "encoders": {c: {"mapping": encoders[c].mapping,
                                                 "global_mean": float(encoders[c].global_mean)}
                                             for c in high_card}},
        }, art)
        meta.write_text(json.dumps({
            "target": TARGET_COL,
            "prediction_note": "Predicts the market LIST price of a residential property; not a guaranteed sale price.",
            "model_type": "price-per-sqft with direct-price fallback",
            "feature_names_used": list(X_train.columns),
            "train_year_rule": f"{YEAR_COL} < {split_year}",
            "test_year_rule": f"{YEAR_COL} >= {split_year}",
            "model_backend": backend, "metrics": result,
        }, indent=2), encoding="utf-8")
        print(f"[market_model] Saved artifact: {art}")
        print(f"[market_model] Saved metadata: {meta}")

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Train the market list-price model (Phase 1).")
    p.add_argument("--data_path", default="data/interim/market_model_table.parquet")
    p.add_argument("--split_year", type=int, default=2025, help="Test = this year onward; train = before.")
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()
    train(Path(args.data_path), args.split_year, save=not args.no_save)


if __name__ == "__main__":
    main()
