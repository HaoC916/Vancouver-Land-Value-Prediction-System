"""Phase 1 — train the market list-price model (accuracy pass).

Improvements over the first cut:
  * subarea/neighbourhood granularity — target-encode subarea (e.g. Metrotown,
    Brentwood) as well as the board area, so the model can price neighbourhood premiums
  * separate models per property segment (detached vs attached/condo), since their
    price dynamics and price-per-sqft differ structurally
  * price-per-sqft target (per segment) with a shared direct-price fallback for the
    few rows without floor area

Forward-in-time split so the reported error reflects out-of-time generalization.
The trained bundle is self-contained (inference defaults + geography metadata baked in).

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
from src.eval.metrics import compute_metrics, neighbourhood_summary, prediction_sanity_stats, scale_warning

TARGET_COL = "listprice"
YEAR_COL = "list_year"
SQFT_COL = "sqft_best"
HIGH_CARD_COLS = ["subarea_name", "region_name", "postal_fsa"]

# Property segment: detached vs everything else (condo/townhouse/other).
DETACHED_TYPES = {"Residential Detached", "Single Family"}
DEFAULT_SEGMENT = "attached"

TUNED = dict(n_estimators=1000, learning_rate=0.04, num_leaves=95,
             min_child_samples=40, subsample=0.8, colsample_bytree=0.8)


def _segment_of(ptype: object) -> str:
    return "detached" if str(ptype) in DETACHED_TYPES else "attached"


def _make_model():
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        return "lightgbm", LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **TUNED)
    except Exception:
        return "hist_gradient_boosting", HistGradientBoostingRegressor(
            random_state=42, max_iter=TUNED["n_estimators"], learning_rate=TUNED["learning_rate"])


def _pipeline(cat_cols, numeric_cols) -> Pipeline:
    transformers = []
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    _, model = _make_model()
    return Pipeline([("preprocess", ColumnTransformer(transformers, remainder="drop")), ("model", model)])


def _inference_metadata(train_df: pd.DataFrame, df: pd.DataFrame, median_ape: float) -> dict:
    raw_feats = [c for c in df.columns if c not in {TARGET_COL, YEAR_COL}]
    raw_num = [c for c in raw_feats if pd.api.types.is_numeric_dtype(train_df[c])]
    raw_cat = [c for c in raw_feats if c not in raw_num]
    numeric_defaults = {c: float(pd.to_numeric(train_df[c], errors="coerce").median())
                        for c in raw_num if pd.to_numeric(train_df[c], errors="coerce").notna().any()}
    categorical_defaults = {}
    for c in raw_cat:
        mode = train_df[c].dropna().astype(str).mode()
        categorical_defaults[c] = str(mode.iloc[0]) if not mode.empty else "Unknown"

    area_hist_cols = ["area_prev_year_median_list", "area_prev_year_listing_count", "area_prev_year_growth"]
    region_area_history = {}
    for reg, g in train_df.groupby("region_name"):
        region_area_history[str(reg)] = {col: float(pd.to_numeric(g[col], errors="coerce").median())
                                         for col in area_hist_cols
                                         if col in g and pd.to_numeric(g[col], errors="coerce").notna().any()}
    # subarea -> its parent area (most common).
    sub2area = {}
    for sub, g in train_df.dropna(subset=["subarea_name"]).groupby("subarea_name"):
        parent = g["region_name"].dropna().astype(str).mode()
        sub2area[str(sub)] = str(parent.iloc[0]) if not parent.empty else None
    area_counts = {str(reg): int(len(g))
                   for reg, g in train_df.dropna(subset=["region_name"]).groupby("region_name")}
    return {
        "numeric_defaults": numeric_defaults, "categorical_defaults": categorical_defaults,
        "region_area_history": region_area_history, "subarea_to_area": sub2area,
        "area_listing_count": area_counts,
        "known_areas": sorted(area_counts.keys()),
        "known_subareas": sorted(sub2area.keys()),
        "default_list_month": int(pd.to_numeric(train_df["list_month"], errors="coerce").median()),
        "median_ape": float(median_ape),
        "user_inputs": ["property_type", "bedrooms", "bathrooms", "floor_area_sqft", "area_name", "year_built"],
    }


def train(data_path: Path, split_year: int, save: bool) -> dict:
    df = pd.read_parquet(data_path).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, YEAR_COL])
    df = df[df[TARGET_COL] > 0].copy()
    df["_segment"] = df["p_propertytype"].map(_segment_of)

    feature_cols = [c for c in df.columns if c not in {TARGET_COL, YEAR_COL, "_segment"}]
    train_df = df[df[YEAR_COL] < split_year].copy()
    test_df = df[df[YEAR_COL] >= split_year].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty split at year {split_year}")
    backend, _ = _make_model()
    print(f"[market_model] train={len(train_df):,} test={len(test_df):,}  backend={backend}")
    print(f"[market_model] segments (train): {train_df['_segment'].value_counts().to_dict()}")

    y_test = test_df[TARGET_COL].copy()
    high_card = [c for c in HIGH_CARD_COLS if c in train_df.columns]
    feature_cols_out = None

    # Fully per-segment: each segment gets its OWN target encoding (so an area's
    # detached price level and its condo price level are encoded separately) and its
    # own price-per-sqft + direct-fallback models.
    segments: dict[str, dict] = {}
    y_pred = np.zeros(len(test_df), dtype=float)
    for seg in ("detached", "attached"):
        tr = train_df[train_df["_segment"] == seg]
        te = test_df[test_df["_segment"] == seg]
        if tr.empty or te.empty:
            continue
        ytr = tr[TARGET_COL]
        enc = fit_target_encoders(tr[feature_cols], np.log1p(ytr), high_card)
        Xtr = apply_target_encoders(tr[feature_cols], enc).drop(columns=high_card)
        Xte = apply_target_encoders(te[feature_cols], enc).drop(columns=high_card)
        numeric_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
        cat_cols = [c for c in Xtr.columns if c not in numeric_cols]
        feature_cols_out = list(Xtr.columns)

        direct = _pipeline(cat_cols, numeric_cols)
        direct.fit(Xtr, np.log1p(ytr))
        pred = np.expm1(direct.predict(Xte))

        # Price-per-sqft is ideal for uniform condos/townhomes, but for detached homes
        # the size↔$/sqft relationship varies too much by area (huge West Van lots vs
        # smaller Surrey builds), which mis-orders areas at a fixed size. So detached
        # uses the direct log-price model; only the attached segment gets a ppsf model.
        sq_tr = pd.to_numeric(tr[SQFT_COL], errors="coerce").values
        sq_te = pd.to_numeric(te[SQFT_COL], errors="coerce").values
        m_tr = ~np.isnan(sq_tr)
        ppsf = None
        if seg == "attached" and m_tr.sum() > 1000:
            ppsf = _pipeline(cat_cols, numeric_cols)
            ppsf.fit(Xtr[m_tr], np.log(ytr.values[m_tr] / sq_tr[m_tr]))
            m_te = ~np.isnan(sq_te)
            pred[m_te] = np.exp(ppsf.predict(Xte[m_te])) * sq_te[m_te]
        y_pred[(test_df["_segment"] == seg).values] = np.maximum(pred, 0.0)
        segments[seg] = {
            "ppsf": ppsf, "direct": direct,
            "encoders": {c: {"mapping": enc[c].mapping, "global_mean": float(enc[c].global_mean)}
                         for c in high_card},
        }

    y_train = train_df[TARGET_COL]

    w = scale_warning(*prediction_sanity_stats(y_test, y_pred))
    if w:
        print(w)
    metrics = compute_metrics(y_test, y_pred, y_train)
    print(f"[market_model] Test Median APE: {metrics['median_ape']:.4f}   MAE: {metrics['mae']:,.0f}")
    # per-segment APE
    seg_te = test_df["_segment"].values
    for seg in ("detached", "attached"):
        m = seg_te == seg
        if m.sum():
            seg_ape = float(np.median(np.abs(y_pred[m] - y_test.values[m]) / y_test.values[m]))
            print(f"[market_model]   {seg}: n={m.sum():,}  median APE {seg_ape:.4f}")

    result = {
        "median_ape": float(metrics["median_ape"]), "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]), "robust_mae": float(metrics["robust_mae"]),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols_out or [])), "model_backend": backend, "split_year": split_year,
    }

    if save:
        figs = Path("reports/figures"); figs.mkdir(parents=True, exist_ok=True)
        art = Path("artifacts/market_price_model.joblib")
        meta = Path("artifacts/market_model_metadata.json")
        art.parent.mkdir(parents=True, exist_ok=True)
        neighbourhood_summary(y_test.values, y_pred, test_df["region_name"].values).rename(
            columns={"NEIGHBOURHOOD_CODE": "region_name"}).to_csv(figs / "market_region_error.csv", index=False)
        pd.DataFrame([{"model": f"market_{backend}", **result}]).to_csv(figs / "market_model_metrics.csv", index=False)

        joblib.dump({
            "model_type": "segmented_ppsf_v2",
            "segments": segments, "high_card_cols": high_card,
            "detached_types": sorted(DETACHED_TYPES), "default_segment": DEFAULT_SEGMENT,
            "sqft_col": SQFT_COL, "feature_cols": feature_cols_out,
            "cat_cols": cat_cols, "numeric_cols": numeric_cols,
            "target_col": TARGET_COL, "year_col": YEAR_COL, "model_backend": backend,
            "inference": _inference_metadata(train_df, df, metrics["median_ape"]),
        }, art)
        meta.write_text(json.dumps({
            "target": TARGET_COL,
            "prediction_note": "Predicts the market LIST price of a residential property; not a guaranteed sale price.",
            "model_type": "per-segment (detached / attached) target-encoding + price-per-sqft with direct fallback",
            "features": "property facts + board area + subarea (neighbourhood), geography target-encoded; coordinates excluded to avoid train/serve skew",
            "train_year_rule": f"{YEAR_COL} < {split_year}", "test_year_rule": f"{YEAR_COL} >= {split_year}",
            "model_backend": backend, "metrics": result,
        }, indent=2), encoding="utf-8")
        print(f"[market_model] Saved artifact: {art}  ({art.stat().st_size/1e6:.0f} MB)")

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Train the market list-price model (Phase 1, accuracy pass).")
    p.add_argument("--data_path", default="data/interim/market_model_table.parquet")
    p.add_argument("--split_year", type=int, default=2025)
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()
    train(Path(args.data_path), args.split_year, save=not args.no_save)


if __name__ == "__main__":
    main()
