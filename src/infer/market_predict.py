"""Feature-driven market list-price predictor.

Given a few property facts (type, bedrooms, bathrooms, floor area, area, year built),
estimate the market list price using the dual-model bundle from train_market_model.py.
Needs only the artifact — inference defaults (medians/modes + per-area history) are
baked into the bundle, so no large lookup is shipped. Covers the whole training
footprint (Greater Vancouver + Fraser Valley), not just addresses in one dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("artifacts/market_price_model.joblib")
REFERENCE_YEAR = 2025  # latest data year; used to derive property age from year built.

# User-friendly property types -> the source's property-type values.
PTYPE_MAP = {
    "house": "Residential Detached", "detached": "Residential Detached",
    "single family": "Residential Detached", "single-family": "Residential Detached",
    "condo": "Residential Attached", "apartment": "Residential Attached",
    "apt": "Residential Attached", "townhouse": "Residential Attached",
    "townhome": "Residential Attached", "attached": "Residential Attached",
}


@dataclass
class MarketEstimate:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    error_band: float
    used_features: dict[str, Any]
    method: str  # "price_per_sqft" or "direct"


class MarketPricePredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Market model not found at {model_path}. Run `python -m src.models.train_market_model`.")
        b = joblib.load(model_path)
        self.direct = b["direct_pipeline"]
        self.ppsf = b.get("ppsf_pipeline")
        self.sqft_col = b.get("sqft_col", "sqft_best")
        self.feature_cols = list(b["feature_cols"])
        self.te_cols = list(b.get("target_encoding", {}).get("cols", []))
        self.te_specs = dict(b.get("target_encoding", {}).get("encoders", {}))
        inf = b.get("inference", {})
        self.numeric_defaults = inf.get("numeric_defaults", {})
        self.categorical_defaults = inf.get("categorical_defaults", {})
        self.region_area_history = inf.get("region_area_history", {})
        self.default_list_month = int(inf.get("default_list_month", 6))
        self.median_ape = float(inf.get("median_ape", 0.13))
        self.known_areas = sorted(self.region_area_history.keys())
        # Raw (pre-encoding) feature columns the model expects, plus the raw
        # geography columns that target encoding consumes.
        self.base_cols = [c for c in self.feature_cols if not c.endswith("_te")]

    def _te(self, source_col: str, raw_value: Any) -> float:
        spec = self.te_specs.get(source_col)
        if not spec:
            return np.nan
        key = "Unknown" if raw_value is None else str(raw_value).strip()
        val = spec.get("mapping", {}).get(key, spec.get("global_mean", np.nan))
        return float(val) if pd.notna(val) else np.nan

    @staticmethod
    def _num(value: Any) -> float:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) else np.nan

    def resolve_area(self, area_name: str | None) -> str | None:
        """Best-effort match of a user area name to a known model area."""
        if not area_name:
            return None
        needle = str(area_name).strip().lower()
        exact = [a for a in self.known_areas if a.lower() == needle]
        if exact:
            return exact[0]
        contains = [a for a in self.known_areas if needle in a.lower() or a.lower() in needle]
        return contains[0] if contains else None

    def _clamp(self, rate: float) -> float:
        return min(max(rate, 0.10), 0.30)

    def predict(self, user: dict[str, Any]) -> MarketEstimate:
        row: dict[str, Any] = {c: np.nan for c in self.base_cols}

        # --- user-provided facts ---
        ptype_raw = str(user.get("property_type", "") or "").strip().lower()
        p_propertytype = PTYPE_MAP.get(ptype_raw, "Residential Attached")
        if "p_propertytype" in row:
            row["p_propertytype"] = p_propertytype

        beds = self._num(user.get("bedrooms"))
        baths = self._num(user.get("bathrooms"))
        sqft = self._num(user.get("floor_area_sqft"))
        year_built = self._num(user.get("year_built"))

        if "p_totalbed" in row and pd.notna(beds):
            row["p_totalbed"] = beds
        if "p_totalbath" in row and pd.notna(baths):
            row["p_totalbath"] = baths
        if pd.notna(sqft) and 200 <= sqft <= 20000:
            for c in ("sqft_best", "p_totalfloorarea", "p_grandtotalfloorarea", "p_floorareamain"):
                if c in row:
                    row[c] = sqft
        else:
            sqft = np.nan
        if "property_age" in row and pd.notna(year_built):
            age = REFERENCE_YEAR - year_built
            row["property_age"] = age if 0 <= age <= 250 else np.nan
        if "list_month" in row:
            row["list_month"] = self.default_list_month
        if "prop_has_prev_listing" in row:
            row["prop_has_prev_listing"] = 0

        # --- geography + per-area history ---
        area = self.resolve_area(user.get("area_name"))
        postal = str(user.get("postal_code", "") or "").upper().replace(" ", "")
        fsa = postal[:3] if len(postal) >= 3 else "Unknown"
        hist = self.region_area_history.get(area, {}) if area else {}
        for c, v in hist.items():
            if c in row:
                row[c] = v

        # --- fill remaining unknowns from baked defaults ---
        for c in self.base_cols:
            if pd.isna(row.get(c, np.nan)) if not isinstance(row.get(c), str) else False:
                if c in self.numeric_defaults:
                    row[c] = self.numeric_defaults[c]
                elif c in self.categorical_defaults:
                    row[c] = self.categorical_defaults[c]

        # --- target-encode geography, assemble model frame ---
        X = pd.DataFrame([row])
        X["region_name"] = area
        X["postal_fsa"] = fsa if fsa != "Unknown" else None
        for src in self.te_cols:
            X[f"{src}_te"] = self._te(src, X[src].iloc[0])
        X = X.reindex(columns=self.feature_cols)

        # --- predict: price-per-sqft where floor area is known, else direct ---
        if self.ppsf is not None and pd.notna(sqft):
            point = float(np.exp(self.ppsf.predict(X)[0]) * sqft)
            method = "price_per_sqft"
        else:
            point = float(np.expm1(self.direct.predict(X)[0]))
            method = "direct"
        point = max(0.0, point)

        band = point * self._clamp(self.median_ape)
        return MarketEstimate(
            point_estimate=point, lower_bound=max(0.0, point - band), upper_bound=point + band,
            error_band=band,
            used_features={"property_type": p_propertytype,
                           "bedrooms": beds if pd.notna(beds) else None,
                           "bathrooms": baths if pd.notna(baths) else None,
                           "floor_area_sqft": sqft if pd.notna(sqft) else None,
                           "area_name": area, "year_built": year_built if pd.notna(year_built) else None},
            method=method)
