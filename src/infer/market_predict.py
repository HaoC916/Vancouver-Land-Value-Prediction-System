"""Feature-driven market list-price predictor.

Given a few property facts (type, bedrooms, bathrooms, floor area, area/neighbourhood),
estimate the market list price using the segmented bundle from train_market_model.py:
a per-segment (detached / attached) price-per-sqft model with a shared direct fallback.
Needs only the artifact — geography metadata (subarea→area, centroids, area history) and
inference defaults are baked in. Covers Greater Vancouver + the Fraser Valley.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("artifacts/market_price_model.joblib")
REFERENCE_YEAR = 2025

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
    method: str


class MarketPricePredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Market model not found at {model_path}. Run `python -m src.models.train_market_model`.")
        b = joblib.load(model_path)
        self.segments = b["segments"]  # {seg: {ppsf, direct, encoders}}
        self.detached_types = set(b.get("detached_types", []))
        self.default_segment = b.get("default_segment", "attached")
        self.sqft_col = b.get("sqft_col", "sqft_best")
        self.feature_cols = list(b["feature_cols"])
        self.te_cols = list(b.get("high_card_cols", []))
        inf = b.get("inference", {})
        self.numeric_defaults = inf.get("numeric_defaults", {})
        self.categorical_defaults = inf.get("categorical_defaults", {})
        self.region_area_history = inf.get("region_area_history", {})
        self.subarea_to_area = inf.get("subarea_to_area", {})
        self.area_listing_count = inf.get("area_listing_count", {})
        self.known_areas = inf.get("known_areas", [])
        self.known_subareas = inf.get("known_subareas", [])
        self.default_list_month = int(inf.get("default_list_month", 6))
        self.median_ape = float(inf.get("median_ape", 0.10))
        self.base_cols = [c for c in self.feature_cols if not c.endswith("_te")]

    # --- helpers ---
    def _te(self, specs: dict, source_col: str, raw_value: Any) -> float:
        spec = specs.get(source_col)
        if not spec:
            return np.nan
        key = "Unknown" if raw_value in (None, "") else str(raw_value).strip()
        val = spec.get("mapping", {}).get(key, spec.get("global_mean", np.nan))
        return float(val) if pd.notna(val) else np.nan

    @staticmethod
    def _num(value: Any) -> float:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) else np.nan

    def resolve(self, area_name: str | None) -> tuple[str | None, str | None]:
        """Resolve a user area/neighbourhood name to (area, subarea).

        Order matters: an EXACT match wins first (so "Surrey"/"Richmond" resolve to the
        board area, not a loosely-matching subarea like "Grandview Surrey" that maps to a
        pricier parent). Only then do we fall back to a substring area (highest-volume when
        ambiguous, e.g. "Burnaby" -> Burnaby South) and finally a substring subarea (so a
        named neighbourhood like "Metrotown" or "Burnaby Metrotown" still resolves).
        """
        if not area_name:
            return None, None
        needle = str(area_name).strip().lower()

        for s in self.known_subareas:            # 1. exact subarea (e.g. Metrotown, Yaletown)
            if s.lower() == needle:
                return self.subarea_to_area.get(s), s
        for a in self.known_areas:               # 2. exact area (e.g. Surrey, Richmond)
            if a.lower() == needle:
                return a, None
        area_hits = [a for a in self.known_areas if needle in a.lower() or a.lower() in needle]
        if area_hits:                            # 3. substring area -> highest volume
            return max(area_hits, key=lambda a: self.area_listing_count.get(a, 0)), None
        sub_hits = [s for s in self.known_subareas if s.lower() in needle or needle in s.lower()]
        if sub_hits:                             # 4. substring subarea (prefer one inside the query)
            contained = [s for s in sub_hits if s.lower() in needle]
            pick = max(contained or sub_hits, key=len)
            return self.subarea_to_area.get(pick), pick
        return None, None

    def neighbourhoods_in(self, area_query: str | None) -> dict:
        """List the known neighbourhoods (subareas) within a city/area, grouped by the
        board area. Lets the agent offer the user concrete neighbourhood choices."""
        q = str(area_query or "").strip().lower()
        areas = [a for a in self.known_areas if q and (q in a.lower() or a.lower() in q)]
        by_area: dict[str, list[str]] = {}
        for sub, parent in self.subarea_to_area.items():
            if parent in areas:
                by_area.setdefault(parent, []).append(sub)
        return {"matched_areas": areas, "neighbourhoods": {a: sorted(v)[:30] for a, v in by_area.items()}}

    def _clamp(self, rate: float) -> float:
        return min(max(rate, 0.08), 0.30)

    def predict(self, user: dict[str, Any]) -> MarketEstimate:
        ptype_raw = str(user.get("property_type", "") or "").strip().lower()
        p_propertytype = PTYPE_MAP.get(ptype_raw, "Residential Attached")
        segment = "detached" if p_propertytype in self.detached_types else "attached"
        seg = (self.segments.get(segment) or self.segments.get(self.default_segment)
               or next(iter(self.segments.values())))

        region, subarea = self.resolve(user.get("area_name"))
        postal = str(user.get("postal_code", "") or "").upper().replace(" ", "")
        fsa = postal[:3] if len(postal) >= 3 else None

        beds, baths = self._num(user.get("bedrooms")), self._num(user.get("bathrooms"))
        sqft, year_built = self._num(user.get("floor_area_sqft")), self._num(user.get("year_built"))
        if not (pd.notna(sqft) and 200 <= sqft <= 20000):
            sqft = np.nan

        row: dict[str, Any] = {c: np.nan for c in self.base_cols}
        if "p_propertytype" in row:
            row["p_propertytype"] = p_propertytype
        if "p_totalbed" in row and pd.notna(beds):
            row["p_totalbed"] = beds
        if "p_totalbath" in row and pd.notna(baths):
            row["p_totalbath"] = baths
        if pd.notna(sqft):
            for c in ("sqft_best", "p_totalfloorarea", "p_grandtotalfloorarea", "p_floorareamain"):
                if c in row:
                    row[c] = sqft
        if "property_age" in row and pd.notna(year_built):
            age = REFERENCE_YEAR - year_built
            row["property_age"] = age if 0 <= age <= 250 else np.nan
        if "list_month" in row:
            row["list_month"] = self.default_list_month
        if "prop_has_prev_listing" in row:
            row["prop_has_prev_listing"] = 0

        # area history
        for c, v in self.region_area_history.get(region, {}).items():
            if c in row:
                row[c] = v
        # fill remaining unknowns from baked defaults
        for c in self.base_cols:
            if not isinstance(row.get(c), str) and pd.isna(row.get(c, np.nan)):
                if c in self.numeric_defaults:
                    row[c] = self.numeric_defaults[c]
                elif c in self.categorical_defaults:
                    row[c] = self.categorical_defaults[c]

        X = pd.DataFrame([row])
        X["region_name"], X["subarea_name"], X["postal_fsa"] = region, subarea, fsa
        for src in self.te_cols:
            X[f"{src}_te"] = self._te(seg["encoders"], src, X[src].iloc[0])
        X = X.reindex(columns=self.feature_cols)

        ppsf = seg.get("ppsf")
        if ppsf is not None and pd.notna(sqft):
            point = float(np.exp(ppsf.predict(X)[0]) * sqft)
            method = "price_per_sqft"
        else:
            point = float(np.expm1(seg["direct"].predict(X)[0]))
            method = "direct"
        point = max(0.0, point)

        band = point * self._clamp(self.median_ape)
        return MarketEstimate(
            point_estimate=point, lower_bound=max(0.0, point - band), upper_bound=point + band,
            error_band=band, method=method,
            used_features={"property_type": p_propertytype, "segment": segment,
                           "bedrooms": beds if pd.notna(beds) else None,
                           "bathrooms": baths if pd.notna(baths) else None,
                           "floor_area_sqft": sqft if pd.notna(sqft) else None,
                           "area_name": region, "subarea": subarea,
                           "year_built": year_built if pd.notna(year_built) else None})
