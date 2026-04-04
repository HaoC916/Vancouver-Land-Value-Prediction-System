from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("artifacts/land_value_model.joblib")
METADATA_PATH = Path("artifacts/model_metadata.json")
LOOKUP_TABLE_PATH = Path("data/processed/model_table.parquet")
METRICS_PATH = Path("reports/figures/model_metrics.csv")
NEIGH_ERROR_PATH = Path("reports/figures/model_neighbourhood_error.csv")
POSTAL_CODE_PATTERN = re.compile(r"^[A-Z]\d[A-Z]\d[A-Z]\d$")


def _normalize_postal_code(value: Any) -> str:
    if value is None:
        return "Unknown"
    postal = re.sub(r"\s+", "", str(value).upper()).strip()
    if postal in {"", "NAN", "NONE", "NAT", "UNKNOWN"}:
        return "Unknown"
    return postal


def _normalize_category_value(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat", "unknown"}:
        return "Unknown"
    return text


def _safe_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    mode = s.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def _building_age_bin(year_built: float | None) -> str:
    if year_built is None or np.isnan(year_built):
        return "unknown"
    if year_built <= 1949:
        return "pre_1950"
    if year_built <= 1979:
        return "1950_1979"
    if year_built <= 1999:
        return "1980_1999"
    return "2000_plus"


def _improvement_recency_bin(years_since_improvement: float | None) -> str:
    if years_since_improvement is None or np.isnan(years_since_improvement):
        return "unknown"
    if years_since_improvement <= 5:
        return "0_5y"
    if years_since_improvement <= 15:
        return "6_15y"
    if years_since_improvement <= 30:
        return "16_30y"
    return "31y_plus"


@dataclass
class PredictionResult:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    error_band: float
    error_band_source: str
    used_features: dict[str, Any]


class LandValuePredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
        lookup_table_path: Path = LOOKUP_TABLE_PATH,
        metrics_path: Path = METRICS_PATH,
        neigh_error_path: Path = NEIGH_ERROR_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run `python -m src.models.train_model` first."
            )
        if not lookup_table_path.exists():
            raise FileNotFoundError(
                f"Lookup model table not found at {lookup_table_path}. "
                "Run `python -m src.data.build_model_table` first."
            )

        self.bundle = joblib.load(model_path)
        self.pipeline = self.bundle["pipeline"]
        self.feature_cols = list(self.bundle["feature_cols"])
        self.cat_cols = list(self.bundle["cat_cols"])
        self.numeric_cols = list(self.bundle["numeric_cols"])
        self.target_encoding = self.bundle.get("target_encoding", {}) or {}
        self.target_encoding_cols = list(self.target_encoding.get("cols", []))
        self.target_encoder_specs = dict(self.target_encoding.get("encoders", {}))

        self.metadata = {}
        if metadata_path.exists():
            self.metadata = pd.read_json(metadata_path, typ="series").to_dict()

        self.lookup_df = pd.read_parquet(lookup_table_path).copy()
        self.lookup_df["REPORT_YEAR"] = pd.to_numeric(
            self.lookup_df["REPORT_YEAR"], errors="coerce"
        ).astype("Int64")
        valid_years = self.lookup_df["REPORT_YEAR"].dropna().astype(int)
        self.lookup_min_year = int(valid_years.min()) if not valid_years.empty else None
        self.lookup_max_year = int(valid_years.max()) if not valid_years.empty else None

        self.metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        self.neigh_error_df = (
            pd.read_csv(neigh_error_path) if neigh_error_path.exists() else pd.DataFrame()
        )

        self.default_report_year = int(
            self.metadata.get("default_report_year", self.lookup_df["REPORT_YEAR"].dropna().max())
        )
        if "PROPERTY_POSTAL_CODE" in self.lookup_df.columns:
            known_postals = (
                self.lookup_df["PROPERTY_POSTAL_CODE"]
                .map(_normalize_postal_code)
                .dropna()
                .astype(str)
            )
            self.known_postal_codes = {
                p for p in known_postals.tolist() if p != "Unknown" and POSTAL_CODE_PATTERN.match(p)
            }
        else:
            self.known_postal_codes = set()

        self.numeric_defaults = {}
        for c in self.numeric_cols:
            if c in self.lookup_df.columns:
                self.numeric_defaults[c] = pd.to_numeric(
                    self.lookup_df[c], errors="coerce"
                ).median()
            else:
                self.numeric_defaults[c] = np.nan

        self.categorical_defaults = {}
        for c in self.cat_cols:
            if c in self.lookup_df.columns:
                self.categorical_defaults[c] = _safe_mode(self.lookup_df[c])
            else:
                self.categorical_defaults[c] = "Unknown"

    @staticmethod
    def normalize_postal_code(value: Any) -> str:
        return _normalize_postal_code(value)

    @staticmethod
    def is_valid_canadian_postal_code(value: Any) -> bool:
        normalized = _normalize_postal_code(value)
        return bool(POSTAL_CODE_PATTERN.match(normalized))

    def is_postal_code_seen(self, value: Any) -> bool:
        normalized = _normalize_postal_code(value)
        if not self.is_valid_canadian_postal_code(normalized):
            return False
        return normalized in self.known_postal_codes

    def get_top_options(self, column: str, top_n: int = 100) -> list[str]:
        if column not in self.lookup_df.columns:
            return []
        s = self.lookup_df[column].dropna().astype(str).str.strip()
        if s.empty:
            return []
        counts = s.value_counts().head(top_n)
        return counts.index.tolist()

    def _pick_numeric(self, feature: str, *frames: pd.DataFrame) -> float:
        for frame in frames:
            if frame is not None and not frame.empty and feature in frame.columns:
                values = pd.to_numeric(frame[feature], errors="coerce").dropna()
                if not values.empty:
                    return float(values.median())
        val = self.numeric_defaults.get(feature, np.nan)
        return float(val) if pd.notna(val) else np.nan

    def _pick_categorical(self, feature: str, *frames: pd.DataFrame):
        for frame in frames:
            if frame is not None and not frame.empty and feature in frame.columns:
                v = _safe_mode(frame[feature])
                if pd.notna(v):
                    return str(v)
        v = self.categorical_defaults.get(feature, "Unknown")
        if pd.isna(v):
            return "Unknown"
        return str(v)

    def _target_encoded_value(self, source_col: str, raw_value: Any) -> float:
        spec = self.target_encoder_specs.get(source_col)
        if not spec:
            return np.nan

        mapping = spec.get("mapping", {})
        global_mean = float(spec.get("global_mean", np.nan))
        if source_col == "PROPERTY_POSTAL_CODE":
            key = _normalize_postal_code(raw_value)
        else:
            key = _normalize_category_value(raw_value)

        value = mapping.get(key, global_mean)
        return float(value) if pd.notna(value) else np.nan

    def _build_row(self, user_input: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        required = [
            "PROPERTY_POSTAL_CODE",
            "LEGAL_TYPE",
            "ZONING_DISTRICT",
            "ZONING_CLASSIFICATION",
            "NEIGHBOURHOOD_CODE",
            "YEAR_BUILT",
        ]
        missing = [k for k in required if not user_input.get(k)]
        if missing:
            raise ValueError(f"Missing required input field(s): {missing}")

        report_year = int(user_input.get("REPORT_YEAR") or self.default_report_year)
        postal = _normalize_postal_code(user_input.get("PROPERTY_POSTAL_CODE"))
        if not self.is_valid_canadian_postal_code(postal):
            raise ValueError(
                "Please enter a valid Canadian postal code (example: V6H2J4 or V6H 2J4)."
            )
        postal_fsa = postal[:3] if postal != "Unknown" and len(postal) >= 3 else "Unknown"
        if postal_fsa != "Unknown" and re.match(r"^[A-Z]\d[A-Z]$", postal_fsa) is None:
            postal_fsa = "Unknown"

        year_built = pd.to_numeric(pd.Series([user_input.get("YEAR_BUILT")]), errors="coerce").iloc[0]
        year_built = float(year_built) if pd.notna(year_built) else np.nan

        big_imp = pd.to_numeric(
            pd.Series([user_input.get("BIG_IMPROVEMENT_YEAR")]), errors="coerce"
        ).iloc[0]
        big_imp = float(big_imp) if pd.notna(big_imp) else np.nan

        property_age = report_year - year_built if pd.notna(year_built) else np.nan
        if pd.notna(property_age) and (property_age < 0 or property_age > 250):
            property_age = np.nan

        years_since_imp = report_year - big_imp if pd.notna(big_imp) else np.nan
        if pd.notna(years_since_imp) and (years_since_imp < 0 or years_since_imp > 250):
            years_since_imp = np.nan

        has_big_improvement = 1 if pd.notna(big_imp) else 0
        building_age_bin = _building_age_bin(year_built)
        imp_recency_bin = _improvement_recency_bin(years_since_imp)

        legal_type = _normalize_category_value(user_input["LEGAL_TYPE"])
        zoning_class = _normalize_category_value(user_input["ZONING_CLASSIFICATION"])
        legal_zoning_combo = f"{legal_type}__{zoning_class}"

        year_df = self.lookup_df[self.lookup_df["REPORT_YEAR"] == report_year]
        neigh = _normalize_category_value(user_input["NEIGHBOURHOOD_CODE"])
        neigh_df = year_df[year_df["NEIGHBOURHOOD_CODE"].astype(str) == neigh] if not year_df.empty else pd.DataFrame()
        fsa_df = year_df[year_df.get("POSTAL_FSA", pd.Series(dtype=str)).astype(str) == postal_fsa] if not year_df.empty and "POSTAL_FSA" in year_df.columns else pd.DataFrame()

        row: dict[str, Any] = {}
        for feature in self.feature_cols:
            row[feature] = np.nan

        # Direct + derived user-visible fields.
        direct_values = {
            "PROPERTY_POSTAL_CODE": postal,
            "LEGAL_TYPE": legal_type,
            "ZONING_DISTRICT": _normalize_category_value(user_input["ZONING_DISTRICT"]),
            "ZONING_CLASSIFICATION": zoning_class,
            "NEIGHBOURHOOD_CODE": neigh,
            "YEAR_BUILT": year_built,
            "BIG_IMPROVEMENT_YEAR": big_imp,
            "REPORT_YEAR_NUM": report_year,
            "REPORT_YEAR_CENTERED": report_year - 2023,
            "PROPERTY_AGE": property_age,
            "YEARS_SINCE_IMPROVEMENT": years_since_imp,
            "HAS_BIG_IMPROVEMENT": has_big_improvement,
            "POSTAL_FSA": postal_fsa,
            "BUILDING_AGE_BIN": building_age_bin,
            "IMPROVEMENT_RECENCY_BIN": imp_recency_bin,
            "LEGAL_ZONING_COMBO": legal_zoning_combo,
        }
        for k, v in direct_values.items():
            if k in row:
                row[k] = v

        # Populate train-only target-encoded features when the model expects them.
        for source_col in self.target_encoding_cols:
            te_col = f"{source_col}_te"
            if te_col not in row:
                continue
            source_val = direct_values.get(source_col, user_input.get(source_col))
            row[te_col] = self._target_encoded_value(source_col, source_val)

        # Fill remaining features from lookup tables/defaults.
        for feature in self.feature_cols:
            if pd.notna(row.get(feature, np.nan)):
                continue

            if feature in self.numeric_cols:
                if feature.startswith("neigh_"):
                    row[feature] = self._pick_numeric(feature, neigh_df, year_df, self.lookup_df)
                elif feature.startswith("fsa_"):
                    row[feature] = self._pick_numeric(feature, fsa_df, year_df, self.lookup_df)
                else:
                    row[feature] = self._pick_numeric(feature, year_df, self.lookup_df)
            else:
                if feature.startswith("neigh_"):
                    row[feature] = self._pick_categorical(feature, neigh_df, year_df, self.lookup_df)
                elif feature.startswith("fsa_"):
                    row[feature] = self._pick_categorical(feature, fsa_df, year_df, self.lookup_df)
                else:
                    row[feature] = self._pick_categorical(feature, year_df, self.lookup_df)

        X = pd.DataFrame([row], columns=self.feature_cols)
        used = {
            "REPORT_YEAR": report_year,
            "POSTAL_FSA": postal_fsa,
            "NEIGHBOURHOOD_CODE": neigh,
        }
        if self.lookup_min_year is not None and self.lookup_max_year is not None:
            out_of_lookup_range = report_year < self.lookup_min_year or report_year > self.lookup_max_year
            used["LOOKUP_SUPPORTED_YEAR_RANGE"] = (
                f"{self.lookup_min_year}-{self.lookup_max_year}"
            )
            used["LOOKUP_YEAR_FALLBACK"] = out_of_lookup_range
        return X, used

    def _error_band(self, prediction: float, neighbourhood_code: str) -> tuple[float, str]:
        if not self.neigh_error_df.empty and {"NEIGHBOURHOOD_CODE", "mean_abs_error"}.issubset(
            self.neigh_error_df.columns
        ):
            match = self.neigh_error_df[
                self.neigh_error_df["NEIGHBOURHOOD_CODE"].astype(str) == str(neighbourhood_code)
            ]
            if not match.empty:
                val = pd.to_numeric(match["mean_abs_error"], errors="coerce").iloc[0]
                if pd.notna(val):
                    return float(val), "neighbourhood_mean_abs_error"

        if not self.metrics_df.empty:
            for col in ["robust_mae", "mae"]:
                if col in self.metrics_df.columns:
                    val = pd.to_numeric(self.metrics_df[col], errors="coerce").iloc[0]
                    if pd.notna(val):
                        return float(val), f"global_{col}"

        fallback = max(100000.0, prediction * 0.20)
        return float(fallback), "fallback_20pct"

    def predict(self, user_input: dict[str, Any]) -> PredictionResult:
        X, used = self._build_row(user_input)
        pred_log = float(self.pipeline.predict(X)[0])
        point = max(0.0, float(np.expm1(pred_log)))

        band, source = self._error_band(point, str(user_input["NEIGHBOURHOOD_CODE"]).strip())
        lower = max(0.0, point - band)
        upper = point + band

        return PredictionResult(
            point_estimate=point,
            lower_bound=lower,
            upper_bound=upper,
            error_band=band,
            error_band_source=source,
            used_features=used,
        )
