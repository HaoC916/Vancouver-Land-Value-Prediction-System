from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infer.predict import LandValuePredictor

# ------------------------------------------------------------
# 1. Create FastAPI app
# ------------------------------------------------------------
# This file is the API layer.
# We keep prediction logic inside src.infer.predict.py.
# React talks to this FastAPI app, and FastAPI calls LandValuePredictor.
app = FastAPI(title="Land Value API")

# ------------------------------------------------------------
# 2. Enable CORS for React dev server
# ------------------------------------------------------------
# Vite usually runs on localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 3. Load predictor once at startup
# ------------------------------------------------------------
# This avoids reloading the model on every request.
predictor = LandValuePredictor()


# ------------------------------------------------------------
# 4. Request model for /predict
# ------------------------------------------------------------
class PredictRequest(BaseModel):
    PROPERTY_POSTAL_CODE: str
    LEGAL_TYPE: str
    ZONING_DISTRICT: str
    ZONING_CLASSIFICATION: str
    NEIGHBOURHOOD_CODE: str
    YEAR_BUILT: int
    BIG_IMPROVEMENT_YEAR: Optional[int] = None
    REPORT_YEAR: Optional[int] = None


# ------------------------------------------------------------
# 5. Small helper functions
# ------------------------------------------------------------
def normalize_text(value: str | None) -> str:
    """
    Convert text to a comparable format:
    - None -> ""
    - strip spaces
    - lower case
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_postal_code(value: str | None) -> str:
    """
    Normalize Canadian postal code:
    - uppercase
    - remove spaces / dashes
    Example:
        "V6B 1A1" -> "V6B1A1"
    """
    if value is None:
        return ""
    return str(value).strip().upper().replace(" ", "").replace("-", "")


def extract_postal_fsa(postal_code: str | None) -> str:
    """
    Extract the first 3 characters of a postal code.
    Example:
        V6B1A1 -> V6B
    """
    postal = normalize_postal_code(postal_code)
    return postal[:3] if len(postal) >= 3 else ""


def get_year_bounds() -> tuple[int, int]:
    """
    Read min/max report year from the lookup table.
    We use the lookup table because prediction logic also depends on it.
    """
    year_series = pd.to_numeric(predictor.lookup_df["REPORT_YEAR"], errors="coerce").dropna()
    min_year = int(year_series.min())
    max_year = int(year_series.max())
    return min_year, max_year


def _filter_lookup_df(
    property_postal_code: str | None,
    legal_type: str | None,
    zoning_district: str | None,
    zoning_classification: str | None,
    neighbourhood_code: str | None,
    report_year: int | None,
) -> pd.DataFrame:
    """
    Build a filtered dataframe for option suggestions.

    IMPORTANT IDEA:
    We do NOT want to show all zoning / neighbourhood options globally.
    Instead, we try to narrow the candidate rows using whatever the user
    has already provided.

    Filtering strategy:
    1. Start from the requested report year (or default report year).
    2. Try exact postal-code match first.
    3. If exact postal match has no rows, fallback to FSA match.
    4. Then further narrow by legal_type / zoning / neighbourhood
       if they are already known.

    If filtering becomes too strict and produces zero rows,
    we fallback to the year-only dataframe.
    """
    df = predictor.lookup_df.copy()

    # Use requested year if provided, else predictor default year.
    target_year = int(report_year) if report_year is not None else predictor.default_report_year
    year_df = df[df["REPORT_YEAR"].astype("Int64") == target_year].copy()

    # If that year has no rows for some reason, fallback to all rows.
    if year_df.empty:
        year_df = df.copy()

    working = year_df.copy()

    # --------------------------------------------------------
    # A. Postal-code-based filtering
    # --------------------------------------------------------
    postal = normalize_postal_code(property_postal_code)
    if postal and "PROPERTY_POSTAL_CODE" in working.columns:
        exact_postal_df = working[
            working["PROPERTY_POSTAL_CODE"].astype(str).str.upper().str.replace(" ", "", regex=False) == postal
        ].copy()

        if not exact_postal_df.empty:
            working = exact_postal_df
        else:
            # Fallback: match on POSTAL_FSA if exact postal code does not exist
            fsa = extract_postal_fsa(postal)
            if fsa and "POSTAL_FSA" in working.columns:
                fsa_df = working[
                    working["POSTAL_FSA"].astype(str).str.upper() == fsa
                ].copy()
                if not fsa_df.empty:
                    working = fsa_df

    # --------------------------------------------------------
    # B. Narrow down using already-known categorical fields
    # --------------------------------------------------------
    filter_pairs = [
        ("LEGAL_TYPE", legal_type),
        ("ZONING_DISTRICT", zoning_district),
        ("ZONING_CLASSIFICATION", zoning_classification),
        ("NEIGHBOURHOOD_CODE", neighbourhood_code),
    ]

    for column, value in filter_pairs:
        normalized_value = normalize_text(value)
        if not normalized_value or column not in working.columns:
            continue

        narrowed = working[
            working[column].astype(str).str.strip().str.lower() == normalized_value
        ].copy()

        # Only apply the filter if it still leaves rows.
        # This avoids becoming too strict too early.
        if not narrowed.empty:
            working = narrowed

    # If filtering accidentally removed everything, fallback to year-level data.
    if working.empty:
        working = year_df.copy()

    return working


def _extract_options_from_frame(frame: pd.DataFrame, column: str, top_n: int = 100) -> list[str]:
    """
    Extract a ranked list of options from a filtered dataframe.
    Ranked by frequency, most common first.
    """
    if column not in frame.columns:
        return []

    series = frame[column].dropna().astype(str).str.strip()
    if series.empty:
        return []

    counts = series.value_counts().head(top_n)
    return counts.index.tolist()


# ------------------------------------------------------------
# 6. Routes
# ------------------------------------------------------------
@app.get("/health")
def health():
    """
    Basic health check for frontend.
    Also returns year bounds so frontend can validate years.
    """
    min_year, max_year = get_year_bounds()

    return {
        "ok": True,
        "default_report_year": predictor.default_report_year,
        "min_report_year": min_year,
        "max_report_year": max_year,
    }


@app.get("/options")
def get_options(
    property_postal_code: Optional[str] = Query(default=None),
    legal_type: Optional[str] = Query(default=None),
    zoning_district: Optional[str] = Query(default=None),
    zoning_classification: Optional[str] = Query(default=None),
    neighbourhood_code: Optional[str] = Query(default=None),
    report_year: Optional[int] = Query(default=None),
):
    """
    Return context-aware options for dropdowns.

    Example:
    - If user already entered postal code, zoning options can be narrowed.
    - If user already chose zoning district, neighbourhood code can be narrowed.
    """
    filtered_df = _filter_lookup_df(
        property_postal_code=property_postal_code,
        legal_type=legal_type,
        zoning_district=zoning_district,
        zoning_classification=zoning_classification,
        neighbourhood_code=neighbourhood_code,
        report_year=report_year,
    )

    min_year, max_year = get_year_bounds()

    return {
        "LEGAL_TYPE": _extract_options_from_frame(filtered_df, "LEGAL_TYPE", top_n=50),
        "ZONING_DISTRICT": _extract_options_from_frame(filtered_df, "ZONING_DISTRICT", top_n=150),
        "ZONING_CLASSIFICATION": _extract_options_from_frame(filtered_df, "ZONING_CLASSIFICATION", top_n=150),
        "NEIGHBOURHOOD_CODE": _extract_options_from_frame(filtered_df, "NEIGHBOURHOOD_CODE", top_n=100),
        "context_row_count": int(len(filtered_df)),
        "default_report_year": predictor.default_report_year,
        "min_report_year": min_year,
        "max_report_year": max_year,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Main prediction endpoint.
    React sends user input here.
    FastAPI passes it to LandValuePredictor.
    """
    result = predictor.predict(req.model_dump())

    return {
        "point_estimate": result.point_estimate,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "error_band": result.error_band,
        "error_band_source": result.error_band_source,
        "used_features": result.used_features,
    }