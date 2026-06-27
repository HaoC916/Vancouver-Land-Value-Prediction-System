from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import re

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infer.predict import LandValuePredictor

# ------------------------------------------------------------
# 1. Create FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Land Value API")

# ------------------------------------------------------------
# 2. Enable CORS for the React frontend
# ------------------------------------------------------------
# Origins are configurable via the ALLOWED_ORIGINS env var (comma-separated) so
# the deployed backend can be locked to the real frontend without a code change.
# When unset, we fall back to local dev + the known Vercel domain.
_DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://vancouver-land-value-prediction-sys.vercel.app",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    or _DEFAULT_ALLOWED_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 3. Load predictor once at startup
# ------------------------------------------------------------
predictor = LandValuePredictor()

# ------------------------------------------------------------
# 4. Address lookup source
# ------------------------------------------------------------
# IMPORTANT:
# This fuzzy lookup layer does NOT change the prediction logic.
# It only helps us resolve a partial street address into the
# full feature set required by /predict.
#
# For demo simplicity, this version reads from the raw property-tax CSV.
# For future improvement, create a pre-processed table.
#ADDRESS_SOURCE_PATH = Path("data/raw/property-tax-report.csv")
ADDRESS_SOURCE_PATH = Path("data/deploy/address_lookup.parquet")


# ------------------------------------------------------------
# 5. Request model for /predict
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
# 6. General text helper functions
# ------------------------------------------------------------
def normalize_text(value: str | None) -> str:
    """
    Convert text to a comparable format:
    - None -> ""
    - trim spaces
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


def normalize_street_name(value: str | None) -> str:
    """
    Normalize street name for fuzzy matching.

    Rules:
    - uppercase
    - replace punctuation with spaces
    - collapse multiple spaces

    Example:
        "26TH Ave W" -> "26TH AVE W"
        "26th-ave. w" -> "26TH AVE W"
    """
    if value is None:
        return ""

    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------
# 7. Civic-number helper functions
# ------------------------------------------------------------
def normalize_civic_number(value: str | None) -> str:
    """
    Normalize civic number text.

    Rules:
    - None -> ""
    - trim spaces
    - uppercase
    - remove internal spaces
    - if the value looks like a float ending in .0, remove the .0

    Examples:
        " 1050 "   -> "1050"
        "1050.0"   -> "1050"
        "1050A"    -> "1050A"
        "1050.50"  -> "1050.50"
    """
    if value is None:
        return ""

    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)

    if text in {"", "NAN", "NONE"}:
        return ""

    # If the civic number looks like "1050.0", convert it to "1050"
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]

    return text


def parse_civic_number_int(value: str | None) -> int | None:
    """
    Try to extract the leading integer part from a civic number.

    Example:
        "1050" -> 1050
        "1050A" -> 1050
        "" -> None
    """
    text = normalize_civic_number(value)
    if not text:
        return None

    match = re.match(r"^(\d+)", text)
    if not match:
        return None

    return int(match.group(1))


def civic_number_matches_range(
    user_street_number: str,
    from_civic_number: str | None,
    to_civic_number: str | None,
) -> bool:
    """
    Check whether the user's street number matches one row.

    Matching rules:
    1. If both FROM and TO exist:
       treat them as an inclusive numeric range.
    2. If only one exists:
       treat that one as the exact civic number.
    3. If neither exists:
       no match.

    Example:
        user=1050, from=1040, to=1060 -> True
        user=1050, from=None, to=1050 -> True
        user=1050, from=1080, to=1100 -> False
    """
    user_num = parse_civic_number_int(user_street_number)
    from_num = parse_civic_number_int(from_civic_number)
    to_num = parse_civic_number_int(to_civic_number)

    if user_num is None:
        return False

    # If both are missing, we cannot match this row
    if from_num is None and to_num is None:
        return False

    # If only one side exists, use exact equality
    if from_num is not None and to_num is None:
        return user_num == from_num

    if from_num is None and to_num is not None:
        return user_num == to_num

    # Both exist -> inclusive range match
    low = min(from_num, to_num)
    high = max(from_num, to_num)
    return low <= user_num <= high


def build_display_address(
    from_civic_number: str | None,
    to_civic_number: str | None,
    street_name: str | None,
    postal_code: str | None,
) -> str:
    """
    Build a user-friendly display address string.

    Rules:
    - If both civic values exist and are equal, show one number
    - If both exist and differ, show a range
    - If only one exists, show that one
    """
    from_text = normalize_civic_number(from_civic_number)
    to_text = normalize_civic_number(to_civic_number)
    street_text = str(street_name).strip() if street_name is not None else ""
    postal_text = str(postal_code).strip() if postal_code is not None else ""

    if from_text and to_text:
        civic_text = from_text if from_text == to_text else f"{from_text}-{to_text}"
    else:
        civic_text = from_text or to_text or "UNKNOWN"

    return f"{civic_text} {street_text}, {postal_text}".strip()


# ------------------------------------------------------------
# 8. Shared lookup helpers
# ------------------------------------------------------------
def get_year_bounds() -> tuple[int, int]:
    """
    Read min/max report year from predictor lookup table.
    """
    year_series = pd.to_numeric(
        predictor.lookup_df["REPORT_YEAR"], errors="coerce"
    ).dropna()

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
    Build a filtered dataframe for precise-mode option suggestions.

    This helper is used by the existing precise-mode frontend.
    """
    df = predictor.lookup_df.copy()

    target_year = (
        int(report_year) if report_year is not None else predictor.default_report_year
    )

    year_df = df[df["REPORT_YEAR"].astype("Int64") == target_year].copy()

    if year_df.empty:
        year_df = df.copy()

    working = year_df.copy()

    postal = normalize_postal_code(property_postal_code)
    if postal and "PROPERTY_POSTAL_CODE" in working.columns:
        exact_postal_df = working[
            working["PROPERTY_POSTAL_CODE"]
            .astype(str)
            .str.upper()
            .str.replace(" ", "", regex=False)
            == postal
        ].copy()

        if not exact_postal_df.empty:
            working = exact_postal_df
        else:
            fsa = extract_postal_fsa(postal)
            if fsa and "POSTAL_FSA" in working.columns:
                fsa_df = working[
                    working["POSTAL_FSA"].astype(str).str.upper() == fsa
                ].copy()
                if not fsa_df.empty:
                    working = fsa_df

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

        if not narrowed.empty:
            working = narrowed

    # If everything becomes empty, fall back to the year-based frame
    if working.empty:
        working = year_df.copy()

    return working


def _extract_options_from_frame(
    frame: pd.DataFrame,
    column: str,
    top_n: int = 100,
) -> list[str]:
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
# 9. Fuzzy lookup source loading
# ------------------------------------------------------------
def load_address_lookup_df() -> pd.DataFrame:
    """
    Load address-level property rows used by fuzzy lookup.

    IMPORTANT:
    This version uses FROM_CIVIC_NUMBER / TO_CIVIC_NUMBER
    instead of STREET_NUMBER, because the raw source does
    not contain STREET_NUMBER.

    The frontend still sends one street number, but backend
    now matches that number against the civic-number range.
    """
    if not ADDRESS_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Address source file not found: {ADDRESS_SOURCE_PATH}"
        )

    address_keep_cols = [
        "PID",
        "REPORT_YEAR",
        "PROPERTY_POSTAL_CODE",
        "FROM_CIVIC_NUMBER",
        "TO_CIVIC_NUMBER",
        "STREET_NAME",
        "LEGAL_TYPE",
        "ZONING_DISTRICT",
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
        "YEAR_BUILT",
        "BIG_IMPROVEMENT_YEAR",
    ]

    # Read only needed columns to reduce memory usage
    if ADDRESS_SOURCE_PATH.suffix.lower() == ".parquet":
        df = pd.read_parquet(ADDRESS_SOURCE_PATH).copy()
        df = df[[c for c in address_keep_cols if c in df.columns]].copy()
    else:
        df = pd.read_csv(
            ADDRESS_SOURCE_PATH,
            sep=";",
            low_memory=False,
            usecols=lambda c: c in address_keep_cols,
        ).copy()

    # For civic number, accept:
    # - FROM only
    # - TO only
    # - or both
    required_min_cols = [
        "REPORT_YEAR",
        "PROPERTY_POSTAL_CODE",
        "STREET_NAME",
        "LEGAL_TYPE",
        "ZONING_DISTRICT",
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
        "YEAR_BUILT",
    ]
    missing = [c for c in required_min_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Address lookup source is missing required columns: {missing}"
        )

    has_from = "FROM_CIVIC_NUMBER" in df.columns
    has_to = "TO_CIVIC_NUMBER" in df.columns

    if not has_from and not has_to:
        raise ValueError(
            "Address lookup source must contain at least one of: "
            "FROM_CIVIC_NUMBER, TO_CIVIC_NUMBER"
        )

    if not has_from:
        df["FROM_CIVIC_NUMBER"] = pd.NA
    if not has_to:
        df["TO_CIVIC_NUMBER"] = pd.NA

    # Numeric cleanup
    df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
    df["YEAR_BUILT"] = pd.to_numeric(df["YEAR_BUILT"], errors="coerce")

    if "BIG_IMPROVEMENT_YEAR" in df.columns:
        df["BIG_IMPROVEMENT_YEAR"] = pd.to_numeric(
            df["BIG_IMPROVEMENT_YEAR"], errors="coerce"
        )
    else:
        df["BIG_IMPROVEMENT_YEAR"] = pd.NA

    # Text cleanup for standard text fields
    for col in [
        "PROPERTY_POSTAL_CODE",
        "STREET_NAME",
        "LEGAL_TYPE",
        "ZONING_DISTRICT",
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Civic-number cleanup uses the dedicated normalizer
    df["FROM_CIVIC_NUMBER"] = df["FROM_CIVIC_NUMBER"].map(normalize_civic_number)
    df["TO_CIVIC_NUMBER"] = df["TO_CIVIC_NUMBER"].map(normalize_civic_number)

    # Remove obviously unusable rows
    df = df.dropna(subset=["REPORT_YEAR", "YEAR_BUILT"]).copy()
    df = df[df["PROPERTY_POSTAL_CODE"].str.len() > 0].copy()
    df = df[df["STREET_NAME"].str.len() > 0].copy()

    # Build normalized helper columns
    df["POSTAL_CODE_NORMALIZED"] = df["PROPERTY_POSTAL_CODE"].map(normalize_postal_code)
    df["POSTAL_FSA"] = df["POSTAL_CODE_NORMALIZED"].str[:3]

    df["FROM_CIVIC_NUMBER_NORMALIZED"] = df["FROM_CIVIC_NUMBER"].map(normalize_civic_number)
    df["TO_CIVIC_NUMBER_NORMALIZED"] = df["TO_CIVIC_NUMBER"].map(normalize_civic_number)
    df["STREET_NAME_NORMALIZED"] = df["STREET_NAME"].map(normalize_street_name)

    # Precompute the leading integer of each civic number once, so the per-request
    # civic-range match can run vectorized instead of a row-wise apply over the
    # whole address lookup table.
    df["FROM_CIVIC_NUM_INT"] = pd.to_numeric(
        df["FROM_CIVIC_NUMBER_NORMALIZED"].str.extract(r"^(\d+)", expand=False),
        errors="coerce",
    )
    df["TO_CIVIC_NUM_INT"] = pd.to_numeric(
        df["TO_CIVIC_NUMBER_NORMALIZED"].str.extract(r"^(\d+)", expand=False),
        errors="coerce",
    )

    # Keep only rows that have at least one civic value
    df = df[
        (df["FROM_CIVIC_NUMBER_NORMALIZED"].str.len() > 0)
        | (df["TO_CIVIC_NUMBER_NORMALIZED"].str.len() > 0)
    ].copy()

    # Build a UI-friendly display address
    df["DISPLAY_ADDRESS"] = df.apply(
        lambda row: build_display_address(
            row.get("FROM_CIVIC_NUMBER"),
            row.get("TO_CIVIC_NUMBER"),
            row.get("STREET_NAME"),
            row.get("PROPERTY_POSTAL_CODE"),
        ),
        axis=1,
    )

    return df.reset_index(drop=True)


# Load once at startup
#address_lookup_df = load_address_lookup_df()
_address_lookup_df: pd.DataFrame | None = None

def get_address_lookup_df() -> pd.DataFrame:
    global _address_lookup_df

    if _address_lookup_df is None:
        _address_lookup_df = load_address_lookup_df()

    return _address_lookup_df


# ------------------------------------------------------------
# 10. Fuzzy lookup helper
# ------------------------------------------------------------
def fuzzy_match_address_candidates(
    street_number: str,
    street_name: str,
    property_postal_code: str | None,
    report_year: int | None,
    limit: int = 10,
) -> tuple[pd.DataFrame, int, str]:
    """
    Match address candidates from the address lookup dataframe.

    Strategy:
    1. Start with one report year.
    2. Filter rows whose civic-number range contains the user street number.
    3. Try exact street-name match first.
    4. If no exact street-name rows, fallback to contains-based matching.
    5. If postal code is provided, try exact postal match first.
    6. If exact postal fails, fallback to FSA match.
    7. Return top candidates.

    Returns:
        matched_df, used_report_year, postal_match_mode
    """
    address_lookup_df = get_address_lookup_df()

    target_year = (
        int(report_year) if report_year is not None else predictor.default_report_year
    )

    working = address_lookup_df[
        address_lookup_df["REPORT_YEAR"].astype("Int64") == target_year
    ].copy()

    if working.empty:
        working = address_lookup_df.copy()

    used_report_year = target_year

    street_name_norm = normalize_street_name(street_name)
    postal_norm = normalize_postal_code(property_postal_code)

    # Step 1: match the user's street number against the civic range. Vectorized
    # over the precomputed integer columns; this is equivalent to applying
    # civic_number_matches_range() row by row, but avoids the per-request apply.
    user_num = parse_civic_number_int(street_number)
    if user_num is None:
        return working.iloc[0:0].copy(), used_report_year, "none"

    from_num = working["FROM_CIVIC_NUM_INT"]
    to_num = working["TO_CIVIC_NUM_INT"]
    has_from = from_num.notna()
    has_to = to_num.notna()

    pair = pd.concat([from_num, to_num], axis=1)
    low = pair.min(axis=1)
    high = pair.max(axis=1)

    match_both = has_from & has_to & (low <= user_num) & (user_num <= high)
    match_from_only = has_from & ~has_to & (from_num == user_num)
    match_to_only = ~has_from & has_to & (to_num == user_num)

    civic_mask = (match_both | match_from_only | match_to_only).fillna(False)
    working = working[civic_mask].copy()

    if working.empty:
        return working, used_report_year, "none"

    # Step 2: exact street-name match first
    exact_name_df = working[
        working["STREET_NAME_NORMALIZED"] == street_name_norm
    ].copy()

    if not exact_name_df.empty:
        working = exact_name_df
    else:
        # Step 3: fallback to contains-based fuzzy name matching
        contains_df = working[
            working["STREET_NAME_NORMALIZED"].str.contains(street_name_norm, na=False)
            | pd.Series(
                [street_name_norm in value for value in working["STREET_NAME_NORMALIZED"]],
                index=working.index,
            )
        ].copy()

        if not contains_df.empty:
            working = contains_df

    postal_match_mode = "none"

    # Step 4: postal filtering
    if postal_norm:
        exact_postal_df = working[
            working["POSTAL_CODE_NORMALIZED"] == postal_norm
        ].copy()

        if not exact_postal_df.empty:
            working = exact_postal_df
            postal_match_mode = "exact"
        else:
            fsa = extract_postal_fsa(postal_norm)
            if fsa:
                fsa_df = working[
                    working["POSTAL_FSA"] == fsa
                ].copy()
                if not fsa_df.empty:
                    working = fsa_df
                    postal_match_mode = "fsa"

    # Step 5: de-duplicate rows for cleaner UI
    candidate_cols = [
        "DISPLAY_ADDRESS",
        "PROPERTY_POSTAL_CODE",
        "LEGAL_TYPE",
        "ZONING_DISTRICT",
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
        "YEAR_BUILT",
        "BIG_IMPROVEMENT_YEAR",
        "REPORT_YEAR",
    ]
    existing_candidate_cols = [c for c in candidate_cols if c in working.columns]

    working = working.drop_duplicates(subset=existing_candidate_cols).copy()

    # Step 6: keep only top N candidates
    working = working.head(limit).copy()

    return working, used_report_year, postal_match_mode


# ------------------------------------------------------------
# 11. Routes
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
    Return context-aware options for precise mode.
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


@app.get("/fuzzy_lookup")
def fuzzy_lookup(
    street_number: str = Query(..., description="Street number, e.g. 1050"),
    street_name: str = Query(..., description="Street name, e.g. 26TH AVE W"),
    property_postal_code: Optional[str] = Query(default=None, description="Optional postal code"),
    report_year: Optional[int] = Query(default=None, description="Optional target report year"),
    limit: int = Query(default=10, ge=1, le=20),
):
    """
    Fuzzy lookup route for address-based matching.

    Frontend sends:
    - street number
    - street name
    - optional postal code
    - optional report year

    Backend returns:
    - 0 candidate rows, or
    - 1 exact/near-exact candidate, or
    - multiple candidates for user selection
    """
    matched_df, used_report_year, postal_match_mode = fuzzy_match_address_candidates(
        street_number=street_number,
        street_name=street_name,
        property_postal_code=property_postal_code,
        report_year=report_year,
        limit=limit,
    )

    candidates = []
    for _, row in matched_df.iterrows():
        candidates.append(
            {
                "candidate_id": len(candidates) + 1,
                "display_address": str(row.get("DISPLAY_ADDRESS", "")),
                "PROPERTY_POSTAL_CODE": normalize_postal_code(row.get("PROPERTY_POSTAL_CODE")),
                "LEGAL_TYPE": str(row.get("LEGAL_TYPE", "")).strip(),
                "ZONING_DISTRICT": str(row.get("ZONING_DISTRICT", "")).strip(),
                "ZONING_CLASSIFICATION": str(row.get("ZONING_CLASSIFICATION", "")).strip(),
                "NEIGHBOURHOOD_CODE": str(row.get("NEIGHBOURHOOD_CODE", "")).strip(),
                "YEAR_BUILT": int(row["YEAR_BUILT"]) if pd.notna(row.get("YEAR_BUILT")) else None,
                "BIG_IMPROVEMENT_YEAR": int(row["BIG_IMPROVEMENT_YEAR"]) if pd.notna(row.get("BIG_IMPROVEMENT_YEAR")) else None,
                "REPORT_YEAR": int(row["REPORT_YEAR"]) if pd.notna(row.get("REPORT_YEAR")) else used_report_year,
            }
        )

    return {
        "match_count": len(candidates),
        "auto_selected": len(candidates) == 1,
        "used_report_year": used_report_year,
        "postal_match_mode": postal_match_mode,
        "candidates": candidates,
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