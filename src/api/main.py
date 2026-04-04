from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infer.predict import LandValuePredictor

app = FastAPI(title="Land Value API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = LandValuePredictor()


class PredictRequest(BaseModel):
    PROPERTY_POSTAL_CODE: str
    LEGAL_TYPE: str
    ZONING_DISTRICT: str
    ZONING_CLASSIFICATION: str
    NEIGHBOURHOOD_CODE: str
    YEAR_BUILT: int
    BIG_IMPROVEMENT_YEAR: Optional[int] = None
    REPORT_YEAR: Optional[int] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/options")
def get_options():
    return {
        "LEGAL_TYPE": predictor.get_top_options("LEGAL_TYPE", top_n=100),
        "ZONING_DISTRICT": predictor.get_top_options("ZONING_DISTRICT", top_n=150),
        "ZONING_CLASSIFICATION": predictor.get_top_options("ZONING_CLASSIFICATION", top_n=200),
        "NEIGHBOURHOOD_CODE": predictor.get_top_options("NEIGHBOURHOOD_CODE", top_n=200),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    result = predictor.predict(req.model_dump())
    return {
        "point_estimate": result.point_estimate,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "error_band": result.error_band,
        "error_band_source": result.error_band_source,
        "used_features": result.used_features,
    }