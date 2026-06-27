import pytest

from src.infer.predict import LandValuePredictor


def _sample(predictor: LandValuePredictor) -> dict:
    def top(col: str, fallback: str) -> str:
        opts = predictor.get_top_options(col, 5)
        return opts[0] if opts else fallback

    return {
        "PROPERTY_POSTAL_CODE": "V6H2J4",
        "LEGAL_TYPE": top("LEGAL_TYPE", "LAND"),
        "ZONING_DISTRICT": top("ZONING_DISTRICT", "RS-1"),
        "ZONING_CLASSIFICATION": top("ZONING_CLASSIFICATION", "Residential"),
        "NEIGHBOURHOOD_CODE": top("NEIGHBOURHOOD_CODE", "1"),
        "YEAR_BUILT": 1990,
        "REPORT_YEAR": 2026,
    }


def test_predictor_loads_in_deploy_lookup_mode():
    p = LandValuePredictor()
    assert p.use_deploy_lookup is True
    assert len(p.feature_cols) == 57


def test_prediction_is_sane():
    p = LandValuePredictor()
    r = p.predict(_sample(p))
    assert r.point_estimate > 0
    assert r.lower_bound <= r.point_estimate <= r.upper_bound
    assert r.error_band > 0


def test_invalid_postal_code_raises():
    p = LandValuePredictor()
    bad = _sample(p)
    bad["PROPERTY_POSTAL_CODE"] = "not-a-postal"
    with pytest.raises(ValueError):
        p.predict(bad)
