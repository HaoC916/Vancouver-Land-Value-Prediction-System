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
    # the per-unit signal must be a model feature
    assert "pid_prev_year_property_value" in p.feature_cols


def test_per_unit_differentiation():
    """Two known units with different prior values get different estimates."""
    p = LandValuePredictor()
    by_value = sorted(p._pid_latest_value.items(), key=lambda kv: kv[1])
    assert len(by_value) > 10
    low_pid = by_value[len(by_value) // 4][0]
    high_pid = by_value[3 * len(by_value) // 4][0]

    base = _sample(p)
    low = p.predict({**base, "PID": low_pid}).point_estimate
    high = p.predict({**base, "PID": high_pid}).point_estimate
    assert high > low


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
