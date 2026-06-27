from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["min_report_year"] <= body["default_report_year"] <= body["max_report_year"]


def test_options_returns_lists():
    r = client.get("/options")
    assert r.status_code == 200
    body = r.json()
    for key in ("LEGAL_TYPE", "ZONING_DISTRICT", "NEIGHBOURHOOD_CODE"):
        assert isinstance(body[key], list)


def test_predict_returns_ordered_range():
    payload = {
        "PROPERTY_POSTAL_CODE": "V6R2P1",
        "LEGAL_TYPE": "LAND",
        "ZONING_DISTRICT": "RS-1",
        "ZONING_CLASSIFICATION": "One-Family Dwelling",
        "NEIGHBOURHOOD_CODE": "7",
        "YEAR_BUILT": 1985,
        "REPORT_YEAR": 2026,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["point_estimate"] > 0
    assert body["lower_bound"] <= body["point_estimate"] <= body["upper_bound"]


def test_fuzzy_lookup_finds_candidates():
    r = client.get(
        "/fuzzy_lookup",
        params={"street_number": "1050", "street_name": "26TH AVE W", "report_year": 2026},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["match_count"] >= 1
    assert len(body["candidates"]) == body["match_count"]
