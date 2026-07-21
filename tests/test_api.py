import pytest
from fastapi.testclient import TestClient

from src.api.main import app, market_predictor

client = TestClient(app)

# The market-price model is a large build/deploy artifact (not committed to git);
# skip its tests where it is absent (e.g. CI) rather than failing.
market_missing = pytest.mark.skipif(
    market_predictor is None, reason="market-price model artifact not present"
)


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


@market_missing
def test_predict_market_returns_ordered_range():
    payload = {
        "property_type": "condo",
        "bedrooms": 2,
        "bathrooms": 2,
        "floor_area_sqft": 900,
        "area_name": "Vancouver West",
        "year_built": 2015,
    }
    r = client.post("/predict_market", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["point_estimate"] > 0
    assert body["lower_bound"] <= body["point_estimate"] <= body["upper_bound"]
    assert body["method"] in ("price_per_sqft", "direct")


@market_missing
def test_predict_market_area_affects_estimate():
    # The same condo should estimate well above in a premium area (Vancouver West) vs a
    # lower-priced one (Surrey) — a large, reliable location premium the model captures.
    base = {"property_type": "condo", "bedrooms": 2, "bathrooms": 2, "floor_area_sqft": 850}
    van_west = client.post("/predict_market", json={**base, "area_name": "Vancouver West"}).json()
    surrey = client.post("/predict_market", json={**base, "area_name": "Surrey"}).json()
    assert van_west["point_estimate"] > surrey["point_estimate"]


def test_fuzzy_lookup_finds_candidates():
    r = client.get(
        "/fuzzy_lookup",
        params={"street_number": "1050", "street_name": "26TH AVE W", "report_year": 2026},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["match_count"] >= 1
    assert len(body["candidates"]) == body["match_count"]


def test_municipality_map_uses_modified_geometry_unions():
    r = client.get("/map/municipalities")
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["geometry_source"] == "modified_geom_union_only"
    assert len(body["features"]) == 21
    assert all(
        feature["properties"]["geometry_source"] == "modified_geom_union"
        for feature in body["features"]
    )


def test_community_map_never_falls_back_to_raw_geometry():
    r = client.get("/map/communities")
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["geometry_source"] == "modified_geom_only"
    assert body["metadata"]["excluded_missing_modified"] == ["308"]
    assert len(body["features"]) == 357
    assert all(
        feature["properties"]["geometry_source"] == "modified_geom"
        for feature in body["features"]
    )


def test_community_map_can_filter_by_municipality():
    r = client.get("/map/communities", params={"municipality": "Burnaby"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["features"]) == 36
    assert {feature["properties"]["municipality"] for feature in body["features"]} == {"Burnaby"}
