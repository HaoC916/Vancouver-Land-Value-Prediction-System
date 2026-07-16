"""AI assistant for the property-value demo — a Claude tool-use agent loop.

POST /assistant/chat receives the visible conversation history and runs a
tool-use loop: the model can estimate market list prices and City of Vancouver
assessed values, recommend neighbourhoods, and look up 2021 census profiles and
market trends across the Greater Vancouver area. Only available
when ANTHROPIC_API_KEY is set (the
Hugging Face Space secret); without it /assistant/status reports offline and
the frontend falls back to the scripted chat flow.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

MODEL_NAME = os.environ.get("ASSISTANT_MODEL", "claude-haiku-4-5")
MAX_TOOL_ROUNDS = 8
MAX_OUTPUT_TOKENS = 1024

# Public-demo guardrails (in-memory; reset on restart, which is fine here).
MAX_MESSAGES = 20
MAX_TOTAL_CHARS = 8000
PER_IP_DAILY_LIMIT = 30
GLOBAL_DAILY_LIMIT = 300

CENSUS_PATH = Path("data/deploy/census_csd.json")
MARKET_PATH = Path("data/deploy/market_trend.parquet")

# Source codes -> what users actually say.
PTYPE_ALIASES = {
    "HOUSE": "HOUSE", "HOUSES": "HOUSE", "DETACHED": "HOUSE",
    "APTU": "APTU", "CONDO": "APTU", "CONDOS": "APTU", "APARTMENT": "APTU", "APT": "APTU",
    "TWIN": "TWIN", "TOWNHOUSE": "TWIN", "TOWNHOME": "TWIN", "ATTACHED": "TWIN",
    "OTHER": "OTHER",
}

SYSTEM_PROMPT = """\
You are the built-in assistant of a property-value demo web app built by Ryan
Chen. You have several abilities, all backed by real data:

1. Home value estimates + neighbourhood guidance — everything below covers the
   Greater Vancouver area (Vancouver, Burnaby, Surrey, Richmond, Coquitlam and the
   Tri-Cities, the North Shore, Delta, Langley, Maple Ridge/Pitt Meadows,
   Abbotsford, Mission, ...). The product does not cover Ontario or anywhere
   outside this region — if asked about elsewhere, say so plainly.

   MARKET LIST PRICE is the core estimate: estimate_market_price predicts what a
   home would LIST for today from its features. It is feature-driven and works for
   EVERY city in the region — NOT looked up by address. Never call it with guessed
   or default values; you need three things first: property type (house / condo /
   townhouse — ask, don't assume), a specific NEIGHBOURHOOD, and floor area in
   sqft (the biggest driver; bedrooms/bathrooms help). Gather them naturally, one
   or two questions at a time. Give the likely range — a model estimate of list
   price, not a guaranteed sale price. (Street ADDRESS in any city? Don't dead-end:
   read the neighbourhood off it, e.g. "6463 Silver Ave, Burnaby" → Metrotown, then
   just get type + size. For City-of-Vancouver addresses ONLY, estimate_property_value
   also returns the BC-assessment value; never tell a Burnaby/Surrey/etc. user you
   "can't" help — you can, via market price.)

   IF THE USER DOESN'T KNOW WHICH NEIGHBOURHOOD, run a short guided buyer diagnosis
   before estimating. Ask ONE thing at a time, offer the choices as a quick menu,
   keep it natural (not a form), and always let them jump ahead if they already
   know their area / type / budget.

   Step 1 — what's the home for? living in it, an investment, or a vacation/getaway.

   • LIVING IN IT (fully supported). Ask what matters MOST, then rank with
     recommend_neighbourhoods using the matching sort_by:
       - easy commute → ask where they work / their key destination (downtown,
         Metrotown...). Use sort_by=commute. (Our transit score is rapid-transit
         access — SkyTrain / SeaBus / West Coast Express — the best proxy for a
         downtown or central commute; say so if their destination is car-oriented.)
       - a good school → ask if they have a school or area in mind; if not, use
         sort_by=schools (best nearby Fraser Institute score, 0-10) to shortlist,
         then look at homes there.
       - lively amenities / a shopping district (商圈) → ask their target area; if
         unsure, offer a vibe and map it to starting points: bustling downtown
         (Downtown / Yaletown / Coal Harbour), a mall hub (Metrotown / Richmond
         Centre / Coquitlam Centre), an artsy strip (Mount Pleasant / Commercial
         Drive), a seaside town (White Rock / Steveston), or foodie/night-market
         (Richmond / Kingsway). Use sort_by=amenities. (No formal 商圈 table yet —
         treat these as starting points, not exhaustive.)
       - safety → sort_by=safety.
       - not sure / a bit of everything → sort_by=livability (our composite of
         amenities, transit, safety and schools).
     Then ask PROPERTY TYPE and BUDGET and call recommend_neighbourhoods with that
     sort_by + max_price. If they named a city, pass it; if they're open to
     anywhere, OMIT city to rank the whole region. Show the top few (the score that
     matters + price), let them pick one, then offer estimate_market_price for it.

   • AN INVESTMENT — be upfront about depth for now: you can show how a market is
     trending (get_market_trend: price direction, plus days-on-market and sales
     volume = liquidity) and typical prices by area, but you do NOT yet compute
     rental yield / cash-flow. Help with appreciation, liquidity and price; say the
     rental-return side is coming.

   • A VACATION / GETAWAY place — resort coverage (ski / lakeside / seaside) is
     limited right now; steer them to in-region getaways (White Rock, Steveston,
     Harrison / Cultus Lake) and say fuller resort-area support is coming.

   Be honest about the data throughout: safety is a CITY-level crime rate shared
   across a municipality (not block-by-block); you don't have per-home listings,
   weather, or user reviews — don't invent them. You can also list options with
   list_neighbourhoods and add city context from get_area_profile.

2. Neighbourhood facts — 2021 Canadian census profiles for 38 Greater Vancouver
   area municipalities (population, income, home values, ownership
   rates, immigration, commuting, and more). Use get_area_profile; some names
   (North Vancouver, Langley) match two municipalities — present both. This is a
   2021 snapshot; say so when asked about "now".

3. Market trends — monthly community-level resale market data (new listings,
   sales, average sold prices, days on market) from May 2021 to May 2026 for
   33 real-estate-board areas across the Greater Vancouver area. Use
   get_market_trend. Area names follow board conventions: "Vancouver West"/
   "Vancouver East", "Burnaby North/South/East", "North Surrey", "Abbotsford" —
   a partial name like "Burnaby" matches all its parts. These are aggregate
   monthly figures, not individual listings.

Rules:
- Amounts are CAD. Reply in plain text only — no markdown, no asterisks.
- Use the numbers the tools return; never estimate from memory. If a tool
  reports not found, say so and suggest what to try (search_addresses helps
  with misspelled streets; the format is like "1128 HASTINGS ST W").
- If a building needs a unit number, ask the user for it.
- You cannot predict future prices and this is not investment advice — trends
  describe what already happened. Politely decline questions unrelated to
  this app's data.
- Keep answers short and concrete — a few sentences. This is a public demo.
"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


# ------------------------------------------------------------
# Census lookup (65 CSD rows, loaded once)
# ------------------------------------------------------------
_census_rows: list[dict[str, Any]] | None = None


def get_census_rows() -> list[dict[str, Any]]:
    global _census_rows
    if _census_rows is None:
        _census_rows = json.loads(CENSUS_PATH.read_text()) if CENSUS_PATH.exists() else []
    return _census_rows


_market_df = None


def get_market_df():
    global _market_df
    if _market_df is None:
        import pandas as pd

        _market_df = pd.read_parquet(MARKET_PATH) if MARKET_PATH.exists() else pd.DataFrame()
    return _market_df


# ------------------------------------------------------------
# Anthropic client (lazy, only when the key exists)
# ------------------------------------------------------------
_client = None


def get_client():
    global _client
    if _client is None and os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
TOOLS = [
    {
        "name": "estimate_property_value",
        "description": (
            "Estimate the BC-assessment value of a CITY OF VANCOUVER property by address "
            "(a bonus that works only inside the City of Vancouver). Give the street/building "
            "number and street name (e.g. 1128 and HASTINGS ST W). If the result is need_unit, "
            "ask the user for their unit number and call again with it. For addresses in any "
            "OTHER city (Burnaby, Surrey, Richmond, ...), do NOT use this — use "
            "estimate_market_price with the address's neighbourhood instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "street_number": {"type": "string", "description": "Building/street number, e.g. 1128"},
                "street_name": {"type": "string", "description": "Street name, e.g. HASTINGS ST W"},
                "unit": {"type": "string", "description": "Unit number for condos/strata, e.g. 2308"},
                "postal_code": {"type": "string", "description": "Optional postal code to disambiguate"},
            },
            "required": ["street_number", "street_name"],
        },
    },
    {
        "name": "estimate_market_price",
        "description": (
            "Estimate the current MARKET LIST price of a home in the Greater Vancouver area "
            "from its features. Different from estimate_property_value (which is "
            "the City of Vancouver assessed/tax value by address): this works across the whole "
            "region and is driven by property facts. Provide at least property_type, "
            "floor_area_sqft, and area_name. Returns a list-price estimate and likely range; "
            "not a guaranteed sale price."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "property_type": {"type": "string", "description": "house / condo / townhouse"},
                "bedrooms": {"type": "number", "description": "Number of bedrooms"},
                "bathrooms": {"type": "number", "description": "Number of bathrooms"},
                "floor_area_sqft": {"type": "number", "description": "Floor area in square feet"},
                "area_name": {"type": "string", "description": "Area or neighbourhood — pass the most specific one the user gives, e.g. Vancouver West, Richmond, Surrey, or a neighbourhood like Metrotown, Yaletown, Brentwood"},
                "year_built": {"type": "integer", "description": "Year the home was built (optional)"},
                "postal_code": {"type": "string", "description": "Optional postal code"},
            },
            "required": ["area_name"],
        },
    },
    {
        "name": "list_market_price_areas",
        "description": "List the Greater Vancouver area names the market-price estimator recognises.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_neighbourhoods",
        "description": (
            "List the neighbourhoods the market-price model knows inside a city/area, e.g. "
            "Burnaby -> Metrotown, Brentwood Park, Deer Lake, ... Use this to offer the user "
            "concrete neighbourhood choices when they are unsure which part of a city they mean."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"area_name": {"type": "string", "description": "City/area, e.g. Burnaby, Surrey, Vancouver West"}},
            "required": ["area_name"],
        },
    },
    {
        "name": "recommend_neighbourhoods",
        "description": (
            "Recommend real neighbourhoods in a city, using per-neighbourhood data. Use when the "
            "user wants a home in a city but is unsure which neighbourhood. Rank by budget "
            "(sort_by=price, with max_price/min_price), by school quality (sort_by=schools — "
            "best nearby Fraser Institute school score, 0-10), by amenities/walkability "
            "(sort_by=amenities — our 0-100 score from nearby groceries, dining, parks and health "
            "services), by transit/commute (sort_by=commute — our 0-100 score from TransLink stop "
            "density + distance to rapid transit), by safety (sort_by=safety — our 0-100 score, "
            "inverse of the official StatCan city-level crime rate), or by our composite livability "
            "score (sort_by=livability — weighted blend of amenities, transit, safety, schools). "
            "Commute, safety and livability cover most of the Greater Vancouver area. Returns each neighbourhood's "
            "typical price, school score, amenity score (with POI counts), transit score, safety "
            "score (with the city crime rate it's based on), livability score, days-on-market, and "
            "recent sales count (a low count = thin/less-reliable data). The safety result "
            "includes safety_year — cite that year for the crime rate; don't guess it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City/area, e.g. Burnaby, Surrey, Coquitlam. OMIT to search the whole Greater Vancouver area region-wide (use this when the user has no city in mind)."},
                "property_type": {"type": "string", "description": "house, condo, or townhouse"},
                "sort_by": {"type": "string", "description": "price (default), schools, amenities, commute, safety, or livability"},
                "max_price": {"type": "number", "description": "Budget ceiling in CAD (optional)"},
                "min_price": {"type": "number", "description": "Budget floor in CAD (optional)"},
            },
            "required": ["property_type"],
        },
    },
    {
        "name": "search_addresses",
        "description": (
            "Fuzzy-search the City of Vancouver address records when an exact match "
            "failed — helps with misspelled or partial street names. Returns up to 5 "
            "candidate display addresses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "street_number": {"type": "string", "description": "Street number"},
                "street_name": {"type": "string", "description": "Street name or part of it"},
                "postal_code": {"type": "string", "description": "Optional postal code"},
            },
            "required": ["street_number", "street_name"],
        },
    },
    {
        "name": "get_area_profile",
        "description": (
            "Get the 2021 census profile of a municipality in the Greater Vancouver area: "
            "population, density, ages, household income, average home value, "
            "owner/renter/condo shares, immigrant share, education, transit commuting. Match "
            "is by name (e.g. Burnaby, Surrey, Abbotsford); may return more than one "
            "municipality for ambiguous names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "area_name": {"type": "string", "description": "Municipality name, e.g. Burnaby"},
            },
            "required": ["area_name"],
        },
    },
    {
        "name": "list_census_areas",
        "description": "List the 38 Greater Vancouver area municipalities that have census profiles.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_market_trend",
        "description": (
            "Monthly resale market trend for a real-estate-board area or a named "
            "community (subarea): sales, new listings, and sales-weighted average "
            "sold price per month, plus a year-over-year price change. Partial names "
            "match ('Burnaby' covers Burnaby North/South/East). property_type: HOUSE "
            "(detached, default), APTU (condo/apartment), TWIN (townhouse/attached), "
            "OTHER. Data: May 2021 - May 2026."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "area_name": {"type": "string", "description": "Board area or community name, e.g. Richmond, Vancouver West, Metrotown"},
                "property_type": {"type": "string", "description": "HOUSE (default), APTU, TWIN, or OTHER; condo/townhouse aliases accepted"},
                "months": {"type": "integer", "description": "How many recent months to return (3-24, default 12)"},
            },
            "required": ["area_name"],
        },
    },
    {
        "name": "list_market_areas",
        "description": "List the 177 real-estate-board area names that have market-trend data.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _tool_estimate(state: dict, street_number: str, street_name: str,
                   unit: Optional[str] = None, postal_code: Optional[str] = None) -> dict:
    from src.api import main as api  # lazy import; main includes this router

    resolved = api.resolve_address(
        street_number=str(street_number),
        street_name=str(street_name),
        unit=str(unit) if unit not in (None, "") else None,
        property_postal_code=str(postal_code) if postal_code else None,
        report_year=None,
    )
    status = resolved.get("status")
    if status == "need_unit":
        return {
            "status": "need_unit",
            "unit_count": resolved.get("unit_count", 0),
            "note": "Multi-unit building - ask the user for their unit number.",
        }
    if status != "single" or not resolved.get("candidate"):
        return {"status": "not_found",
                "note": "No exact match. Try search_addresses, or the address may be outside the City of Vancouver."}

    cand = resolved["candidate"]
    result = api.predictor.predict({
        "PROPERTY_POSTAL_CODE": cand["PROPERTY_POSTAL_CODE"],
        "LEGAL_TYPE": cand["LEGAL_TYPE"],
        "ZONING_DISTRICT": cand["ZONING_DISTRICT"],
        "ZONING_CLASSIFICATION": cand["ZONING_CLASSIFICATION"],
        "NEIGHBOURHOOD_CODE": cand["NEIGHBOURHOOD_CODE"],
        "YEAR_BUILT": cand["YEAR_BUILT"],
        "BIG_IMPROVEMENT_YEAR": cand["BIG_IMPROVEMENT_YEAR"],
        "REPORT_YEAR": cand["REPORT_YEAR"],
        "PID": cand["PID"],
    })
    estimate = {
        "point_estimate": result.point_estimate,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "error_band": result.error_band,
        "error_band_source": result.error_band_source,
        "used_features": result.used_features,
    }
    # Stash for the UI side panels; returned with the chat reply.
    state["property"] = cand
    state["estimate"] = estimate
    return {
        "status": "ok",
        "address": cand["display_address"],
        "estimate_cad": round(result.point_estimate),
        "likely_range_cad": [round(result.lower_bound), round(result.upper_bound)],
        "property": {
            "legal_type": cand["LEGAL_TYPE"],
            "zoning_district": cand["ZONING_DISTRICT"],
            "neighbourhood_code": cand["NEIGHBOURHOOD_CODE"],
            "year_built": cand["YEAR_BUILT"],
            "assessment_year": cand["REPORT_YEAR"],
            "unit": cand.get("UNIT") or None,
        },
    }


def _tool_market_price(area_name: str, property_type: Optional[str] = None,
                       bedrooms: Optional[float] = None, bathrooms: Optional[float] = None,
                       floor_area_sqft: Optional[float] = None, year_built: Optional[int] = None,
                       postal_code: Optional[str] = None) -> dict:
    from src.api import main as api

    if api.market_predictor is None:
        return {"status": "unavailable", "note": "The market-price model is not loaded in this deployment."}
    r = api.market_predictor.predict({
        "property_type": property_type, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "floor_area_sqft": floor_area_sqft, "area_name": area_name,
        "year_built": year_built, "postal_code": postal_code,
    })
    resolved = r.used_features.get("area_name")
    return {
        "status": "ok",
        "estimate_list_price_cad": round(r.point_estimate),
        "likely_range_cad": [round(r.lower_bound), round(r.upper_bound)],
        "method": r.method,
        "resolved_area": resolved,
        "inputs_used": r.used_features,
        "note": ("This is a market list-price estimate, not a guaranteed sale price."
                 if resolved else
                 "Area not recognised — used a region-wide average. Call list_market_price_areas for valid areas."),
    }


def _tool_list_market_price_areas() -> dict:
    from src.api import main as api

    if api.market_predictor is None:
        return {"note": "The market-price model is not loaded in this deployment."}
    return {"count": len(api.market_predictor.known_areas), "areas": api.market_predictor.known_areas}


def _tool_list_neighbourhoods(area_name: str) -> dict:
    from src.api import main as api

    if api.market_predictor is None:
        return {"note": "The market-price model is not loaded in this deployment."}
    res = api.market_predictor.neighbourhoods_in(area_name)
    if not res.get("matched_areas"):
        return {"count": 0, "note": f"No area matched '{area_name}'. Call list_market_price_areas for valid city/area names."}
    return res


def _tool_recommend_neighbourhoods(property_type: str, city: Optional[str] = None,
                                   sort_by: str = "price",
                                   max_price: Optional[float] = None,
                                   min_price: Optional[float] = None) -> dict:
    from src.api import main as api

    if api.neighbourhood_profiles is None:
        return {"status": "unavailable", "note": "Neighbourhood data is not loaded in this deployment."}
    return api.neighbourhood_profiles.recommend(
        city=city, property_type=property_type, sort_by=sort_by, max_price=max_price, min_price=min_price)


def _tool_search(street_number: str, street_name: str, postal_code: Optional[str] = None) -> dict:
    from src.api import main as api

    matched, used_year, postal_mode = api.fuzzy_match_address_candidates(
        street_number=str(street_number),
        street_name=str(street_name),
        property_postal_code=str(postal_code) if postal_code else None,
        report_year=None,
        limit=5,
    )
    addresses = matched["DISPLAY_ADDRESS"].astype(str).drop_duplicates().tolist() if not matched.empty else []
    return {"count": len(addresses), "candidates": addresses, "assessment_year": used_year}


def _tool_area_profile(area_name: str) -> dict:
    rows = get_census_rows()
    needle = str(area_name).strip().lower()
    matches = [r for r in rows if r["name"].lower() == needle]
    if not matches:
        matches = [r for r in rows if needle in r["name"].lower() or r["name"].lower() in needle]
    if not matches:
        return {"count": 0,
                "note": "No municipality with that name in the 2021 census set. Use list_census_areas."}
    return {"count": len(matches), "profiles": matches[:3]}


def _tool_list_areas() -> dict:
    rows = get_census_rows()
    by_region: dict[str, list[str]] = {}
    for r in rows:
        label = f"{r['name']} ({r['kind']})" if r["kind"] else r["name"]
        by_region.setdefault(r["region"], []).append(label)
    return {"count": len(rows), "regions": by_region}


def _monthly_frame(rows):
    """Aggregate subarea rows to one row per month (sales-weighted avg price)."""
    import pandas as pd

    def agg(group: "pd.DataFrame") -> "pd.Series":
        priced = group.dropna(subset=["avg_sold_price"])
        priced = priced[priced["sold_count"] > 0]
        weight = priced["sold_count"].sum()
        price = float((priced["avg_sold_price"] * priced["sold_count"]).sum() / weight) if weight else None
        return pd.Series({
            "sold": int(group["sold_count"].fillna(0).sum()),
            "new_listings": int(group["new_listing_count"].fillna(0).sum()),
            "avg_sold_price": price,
        })

    monthly = rows.groupby("period_start").apply(agg, include_groups=False).reset_index()
    return monthly.sort_values("period_start")


def _window_price(monthly, start: int, end: int):
    window = monthly.iloc[start:end].dropna(subset=["avg_sold_price"])
    if window.empty or window["sold"].sum() == 0:
        return None
    return float((window["avg_sold_price"] * window["sold"]).sum() / window["sold"].sum())


def _trend_block(label: str, rows, months: int) -> dict:
    monthly = _monthly_frame(rows)
    recent = monthly.tail(months)
    series = [
        {
            "month": r["period_start"],
            "sold": int(r["sold"]),
            "new_listings": int(r["new_listings"]),
            "avg_sold_price": round(r["avg_sold_price"]) if r["avg_sold_price"] else None,
        }
        for _, r in recent.iterrows()
    ]
    block: dict[str, Any] = {
        "name": label,
        "monthly": series,
        "sold_last_12_months": int(monthly.tail(12)["sold"].sum()),
    }
    latest = _window_price(monthly, -3, len(monthly))
    year_ago = _window_price(monthly, -15, -12)
    if latest and year_ago:
        block["avg_price_last_3_months"] = round(latest)
        block["avg_price_same_3_months_last_year"] = round(year_ago)
        block["price_change_pct_year_over_year"] = round((latest / year_ago - 1) * 100, 1)
    return block


def _tool_market_trend(area_name: str, property_type: Optional[str] = None, months: Optional[int] = None) -> dict:
    df = get_market_df()
    if df.empty:
        return {"note": "Market-trend data is not available in this deployment."}

    months = max(3, min(int(months or 12), 24))
    ptype_raw = str(property_type or "HOUSE").strip().upper()
    ptype = PTYPE_ALIASES.get(ptype_raw)
    ptype_note = None
    if ptype is None:
        ptype = "HOUSE"
        ptype_note = f"Unknown property_type '{property_type}' - defaulted to HOUSE. Valid: HOUSE, APTU (condo), TWIN (townhouse), OTHER."
    typed = df[df["property_type"] == ptype]

    needle = str(area_name).strip().lower()
    area_names = sorted(typed["area_name"].dropna().unique())
    matches = [a for a in area_names if a.lower() == needle] or [a for a in area_names if needle in a.lower()]

    blocks = []
    if matches:
        for area in matches[:3]:
            blocks.append(_trend_block(area, typed[typed["area_name"] == area], months))
    else:
        # No board area matched - try community (subarea) names.
        pairs = typed[["area_name", "subarea_name"]].dropna().drop_duplicates()
        sub_matches = pairs[pairs["subarea_name"].str.lower().str.contains(needle, regex=False)]
        if sub_matches.empty:
            return {"count": 0,
                    "note": "No board area or community matched that name. Call list_market_areas for valid area names."}
        for _, pair in sub_matches.head(3).iterrows():
            rows = typed[(typed["area_name"] == pair["area_name"]) & (typed["subarea_name"] == pair["subarea_name"])]
            blocks.append(_trend_block(f"{pair['subarea_name']} ({pair['area_name']})", rows, months))
        matches = list(sub_matches["subarea_name"])

    result: dict[str, Any] = {
        "property_type": ptype,
        "count": len(matches),
        "trends": blocks,
    }
    if len(matches) > 3:
        result["note"] = f"{len(matches)} areas matched; showing the first 3. Ask with a more specific name for others."
    if property_type is None:
        result["default_note"] = "property_type defaulted to HOUSE (detached). Also available: APTU (condo/apartment), TWIN (townhouse), OTHER."
    if ptype_note:
        result["default_note"] = ptype_note
    return result


def _tool_list_market_areas() -> dict:
    df = get_market_df()
    if df.empty:
        return {"note": "Market-trend data is not available in this deployment."}
    names = sorted(df["area_name"].dropna().unique().tolist())
    return {"count": len(names), "areas": names}


def run_tool(name: str, tool_input: dict, state: dict) -> tuple[str, bool]:
    """Returns (json_result, is_error)."""
    try:
        if name == "estimate_property_value":
            payload = _tool_estimate(state, **tool_input)
        elif name == "estimate_market_price":
            payload = _tool_market_price(**tool_input)
        elif name == "list_market_price_areas":
            payload = _tool_list_market_price_areas()
        elif name == "list_neighbourhoods":
            payload = _tool_list_neighbourhoods(**tool_input)
        elif name == "recommend_neighbourhoods":
            payload = _tool_recommend_neighbourhoods(**tool_input)
        elif name == "search_addresses":
            payload = _tool_search(**tool_input)
        elif name == "get_area_profile":
            payload = _tool_area_profile(**tool_input)
        elif name == "list_census_areas":
            payload = _tool_list_areas()
        elif name == "get_market_trend":
            payload = _tool_market_trend(**tool_input)
        elif name == "list_market_areas":
            payload = _tool_list_market_areas()
        else:
            return f"Error: unknown tool {name}", True
        return json.dumps(payload, default=str), False
    except TypeError:
        return "Error: bad arguments for this tool.", True
    except Exception:
        return "Error: this lookup failed - tell the user and suggest rephrasing.", True


# ------------------------------------------------------------
# Rate limiting
# ------------------------------------------------------------
_rate_lock = threading.Lock()
_requests_per_ip: dict[str, int] = {}
_requests_today = 0
_counter_day: date = datetime.now(timezone.utc).date()


def check_rate_limits(ip: str) -> None:
    global _requests_today, _counter_day
    today = datetime.now(timezone.utc).date()
    with _rate_lock:
        if today != _counter_day:
            _requests_per_ip.clear()
            _requests_today = 0
            _counter_day = today
        _requests_today += 1
        if _requests_today > GLOBAL_DAILY_LIMIT:
            raise HTTPException(429, "The assistant hit its daily budget - try again tomorrow")
        _requests_per_ip[ip] = _requests_per_ip.get(ip, 0) + 1
        if _requests_per_ip[ip] > PER_IP_DAILY_LIMIT:
            raise HTTPException(429, "Daily message limit reached for this demo - try again tomorrow")


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def validate(req: ChatRequest) -> None:
    msgs = req.messages
    if not msgs:
        raise HTTPException(400, "messages must not be empty")
    if len(msgs) > MAX_MESSAGES:
        raise HTTPException(400, "Conversation too long for this demo - start a new chat")
    total = 0
    for m in msgs:
        if m.role not in ("user", "assistant") or not m.content.strip():
            raise HTTPException(400, "Invalid message in request")
        total += len(m.content)
    if total > MAX_TOTAL_CHARS:
        raise HTTPException(400, "Conversation too large for this demo - start a new chat")
    if msgs[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user")


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@router.get("/assistant/status")
def assistant_status():
    return {"available": is_available(), "model": MODEL_NAME}


@router.post("/assistant/chat")
def assistant_chat(req: ChatRequest, request: Request):
    if not is_available():
        raise HTTPException(503, "The assistant is offline (no API key configured)")
    validate(req)
    check_rate_limits(client_ip(request))

    import anthropic

    client = get_client()
    conversation: list[dict[str, Any]] = [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    state: dict[str, Any] = {}

    try:
        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=conversation,
            )
            if response.stop_reason != "tool_use":
                reply = "".join(b.text for b in response.content if b.type == "text").strip()
                if not reply:
                    reply = "Sorry, I could not answer that. Try asking about a Vancouver address or a neighbourhood."
                return {"reply": reply,
                        "property": state.get("property"),
                        "estimate": state.get("estimate")}

            conversation.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output, is_error = run_tool(block.name, dict(block.input), state)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
            conversation.append({"role": "user", "content": results})

        return {"reply": "Sorry, that took more lookups than this demo allows. Try something narrower.",
                "property": state.get("property"),
                "estimate": state.get("estimate")}
    except anthropic.APIStatusError as e:
        if e.status_code == 429:
            raise HTTPException(429, "The assistant is busy right now - try again in a minute")
        raise HTTPException(502, "Assistant is temporarily unavailable")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Assistant is temporarily unavailable")
