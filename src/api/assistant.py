"""AI assistant for the property-value demo — a Claude tool-use agent loop.

POST /assistant/chat receives the visible conversation history and runs a
tool-use loop: the model can resolve City of Vancouver addresses, run the
value model, and look up 2021 census profiles for Greater Vancouver / Greater
Toronto municipalities. Only available when ANTHROPIC_API_KEY is set (the
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

1. Property value estimates — a machine-learning model trained on City of
   Vancouver property-tax assessment records. Use estimate_property_value with
   a street number and street name (unit number for condos). IMPORTANT scope:
   only addresses inside the City of Vancouver work. Burnaby, Richmond, Surrey,
   Toronto etc. are NOT in the assessment data — for those you can only offer
   census facts. Estimates are model estimates of assessed value (land plus
   building), not appraisals or sale prices; always mention the likely range.

   Market list price — a second machine-learning model that estimates what a
   home would LIST for today from its features (property type, bedrooms,
   bathrooms, floor area, neighbourhood) across Greater Vancouver and the Fraser
   Valley. Use estimate_market_price. It is NOT looked up by address — it is
   driven by property features, so gather them first. Never call it with guessed
   or default values, and never answer a vague question ("what's a Burnaby condo
   worth?") with one number — prices depend heavily on type, size and
   neighbourhood, so ask a couple of quick questions first (keep it natural, one
   or two at a time, not a form):
     a) Property type — house/detached, condo/apartment, or townhouse? Ask, don't
        assume; they are different markets.
     b) Neighbourhood — which part of the city? A place like Burnaby ranges from
        pricier areas (Metrotown) to cheaper ones, so it matters a lot. If the
        user is unsure, ask what they care about (budget, commute/transit,
        schools, or lifestyle/amenities), then offer concrete options with
        list_neighbourhoods and city-level context from get_area_profile, and let
        them pick one. (Be honest: you can list neighbourhoods and give
        city-level facts, but you do not yet have per-neighbourhood school or
        amenity scores — don't invent them.)
     c) Size — floor area in square feet (the biggest driver), plus bedrooms and
        bathrooms.
   Only once you have type, a specific neighbourhood, and size should you call
   estimate_market_price. Give the likely range; it's a model estimate of list
   price, not a guaranteed sale price. Shortcut: if the user gives a precise City
   of Vancouver street address, you can also offer the assessed value via
   estimate_property_value.

2. Neighbourhood facts — 2021 Canadian census profiles for 65 municipalities
   across Greater Vancouver and the Greater Toronto Area (population, income,
   home values, ownership rates, immigration, commuting, and more). Use
   get_area_profile; some names (North Vancouver, Langley) match two
   municipalities — present both. This is a 2021 snapshot; say so when asked
   about "now".

3. Market trends — monthly community-level resale market data (new listings,
   sales, average sold prices, days on market) from May 2021 to May 2026 for
   177 real-estate-board areas around Greater Vancouver, the Greater Toronto
   Area, and nearby cities. Use get_market_trend. Area names follow board
   conventions: "Vancouver West"/"Vancouver East", "Burnaby North/South/East",
   "Toronto C01".."W10" districts — a partial name like "Burnaby" matches all
   its parts. These are aggregate monthly figures, not individual listings.

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
            "Estimate the assessed value of a City of Vancouver property. Give the "
            "street/building number and street name (e.g. 1128 and HASTINGS ST W). "
            "If the result is need_unit, ask the user for their unit number and call "
            "again with it. Returns the estimate plus property facts on success."
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
            "Estimate the current MARKET LIST price of a home in Greater Vancouver or the "
            "Fraser Valley from its features. Different from estimate_property_value (which is "
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
        "description": "List the Greater Vancouver / Fraser Valley areas the market-price estimator recognises.",
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
            "Get the 2021 census profile of a municipality in Greater Vancouver or the "
            "Greater Toronto Area: population, density, ages, household income, average "
            "home value, owner/renter/condo shares, immigrant share, education, transit "
            "commuting. Match is by name (e.g. Burnaby, Richmond Hill); may return more "
            "than one municipality for ambiguous names."
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
        "description": "List the 65 municipalities that have census profiles, grouped by region.",
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
