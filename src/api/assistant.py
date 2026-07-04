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

SYSTEM_PROMPT = """\
You are the built-in assistant of a property-value demo web app built by Ryan
Chen. You have two abilities, both backed by real data:

1. Property value estimates — a machine-learning model trained on City of
   Vancouver property-tax assessment records. Use estimate_property_value with
   a street number and street name (unit number for condos). IMPORTANT scope:
   only addresses inside the City of Vancouver work. Burnaby, Richmond, Surrey,
   Toronto etc. are NOT in the assessment data — for those you can only offer
   census facts. Estimates are model estimates of assessed value (land plus
   building), not appraisals or sale prices; always mention the likely range.

2. Neighbourhood facts — 2021 Canadian census profiles for 65 municipalities
   across Greater Vancouver and the Greater Toronto Area (population, income,
   home values, ownership rates, immigration, commuting, and more). Use
   get_area_profile; some names (North Vancouver, Langley) match two
   municipalities — present both. This is a 2021 snapshot; say so when asked
   about "now".

Rules:
- Amounts are CAD. Reply in plain text only — no markdown, no asterisks.
- Use the numbers the tools return; never estimate from memory. If a tool
  reports not found, say so and suggest what to try (search_addresses helps
  with misspelled streets; the format is like "1128 HASTINGS ST W").
- If a building needs a unit number, ask the user for it.
- You have no listing prices, sales, or market-trend data. You cannot predict
  future prices. Politely decline questions unrelated to this app's data.
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


def run_tool(name: str, tool_input: dict, state: dict) -> tuple[str, bool]:
    """Returns (json_result, is_error)."""
    try:
        if name == "estimate_property_value":
            payload = _tool_estimate(state, **tool_input)
        elif name == "search_addresses":
            payload = _tool_search(**tool_input)
        elif name == "get_area_profile":
            payload = _tool_area_profile(**tool_input)
        elif name == "list_census_areas":
            payload = _tool_list_areas()
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
