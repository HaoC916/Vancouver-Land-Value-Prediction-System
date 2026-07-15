"""Geographic scope of the product: Greater Vancouver + Fraser Valley.

Single source of truth for which real-estate "area" names are in scope. The market-price
model is trained on the Greater Vancouver + Fraser Valley boards, and the product is scoped
to match — Ontario/GTA and the far/resort BC areas (Sea-to-Sky, Sunshine Coast, Gulf Islands,
Cariboo) are deliberately excluded so every surface (neighbourhood recommendations, market
trends, census profiles) covers the same, consistent region.

Used by the deploy-lookup builders to filter their output; keep this list and the census
`source_region == "greater_vancouver"` filter in sync.
"""
from __future__ import annotations

# Metro Vancouver Regional District.
_METRO_VANCOUVER = {
    "Vancouver West", "Vancouver East",
    "Burnaby East", "Burnaby North", "Burnaby South",
    "Richmond",
    "Coquitlam", "Port Coquitlam", "Port Moody",
    "New Westminster",
    "North Vancouver", "West Vancouver",
    "N. Delta", "Ladner", "Tsawwassen",              # Delta
    "Surrey", "North Surrey", "South Surrey White Rock", "Cloverdale",
    "Langley",
    "Maple Ridge", "Pitt Meadows",
    "Bowen Island",
}

# Fraser Valley (Abbotsford, Mission, Chilliwack + the eastern valley communities).
_FRASER_VALLEY = {
    "Abbotsford", "Mission",
    "Chilliwack", "East Chilliwack", "Sardis", "Yarrow", "Cultus Lake & Area",
    "Agassiz", "Harrison Lake", "Hope & Area",
}

GREATER_VANCOUVER_AREAS: frozenset[str] = frozenset(_METRO_VANCOUVER | _FRASER_VALLEY)

# Deliberately EXCLUDED (documented so the intent is clear if these reappear in source data):
#   Sea-to-Sky:      Squamish, Whistler, Pemberton
#   Sunshine Coast:  Sunshine Coast
#   Islands:         Islands-Van. & Gulf
#   Far/rural BC:    Fraser Canyon, 100 Mile House, PG Rural North, PG Rural South
#   Junk/placeholder: VAN Fake Area, FVREB Out of Town, Out of Town
#   All of Ontario / Greater Toronto Area.


def in_scope(area_name: str) -> bool:
    return area_name in GREATER_VANCOUVER_AREAS
