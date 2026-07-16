"""Distill a per-neighbourhood safety lookup for the agent.

Assigns each community (subarea) a 0-100 safety score from official city-level crime rates,
so the agent can recommend neighbourhoods by safety (sort_by=safety), like subarea_school /
subarea_amenities / subarea_transit power the other livability dimensions.

Source: Statistics Canada table 35-10-0184-01 (incident-based crime, police services), the
ONLY source here that may legitimately be called a "crime rate" — it carries an official
population denominator and covers all Metro Vancouver municipalities. (The municipal police
dashboards in data/raw/crime/ are property-crime subsets with NO denominator; they are for
future point-level maps, not rates, and must NOT be reconciled with these — see
docs/crime_data_sources.md.)

Metrics per subarea (all from the subarea's municipality, latest year):
  crime_rate_per_100k     Total Criminal Code violations (excl. traffic), rate / 100,000
  violent_rate_per_100k   Total violent Criminal Code violations, rate / 100,000
  property_rate_per_100k  Total property crime violations, rate / 100,000
  safety_score            our own 0-100 score (higher = safer), inverse-scaled from the rate
  safety_basis            the municipality the rate is for

This is a CITY-LEVEL signal: every subarea in a municipality shares its rate (subarea-level
crime is only available for Vancouver [VPD coords] + Maple Ridge, a future refinement). Only
Metro Vancouver municipalities are mapped; subareas elsewhere (Fraser Valley, GTA, Sunshine
Coast...) are left out and read back as null, not a misleading number.

Join key: region_id == community_region_lookup.region_id (same id space as the other lookups).

Run:  python -m src.data.build_safety_deploy
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

V_TOTAL = "Total, all Criminal Code violations (excluding traffic) [50]"
V_VIOLENT = "Total violent Criminal Code violations [100]"
V_PROPERTY = "Total property crime violations [200]"
RATE_STAT = "Rate per 100,000 population"
COUNT_STAT = "Actual incidents"

# Our-own LOG inverse scale for the total crime rate → safety_score (documented, not
# AreaVibes'): a rate of SAFE_RATE/100k or below scores 100; UNSAFE_RATE/100k or above → 0,
# log-spaced between. Log (not linear) because Greater-Van rates span an order of magnitude —
# safe suburbs ~2900/100k up to small Fraser Valley highway towns >20000/100k — so a linear
# scale would clamp every high city to 0. Roughly: Port Moody ~92, Vancouver ~55, Chilliwack
# ~22, Hope (extreme) → 0.
SAFE_RATE = 2500.0
UNSAFE_RATE = 18000.0

# Our MLS "area" names → StatCan jurisdiction DGUID code(s). Multi-code areas (e.g. North
# Vancouver = City + District) are combined population-weighted. DGUID codes are unambiguous
# (avoid the municipal/rural GEO-string duplicates). See docs/crime_data_sources.md.
AREA_TO_CODES: dict[str, list[int]] = {
    # Metro Vancouver
    "Vancouver West": [59023], "Vancouver East": [59023],
    "Burnaby East": [59703], "Burnaby North": [59703], "Burnaby South": [59703],
    "Richmond": [59711],
    "Coquitlam": [59705], "Port Coquitlam": [59708], "Port Moody": [59016],
    "New Westminster": [59012],
    "North Vancouver": [59706, 59707],  # City + District (pop-weighted)
    "West Vancouver": [59026],
    "Surrey": [59704], "North Surrey": [59704], "Cloverdale": [59704],
    "South Surrey White Rock": [59704],  # dominated by South Surrey (= Surrey jurisdiction)
    "Delta": [59004], "N. Delta": [59004], "Ladner": [59004], "Tsawwassen": [59004],
    "Langley": [59930, 59731],  # City + Township (pop-weighted)
    "Maple Ridge": [59727], "Pitt Meadows": [59818],
    # Fraser Valley
    "Abbotsford": [59009],
    "Mission": [59734],
    "Chilliwack": [59724], "East Chilliwack": [59724], "Sardis": [59724],
    "Yarrow": [59724], "Cultus Lake & Area": [59724],
    "Agassiz": [59029], "Harrison Lake": [59029],  # Kent / Agassiz RCMP
    "Hope & Area": [59749],
}
BASIS_NAME = {
    59023: "Vancouver", 59703: "Burnaby", 59711: "Richmond", 59705: "Coquitlam",
    59708: "Port Coquitlam", 59016: "Port Moody", 59012: "New Westminster",
    59706: "North Vancouver", 59707: "North Vancouver", 59026: "West Vancouver",
    59704: "Surrey", 59004: "Delta", 59727: "Maple Ridge", 59818: "Pitt Meadows",
    59930: "Langley", 59731: "Langley", 59880: "White Rock",
    59009: "Abbotsford", 59734: "Mission", 59724: "Chilliwack",
    59029: "Kent (Agassiz)", 59749: "Hope",
}


_LOG_SAFE = math.log(SAFE_RATE)
_LOG_SPAN = math.log(UNSAFE_RATE) - _LOG_SAFE


def _safety_score(rate: float) -> float:
    if rate <= 0:
        return 100.0
    frac = (math.log(rate) - _LOG_SAFE) / _LOG_SPAN  # 0 at SAFE_RATE, 1 at UNSAFE_RATE
    return round(float(np.clip(100.0 * (1.0 - frac), 0.0, 100.0)), 1)


def _municipal_rates(statcan_path: Path, year: int) -> dict[int, dict]:
    """Per DGUID code: total/violent/property rate + derived population, for `year`."""
    df = pd.read_csv(statcan_path, low_memory=False)
    df["code"] = df["GEO"].str.extract(r"\[(\d+)\]").astype("Int64")
    d = df[df["REF_DATE"] == year]

    def val(code, viol, stat):
        r = d[(d["code"] == code) & (d["Violations"] == viol) & (d["Statistics"] == stat)]
        v = r["VALUE"].iloc[0] if len(r) else None
        return float(v) if v is not None and pd.notna(v) else None

    out: dict[int, dict] = {}
    for code in set(c for cs in AREA_TO_CODES.values() for c in cs):
        rate = val(code, V_TOTAL, RATE_STAT)
        inc = val(code, V_TOTAL, COUNT_STAT)
        if rate is None or inc is None or rate == 0:
            continue
        out[code] = {
            "pop": inc / rate * 1e5,
            "total_inc": inc,
            "violent_inc": val(code, V_VIOLENT, COUNT_STAT) or 0.0,
            "property_inc": val(code, V_PROPERTY, COUNT_STAT) or 0.0,
        }
    return out


def _combine(codes: list[int], rates: dict[int, dict]) -> dict | None:
    present = [c for c in codes if c in rates]
    if not present:
        return None
    pop = sum(rates[c]["pop"] for c in present)
    if pop == 0:
        return None
    tot = sum(rates[c]["total_inc"] for c in present)
    vio = sum(rates[c]["violent_inc"] for c in present)
    pro = sum(rates[c]["property_inc"] for c in present)
    return {
        "crime_rate_per_100k": round(tot / pop * 1e5, 1),
        "violent_rate_per_100k": round(vio / pop * 1e5, 1),
        "property_rate_per_100k": round(pro / pop * 1e5, 1),
        "safety_basis": BASIS_NAME[present[0]],
    }


def build(statcan_path: Path, lookup_path: Path, out_path: Path, year: int) -> pd.DataFrame:
    rates = _municipal_rates(statcan_path, year)
    # One resolved record per mapped area name.
    area_rec: dict[str, dict] = {}
    for area, codes in AREA_TO_CODES.items():
        rec = _combine(codes, rates)
        if rec:
            rec = dict(rec)
            rec["safety_score"] = _safety_score(rec["crime_rate_per_100k"])
            area_rec[area] = rec

    lk = pd.read_csv(lookup_path)
    lk["region_id"] = lk["region_id"].astype(str)
    rows = []
    for _, r in lk.drop_duplicates("region_id").iterrows():
        rec = area_rec.get(r["area_name"])
        if rec:
            rows.append({"region_id": r["region_id"], "safety_year": year, **rec})

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    covered = out["safety_basis"].nunique()
    print(f"[safety] {len(out)} subareas across {covered} municipalities ({year}) -> {out_path}")
    print(out.groupby("safety_basis")[["crime_rate_per_100k", "safety_score"]]
          .first().sort_values("safety_score", ascending=False).to_string())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the per-neighbourhood safety lookup (StatCan).")
    p.add_argument("--statcan_path",
                   default="data/raw/crime/statcan/35100184_greater_van_subset.csv")
    p.add_argument("--lookup_path", default="data/processed/community_region_lookup.csv")
    p.add_argument("--out_path", default="data/deploy/subarea_safety.parquet")
    p.add_argument("--year", type=int, default=2024)
    a = p.parse_args()
    build(Path(a.statcan_path), Path(a.lookup_path), Path(a.out_path), a.year)


if __name__ == "__main__":
    main()
