"""Distill a per-neighbourhood amenities/walkability lookup for the agent.

For each community (subarea) centre, count nearby everyday amenities from OpenStreetMap and
fold them into a single 0-100 amenity score. Powers the agent recommending neighbourhoods by
amenities (sort_by=amenities), the same way subarea_school.parquet powers sort_by=schools.

Metrics per subarea (from the community centre):
  grocery_count_1km   supermarkets / grocery / convenience within 1 km
  food_count_1km      restaurants / cafes / fast food within 1 km
  park_count_1km      parks within 1 km  (+ park_dist_km to the nearest)
  health_count_2km    pharmacies / clinics / doctors within 2 km
  hospital_dist_km    distance to the nearest hospital
  amenity_score       our own weighted, saturating composite (0-100) over the above

Data: OpenStreetMap via the Overpass API (open, ODbL — attribution required; do NOT scrape
AreaVibes' proprietary scores). To be polite to Overpass we do NOT hit it once per subarea:
we cluster the 700-odd centres into a couple of regional bounding boxes and pull each POI
category once per region (cached to data/raw/osm/), then measure distances locally. Ships a
tiny parquet (aggregate counts + score only, no raw POIs).

Join key: region_id == community_boundary.subarea_id (same as build_school_deploy).

Run:
    python -m src.data.build_amenities_deploy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.data._community import (COMMUNITY_BOUNDARY_PATH, COMMUNITY_LOOKUP_PATH,
                                 decode_point, load_boundary)

csv.field_size_limit(1 << 28)

# POI categories -> Overpass tag filters. One union query per category per region.
CATEGORIES: dict[str, list[str]] = {
    "grocery": ['nwr["shop"~"^(supermarket|grocery|convenience|greengrocer)$"]'],
    "food": ['nwr["amenity"~"^(restaurant|cafe|fast_food)$"]'],
    "park": ['nwr["leisure"="park"]'],
    "health": ['nwr["amenity"~"^(pharmacy|clinic|doctors)$"]'],
    "hospital": ['nwr["amenity"="hospital"]'],
}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "vancouver-livability/1.0 (neighbourhood amenity aggregation; OSM ODbL)"
REGION_LON_GAP = 2.0   # split centres into regions where longitude jumps by more than this
BBOX_BUFFER_DEG = 0.03  # ~3 km padding so edge subareas still see POIs just outside the box


def _load_centres(boundary_path: Path,
                  lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> list[tuple[str, float, float]]:
    out = []
    for r in load_boundary(boundary_path, lookup_path).to_dict("records"):
        try:
            lon, lat = decode_point(r["center_geom"])
        except (ValueError, KeyError, TypeError):
            continue
        out.append((str(r["subarea_id"]), lat, lon))
    return out


def _cluster_regions(centres: list[tuple[str, float, float]]) -> list[dict]:
    """Group centres into regional bounding boxes by longitude gaps (Metro Van vs GTA, etc.),
    so each POI category is fetched once per region instead of once per subarea."""
    by_lon = sorted(centres, key=lambda c: c[2])
    clusters: list[list[tuple[str, float, float]]] = [[]]
    prev = None
    for c in by_lon:
        if prev is not None and c[2] - prev > REGION_LON_GAP:
            clusters.append([])
        clusters[-1].append(c)
        prev = c[2]
    regions = []
    for cl in clusters:
        lats = [c[1] for c in cl]
        lons = [c[2] for c in cl]
        regions.append({
            "n": len(cl),
            "bbox": (min(lats) - BBOX_BUFFER_DEG, min(lons) - BBOX_BUFFER_DEG,
                     max(lats) + BBOX_BUFFER_DEG, max(lons) + BBOX_BUFFER_DEG),
        })
    return regions


def _overpass(query: str, timeout: int = 180) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(4):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            req = urllib.request.Request(endpoint, data=data,
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - network flakiness; back off and retry
            last = exc
            wait = 5 * (attempt + 1)
            print(f"[amenities]   overpass {endpoint} failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Overpass failed after retries: {last}")


def _fetch_pois(region_idx: int, bbox: tuple[float, float, float, float], category: str,
                filters: list[str], cache_dir: Path) -> np.ndarray:
    """Return an (N, 2) array of (lat, lon) for one category in one region, cached to disk."""
    # Include the rounded bbox in the cache key.  The previous r0/r1 names silently reused
    # Metro-Vancouver-only POIs after the boundary expanded east into Fraser Valley.
    bbox_key = "_".join(f"{v:.3f}".replace("-", "m").replace(".", "p") for v in bbox)
    cache = cache_dir / f"osm_{bbox_key}_{category}.json"
    if cache.exists():
        elements = json.loads(cache.read_text())["elements"]
    else:
        s, w, n, e = bbox
        body = "".join(f"{flt}({s},{w},{n},{e});" for flt in filters)
        query = f"[out:json][timeout:180];({body});out center;"
        print(f"[amenities]   fetching region {region_idx} / {category} ...")
        result = _overpass(query)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
        elements = result["elements"]
        time.sleep(2)  # be polite between Overpass calls
    pts = []
    for el in elements:
        if "lat" in el and "lon" in el:
            pts.append((el["lat"], el["lon"]))
        elif "center" in el:
            pts.append((el["center"]["lat"], el["center"]["lon"]))
    return np.array(pts, dtype=float).reshape(-1, 2)


def _dists_km(lat: float, lon: float, pts: np.ndarray) -> np.ndarray:
    """Equirectangular km distances from (lat, lon) to every point (same approx as schools)."""
    if len(pts) == 0:
        return np.array([])
    dlat = (pts[:, 0] - lat) * 111.0
    dlon = (pts[:, 1] - lon) * 111.0 * math.cos(math.radians(lat))
    return np.sqrt(dlat ** 2 + dlon ** 2)


def _sat(x: float, k: float) -> float:
    """Saturating 0..1 curve: ~0.63 at x=k, approaches 1. Diminishing returns on counts."""
    return 1.0 - math.exp(-x / k)


def _amenity_score(grocery: int, food: int, park_cnt: int, park_dist: float,
                   health: int, hosp_dist: float) -> float:
    """Our own 0-100 walkable-amenity composite (NOT AreaVibes' formula/weights).
    Everyday errands + dining + greenspace + health access + hospital proximity, each
    normalised with a saturating curve so a handful of nearby options already scores well."""
    grocery_n = _sat(grocery, 3)                       # ~3 groceries within 1km is plenty
    food_n = _sat(food, 12)                             # dining density
    park_prox = math.exp(-park_dist / 1.0) if math.isfinite(park_dist) else 0.0
    park_n = 0.6 * _sat(park_cnt, 3) + 0.4 * park_prox
    health_n = _sat(health, 6)                          # pharmacies/clinics within 2km
    hosp_n = math.exp(-hosp_dist / 8.0) if math.isfinite(hosp_dist) else 0.0
    score = (0.25 * grocery_n + 0.20 * food_n + 0.20 * park_n
             + 0.20 * health_n + 0.15 * hosp_n)
    return round(100.0 * score, 1)


def build(boundary_path: Path, out_path: Path, cache_dir: Path,
          lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> pd.DataFrame:
    centres = _load_centres(boundary_path, lookup_path)
    regions = _cluster_regions(centres)
    print(f"[amenities] {len(centres)} centres in {len(regions)} region(s): "
          + ", ".join(f"{r['n']}@{r['bbox'][0]:.1f},{r['bbox'][1]:.1f}" for r in regions))

    # Which region a centre belongs to (by bbox containment on longitude clusters).
    def region_of(lat: float, lon: float) -> int:
        for i, r in enumerate(regions):
            s, w, n, e = r["bbox"]
            if s <= lat <= n and w <= lon <= e:
                return i
        return -1

    # Fetch every category once per region up front (cached).
    poi: dict[tuple[int, str], np.ndarray] = {}
    for i, r in enumerate(regions):
        for cat, filters in CATEGORIES.items():
            poi[(i, cat)] = _fetch_pois(i, r["bbox"], cat, filters, cache_dir)
        counts = {c: len(poi[(i, c)]) for c in CATEGORIES}
        print(f"[amenities] region {i} POIs: {counts}")

    rows = []
    for sid, lat, lon in centres:
        ri = region_of(lat, lon)
        if ri < 0:
            continue
        gd = _dists_km(lat, lon, poi[(ri, "grocery")])
        fd = _dists_km(lat, lon, poi[(ri, "food")])
        pd_ = _dists_km(lat, lon, poi[(ri, "park")])
        hd = _dists_km(lat, lon, poi[(ri, "health")])
        hosp = _dists_km(lat, lon, poi[(ri, "hospital")])

        grocery = int((gd < 1.0).sum())
        food = int((fd < 1.0).sum())
        park_cnt = int((pd_ < 1.0).sum())
        park_dist = float(pd_.min()) if len(pd_) else math.inf
        health = int((hd < 2.0).sum())
        hosp_dist = float(hosp.min()) if len(hosp) else math.inf

        rows.append({
            "region_id": sid,
            "grocery_count_1km": grocery,
            "food_count_1km": food,
            "park_count_1km": park_cnt,
            "park_dist_km": round(park_dist, 2) if math.isfinite(park_dist) else None,
            "health_count_2km": health,
            "hospital_dist_km": round(hosp_dist, 1) if math.isfinite(hosp_dist) else None,
            "amenity_score": _amenity_score(grocery, food, park_cnt, park_dist,
                                             health, hosp_dist),
        })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[amenities] {len(out)} subareas -> {out_path} "
          f"(amenity_score: min {out['amenity_score'].min()}, "
          f"median {out['amenity_score'].median()}, max {out['amenity_score'].max()})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the per-neighbourhood amenities lookup (OSM).")
    p.add_argument("--boundary_path", default=str(COMMUNITY_BOUNDARY_PATH))
    p.add_argument("--lookup_path", default=str(COMMUNITY_LOOKUP_PATH))
    p.add_argument("--out_path", default="data/deploy/subarea_amenities.parquet")
    p.add_argument("--cache_dir", default="data/raw/osm")
    a = p.parse_args()
    build(Path(a.boundary_path), Path(a.out_path), Path(a.cache_dir), Path(a.lookup_path))


if __name__ == "__main__":
    main()
