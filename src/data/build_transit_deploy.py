"""Distill a per-neighbourhood transit/commute lookup for the agent.

For each community (subarea) centre, measure transit access from open GTFS feeds and fold it
into a 0-100 transit score. Powers the agent recommending neighbourhoods by commute
(sort_by=commute), the same way subarea_amenities.parquet powers sort_by=amenities.

Metrics per subarea (from the community centre):
  transit_stops_800m       boarding stops within 800 m (walkable transit density)
  rapid_transit_dist_km    distance to the nearest rapid-transit stop
                           (SkyTrain / SeaBus / West Coast Express)
  transit_score            our own weighted, saturating composite (0-100)

Data: TransLink open GTFS (Metro Vancouver) + BC Transit's Fraser Valley operator feed
(Abbotsford/Mission/Chilliwack local buses) for the walkable stop-density count. Rapid transit
comes ONLY from TransLink (GTFS route_type {1 SkyTrain, 2 West Coast Express, 4 SeaBus}) — the
Fraser Valley has none of its own, though Mission's WCE station is in the TransLink feed. We
score every in-scope subarea that has a centre. A community with no nearby stops receives a
real zero-density score instead of being treated as missing data. Bus (route_type 3) and
HandyDART are excluded from the
rapid-transit distance but their stops still count toward stops_800m.

Join key: region_id == community_boundary.subarea_id (same as build_school_deploy).

Run:
    python -m src.data.build_transit_deploy
"""
from __future__ import annotations

import argparse
import csv
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data._community import (COMMUNITY_BOUNDARY_PATH, COMMUNITY_LOOKUP_PATH,
                                 decode_point, load_boundary)

csv.field_size_limit(1 << 28)

GTFS_URL = "https://gtfs-static.translink.ca/gtfs/google_transit.zip"
# BC Transit "Fraser Valley Region" operator feed (Abbotsford/Mission/Chilliwack/Agassiz-Harrison
# local buses) — adds Fraser Valley bus stops to the walkable stop-density count. Same stops.txt
# format as TransLink. Rapid transit (SkyTrain/SeaBus/West Coast Express) still comes only from
# TransLink; the Fraser Valley has none of its own (Mission's WCE station is in the TransLink feed).
BCT_FRASER_VALLEY_URL = "https://bct.tmix.se/Tmix.Cap.TdExport.WebApi/gtfs/?operatorIds=13"
USER_AGENT = "vancouver-livability/1.0 (neighbourhood transit aggregation; TransLink + BC Transit GTFS)"
RAPID_ROUTE_TYPES = {"1", "2", "4"}  # SkyTrain, West Coast Express, SeaBus
STOP_RADIUS_KM = 0.8


def _dists_km(lat: float, lon: float, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.array([])
    dlat = (pts[:, 0] - lat) * 111.0
    dlon = (pts[:, 1] - lon) * 111.0 * math.cos(math.radians(lat))
    return np.sqrt(dlat ** 2 + dlon ** 2)


def _sat(x: float, k: float) -> float:
    return 1.0 - math.exp(-x / k)


def _transit_score(stops_800m: int, rapid_dist_km: float) -> float:
    """Our own 0-100 commute/transit composite (NOT AreaVibes' formula). Walkable stop
    density + proximity to rapid transit, which drives commute value most in Metro Van."""
    stop_n = _sat(stops_800m, 10)                       # ~10 stops within 800m is well served
    rapid_n = math.exp(-rapid_dist_km / 1.2) if math.isfinite(rapid_dist_km) else 0.0
    return round(100.0 * (0.45 * stop_n + 0.55 * rapid_n), 1)


def _ensure_gtfs(cache_zip: Path, url: str, label: str) -> Path:
    if not cache_zip.exists():
        cache_zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"[transit] downloading {label} GTFS -> {cache_zip}")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=180) as resp, open(cache_zip, "wb") as f:
            f.write(resp.read())
    return cache_zip


def _load_boarding_stops(cache_zip: Path) -> np.ndarray:
    """Boarding-stop coords (location_type 0/blank) from any GTFS zip's stops.txt."""
    with zipfile.ZipFile(cache_zip) as z:
        stops = pd.read_csv(z.open("stops.txt"), dtype=str)
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    loc = stops.get("location_type", pd.Series("0", index=stops.index)).fillna("0").replace("", "0")
    boarding = stops[loc == "0"]
    return boarding[["stop_lat", "stop_lon"]].to_numpy(dtype=float)


def _load_gtfs(cache_zip: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (all boarding-stop coords, rapid-transit-stop coords) as (N, 2) lat/lon arrays."""
    with zipfile.ZipFile(cache_zip) as z:
        with z.open("stops.txt") as f:
            stops = pd.read_csv(f, dtype=str)
        with z.open("routes.txt") as f:
            routes = pd.read_csv(f, dtype=str)
        with z.open("trips.txt") as f:
            trips = pd.read_csv(f, usecols=["route_id", "trip_id"], dtype=str)
        rapid_routes = set(routes.loc[routes["route_type"].isin(RAPID_ROUTE_TYPES), "route_id"])
        rapid_trips = set(trips.loc[trips["route_id"].isin(rapid_routes), "trip_id"])
        rapid_stop_ids: set[str] = set()
        with z.open("stop_times.txt") as f:  # ~90MB — scan in chunks, keep only rapid stops
            for chunk in pd.read_csv(f, usecols=["trip_id", "stop_id"], dtype=str,
                                     chunksize=500_000):
                hit = chunk.loc[chunk["trip_id"].isin(rapid_trips), "stop_id"]
                rapid_stop_ids.update(hit.tolist())

    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    # location_type 0/blank = actual boarding stops (1 = station, 2 = entrance) — count only 0.
    loc = stops["location_type"].fillna("0").replace("", "0")
    boarding = stops[loc == "0"]
    all_pts = boarding[["stop_lat", "stop_lon"]].to_numpy(dtype=float)
    rapid = stops[stops["stop_id"].isin(rapid_stop_ids)]
    rapid_pts = rapid[["stop_lat", "stop_lon"]].to_numpy(dtype=float)
    print(f"[transit] {len(all_pts):,} boarding stops; {len(rapid_pts)} rapid-transit stops "
          f"({len(rapid_routes)} rapid routes)")
    return all_pts, rapid_pts


def build(boundary_path: Path, out_path: Path, cache_zip: Path, bct_cache_zip: Path,
          lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> pd.DataFrame:
    all_pts, rapid_pts = _load_gtfs(_ensure_gtfs(cache_zip, GTFS_URL, "TransLink"))
    # Add Fraser Valley bus stops (BC Transit) to the walkable stop-density set, so Abbotsford /
    # Mission subareas get scored too. Rapid-transit stops stay TransLink-only.
    try:
        bct_pts = _load_boarding_stops(_ensure_gtfs(bct_cache_zip, BCT_FRASER_VALLEY_URL, "BC Transit FV"))
        all_pts = np.vstack([all_pts, bct_pts])
        print(f"[transit] + {len(bct_pts):,} Fraser Valley (BC Transit) boarding stops")
    except Exception as exc:  # noqa: BLE001 - FV is optional; TransLink-only still valid
        print(f"[transit] WARNING: BC Transit FV feed unavailable ({exc}); Metro Van only")

    rows = []
    for r in load_boundary(boundary_path, lookup_path).to_dict("records"):
        try:
            lon, lat = decode_point(r["center_geom"])
        except (ValueError, KeyError, TypeError):
            continue
        sd = _dists_km(lat, lon, all_pts)
        rd = _dists_km(lat, lon, rapid_pts)
        stops_800m = int((sd < STOP_RADIUS_KM).sum())
        rapid_dist = float(rd.min()) if len(rd) else math.inf
        rows.append({
            "region_id": str(r["subarea_id"]),
            "transit_stops_800m": stops_800m,
            "rapid_transit_dist_km": round(rapid_dist, 1) if math.isfinite(rapid_dist) else None,
            "transit_score": _transit_score(stops_800m, rapid_dist),
        })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[transit] {len(out)} subareas -> {out_path} "
          f"(transit_score: min {out['transit_score'].min()}, "
          f"median {out['transit_score'].median()}, max {out['transit_score'].max()})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the per-neighbourhood transit lookup (TransLink + BC Transit GTFS).")
    p.add_argument("--boundary_path", default=str(COMMUNITY_BOUNDARY_PATH))
    p.add_argument("--lookup_path", default=str(COMMUNITY_LOOKUP_PATH))
    p.add_argument("--out_path", default="data/deploy/subarea_transit.parquet")
    p.add_argument("--cache_zip", default="data/raw/gtfs/translink_google_transit.zip")
    p.add_argument("--bct_cache_zip", default="data/raw/gtfs/bctransit_fraser_valley.zip")
    a = p.parse_args()
    build(Path(a.boundary_path), Path(a.out_path), Path(a.cache_zip), Path(a.bct_cache_zip),
          Path(a.lookup_path))


if __name__ == "__main__":
    main()
