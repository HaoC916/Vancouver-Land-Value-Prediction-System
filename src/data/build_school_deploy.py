"""Distill a per-neighbourhood school-quality lookup for the agent.

For each community (subarea) center, find ranked schools within a radius and aggregate their
Fraser-Institute score (0-10, higher is better) into best / average / count. Powers the
agent recommending neighbourhoods by school quality. Public school data (locations + Fraser
rankings) — not MLS/listing data. Ships a tiny parquet (aggregate scores only).

Join keys: school location id "HS_<n>" <-> school_rank.school_id "<n>"; community center from
community_boundary.center_geom (EWKB hex, SRID 4326). Uses schools within RADIUS_KM of the
subarea center (a proxy for the catchment; school.csv also has catchment polygons for a
future point-in-polygon refinement).

Run:
    python -m src.data.build_school_deploy
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data._community import (COMMUNITY_BOUNDARY_PATH, COMMUNITY_LOOKUP_PATH,
                                 SCHOOL_PATH, SCHOOL_RANK_PATH, decode_point, load_boundary)

csv.field_size_limit(1 << 28)
RADIUS_KM = 2.5


def _year(s: str) -> int:
    m = re.search(r"(\d{4})", s or "")
    return int(m.group(1)) if m else 0


def build(rank_path: Path, school_path: Path, boundary_path: Path, out_path: Path,
          lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> pd.DataFrame:
    # Best recent Fraser 0-10 score per numeric school id.
    best: dict[str, tuple[int, float]] = {}
    with open(rank_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["score_value"])
            except (TypeError, ValueError):
                continue
            if not 0 <= v <= 10:
                continue
            sid, y = r["school_id"], _year(r["year"])
            if sid not in best or y > best[sid][0]:
                best[sid] = (y, v)

    schools = []
    with open(school_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                la, lo = float(r["latitude"]), float(r["longitude"])
            except (TypeError, ValueError):
                continue
            num = r["id"].split("_")[-1]
            if num in best:
                schools.append((la, lo, best[num][1]))
    S = np.array(schools)
    print(f"[school] schools with score+coords: {len(S):,}")

    rows = []
    for r in load_boundary(boundary_path, lookup_path).to_dict("records"):
        lon, lat = decode_point(r["center_geom"])
        dlat = (S[:, 0] - lat) * 111.0
        dlon = (S[:, 1] - lon) * 111.0 * math.cos(math.radians(lat))
        d = np.sqrt(dlat ** 2 + dlon ** 2)
        near = S[d < RADIUS_KM]
        if len(near):
            rows.append({
                "region_id": str(r["subarea_id"]),
                "best_school_score": round(float(near[:, 2].max()), 1),
                "avg_school_score": round(float(near[:, 2].mean()), 1),
                "school_count": int(len(near)),
            })
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[school] {len(out)} subareas with a school within {RADIUS_KM}km -> {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build the per-neighbourhood school-quality lookup.")
    p.add_argument("--rank_path", default=str(SCHOOL_RANK_PATH))
    p.add_argument("--school_path", default=str(SCHOOL_PATH))
    p.add_argument("--boundary_path", default=str(COMMUNITY_BOUNDARY_PATH))
    p.add_argument("--lookup_path", default=str(COMMUNITY_LOOKUP_PATH))
    p.add_argument("--out_path", default="data/deploy/subarea_school.parquet")
    a = p.parse_args()
    build(Path(a.rank_path), Path(a.school_path), Path(a.boundary_path), Path(a.out_path),
          Path(a.lookup_path))


if __name__ == "__main__":
    main()
