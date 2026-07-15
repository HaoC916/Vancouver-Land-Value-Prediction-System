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
import struct
from pathlib import Path

import numpy as np
import pandas as pd

csv.field_size_limit(1 << 28)
RADIUS_KM = 2.5


def _year(s: str) -> int:
    m = re.search(r"(\d{4})", s or "")
    return int(m.group(1)) if m else 0


def _decode_point(hexstr: str) -> tuple[float, float]:
    b = bytes.fromhex(hexstr)
    e = "<" if b[0] == 1 else ">"
    t = struct.unpack(e + "I", b[1:5])[0]
    o = 9 if t & 0x20000000 else 5
    lon = struct.unpack(e + "d", b[o:o + 8])[0]
    lat = struct.unpack(e + "d", b[o + 8:o + 16])[0]
    return lon, lat


def build(rank_path: Path, school_path: Path, boundary_path: Path, out_path: Path) -> pd.DataFrame:
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
    with open(boundary_path, newline="") as f:
        for r in csv.DictReader(f):
            lon, lat = _decode_point(r["center_geom"])
            dlat = (S[:, 0] - lat) * 111.0
            dlon = (S[:, 1] - lon) * 111.0 * math.cos(math.radians(lat))
            d = np.sqrt(dlat ** 2 + dlon ** 2)
            near = S[d < RADIUS_KM]
            if len(near):
                rows.append({
                    "region_id": r["subarea_id"],
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
    p.add_argument("--rank_path", default="data/processed/updated/school_rank.csv")
    p.add_argument("--school_path", default="data/processed/updated/school.csv")
    p.add_argument("--boundary_path", default="data/processed/community_boundary.csv")
    p.add_argument("--out_path", default="data/deploy/subarea_school.parquet")
    a = p.parse_args()
    build(Path(a.rank_path), Path(a.school_path), Path(a.boundary_path), Path(a.out_path))


if __name__ == "__main__":
    main()
