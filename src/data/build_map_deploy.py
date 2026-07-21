"""Build lightweight city/community GeoJSON for the interactive map.

Only ``modified_geom`` is used. A community without a modified boundary is deliberately
excluded; this builder never falls back to the raw hull.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from shapely import make_valid, wkt
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

from src.data._community import (COMMUNITY_BOUNDARY_PATH, COMMUNITY_LOOKUP_PATH,
                                 REGION_PATH, load_boundary, load_subarea_lookup)
from src.infer.neighbourhoods import NeighbourhoodProfiles

COMMUNITY_OUTPUT = Path("data/deploy/community_map.geojson")
MUNICIPALITY_OUTPUT = Path("data/deploy/municipality_map.geojson")
MUNICIPALITY_PROFILE_PATH = Path("data/deploy/municipality_profile.parquet")
SIMPLIFY_TOLERANCE = 0.00006  # roughly 5-7 metres at Greater Vancouver latitudes
DISPLAY_CLOSE_TOLERANCE = 0.00015  # closes roughly 10-17 metre-wide display slits
MAX_DISPLAY_AREA_CHANGE_RATIO = 0.001  # never alter a city display area by more than 0.1%


def _display_municipality_name(name: object) -> str:
    """Use the concise product label while preserving the source municipality ID."""
    value = str(name)
    return "Surrey" if value == "Surrey and Whiterock" else value


def _remove_interior_rings(geom):
    """Fill display-only polygon holes without introducing any raw geometry.

    The modified boundaries contain hundreds of very narrow interior rings. Leaflet strokes every
    ring, which makes them look like stray internal lines. The exterior shells remain entirely from
    ``modified_geom``; only their holes are removed for the visual map artifact.
    """
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        filled = [Polygon(part.exterior) for part in geom.geoms]
        merged = unary_union(filled)
        if merged.geom_type == "Polygon":
            return merged
        if merged.geom_type == "MultiPolygon":
            return MultiPolygon(list(merged.geoms))
    return geom


def _close_narrow_slits(geom):
    """Close hairline exterior notches that Leaflet otherwise strokes as boundary spikes."""
    closed = geom.buffer(DISPLAY_CLOSE_TOLERANCE, join_style=2).buffer(
        -DISPLAY_CLOSE_TOLERANCE, join_style=2
    )
    if closed.is_empty or closed.geom_type not in {"Polygon", "MultiPolygon"}:
        return geom, False
    closed = _remove_interior_rings(closed)
    area_change = abs(closed.area - geom.area) / geom.area if geom.area else 0.0
    if area_change > MAX_DISPLAY_AREA_CHANGE_RATIO:
        return geom, False
    return closed, True


def _id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def _geometry(value: object):
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper().startswith("SRID="):
        text = text.split(";", 1)[1]
    geom = wkt.loads(text)
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    geom = _remove_interior_rings(geom)
    geom = geom.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    return geom if geom.geom_type in {"Polygon", "MultiPolygon"} else None


def _optional(value: object, digits: int = 1):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _market_by_id(profile: pd.DataFrame, id_col: str) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in profile.iterrows():
        rid = str(row[id_col])
        result.setdefault(rid, {})[str(row["property_type"])] = {
            "typical_price_cad": int(row["median_price"]),
            "sales_last_12m": int(row["sold_12m"]),
            "median_days_on_market": None if pd.isna(row["median_dom"]) else int(row["median_dom"]),
            "price_change_pct_year_over_year": _optional(row.get("price_change_yoy_pct")),
        }
    return result


def build(
    boundary_path: Path = COMMUNITY_BOUNDARY_PATH,
    lookup_path: Path = COMMUNITY_LOOKUP_PATH,
    region_path: Path = REGION_PATH,
    community_output: Path = COMMUNITY_OUTPUT,
    municipality_output: Path = MUNICIPALITY_OUTPUT,
) -> tuple[dict, dict]:
    boundary = load_boundary(boundary_path, lookup_path)
    missing_modified = boundary[boundary["modified_geom"].isna()]["subarea_id"].astype(str).tolist()
    boundary = boundary[boundary["modified_geom"].notna()].copy()

    lookup = load_subarea_lookup(lookup_path)
    lookup["area_id"] = _id(lookup["area_id"])
    region = pd.read_csv(region_path, low_memory=False)
    region["id"] = _id(region["id"])
    region["parent_id"] = _id(region["parent_id"])
    area_to_municipality = dict(region.loc[region["region_type"].eq("Area"), ["id", "parent_id"]].values)
    municipality_names = dict(region.loc[
        region["region_type"].eq("Municipality"), ["id", "name"]
    ].values)

    names = lookup.set_index("region_id")
    neighbourhoods = NeighbourhoodProfiles().df
    if neighbourhoods is None:
        raise ValueError("Neighbourhood profile is required to build map properties")
    market_by_community = _market_by_id(neighbourhoods, "region_id")
    scores = neighbourhoods.drop_duplicates("region_id").set_index("region_id")

    community_features = []
    geometry_groups: dict[str, list] = {}
    city_community_ids: dict[str, list[str]] = {}
    for _, row in boundary.iterrows():
        rid = str(row["subarea_id"])
        if rid not in names.index:
            continue
        geom = _geometry(row["modified_geom"])
        if geom is None:
            continue
        name_row = names.loc[rid]
        area_id = str(name_row["area_id"])
        municipality_id = str(area_to_municipality.get(area_id, ""))
        municipality = _display_municipality_name(
            municipality_names.get(municipality_id, name_row["area_name"])
        )
        score = scores.loc[rid] if rid in scores.index else None
        props = {
            "region_id": rid,
            "name": str(name_row["subarea_name"]),
            "area": str(name_row["area_name"]),
            "municipality_id": municipality_id,
            "municipality": municipality,
            "geometry_source": "modified_geom",
            "market": market_by_community.get(rid, {}),
            "livability_score": _optional(score.get("livability_score")) if score is not None else None,
            "amenity_score": _optional(score.get("amenity_score")) if score is not None else None,
            "transit_score": _optional(score.get("transit_score")) if score is not None else None,
            "safety_score": _optional(score.get("safety_score")) if score is not None else None,
            "best_school_score": _optional(score.get("best_school_score")) if score is not None else None,
        }
        community_features.append({
            "type": "Feature", "id": rid, "properties": props, "geometry": mapping(geom)
        })
        geometry_groups.setdefault(municipality_id, []).append(geom)
        city_community_ids.setdefault(municipality_id, []).append(rid)

    municipality_profile = pd.read_parquet(MUNICIPALITY_PROFILE_PATH)
    market_by_municipality = _market_by_id(municipality_profile, "region_id")
    municipality_features = []
    cleanup_guard_skipped_ids = []
    for municipality_id, geoms in geometry_groups.items():
        city_geom = unary_union(geoms).simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        city_geom, cleanup_applied = _close_narrow_slits(_remove_interior_rings(city_geom))
        if not cleanup_applied:
            cleanup_guard_skipped_ids.append(municipality_id)
        ids = city_community_ids[municipality_id]
        city_scores = scores.loc[scores.index.intersection(ids), "livability_score"]
        municipality_features.append({
            "type": "Feature",
            "id": municipality_id,
            "properties": {
                "region_id": municipality_id,
                "name": _display_municipality_name(
                    municipality_names.get(municipality_id, "Unknown")
                ),
                "geometry_source": "modified_geom_union",
                "community_count": len(ids),
                "market": market_by_municipality.get(municipality_id, {}),
                "livability_score": _optional(city_scores.mean()) if city_scores.notna().any() else None,
            },
            "geometry": mapping(city_geom),
        })

    communities = {
        "type": "FeatureCollection",
        "metadata": {
            "geometry_source": "modified_geom_only",
            "geometry_cleanup": ["interior_rings_removed"],
            "excluded_missing_modified": missing_modified,
        },
        "features": community_features,
    }
    municipalities = {
        "type": "FeatureCollection",
        "metadata": {
            "geometry_source": "modified_geom_union_only",
            "geometry_cleanup": ["interior_rings_removed", "narrow_slits_closed"],
            "cleanup_tolerance_degrees": DISPLAY_CLOSE_TOLERANCE,
            "max_cleanup_area_change_ratio": MAX_DISPLAY_AREA_CHANGE_RATIO,
            "cleanup_guard_skipped_ids": sorted(cleanup_guard_skipped_ids),
        },
        "features": sorted(municipality_features, key=lambda f: f["properties"]["name"]),
    }
    community_output.parent.mkdir(parents=True, exist_ok=True)
    municipality_output.parent.mkdir(parents=True, exist_ok=True)
    community_output.write_text(json.dumps(communities, separators=(",", ":")), encoding="utf-8")
    municipality_output.write_text(json.dumps(municipalities, separators=(",", ":")), encoding="utf-8")
    print(
        f"[map] municipalities={len(municipality_features)}, communities={len(community_features)}, "
        f"excluded_missing_modified={missing_modified}"
    )
    return municipalities, communities


def main() -> None:
    p = argparse.ArgumentParser(description="Build modified-geometry-only map GeoJSON.")
    p.add_argument("--boundary_path", default=str(COMMUNITY_BOUNDARY_PATH))
    p.add_argument("--lookup_path", default=str(COMMUNITY_LOOKUP_PATH))
    p.add_argument("--region_path", default=str(REGION_PATH))
    p.add_argument("--community_output", default=str(COMMUNITY_OUTPUT))
    p.add_argument("--municipality_output", default=str(MUNICIPALITY_OUTPUT))
    a = p.parse_args()
    build(Path(a.boundary_path), Path(a.lookup_path), Path(a.region_path),
          Path(a.community_output), Path(a.municipality_output))


if __name__ == "__main__":
    main()
