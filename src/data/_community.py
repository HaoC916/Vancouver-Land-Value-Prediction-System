"""Canonical community-boundary and market-trend inputs.

The July 2026 refresh separates IDs from names.  This module keeps the source paths and
the ID-to-name join in one place so every deploy builder reads the same data revision.
"""
from __future__ import annotations

from pathlib import Path
import re
import struct

import pandas as pd

from src.data._scope import GREATER_VANCOUVER_AREAS

UPDATED_JULY_20_DIR = Path("data/processed/updated/july 20")
COMMUNITY_BOUNDARY_PATH = UPDATED_JULY_20_DIR / "community_boundary_bc_modified.csv"
COMMUNITY_TREND_PATH = UPDATED_JULY_20_DIR / "community_market_trend.csv"
COMMUNITY_LOOKUP_PATH = Path("data/processed/community_region_lookup.csv")
REGION_PATH = Path("data/processed/updated/july 6/region.csv")
SCHOOL_PATH = Path("data/processed/updated/july 6/school.csv")
SCHOOL_RANK_PATH = Path("data/processed/updated/july 6/school_rank.csv")


def decode_point(value: object) -> tuple[float, float]:
    """Return ``(longitude, latitude)`` from PostGIS EWKB hex or EWKT text."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("Empty point geometry")
    if "POINT" in text.upper():
        match = re.search(
            r"POINT\s*(?:Z\s*)?\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)(?:\s+[-+0-9.eE]+)?\s*\)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Invalid EWKT point: {text[:80]}")
        return float(match.group(1)), float(match.group(2))

    raw = bytes.fromhex(text)
    endian = "<" if raw[0] == 1 else ">"
    geom_type = struct.unpack(endian + "I", raw[1:5])[0]
    offset = 9 if geom_type & 0x20000000 else 5
    lon = struct.unpack(endian + "d", raw[offset:offset + 8])[0]
    lat = struct.unpack(endian + "d", raw[offset + 8:offset + 16])[0]
    return lon, lat


def _normalise_id(series: pd.Series) -> pd.Series:
    """Stable string IDs without turning integer-like IDs into ``123.0``."""
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def load_subarea_lookup(path: Path = COMMUNITY_LOOKUP_PATH) -> pd.DataFrame:
    lookup = pd.read_csv(path, low_memory=False)
    required = {"region_id", "subarea_name", "area_name"}
    missing = required - set(lookup.columns)
    if missing:
        raise ValueError(f"Community lookup is missing columns: {sorted(missing)}")
    lookup = lookup.copy()
    lookup["region_id"] = _normalise_id(lookup["region_id"])
    if lookup["region_id"].duplicated().any():
        raise ValueError("Community lookup has duplicate region_id values")
    return lookup


def scoped_subarea_ids(lookup_path: Path = COMMUNITY_LOOKUP_PATH) -> set[str]:
    lookup = load_subarea_lookup(lookup_path)
    return set(lookup.loc[lookup["area_name"].isin(GREATER_VANCOUVER_AREAS), "region_id"])


def load_boundary(
    boundary_path: Path = COMMUNITY_BOUNDARY_PATH,
    lookup_path: Path = COMMUNITY_LOOKUP_PATH,
    *,
    scoped: bool = True,
) -> pd.DataFrame:
    boundary = pd.read_csv(boundary_path, low_memory=False)
    required = {"subarea_id", "geom", "modified_geom", "center_geom"}
    missing = required - set(boundary.columns)
    if missing:
        raise ValueError(f"Community boundary is missing columns: {sorted(missing)}")
    boundary = boundary.copy()
    boundary["subarea_id"] = _normalise_id(boundary["subarea_id"])
    if boundary["subarea_id"].duplicated().any():
        raise ValueError("Community boundary has duplicate subarea_id values")
    if scoped:
        boundary = boundary[boundary["subarea_id"].isin(scoped_subarea_ids(lookup_path))].copy()
    return boundary


def load_subarea_trend(
    trend_path: Path = COMMUNITY_TREND_PATH,
    lookup_path: Path = COMMUNITY_LOOKUP_PATH,
    *,
    scoped: bool = True,
) -> pd.DataFrame:
    """Load Subarea rows and attach subarea/board-area names from the master lookup."""
    trend = pd.read_csv(trend_path, na_values=["NULL"], low_memory=False)
    required = {"geo_level", "region_id", "period_start", "property_type"}
    missing = required - set(trend.columns)
    if missing:
        raise ValueError(f"Community trend is missing columns: {sorted(missing)}")
    trend = trend[trend["geo_level"].eq("Subarea")].copy()
    trend["region_id"] = _normalise_id(trend["region_id"])

    # Refreshed files intentionally contain IDs only.  Drop any stale name columns if an
    # enriched input is supplied, then make the lookup the single source of truth.
    name_cols = [c for c in ("subarea_name", "area_id", "area_name", "area_region_type")
                 if c in trend.columns]
    trend = trend.drop(columns=name_cols)
    lookup = load_subarea_lookup(lookup_path)
    trend = trend.merge(lookup, on="region_id", how="left", validate="many_to_one")
    if trend[["subarea_name", "area_name"]].isna().any(axis=None):
        missing_ids = trend.loc[trend["subarea_name"].isna(), "region_id"].nunique()
        raise ValueError(f"Community trend has {missing_ids} unmapped Subarea region_id values")
    if scoped:
        trend = trend[trend["area_name"].isin(GREATER_VANCOUVER_AREAS)].copy()
    return trend


def load_municipality_trend(
    trend_path: Path = COMMUNITY_TREND_PATH,
    region_path: Path = REGION_PATH,
) -> pd.DataFrame:
    """Load Municipality rows and attach English/Chinese names from ``region.csv``."""
    trend = pd.read_csv(trend_path, na_values=["NULL"], low_memory=False)
    trend = trend[trend["geo_level"].eq("Municipality")].copy()
    trend["region_id"] = _normalise_id(trend["region_id"])
    region = pd.read_csv(region_path, low_memory=False)
    region = region[region["region_type"].eq("Municipality")].copy()
    region["id"] = _normalise_id(region["id"])
    names = region[["id", "name", "name_cn", "province", "board", "latitude", "longitude"]].rename(
        columns={"id": "region_id", "name": "municipality_name"}
    )
    out = trend.merge(names, on="region_id", how="left", validate="many_to_one")
    if out["municipality_name"].isna().any():
        raise ValueError("Municipality trend contains IDs missing from region.csv")
    return out
