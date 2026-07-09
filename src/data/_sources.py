"""Resolve logical source names to local file paths via a gitignored manifest.

The underlying data is licensed and private, so its real filenames/paths and the
region-board codes used to scope it must never be committed. Committed code refers
to sources by logical name only; the real mapping lives in
``config/sources.local.json`` (gitignored). See ``config/sources.example.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MANIFEST = Path("config/sources.local.json")


def load_manifest(path: Path | None = None) -> dict:
    path = path or DEFAULT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {path}\n"
            "Copy config/sources.example.json to config/sources.local.json and point "
            "it at your local (private, non-committed) data files."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def source_path(manifest: dict, logical_name: str) -> Path:
    """Return the local Path for a logical source name."""
    sources = manifest.get("sources", {})
    if logical_name not in sources:
        raise KeyError(
            f"Logical source '{logical_name}' is not defined in the manifest 'sources'."
        )
    return Path(manifest.get("data_root", ".")) / sources[logical_name]


def region_filter(manifest: dict, region_name: str) -> dict:
    """Return the {boards, province} scoping config for a logical region name."""
    regions = manifest.get("regions", {})
    if region_name not in regions:
        raise KeyError(
            f"Region '{region_name}' is not defined in the manifest 'regions'."
        )
    return regions[region_name]
