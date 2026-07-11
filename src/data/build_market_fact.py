"""Phase 0 — assemble a clean market fact table from the licensed listings source.

Analogous to ``clean_property_tax.py`` (raw -> interim clean parquet), but for the
market model: it joins listing records to their property features and geography and
writes one leakage-free row per listing event. Feature engineering / leakage-safe
history (``shift(1)``) is layered on top later in ``build_market_table.py``.

Naming: the licensed source is private. Its record key is renamed to ``market_key``
on read; link tables and the source itself are referenced by logical name only (via
the gitignored manifest, see ``src/data/_sources.py``). Nothing here hardcodes the
real source identity, filenames, or board codes.

Join backbone (verified ~90% connectivity): the ``propertyid`` column in the listing
records is empty, so everything routes through ``market_key``:
    listings.market_key -> link_property -> property   (features)
    listings.market_key -> link_address -> address -> region   (geography / scope)

Run (from repo root):
    python -m src.data.build_market_fact --region greater_vancouver
    python -m src.data.build_market_fact --region greater_vancouver --limit 200000  # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data._sources import load_manifest, region_filter, source_path

# Source key renamed on read so the private source's key name never propagates.
KEY = "market_key"

# Columns pulled from each source. Free-text remarks/URLs are intentionally NOT
# read (memory + they are the most licensing-sensitive fields; NLP is a later wave).
MARKET_USECOLS = [
    "id", "mlsid", "listdate", "listprice", "listpriceum", "originalprice",
    "soldprice", "solddate", "dom", "cdom", "status", "transactiontype", "closingdate",
]
PROPERTY_USECOLS = [
    "id", "yearbuilt", "totalbed", "bedplus", "bedinbase", "bednobase",
    "fullbath", "halfbath", "totalbath", "totalfloorarea", "grandtotalfloorarea",
    "floorareamain", "totalparking", "propertytype", "dwelltype", "dwellclass",
    "style", "storeys", "view", "category", "transactiontype", "constructiontype",
    "type", "totalareaum", "addressid",
]

MIN_PRICE = 50_000
MAX_PRICE = 100_000_000
MIN_YEAR = 2015
MAX_YEAR = 2025
CHUNK = 500_000


def _append(summary: list[dict], section: str, metric: str, value: object) -> None:
    summary.append({"section": section, "metric": metric, "value": value})


def _usecols_present(path: Path, usecols: list[str]) -> list[str]:
    """Intersect requested columns with the real header so schema drift is soft."""
    header = pd.read_csv(path, nrows=0)
    present = [c for c in usecols if c in header.columns]
    missing = [c for c in usecols if c not in header.columns]
    if missing:
        print(f"[market_fact]   note: {path.name} missing {missing} (skipped)")
    return present


def _resolve_scope(region_path: Path, cfg: dict) -> tuple[set[str], dict[str, str]]:
    """Region ids in scope + id->area-name map. Board codes come from the manifest."""
    reg = pd.read_csv(region_path, usecols=["id", "province", "board", "name"], dtype=str)
    boards = {b.upper() for b in cfg.get("boards", [])}
    prov = cfg.get("province")
    board_u = reg["board"].fillna("").str.upper()
    if boards:
        mask = board_u.isin(boards)
    elif prov:
        mask = reg["province"] == prov
    else:
        raise ValueError("Region filter must define 'boards' or 'province'.")
    ids = set(reg.loc[mask, "id"].dropna())
    name_map = dict(zip(reg["id"], reg["name"]))
    print(f"[market_fact] scope regions: {len(ids):,}")
    return ids, name_map


def _load_link(path: Path, key_col: str, val_col: str, keep: set[str] | None,
               keep_on: str) -> pd.DataFrame:
    """Load a two-column link table, optionally filtered, deduped 1:1 on key_col."""
    df = pd.read_csv(path, usecols=[key_col, val_col], dtype=str)
    if keep is not None:
        df = df[df[keep_on].isin(keep)]
    before = len(df)
    df = df.drop_duplicates(subset=[key_col], keep="first")
    if len(df) != before:
        print(f"[market_fact]   {path.name}: dropped {before - len(df):,} dup {key_col} rows")
    return df


def build_market_fact(manifest: dict, region_name: str, limit: int | None) -> tuple[pd.DataFrame, list[dict]]:
    summary: list[dict] = []
    cfg = region_filter(manifest, region_name)

    # 1) Geographic scope: which region ids, then which addresses, then which keys.
    region_ids, region_names = _resolve_scope(source_path(manifest, "region"), cfg)

    addr = pd.read_csv(source_path(manifest, "address"),
                       usecols=["id", "area_id", "subarea_id", "postal_code"], dtype=str)
    addr = addr[addr["area_id"].isin(region_ids)]
    scope_address_ids = set(addr["id"])
    addr["region_name"] = addr["area_id"].map(region_names)
    addr["subarea_name"] = addr["subarea_id"].map(region_names)
    print(f"[market_fact] in-scope addresses: {len(scope_address_ids):,}")

    link_addr = _load_link(source_path(manifest, "link_address"),
                           key_col="mlsid", val_col="addressid",
                           keep=scope_address_ids, keep_on="addressid").rename(columns={"mlsid": KEY})
    scope_keys = set(link_addr[KEY])
    print(f"[market_fact] in-scope market keys: {len(scope_keys):,}")

    # 2) Key -> property id, then load only the in-scope property features.
    link_prop = _load_link(source_path(manifest, "link_property"),
                           key_col="mlsid", val_col="propertyid",
                           keep=scope_keys, keep_on="mlsid").rename(
                               columns={"mlsid": KEY, "propertyid": "property_id"})
    scope_property_ids = set(link_prop["property_id"])

    prop_path = source_path(manifest, "property")
    prop_cols = _usecols_present(prop_path, PROPERTY_USECOLS)
    prop = pd.read_csv(prop_path, usecols=prop_cols, dtype={"id": str, "addressid": str}, low_memory=False)
    prop = prop[prop["id"].isin(scope_property_ids)]
    prop = prop.drop_duplicates(subset=["id"], keep="first")
    prop = prop.rename(columns={c: f"p_{c}" for c in prop.columns if c != "id"})
    prop = prop.rename(columns={"id": "property_id"})
    print(f"[market_fact] property feature rows: {len(prop):,}")

    # 2b) Key -> land parcel -> coordinates (sparse; a spatial signal where present).
    link_land = _load_link(source_path(manifest, "link_land"),
                           key_col="mlsid", val_col="landid",
                           keep=scope_keys, keep_on="mlsid").rename(
                               columns={"mlsid": KEY, "landid": "land_id"})
    land = pd.read_csv(source_path(manifest, "land"),
                       usecols=["id", "latitude", "longitude"], dtype={"id": str})
    land = land[land["id"].isin(set(link_land["land_id"]))].drop_duplicates(subset=["id"], keep="first")
    land = land.rename(columns={"id": "land_id"})
    for c in ("latitude", "longitude"):
        land[c] = pd.to_numeric(land[c], errors="coerce")
        land.loc[land[c] == 0, c] = np.nan
    key_coords = link_land.merge(land, on="land_id", how="left")[[KEY, "latitude", "longitude"]]
    print(f"[market_fact] key->coords rows: {len(key_coords):,}  "
          f"(lat present {key_coords['latitude'].notna().mean()*100:.0f}%)")

    # 3) Listing records — chunked, in-scope only, no free-text fields.
    mkt_path = source_path(manifest, "market_records")
    mkt_cols = _usecols_present(mkt_path, MARKET_USECOLS)
    frames, scanned = [], 0
    for chunk in pd.read_csv(mkt_path, usecols=mkt_cols, dtype={"id": str, "mlsid": str},
                             chunksize=CHUNK, low_memory=False):
        scanned += len(chunk)
        chunk = chunk.rename(columns={"mlsid": KEY, "id": "listing_id"})
        chunk = chunk[chunk[KEY].isin(scope_keys)]
        frames.append(chunk)
        if limit is not None and scanned >= limit:
            print(f"[market_fact] --limit: stopped after scanning {scanned:,} source rows")
            break
    mkt = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=mkt_cols)
    print(f"[market_fact] in-scope listing rows: {len(mkt):,}")
    _append(summary, "assemble", "listing_rows_in_scope", len(mkt))

    # 4) Assemble — every merge is 1:many from the listing side and guarded.
    n0 = len(mkt)
    out = mkt.merge(link_prop, on=KEY, how="left").merge(prop, on="property_id", how="left")
    out = out.merge(link_addr, on=KEY, how="left")
    out = out.merge(
        addr[["id", "region_name", "subarea_name", "postal_code"]].rename(columns={"id": "addressid"}),
        on="addressid", how="left")
    out = out.merge(key_coords, on=KEY, how="left")
    if len(out) != n0:
        raise ValueError(
            f"Row count changed after joins: {n0:,} -> {len(out):,} (non-unique merge = leakage risk)."
        )

    # 5) Clean + derive. Target is list price; sold price kept as a secondary label.
    out["listprice"] = pd.to_numeric(out["listprice"], errors="coerce")
    out["soldprice"] = pd.to_numeric(out.get("soldprice"), errors="coerce")
    out["listdate"] = pd.to_datetime(out["listdate"], errors="coerce")
    out = out[out["listprice"].between(MIN_PRICE, MAX_PRICE)]
    out = out.dropna(subset=["listdate"])
    out["list_year"] = out["listdate"].dt.year
    out["list_month"] = out["listdate"].dt.month
    out = out[out["list_year"].between(MIN_YEAR, MAX_YEAR)]

    # 6) Dedup exact re-syncs / relist artifacts of the same event.
    before = len(out)
    out = out.drop_duplicates(subset=["listing_id"], keep="first")
    out = out.drop_duplicates(subset=["property_id", "listdate", "listprice", "status"], keep="first")
    _append(summary, "dedup", "dropped_duplicate_rows", before - len(out))

    _append(summary, "table", "total_rows", len(out))
    _append(summary, "table", "total_columns", len(out.columns))
    _append(summary, "coverage", "has_property_features_rate", float(out["property_id"].notna().mean()))
    _append(summary, "coverage", "has_region_rate", float(out["region_name"].notna().mean()))
    _append(summary, "coverage", "soldprice_present_rate", float(out["soldprice"].notna().mean()))
    _append(summary, "coverage", "year_min", int(out["list_year"].min()) if len(out) else None)
    _append(summary, "coverage", "year_max", int(out["list_year"].max()) if len(out) else None)
    for c in out.columns:
        _append(summary, "missing_rate", c, float(out[c].isna().mean()))

    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the clean market fact table (Phase 0).")
    parser.add_argument("--region", default="greater_vancouver",
                        help="Logical region name defined in the manifest.")
    parser.add_argument("--manifest", default=None, help="Path to sources.local.json.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: stop after scanning this many source listing rows.")
    parser.add_argument("--out_path", default="data/interim/market_fact_gv.parquet")
    parser.add_argument("--summary_path", default="reports/figures/market_fact_summary.csv")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    print(f"[market_fact] region={args.region} limit={args.limit}")
    fact, summary = build_market_fact(manifest, args.region, args.limit)

    out_path, summary_path = Path(args.out_path), Path(args.summary_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fact.to_parquet(out_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"[market_fact] Saved: {out_path}  (rows={len(fact):,}, cols={len(fact.columns):,})")
    print(f"[market_fact] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
