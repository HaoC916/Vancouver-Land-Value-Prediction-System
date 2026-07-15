#!/usr/bin/env python3
"""Fetch all rows from public ArcGIS FeatureServer layers into raw/ as GeoJSON + CSV.

Keyset pagination on the object-id field (where OBJECTID > last, ordered ASC),
which is stable across pages unlike resultOffset. Layer metadata is saved
alongside the data for provenance. Re-running overwrites the snapshot files;
each snapshot is also recorded in _fetch_metadata.json.

Usage (from the repo root): python -m src.data.crime_fetch_arcgis [source_name ...]
(default: all)
"""
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW = Path("data/raw/crime")

SOURCES = [
    {
        "name": "burnaby_anonymized_crime",
        "service": "https://services5.arcgis.com/NgSjNljtJn9hphOU/arcgis/rest/services/AnonymizedCrime/FeatureServer",
        "out_dir": RAW / "burnaby",
        "note": "Burnaby RCMP Property Crime Dashboard backing layer. Property crime only "
                "(5 File_Type values), data from 2025-01 onward, anonymized point locations.",
    },
    {
        "name": "coquitlam_poco_property_crime",
        "service": "https://services2.arcgis.com/Q6Lq3evZUGfPrN7o/arcgis/rest/services/Property_Crime_Data_Layer_Coquitlam/FeatureServer",
        "out_dir": RAW / "coquitlam_poco",
        "note": "Coquitlam RCMP Property Crime Dashboard layer (Coquitlam + Port Coquitlam). "
                "4 property crime categories, ~2-year rolling window, approximate locations.",
    },
    {
        "name": "port_coquitlam_property_crime",
        "service": "https://services2.arcgis.com/Q6Lq3evZUGfPrN7o/arcgis/rest/services/Property_Crime_Data_Layer_Port_Coquitlam_2/FeatureServer",
        "out_dir": RAW / "coquitlam_poco",
        "note": "Port Coquitlam layer behind the Coquitlam RCMP PoCo dashboard "
                "(resolved via web map item 60d2e5cca5d94ebcad17f0f3bdd068ce). "
                "Same 4 property crime categories as the Coquitlam layer.",
    },
    {
        "name": "maple_ridge_rcmp_crime",
        "service": "https://services8.arcgis.com/5BI85L7dtV6PVZGB/arcgis/rest/services/RCMP%20Crime%20Dashboard%20Data/FeatureServer",
        "out_dir": RAW / "maple_ridge",
        "note": "Ridge Meadows RCMP crime layer behind the Maple Ridge Property/Persons Crime "
                "Dashboards (item 1e6a00ac75864cd0a7c32dc8b3f125ad). Founded crime, four main "
                "categories (person, property, other CC, drugs), 2016 to current per item description.",
    },
    {
        "name": "langley_township_property_crime",
        "service": "https://services5.arcgis.com/frpHL0Fv8koQRVWY/arcgis/rest/services/PropertyCrime/FeatureServer",
        "out_dir": RAW / "langley_township",
        "note": "Township of Langley Open Data 'Property Crimes' feature service.",
    },
]

UA = {"User-Agent": "nustream-crime-data-etl/0.1 (research; contact: chehd1502@gmail.com)"}


def get_json(url: str, params: dict) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"ArcGIS error from {full}: {data['error']}")
    return data


def epoch_ms_to_iso(value):
    if value is None or value == "":
        return ""
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError, OSError):
        return value  # DateOnly fields may already be 'YYYY-MM-DD' strings


def fetch_layer(service: str, layer: dict, out_dir: Path, manifest: list):
    lid = layer["id"]
    meta = get_json(f"{service}/{lid}", {"f": "json"})
    lname = meta.get("name", f"layer{lid}").replace("/", "_").replace(" ", "_")
    (out_dir / f"{lname}.layer_metadata.json").write_text(json.dumps(meta, indent=2))

    oid_field = meta.get("objectIdField") or next(
        (f["name"] for f in meta.get("fields", []) if f["type"] == "esriFieldTypeOID"), "OBJECTID")
    page_size = min(int(meta.get("maxRecordCount") or 1000), 2000)
    date_fields = {f["name"] for f in meta.get("fields", [])
                   if f["type"] in ("esriFieldTypeDate", "esriFieldTypeDateOnly")}

    is_table = meta.get("type") == "Table" or not meta.get("geometryType")
    features, last_oid = [], None
    while True:
        where = "1=1" if last_oid is None else f"{oid_field}>{last_oid}"
        params = {
            "f": "json" if is_table else "geojson", "where": where, "outFields": "*",
            "orderByFields": f"{oid_field} ASC", "resultRecordCount": page_size,
            "returnGeometry": "false" if is_table else "true", "outSR": 4326,
        }
        page = get_json(f"{service}/{lid}/query", params)
        feats = page.get("features", [])
        if is_table:  # normalize esri {attributes} rows to geojson-style features
            feats = [{"type": "Feature", "id": f["attributes"].get(oid_field),
                      "properties": f["attributes"], "geometry": None} for f in feats]
        if not feats:
            break
        features.extend(feats)
        last = feats[-1]
        last_oid = last.get("id", (last.get("properties") or {}).get(oid_field))
        if last_oid is None:
            raise RuntimeError(f"{lname}: cannot find object id on feature for pagination")
        print(f"  layer {lid} ({lname}): +{len(feats)} (total {len(features)})", flush=True)
        if len(feats) < page_size:
            break

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    geojson_path = out_dir / f"{lname}.geojson"
    geojson_path.write_text(json.dumps(
        {"type": "FeatureCollection",
         "fetched_at": fetched_at, "source": f"{service}/{lid}",
         "features": features}))

    # Flattened CSV: all property keys (epoch-ms dates converted to ISO) + lon/lat
    keys = []
    for f in features:
        for k in (f.get("properties") or {}):
            if k not in keys:
                keys.append(k)
    csv_path = out_dir / f"{lname}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["longitude", "latitude"])
        for f in features:
            props = f.get("properties") or {}
            row = [epoch_ms_to_iso(props.get(k)) if k in date_fields else props.get(k) for k in keys]
            geom = f.get("geometry") or {}
            coords = geom.get("coordinates") if geom.get("type") == "Point" else None
            row += [coords[0], coords[1]] if coords else ["", ""]
            w.writerow(row)

    manifest.append({
        "layer_id": lid, "layer_name": lname, "rows": len(features),
        "object_id_field": oid_field, "page_size": page_size,
        "files": [geojson_path.name, csv_path.name, f"{lname}.layer_metadata.json"],
        "fetched_at": fetched_at,
    })


def main():
    only = set(sys.argv[1:])
    for src in SOURCES:
        if only and src["name"] not in only:
            continue
        print(f"== {src['name']} ==", flush=True)
        src["out_dir"].mkdir(parents=True, exist_ok=True)
        root = get_json(src["service"], {"f": "json"})
        manifest = []
        for layer in root.get("layers", []) + root.get("tables", []):
            fetch_layer(src["service"], layer, src["out_dir"], manifest)
        (src["out_dir"] / f"_fetch_metadata.{src['name']}.json").write_text(json.dumps({
            "source_name": src["name"], "service_url": src["service"],
            "note": src["note"], "layers": manifest,
        }, indent=2))
        print(f"   -> {sum(m['rows'] for m in manifest)} rows across {len(manifest)} layer(s)")


if __name__ == "__main__":
    main()
