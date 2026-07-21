# Vancouver Property Project — Current Handoff Log

This is the short, code-adjacent handoff. Historical design decisions remain in the portfolio
handoff; this file records the current data revision, reproducible build entry points, validation
results, and known limitations.

## 2026-07-20 — July community-data refresh

### Canonical local inputs

- `data/processed/updated/july 20/community_boundary_bc_modified.csv`
- `data/processed/updated/july 20/community_market_trend.csv`
- `data/processed/community_region_lookup.csv`
- `data/processed/updated/july 6/region.csv`
- `data/processed/updated/july 6/school.csv`
- `data/processed/updated/july 6/school_rank.csv`

These large/private inputs stay gitignored. `src/data/_community.py` is the single source of truth
for paths, Subarea name joins, scope filtering, and the refreshed EWKT/legacy EWKB centre parser.
The accidental leading space in the old `updated/ july 20` folder was removed.

### Rebuilt deploy artifacts

- `market_trend.parquet`: 44,086 scoped rows, 33 board areas, 2021-05 through **2026-06**.
- `neighbourhood_profile.parquet`: 844 rows across 366 recently active neighbourhoods.
- `subarea_school.parquet`: 276 scoped communities with a ranked school within 2.5 km.
- `subarea_amenities.parquet`: 358 scoped boundary centres; OSM bbox now extends east through Hope.
- `subarea_transit.parquet`: 358 scoped boundary centres using TransLink + BC Transit Fraser Valley.
- `subarea_safety.parquet`: unchanged; 2024 StatCan city rates with the deployed inverse-log score.
- Composite `livability_score` remains runtime-computed in `src/infer/neighbourhoods.py`.

Rebuild commands:

```bash
.venv/bin/python -m src.data.build_school_deploy
.venv/bin/python -m src.data.build_amenities_deploy
.venv/bin/python -m src.data.build_transit_deploy
.venv/bin/python -m src.data.build_neighbourhood_deploy
.venv/bin/python -m src.data.build_market_trend_deploy
```

OSM cache filenames include the requested bbox. This prevents a wider Fraser Valley build from
silently reusing the old Metro-Vancouver-only cache.

### Validation

- 17 tests pass (including strict modified-only map API coverage).
- Chilliwack: 11/11 profile communities have amenities, transit, safety, and livability.
- Sardis: 9/9; Mission: 9/9; Abbotsford: 9/9.
- Fraser Valley boundary-backed profiles now receive spatial scores; communities with no nearby
  transit get a real low/zero access score rather than a misleading null.

Known boundary mismatch: 12 recently active trend Subareas are not present in the 409-row refreshed
boundary export (for example Harrison Mills, Laidlaw, Passage Island, and several small Metro
Vancouver subareas). They retain market price/trend data but correctly remain null for spatial
scores. Do not fabricate centres for them; resolve in the next source boundary export.

### Model-report cleanup

The previously untracked market-model metadata and aggregate evaluation reports are intentional,
small reproducibility artifacts and should be versioned. The large model joblib remains ignored and
is deployed through Hugging Face LFS. Before versioning, metadata was corrected to state that
coordinates are excluded (avoids train/serve skew). `src/eval/market_feature_importance.py` rebuilds
importance from the current segmented bundle and weights detached-direct / attached-ppsf / attached-
direct routes by their out-of-time usage; it replaces the stale pre-segmentation importance report.

## 2026-07-20 — Municipality market capability

Implemented as a separate change from the Subarea refresh:

- `build_municipality_deploy.py` filters the refreshed `geo_level=Municipality` rows through the
  21 in-scope municipalities derived from the Area→Municipality hierarchy in `region.csv`.
- `municipality_trend.parquet`: 4,214 monthly/type rows through 2026-06.
- `municipality_profile.parquet`: 83 municipality×property-type recommendation rows.
- Agent tools: `get_municipality_trend` for a whole-city aggregate and
  `recommend_municipalities` by budget/price, observed year-over-year appreciation, or liquidity.
- Thin markets are excluded when at least three reliable alternatives (24+ sales/year) exist.
  Appreciation is explicitly historical, not a forecast.

The next product branches remain investment rental-yield data and vacation/resort coverage.

### Verified deployment

- GitHub branch `market-price-v2`: `4ddd358`.
- Hugging Face backend Space: `3ffa394`.
- Hugging Face frontend Space: `3db8183`.
- Live checks: Chilliwack neighbourhood livability, Burnaby June-2026 Subarea/board trend,
  Burnaby whole-municipality condo trend, and municipality recommendations under a $500k condo
  budget all returned the refreshed values.

## 2026-07-20 — Interactive Map

The frontend now has a third top-level `Map` entry beside Chat and Search. It provides:

- Cities and Communities views with city-to-community drill-down.
- Condo / house / townhouse market snapshots.
- Choropleths for typical price and livability; the Cities view also exposes observed year-over-year
  price change.
- A selected-area detail panel and `Ask in Chat`, which opens Chat with an editable contextual prompt.

Map geometry has a stricter rule than the general spatial-data pipeline: **only `modified_geom` is
rendered**. `build_map_deploy.py` produces 357 community features; the one in-scope boundary without
modified geometry (Kawkawa Lake, region 308) is intentionally omitted, never replaced by raw geometry.
The 21 city shapes are unions of those same modified community polygons. Generated artifacts are
`community_map.geojson` and `municipality_map.geojson`; the API serves them from
`/map/communities` and `/map/municipalities`.

Display-only interior rings are removed from both GeoJSON artifacts. These rings were exclusions and
thin gaps inside the modified polygons, and Leaflet rendered all 889 city-level rings as distracting
internal lines. Exterior shells still come exclusively from `modified_geom`. The product display label
for source municipality `Surrey and Whiterock` is normalized to `Surrey` while retaining its source ID.

City display boundaries also receive a 0.00015-degree morphological closing to remove the narrow
exterior slits that Leaflet rendered as dangling spikes. A 0.1% per-city area-change guard preserves the
pre-cleaned shape when closing would be too invasive (currently Chilliwack). Validation keeps 21 city and
357 community IDs/properties unchanged, all geometries valid, zero interior rings, unchanged total bounds,
and a maximum city-area change of 0.0348%. Pre-spike backups, checksums, and reviewed cleaned copies are in
`data/processed/updated/july 20/` and remain ignored/private with the canonical inputs.

West Vancouver and North Vancouver use an urban-focused **Cities-layer display footprint** capped at
latitude 49.381. This removes the large Furry Creek/Lions Bay/Cypress and Indian Arm natural/rural reaches
from the city overview while retaining all 34/33 market communities, metrics, and untrimmed Community-layer
polygons. West Vancouver retains 26.827% and North Vancouver 83.381% of their former display-union area;
the other 19 city shapes are exactly unchanged. The pre-trim and validated post-trim GeoJSON files are
checksummed in the July 20 backup manifest.

Mission uses the same urban-focused Cities-layer principle, but an explicit community allowlist is more
accurate than a latitude cut: Mission BC, Mission-West, and Hatzic form the displayed city footprint.
Lake Errock, Dewdney Deroche, Hemlock, Durieu, Stave Falls, and Steelhead remain intact in Community view
and in all 9-community municipality market/livability metrics. The city card reports “3 urban communities
shown · 9 market communities.” Mission's display area is 3.592% of its former rural-inclusive union and
the other 20 city shapes are exactly unchanged.
