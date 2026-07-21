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

- 12 tests pass.
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
