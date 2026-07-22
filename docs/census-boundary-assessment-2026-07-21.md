# Statistics Canada boundary assessment (2026-07-21)

## Source inventory

- Source: `data/processed/updated/july21/census_boundary.csv`
- Census year: 2021
- Rows: 29,641
- Geographic levels: 1,328 Census Subdivisions (`CSD`) and 28,313 Dissemination Areas (`DA`)
- Provincial coverage in this extract: British Columbia (`59`) and Ontario (`35`)
- BC CSD records: 751
- Geometry encoding: EWKB hex, SRID 4326
- Join keys: `geo_code` and `dguid`
- Important limitation: this file does not contain a readable geographic name or CSD type, so it needs an official code/name crosswalk before it can be presented to users.

The file is a CSD/DA boundary extract, not a single Greater Vancouver combined statistical area boundary.

## Comparison with the current Cities layer

The current Cities layer is a union of refined MLS `modified_geom` community boundaries. The comparison below uses 2021 CSD geometry, combining the two official CSDs for Langley and North Vancouver, and combining Surrey with White Rock to match the current market grouping.

| Result | Cities | Interpretation |
| --- | --- | --- |
| Strong alignment (intersection-over-union at least 0.85) | Abbotsford, Bowen Island, Burnaby, Coquitlam, Delta, Hope, Langley, Maple Ridge, New Westminster, Pitt Meadows, Richmond, Surrey, Vancouver | Official CSD geometry is suitable for validation and could support an optional administrative-boundary view. |
| Moderate difference | Chilliwack, Kent, Port Coquitlam | The official administrative boundary and the market-community union differ enough that replacement would visibly change the product meaning. |
| Deliberately smaller urban display | Mission, North Vancouver, West Vancouver | The current display intentionally omits large rural/natural areas. Replacing it with full CSD geometry would reverse the requested urban-focused cleanup. |
| Material semantic difference | Harrison Hot Springs, Port Moody | The current MLS union is respectively about 10.69× and 1.87× the official CSD area; these require a product decision rather than an automatic swap. |

Selected quantitative checks:

| City | Current / official area | Intersection-over-union | Notes |
| --- | ---: | ---: | --- |
| Maple Ridge | 0.99 | 0.974 | Near match; useful as an authoritative QA reference. |
| Mission | 0.34 | 0.322 | Expected: current Cities layer is the three-community urban footprint. |
| North Vancouver | 0.64 | 0.619 | Expected: current Cities layer is cropped to the urban North Shore. |
| West Vancouver | 0.71 | 0.674 | Expected: current Cities layer omits the large northern natural area. |
| Harrison Hot Springs | 10.69 | 0.094 | MLS market aggregation is much larger than the municipal CSD. |
| Port Moody | 1.87 | 0.526 | MLS market aggregation extends materially beyond the municipal CSD. |

## Recommendation

Do not replace the current Cities layer wholesale.

1. Keep refined MLS `modified_geom` boundaries as the market and livability display geometry.
2. Use Statistics Canada CSD geometry as an authoritative validation/reference layer.
3. If an administrative view is added later, expose it as an explicit `Official city boundaries (2021 Census)` overlay so users are not led to treat administrative and market boundaries as identical.
4. Maintain an explicit mapping table because several product cities combine multiple CSDs: Langley (city + district), North Vancouver (city + district), and Surrey (Surrey + White Rock for the current market grouping).
5. Keep Mission, North Vancouver, and West Vancouver urban-display rules when the default Cities layer is shown.

