---
title: Vancouver Property Value API
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Vancouver Property Value API

FastAPI backend for the Vancouver Property Value Prediction System. It estimates
the **total assessed property value** (land + improvement) for a City of Vancouver
address, with per-unit support for condos/strata.

This Space runs the container defined by the root `Dockerfile` (Python 3.13,
`uvicorn` on port 7860). Endpoints:

- `GET /health`
- `GET /resolve_address` — exact address → property / unit resolution
- `GET /fuzzy_lookup`, `GET /options` — lookup helpers
- `POST /predict` — property value estimate

Source: https://github.com/HaoC916/Vancouver-Land-Value-Prediction-System
