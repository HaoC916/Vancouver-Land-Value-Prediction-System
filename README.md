# CMPT 733 Final Project — WEB CRaWLer

## Project Overview
This project predicts Vancouver land **assessment** value (`CURRENT_LAND_VALUE`) and reports where prediction error is larger or smaller across neighbourhoods.

This repository uses one consolidated final pipeline:
- clean property data
- standardize external sources
- build one merged model table
- train and evaluate one final model

## At a Glance
- Target: `CURRENT_LAND_VALUE`
- Final merged table: `data/processed/model_table.parquet`
- Final trainer: `python -m src.models.train_model`
- Notebook demo: `final_submission_demo.ipynb`
- Web demo apps:
  - React frontend + REST API backend
  - Streamlit demo: `app.py`
- Census status: deferred by default

## Notebook Demo Entry Point
This repository includes a root-level notebook: `final_submission_demo.ipynb`.

It is the easiest entry point for instructors or readers who prefer Jupyter Notebook. The notebook does not duplicate core project logic. Instead, it runs the existing Python modules in sequence and displays final outputs. It checks raw files, runs the pipeline steps, and shows final metrics, tables, and figures.

If you prefer a step-by-step notebook workflow, open `final_submission_demo.ipynb` and run the cells from top to bottom.

## Google Drive Raw Data Link
Raw data is shared outside git and should be downloaded from:  
https://drive.google.com/file/d/1ENhMgJCcVpjG5r3AhCcJd4pXKMpqBuBP/view?usp=sharing

## Target and Evaluation Protocol
- Prediction target: `CURRENT_LAND_VALUE`
- Train split: `REPORT_YEAR < 2024`
- Test split: `REPORT_YEAR >= 2024`
- Official final model (`src/models/train_model.py`) uses **train-only target encoding** on:
  - `PROPERTY_POSTAL_CODE`
  - `NEIGHBOURHOOD_CODE`
  - `LEGAL_TYPE`
- Default encoding behavior is Mode B:
  - replace raw high-cardinality versions of those 3 fields with encoded numeric versions
- Metrics:
  - RMSE
  - MAE
  - Median APE
  - robust RMSE/MAE (capped using training p99.5)

## Which Data Are Used in the Final Pipeline?
### Directly used
- Property tax:
  - Raw: `data/raw/property-tax-report.csv`
  - Cleaned fact table: `data/interim/property_tax_clean.parquet`
- Mortgage yearly features:
  - Raw: `data/raw/statcan_mortgage_rate_5yr_20260228.csv`
  - Standardized: `data/interim/mortgage_rate_yearly.parquet`
- IRCC permanent resident yearly features:
  - Raw: `data/raw/ircc_pr_cma_20260228.xlsx`
  - Standardized: `data/interim/ircc_pr_yearly.parquet`
- IRCC study permit yearly features:
  - Raw: `data/raw/ircc_studypermits_pt_studylevel_20260228.xlsx`
  - Standardized: `data/interim/ircc_study_permits_yearly.parquet`
- CMHC rental yearly features:
  - Raw: `data/raw/cmhc_vancouver_rental_supply_change_20260228.csv`
  - Standardized: `data/interim/cmhc_rental_yearly.parquet`

### Prepared but deferred by default
- Census profile:
  - Raw:
    - `data/raw/statcan_censusprofile2021_data_20260228.csv`
    - `data/raw/statcan_censusprofile2021_geoindex_20260228.csv`
    - `data/raw/statcan_censusprofile2021_meta_20260228.txt`
    - `data/raw/optional/statcan_censusprofile2021_single_geo_20260228.csv`
  - Standardized:
    - `data/interim/census_profile_standardized.parquet`
- Census merge is attempted only when `--merge_census` is passed and a robust shared geography key exists.
- In current data, census remains deferred because no robust shared key is available.

## Final Data Pipeline Outputs
- Final merged table:
  - `data/processed/model_table.parquet`
- Final table summary:
  - `reports/figures/model_table_summary.csv`

Postal-code normalization note:
- `PROPERTY_POSTAL_CODE` is normalized internally to **uppercase with no spaces** (for example, `V6H 2J4` and `V6H2J4` are treated the same).
- This avoids inconsistent predictions caused by formatting differences.

The final merged table includes:
- property-level derived features
- lag/yoy macro features
- leakage-safe neighbourhood/FSA historical features (previous-year and rolling past windows only)

## What the Feature-Importance Figure Means
The feature-importance chart (`reports/figures/model_feature_importance_top20.png`) shows which inputs the model relied on most for prediction.

- Larger bar = feature contributed more to predictive performance.
- Smaller bar = feature contributed less in this setup.
- Importance is relative.
- Importance is not causation.

Plain-language takeaway:
- property location/structure features dominate
- local historical context helps
- broad yearly macro signals contribute less
- smarter encoding of high-cardinality location/type fields gives a meaningful accuracy gain

## Plain-Language Description of Key Features
| Feature | Plain-language meaning | Why it may matter | Source dataset |
|---|---|---|---|
| `LEGAL_TYPE` | Legal form of the property. | Different legal forms align with different value patterns. | `property-tax-report.csv` |
| `PROPERTY_POSTAL_CODE` | Full postal code. | Captures very local location differences. | `property-tax-report.csv` |
| `ZONING_DISTRICT` | City zoning district. | Zoning rules influence land use/value potential. | `property-tax-report.csv` |
| `ZONING_CLASSIFICATION` | Detailed zoning category. | Adds finer land-use context. | `property-tax-report.csv` |
| `NEIGHBOURHOOD_CODE` | City neighbourhood code. | Neighbourhoods have distinct market levels. | `property-tax-report.csv` |
| `YEAR_BUILT` | Year built. | Property age profile often correlates with value patterns. | `property-tax-report.csv` |
| `POSTAL_FSA` | First 3 postal characters (Forward Sortation Area). | Coarse local area grouping. | Derived from columns in `property-tax-report.csv` |
| `neigh_prev_year_median_land_value` | Previous-year neighbourhood median value. | Local historical market level. | Calculated from earlier years of the property-tax dataset (previous year only) |
| `neigh_prev_3yr_rolling_mean_land_value` | Rolling average over prior years in same neighbourhood. | Smoother local trend signal. | Calculated from earlier years of the property-tax dataset |
| `fsa_prev_year_median_land_value` | Previous-year median value in same FSA. | Postal-area historical context. | Calculated from earlier years of the property-tax dataset (previous year only) |
| `cmhc_rental_supply_existing_converted_lag1` | Previous-year CMHC converted rental supply indicator. | Lagged rental-market context. | `cmhc_vancouver_rental_supply_change_20260228.csv` |

## Final Model Outputs
### Core outputs
- `reports/figures/model_metrics.csv`
- `reports/figures/model_neighbourhood_error.csv`
- `reports/figures/model_feature_importance.csv`
- `reports/figures/model_feature_importance_grouped.csv`

### Public-friendly figures
![Model Scatter](reports/figures/model_scatter_public.png)
![Model Error Distribution](reports/figures/model_error_distribution_public.png)
![Model Neighbourhood Difficulty](reports/figures/model_neighbourhood_difficulty_public.png)
![Model Feature Importance Top20](reports/figures/model_feature_importance_top20.png)

## What Changed from the Earlier Model?
Earlier model:
- used the same split/target protocol but treated all categorical features with standard encoding only
- handled very high-cardinality fields (`PROPERTY_POSTAL_CODE`, `NEIGHBOURHOOD_CODE`, `LEGAL_TYPE`) more naively
- produced weaker error metrics

Current official model:
- keeps the same reproducible pipeline and evaluation protocol
- uses **train-only target encoding** (on `log1p(CURRENT_LAND_VALUE)`) for the 3 highest-impact high-cardinality categorical features
- defaults to Mode B, which uses encoded numeric versions instead of raw high-cardinality categorical versions in the model input
- improves RMSE, MAE, Median APE, and robust errors

| Model | RMSE | MAE | Median APE | Robust RMSE | Robust MAE |
|---|---:|---:|---:|---:|---:|
| Earlier model | 9,998,813.17 | 691,434.48 | 0.1818 | 1,657,883.65 | 489,160.08 |
| Current encoded model | 9,577,667.34 | 590,037.32 | 0.1389 | 1,414,066.15 | 396,533.43 |

## Why Accuracy Improved
The strongest signals in this project are fine-grained location and property-type fields. These fields have many distinct categories, so simple categorical handling leaves useful structure on the table.

Train-only target encoding gives the model a cleaner numeric summary of how each category behaved in historical training data, while still avoiding leakage from the test period. In practice, this helped more than adding extra coarse yearly macro indicators.

## Demo Web App
The project includes **two local demo interfaces** for presentation and educational use.

Both demos use the same trained final model artifact (`artifacts/land_value_model.joblib`) and estimate **assessed land value**, not guaranteed market sale price. The two demos differ mainly in interface style and target users rather than underlying model logic.

### Option A — React Frontend + REST API Backend
This demo is a web-style interactive system built with:
- a React frontend
- a Python REST API backend

It is designed to provide a more user-facing prediction experience and currently supports two modes:

- **Precise Mode**  
  A guided, step-by-step prediction workflow.  
  The frontend applies context-aware filtering, so valid options are narrowed based on previous user inputs.  
  This helps reduce unrealistic or non-existent property combinations.

- **Fuzzy Mode**  
  A simplified prediction workflow for non-expert users.  
  Users can start from address + postal code, retrieve matched candidate properties, and then generate a prediction from the selected match.

App notes:
- predicts **assessed land value (`CURRENT_LAND_VALUE`)**
- uses the same trained model as the final pipeline
- supports two interaction modes:
  - **Precise Mode** for guided step-by-step prediction
  - **Fuzzy Mode** for address-based prediction
- reconstructs part of the full model feature vector using:
  - derived fields
  - lookup values from `data/processed/model_table.parquet`
- returns:
  - a point estimate
  - an estimated value range

### Option B — Streamlit Demo
This demo is a lightweight local interface implemented in Streamlit at `app.py`.

- It estimates **assessed land value** from a small set of user inputs.
- It uses the trained final model artifact (`artifacts/land_value_model.joblib`).
- It is for presentation and educational demo use, not professional appraisal.

App notes:
- The app predicts assessed land value, not guaranteed market sale price.
- The app is designed for near-range estimation only (`REPORT_YEAR` is limited to `2024`–`2027`).
- This limit prevents misleading far-future predictions when lookup-based features are unavailable beyond the observed horizon.
- Some internal model features are filled using derived fields and lookup values from `data/processed/model_table.parquet`.
- The app returns a point estimate and an estimated range.  
  The range uses neighbourhood-level typical error when available; otherwise it falls back to global model error.

## Current Finding
- The encoded model in `src/models/train_model.py` is now the official final model.
- Location/structure variables remain the main predictive drivers.
- Better handling of high-cardinality categorical fields substantially improved performance versus the earlier main model.
- Extra coarse macro information still adds limited gain compared with strong property-level signals.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Final Pipeline
### Option A — Notebook demo entry point
Open `final_submission_demo.ipynb` and run cells from top to bottom.  
The notebook calls the same Python modules below and displays final metrics/figures.

### Option B — Python pipeline directly
```bash
python -m src.data.clean_property_tax --in_path data/raw/property-tax-report.csv --out_path data/interim/property_tax_clean.parquet

python -m src.data.standardize_mortgage_rate
python -m src.data.standardize_ircc_pr
python -m src.data.standardize_ircc_study_permits
python -m src.data.standardize_cmhc_rental
python -m src.data.standardize_census_profile

python -m src.data.build_model_table
python -m src.models.train_model
```

Optional census merge attempt:
```bash
python -m src.data.build_model_table --merge_census
```

## Run the demos
### Run Option 1 — React + REST API
Start the backend API first, then run the React frontend.
Backend:
```bash
python -m src.models.train_model
python -m uvicorn src.api.main:app --reload --port 8000
```
Frontend:
```bash
cd frontend
npm install
npm run dev
```

### Run Option 2 — Streamlit
```bash
python -m src.models.train_model
python -m streamlit run app.py
```

If the artifact is missing, the app will show a message asking you to run training first.

## Data Governance
- Raw/interim/processed data files are not committed to git.
- Do not commit local data artifacts under `data/`.

## Project Evolution Note
Earlier experimental entry points were consolidated into this final submission pipeline to reduce maintenance overhead and avoid version confusion.  
The previous non-encoded main trainer is retained only as historical reference through documented metrics; the official path is now `python -m src.models.train_model`.

## License
MIT (see `LICENSE`).
