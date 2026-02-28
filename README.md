# CMPT 733 Final Project

## Project Overview
Placeholder: One or two sentences describing the problem, objective, and expected impact.

## Team & Roles
- Chenzheng Li
- Luna Sang
- Ryan Chen
- Wenxiang He

## Dataset Sources
- Placeholder: dataset name + link
- Placeholder: dataset name + link

## Repo Structure

### Top-level files
- `README.md` — Project landing page: what we do, datasets, how to set up, and where things live.
- `requirements.txt` — Python dependencies for `pip install -r requirements.txt` (keep minimal; add only when needed).
- `.gitignore` — Prevents committing large data files, caches, secrets, and notebook checkpoints.
- `LICENSE` — Usage/license terms for the repository (e.g., MIT).
- `CONTRIBUTING.md` — Team workflow rules (branching, PR checklist, commit message conventions).

### GitHub automation
- `.github/workflows/ci.yml` — Minimal CI checks on push/PR (e.g., install deps / basic Python sanity checks). Helps catch “works on my machine” issues early.

### Data
- `data/raw/` — Original datasets as downloaded. **Not tracked in git** (store download links/scripts instead).
- `data/interim/` — Intermediate outputs during cleaning/integration (temporary artifacts).
- `data/processed/` — Final modeling-ready datasets. Typically **not tracked in git** unless very small samples.

### Analysis & experiments
- `notebooks/` — Jupyter notebooks for EDA, quick experiments, and visual exploration. (Notebooks can be messy; final reproducible steps should eventually live in `src/`.)

### Source code (reproducible pipeline lives here)
- `src/` — Main Python package for reusable code.
  - `src/data/` — Data loading, cleaning, joining/integration, and feature construction helpers.
  - `src/models/` — Model training, prediction, and serialization utilities.
  - `src/eval/` — Metrics, error analysis (e.g., subgroup/neighborhood breakdown), and comparison utilities.
  - `src/viz/` — Plotting functions used by notebooks/reports to keep visuals consistent.

### Outputs & writing assets
- `reports/` — Report-related materials.
  - `reports/figures/` — Exported plots/images used in the report/slides (so outputs are centralized).
- `slides/` — Presentation materials.
  - `slides/milestone/` — Milestone presentation (5 min) assets.
  - `slides/final/` — Final presentation assets.
- `docs/` — Extra documentation: data dictionary, decisions/assumptions, meeting notes, references, etc.

## Setup
Using `pip` (placeholder commands):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
Placeholder: Add minimal steps for preprocessing, training, and evaluation once defined.

## Milestones & Deliverables
- Milestone slides
- Final presentation
- Final report
- Poster
- Code

## License
Placeholder: MIT (see `LICENSE`).