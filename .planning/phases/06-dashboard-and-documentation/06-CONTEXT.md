# Phase 6: Dashboard and Documentation - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Interactive Streamlit multi-page dashboard for exploring model results, materials, and predictions, plus a comprehensive README for the GitHub repository. Covers requirements DASH-01 through DASH-07 and DOCS-01 through DOCS-04.

</domain>

<decisions>
## Implementation Decisions

### Dashboard Pages & Navigation
- Multi-page Streamlit app with sidebar navigation
- 6 pages: Overview, Data Explorer, Model Comparison, Materials Explorer, Predict, Crystal Viewer
- Overview/landing page is a metrics dashboard: model comparison table (reusing metrics.py data), bar chart of MAE across models, key findings summary
- Materials Explorer and Discovery Panel combined on one page: searchable/filterable table (DASH-05) on top, "Top Candidates" panel (DASH-06) below
- Training curves (DASH-04) live on the Model Comparison page as a section alongside metrics table and bar chart

### Interactive Prediction UX
- Two input modes on one page: composition string (e.g., "LiFePO4") for quick baseline predictions, plus optional CIF file upload for full GNN predictions
- All available models run predictions: RF, XGBoost from composition; CGCNN, MEGNet additionally if CIF provided
- Results displayed as property cards — one card per predicted property (voltage, capacity, formation energy, stability) showing values with units
- CIF uploads also render an inline 3D crystal structure preview (py3Dmol) before predictions, confirming the user uploaded the correct structure

### Charting Approach
- Plotly for all dashboard charts (interactive hover, zoom, pan) via st.plotly_chart()
- Matplotlib PNGs from evaluation/plots.py still generated separately for README and paper figures — both coexist
- Wong colorblind-safe palette carried over to Plotly: RF=#0072B2, XGB=#D55E00, CGCNN=#009E73, MEGNet=#CC79A7 (reuse MODEL_COLORS from metrics.py)
- Parity plots: interactive Plotly in dashboard (hover to see material ID), static matplotlib PNGs for export
- Data Explorer visualizations: histograms for each property distribution, plus interactive scatter matrix (property vs property) for correlation exploration

### README Structure & Depth
- Academic portfolio tone — written for ML researchers and hiring managers
- Embedded key figures: bar comparison chart, one parity plot, and a dashboard screenshot (2-3 figures total)
- Methodology section: architecture summaries + key design choices (compositional splitting, pretrained MEGNet fine-tuning, data sources) — 2-3 paragraphs, not a full paper
- Results summary table: compact table showing best-performing model and its MAE/R-squared for each property
- Sections: Introduction/Motivation, Data Sources, Methodology, Results, Dashboard, How to Run/Reproduce

### Claude's Discretion
- Exact Streamlit page file naming and directory structure (pages/ convention)
- Plotly chart styling details (axis formatting, legend placement)
- Crystal viewer integration specifics (py3Dmol vs alternative if compatibility issues)
- README section ordering and transition prose
- Dashboard theme/styling (Streamlit config)
- Materials Explorer filter widget design (sliders, dropdowns, multiselect)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation/metrics.py`: `load_all_results()`, `MODEL_COLORS`, `MODEL_LABELS`, `MODELS_ORDER`, `PROPERTIES` — directly reusable for dashboard data loading and consistent color mapping
- `evaluation/plots.py`: matplotlib figure generation (parity, bar, learning curves) — continues generating static PNGs for README/paper
- `pipeline.py`: Full stage orchestration with `build_parser()` and stage functions — documents the pipeline for README
- `data/cache.py`: `DataCache` for loading cached dataset records — reusable for Data Explorer page
- `features/split.py`: `compositional_split()` — documents methodology for README
- `models/utils.py`: `compute_metrics()` — shared evaluation utility

### Established Patterns
- JSON results in `data/results/{model_name}/` with keys: mae, rmse, r2, n_train, n_test
- CSV per-epoch metrics for GNNs (columns: epoch, train_loss, val_loss, val_mae, lr)
- Lazy imports for heavy dependencies (matgl, xgboost, torch)
- YAML config files in `configs/` directory
- Per-property sequential processing with skip if <5 records

### Integration Points
- Input: `data/results/` (JSON artifacts, CSV training logs) for dashboard metrics and charts
- Input: `data/cache/` (cached cleaned records) for Data Explorer and Materials Explorer
- Input: saved model checkpoints for interactive prediction
- Output: `dashboard/` directory (new Streamlit app)
- Output: `README.md` (new or rewritten)
- New dependency: `streamlit`, `plotly`, `py3Dmol` (or `stmol`) added to requirements.txt

</code_context>

<specifics>
## Specific Ideas

- Landing page should immediately answer "how well do the models perform?" — metrics-first design
- Prediction page offers dual-mode input: quick composition string OR full CIF upload, so both casual explorers and researchers with structures in hand are served
- Crystal viewer on the Predict page acts as upload validation — user sees the structure before predictions run

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-dashboard-and-documentation*
*Context gathered: 2026-03-07*
