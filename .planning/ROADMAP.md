# Roadmap: Lithium-Ion Battery Cathode Performance Prediction

**Created:** 2026-03-05
**Phases:** 6
**Requirements:** 37 mapped

## Phases

- [x] **Phase 1: Data Pipeline and Project Foundation** - Reproducible data ingestion, cleaning, caching, and project infrastructure (completed 2026-03-06)
- [x] **Phase 2: Featurization and Baseline Models** - Crystal-to-graph conversion, Magpie descriptors, and sklearn baselines that validate the pipeline end-to-end (completed 2026-03-06)
- [ ] **Phase 3: CGCNN Implementation** - First GNN architecture with training infrastructure, transfer learning, and multi-property prediction
- [ ] **Phase 4: MEGNet Implementation** - Second GNN architecture via matgl, isolated to contain DGL dependency risk
- [ ] **Phase 5: Evaluation and Benchmarking** - Rigorous cross-model comparison with publication-quality figures and CLI pipeline
- [ ] **Phase 6: Dashboard and Documentation** - Interactive Streamlit dashboard and comprehensive README

## Phase Details

### Phase 1: Data Pipeline and Project Foundation
**Goal:** A researcher can fetch, clean, and cache cathode material data from all three sources with a single command, producing validated pymatgen Structures ready for featurization
**Depends on:** Nothing (first phase)
**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, REPR-01, REPR-02, REPR-03
**Success Criteria** (what must be TRUE):
  1. Running `python -m cathode_ml.data.fetch` downloads cathode data from Materials Project, OQMD, and Battery Data Genome, caching results locally so subsequent runs skip API calls
  2. The cleaned dataset contains only valid crystal structures with target properties (formation energy, voltage, stability) and a log documents every filter applied with rationale
  3. A YAML config file controls all pipeline parameters (API keys, filter thresholds, random seeds) and changing a config value changes pipeline behavior
  4. `pip install -r requirements.txt` (or `conda env create`) installs all dependencies with pinned versions on a fresh machine
  5. All random operations use fixed seeds from config, producing identical outputs across runs
**Plans:** 3/3 plans complete

Plans:
- [x] 01-01-PLAN.md — Project scaffolding, config system, cache module, and dependency files
- [x] 01-02-PLAN.md — Materials Project and OQMD data fetchers with caching
- [x] 01-03-PLAN.md — Battery Data Genome fetcher, cleaning pipeline, and CLI entry point

### Phase 2: Featurization and Baseline Models
**Goal:** Crystal structures are converted to both graph representations and tabular features, and baseline models establish a performance floor on compositional splits
**Depends on:** Phase 1
**Requirements:** FEAT-01, FEAT-02, FEAT-03, FEAT-04, MODL-03, MODL-04
**Success Criteria** (what must be TRUE):
  1. Each crystal structure produces a PyG Data object with atom nodes, bond edges, and Gaussian distance expansion edge features -- and zero graphs are disconnected (validated automatically)
  2. Each composition produces a Magpie descriptor vector via matminer suitable for sklearn input
  3. Train/validation/test splits group all entries sharing a reduced composition formula into the same fold, preventing polymorph leakage
  4. Random Forest and XGBoost baselines produce MAE, RMSE, and R-squared on held-out test data for each target property, with results saved as JSON artifacts
**Plans:** 3/3 plans complete

Plans:
- [ ] 02-01-PLAN.md — Crystal structure to PyG graph conversion with Gaussian edge features
- [ ] 02-02-PLAN.md — Magpie composition featurizer and compositional group splitting
- [ ] 02-03-PLAN.md — Random Forest and XGBoost baseline models with JSON results

### Phase 3: CGCNN Implementation
**Goal:** CGCNN predicts cathode properties with proper training infrastructure that MEGNet will reuse
**Depends on:** Phase 2
**Requirements:** MODL-01, MODL-05, MODL-06, MODL-07
**Success Criteria** (what must be TRUE):
  1. CGCNN model using PyG CGConv layers trains on crystal graphs and produces predictions for formation energy, voltage, stability, and capacity as separate per-property models
  2. Training uses CGCNN-appropriate hyperparameters (LR ~1e-3, ~400 epochs, early stopping) configured via YAML, not hardcoded
  3. Model checkpoints, training loss curves, and per-epoch validation metrics are saved as artifacts (pt files, JSON, CSV) after each training run
  4. Cross-validated evaluation on the same compositional folds as baselines produces comparable metrics (MAE, RMSE, R-squared)
**Plans:** TBD

### Phase 4: MEGNet Implementation
**Goal:** MEGNet produces results on identical data splits as CGCNN for a fair head-to-head comparison
**Depends on:** Phase 2 (data/features), Phase 3 (training infrastructure)
**Requirements:** MODL-02
**Success Criteria** (what must be TRUE):
  1. MEGNet model via matgl trains successfully (in a separate conda environment if necessary) and produces predictions for all target properties
  2. MEGNet uses its own appropriate hyperparameters (LR ~1e-4, ~1000 epochs, larger batches) independent of CGCNN settings
  3. Evaluation uses the exact same compositional folds and test set as CGCNN and baselines, with results saved in the same artifact format
**Plans:** TBD

### Phase 5: Evaluation and Benchmarking
**Goal:** All models are rigorously compared with publication-quality figures and the full pipeline is runnable end-to-end from CLI
**Depends on:** Phase 3, Phase 4
**Requirements:** EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, REPR-04
**Success Criteria** (what must be TRUE):
  1. A unified evaluation script loads all model predictions and computes MAE, RMSE, and R-squared across all models and properties in one run
  2. Parity plots (predicted vs actual) for each model-property combination are saved as publication-quality figures (300+ DPI, labeled axes, tight layout)
  3. Bar chart comparisons show model performance side-by-side across all properties, making relative strengths immediately visible
  4. Training/learning curves (loss and validation MAE per epoch) are plotted for both GNN models
  5. A CLI command runs the full pipeline (fetch, clean, featurize, train, evaluate) end-to-end with one invocation
**Plans:** TBD

### Phase 6: Dashboard and Documentation
**Goal:** Users explore results, materials, and predictions interactively through a web dashboard, and the README provides complete project context
**Depends on:** Phase 5
**Requirements:** DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06, DASH-07, DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):
  1. `streamlit run dashboard/app.py` launches a multi-page web app displaying model comparison metrics in tables and interactive charts
  2. A user can input a composition or structure and receive predicted properties from trained models in the dashboard
  3. The data explorer page lets users browse, filter, and visualize dataset distributions (histograms, scatter plots)
  4. The materials explorer page provides a searchable, filterable table of materials by voltage, formation energy, capacity, elements, and stability -- with a discovery panel highlighting top candidates
  5. Crystal structures render as interactive 3D viewers (rotate, zoom) using py3Dmol
  6. Training curves (loss, learning rate, convergence) display per model in the dashboard
  7. README contains introduction, methodology, pipeline details, and results summary with key findings

**Plans:** TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 1. Data Pipeline and Project Foundation | 3/3 | Complete   | 2026-03-06 |
| 2. Featurization and Baseline Models | 3/3 | Complete   | 2026-03-06 |
| 3. CGCNN Implementation | 0/0 | Not started | - |
| 4. MEGNet Implementation | 0/0 | Not started | - |
| 5. Evaluation and Benchmarking | 0/0 | Not started | - |
| 6. Dashboard and Documentation | 0/0 | Not started | - |

---
*Last updated: 2026-03-05*
