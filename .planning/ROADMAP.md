# Roadmap: Lithium-Ion Battery Cathode Performance Prediction

**Created:** 2026-03-05
**Phases:** 8
**Requirements:** 37 mapped

## Phases

- [x] **Phase 1: Data Pipeline and Project Foundation** - Reproducible data ingestion, cleaning, caching, and project infrastructure (completed 2026-03-06)
- [x] **Phase 2: Featurization and Baseline Models** - Crystal-to-graph conversion, Magpie descriptors, and sklearn baselines that validate the pipeline end-to-end (completed 2026-03-06)
- [ ] **Phase 3: CGCNN Implementation** - First GNN architecture with training infrastructure, transfer learning, and multi-property prediction
- [ ] **Phase 4: MEGNet Implementation** - Second GNN architecture via matgl, isolated to contain DGL dependency risk
- [x] **Phase 5: Evaluation and Benchmarking** - Rigorous cross-model comparison with publication-quality figures and CLI pipeline (completed 2026-03-07)
- [x] **Phase 6: Dashboard and Documentation** - Interactive Streamlit dashboard and comprehensive README (completed 2026-03-07)
- [x] **Phase 7: Fix Pipeline Orchestrator Wiring** - Fix config loading, data handoff, and baseline results path in pipeline.py (completed 2026-03-07)
- [ ] **Phase 8: Fix Dashboard Cross-Phase Wiring** - Fix page guards, import errors, checkpoint format, and data path in dashboard

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
- [x] 02-01-PLAN.md — Crystal structure to PyG graph conversion with Gaussian edge features
- [x] 02-02-PLAN.md — Magpie composition featurizer and compositional group splitting
- [x] 02-03-PLAN.md — Random Forest and XGBoost baseline models with JSON results

### Phase 3: CGCNN Implementation
**Goal:** CGCNN predicts cathode properties with proper training infrastructure that MEGNet will reuse
**Depends on:** Phase 2
**Requirements:** MODL-01, MODL-05, MODL-06, MODL-07
**Success Criteria** (what must be TRUE):
  1. CGCNN model using PyG CGConv layers trains on crystal graphs and produces predictions for formation energy, voltage, stability, and capacity as separate per-property models
  2. Training uses CGCNN-appropriate hyperparameters (LR ~1e-3, ~400 epochs, early stopping) configured via YAML, not hardcoded
  3. Model checkpoints, training loss curves, and per-epoch validation metrics are saved as artifacts (pt files, JSON, CSV) after each training run
  4. Cross-validated evaluation on the same compositional folds as baselines produces comparable metrics (MAE, RMSE, R-squared)
**Plans:** 2 plans

Plans:
- [x] 03-01-PLAN.md — CGCNN model class, shared evaluation utils, and YAML config
- [ ] 03-02-PLAN.md — GNNTrainer and CGCNN training orchestrator with artifacts

### Phase 4: MEGNet Implementation
**Goal:** MEGNet produces results on identical data splits as CGCNN for a fair head-to-head comparison
**Depends on:** Phase 2 (data/features), Phase 3 (training infrastructure)
**Requirements:** MODL-02
**Success Criteria** (what must be TRUE):
  1. MEGNet model via matgl trains successfully (in a separate conda environment if necessary) and produces predictions for all target properties
  2. MEGNet uses its own appropriate hyperparameters (LR ~1e-4, ~1000 epochs, larger batches) independent of CGCNN settings
  3. Evaluation uses the exact same compositional folds and test set as CGCNN and baselines, with results saved in the same artifact format
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md — MEGNet wrapper module, YAML config, and test scaffolding
- [ ] 04-02-PLAN.md — MEGNet training orchestrator with Lightning integration and CLI

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
**Plans:** 3/3 plans complete

Plans:
- [ ] 05-01-PLAN.md — Evaluation metrics module with unified result loading and comparison tables
- [ ] 05-02-PLAN.md — Publication-quality plotting (parity, bar charts, learning curves) and evaluation CLI
- [ ] 05-03-PLAN.md — Full pipeline orchestrator with argparse CLI and stage execution

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

**Plans:** 4/4 plans complete

Plans:
- [ ] 06-01-PLAN.md — Dashboard scaffolding, utilities, Overview and Model Comparison pages
- [ ] 06-02-PLAN.md — Baseline model persistence, Predict page, and Crystal Viewer
- [ ] 06-03-PLAN.md — Data Explorer and Materials Explorer pages
- [ ] 06-04-PLAN.md — Comprehensive README documentation

### Phase 7: Fix Pipeline Orchestrator Wiring
**Goal:** The unified pipeline CLI (`python -m cathode_ml`) correctly loads separate config files, reads processed data, and baseline results are found by the evaluation loader
**Depends on:** Phase 5, Phase 6
**Requirements:** EVAL-01, EVAL-02, EVAL-03, DATA-04
**Gap Closure:** Closes audit findings 1, 2, 3
**Success Criteria** (what must be TRUE):
  1. `pipeline.py` loads `features.yaml`, `baselines.yaml`, `cgcnn.yaml`, `megnet.yaml` separately instead of extracting sections from `data.yaml`
  2. Pipeline train stage reads records from `data/processed/materials.json` matching standalone CLIs
  3. Baseline results JSON is saved to (or read from) a consistent path so `load_all_results` includes RF and XGBoost metrics
**Plans:** 1 plan

Plans:
- [x] 07-01-PLAN.md — Fix config loading, data handoff, and baseline results path

### Phase 8: Fix Dashboard Cross-Phase Wiring
**Goal:** All 6 dashboard pages render correctly with proper data loading, model prediction, and crystal viewing
**Depends on:** Phase 7
**Requirements:** DASH-01, DASH-02, DASH-03, DASH-05, DASH-06, DASH-07
**Gap Closure:** Closes audit findings 4, 5, 6, 7
**Success Criteria** (what must be TRUE):
  1. Predict and Crystal Viewer pages call `main()` at module level (not behind `__name__` guard) and render in Streamlit
  2. `model_loader.py` imports `structure_to_graph` (not `structure_to_pyg_data`) and passes config dict correctly
  3. MEGNet checkpoint loading handles raw state_dict format (no `model_state_dict` key required)
  4. `get_cached_records()` reads from `data/processed/materials.json` so Data Explorer and Materials Explorer display actual data
**Plans:** 1 plan

Plans:
- [ ] 08-01-PLAN.md — Fix page guards, graph import, checkpoint format, and data path

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 1. Data Pipeline and Project Foundation | 3/3 | Complete   | 2026-03-06 |
| 2. Featurization and Baseline Models | 3/3 | Complete   | 2026-03-06 |
| 3. CGCNN Implementation | 1/2 | In progress | - |
| 4. MEGNet Implementation | 0/2 | Not started | - |
| 5. Evaluation and Benchmarking | 3/3 | Complete   | 2026-03-07 |
| 6. Dashboard and Documentation | 4/4 | Complete   | 2026-03-07 |
| 7. Fix Pipeline Orchestrator Wiring | 1/1 | Complete   | 2026-03-07 |
| 8. Fix Dashboard Cross-Phase Wiring | 0/1 | Not started | - |

### Phase 9: Replace MEGNet with M3GNet and TensorNet from matgl 2.x

**Goal:** MEGNet is replaced by M3GNet (pretrained, fine-tuned) and TensorNet (from-scratch, O(3)-equivariant) using matgl 2.x APIs, with full pipeline integration, evaluation, dashboard support, and test coverage
**Depends on:** Phase 8
**Requirements:** MODL-02
**Success Criteria** (what must be TRUE):
  1. M3GNet model loads pretrained weights from matgl 2.x (M3GNet-MP-2018.6.1-Eform), trains with include_line_graph=True and threebody_cutoff, and produces predictions for all target properties
  2. TensorNet model constructs from config (no pretrained property models), trains with O(3) equivariance, and produces predictions for all target properties
  3. Both models use identical compositional splits as CGCNN and baselines, with results in the same JSON artifact format
  4. Pipeline CLI offers m3gnet and tensornet as model choices (megnet removed), and all five models run end-to-end
  5. Dashboard loads M3GNet and TensorNet checkpoints, displays their metrics, and supports structure-based prediction
  6. Zero MEGNet references remain in the active codebase; old MEGNet files are deleted
**Plans:** 4 plans

Plans:
- [ ] 09-01-PLAN.md -- Core model wrappers (m3gnet.py, tensornet.py), YAML configs, requirements.txt
- [ ] 09-02-PLAN.md -- Training orchestrators (train_m3gnet.py, train_tensornet.py) and pipeline.py update
- [ ] 09-03-PLAN.md -- Evaluation, dashboard, and documentation updates (metrics, plots, model_loader, README)
- [ ] 09-04-PLAN.md -- Tests (test_m3gnet.py, test_tensornet.py), integration test updates, MEGNet file deletion

---
*Last updated: 2026-03-07*
