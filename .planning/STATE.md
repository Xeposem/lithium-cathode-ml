---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: Plan 02 of 04
status: in-progress
last_updated: "2026-03-07T09:11:35.913Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 17
  completed_plans: 15
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Accurate, reproducible prediction of cathode performance properties from crystal structure, with clear model comparison and publication-quality results
**Current focus:** Phase 6

## Progress

| Phase | Name | Status | Requirements |
|-------|------|--------|-------------|
| 1 | Data Pipeline and Project Foundation | Complete (3/3 plans) | DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, REPR-01, REPR-02, REPR-03 |
| 2 | Featurization and Baseline Models | Complete (3/3 plans) | FEAT-01, FEAT-02, FEAT-03, FEAT-04, MODL-03, MODL-04 |
| 3 | CGCNN Implementation | Complete (2/2 plans) | MODL-01, MODL-05, MODL-06, MODL-07 |
| 4 | MEGNet Implementation | Complete (2/2 plans) | MODL-02 |
| 5 | Evaluation and Benchmarking | Complete (3/3 plans) | EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, REPR-04 |
| 6 | Dashboard and Documentation | In Progress (1/4 plans) | DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06, DASH-07, DOCS-01, DOCS-02, DOCS-03, DOCS-04 |

## Current Phase

**Phase 6: Dashboard and Documentation**
Status: In Progress
Plans: 2/4
Current Plan: Plan 02 of 04

## Accumulated Context

### Key Decisions
- DataCache.load() returns only data field, not wrapper (clean API for fetchers)
- MD5 hash for cache keys (deterministic, acceptable collision risk at project scale)
- Explicit FileNotFoundError in load_config for clear error messages
- Lazy import pattern for heavy science deps (mp-api, qmpy-rester) to avoid version conflicts at import time
- OQMD structure_dict is empty dict (REST API does not return full crystal structure)
- Electrode join via material_ids list iteration from each electrode doc
- BDG fetcher is a CSV file downloader, not API client (BDG is not a single API)
- Deduplication uses source priority: MP > OQMD > BDG
- IQR outlier removal skips when fewer than 4 data points
- Fetcher imports in fetch.py use try/except for robustness
- Separate models per property (not multi-output) per research recommendation
- CGCNN before MEGNet (zero dependency conflicts vs matgl/DGL risk)
- Baselines with featurization (fast end-to-end validation)
- Compositional group splitting from day one (prevents leakage)
- Streamlit for dashboard (not Dash -- simpler, fast development, sufficient for dataset scale)
- Identity decorator fallback for st.cache_resource in test environments
- Dynamic model discovery by scanning baselines/*.joblib filenames
- Tabs layout for composition vs CIF input modes on Predict page
- matgl v1.3.0 for MEGNet (NOT v2.0.0 -- MEGNet not yet ported to PyG in v2)
- Matminer v0.9.3 Magpie preset produces no all-NaN columns for single-element compositions; drop logic retained for robustness
- Two-stage GroupShuffleSplit: test first, then val from remainder with adjusted fraction
- scipy KDTree with periodic images instead of pymatgen get_all_neighbors (Cython dtype bug on Windows)
- Lazy import for xgboost in train_baseline (only loaded when model_type='xgb')
- Minimum 5 valid records per property to train baselines (skip otherwise)
- Softplus activation in CGCNN FC head (smooth, differentiable, standard for regression)
- compute_metrics accepts raw arrays (not model object) for GNN compatibility
- Config-driven model construction pattern: build_X_from_config(model_config, features_config)
- GNNTrainer is model-agnostic (any nn.Module + PyG DataLoader) for MEGNet reuse in Phase 4
- Seeds reset before each property model init for reproducible weight initialization
- Per-property sequential training loop matching baselines pattern (skip if <5 records)
- MEGNet-MP-2018.6.1-Eform as default pretrained model (confirmed in matgl tutorials)
- Lazy imports for all matgl usage with centralized _import_matgl() helper
- get_megnet_state_dict extracts model.model.state_dict() for .pt format compatibility
- Separated _run_lightning_training for clean mocking; train_megnet_for_property accepts pre-computed indices
- Heading per property in markdown comparison tables (### property_name) for structured output
- Italic footnote for MEGNet dagger symbol in comparison tables
- Parity plots deferred in CLI until prediction arrays available (Plan 03 integration)
- Learning curves grid: rows=properties, cols=models (CGCNN, MEGNet only)
- Evaluate stage gracefully skips plots module if not yet available (plan 05-02 not yet executed)
- Featurize stage is pass-through log since featurization happens inline in training orchestrators
- _render() function pattern in dashboard pages for safe import outside Streamlit runtime
- use_container_width=True for Plotly charts in Streamlit dashboard
- Re-export MODEL_COLORS/LABELS/ORDER/PROPERTIES from data_loader for dashboard convenience

### Research Flags
- Phase 4 (MEGNet): matgl v1.3.0 + PyTorch compatibility untested; may need separate conda env
- Phase 3 (CGCNN): Transfer learning from full MP (~150K entries) needs strategy research during planning
- OQMD: qmpy_rester unmaintained since 2019; may need direct HTTP fallback
- Battery Data Genome: Resolved -- implemented as CSV file downloader with graceful degradation

### Blockers
None

### Todos
None

---
*Last updated: 2026-03-07*
*Last session: Completed 06-02-PLAN.md (Predict page and Crystal Viewer)*
