# Roadmap: Lithium-Ion Battery Cathode Performance Prediction

## Milestones

- ✅ **v1.0 MVP** - Phases 1-9 (shipped 2026-03-09)
- 🚧 **v1.1 Polish & Correctness** - Phases 10-12 (in progress)

## Phases

<details>
<summary>v1.0 MVP (Phases 1-9) - SHIPPED 2026-03-09</summary>

- [x] Phase 1: Data Pipeline and Project Foundation (3/3 plans) - completed 2026-03-06
- [x] Phase 2: Featurization and Baseline Models (3/3 plans) - completed 2026-03-06
- [x] Phase 3: CGCNN Implementation (2/2 plans) - completed 2026-03-06
- [x] Phase 4: MEGNet Implementation (2/2 plans) - completed 2026-03-07
- [x] Phase 5: Evaluation and Benchmarking (3/3 plans) - completed 2026-03-07
- [x] Phase 6: Dashboard and Documentation (4/4 plans) - completed 2026-03-07
- [x] Phase 7: Fix Pipeline Orchestrator Wiring (1/1 plan) - completed 2026-03-07
- [x] Phase 8: Fix Dashboard Cross-Phase Wiring (1/1 plan) - completed 2026-03-08
- [x] Phase 9: Replace MEGNet with M3GNet and TensorNet (4/4 plans) - completed 2026-03-09

</details>

### v1.1 Polish & Correctness (In Progress)

**Milestone Goal:** Fix known bugs, validate all data sources, retrain models, and update all project surfaces to reflect accurate state.

- [x] **Phase 10: Model Bug Fixes** - Fix M3GNet/TensorNet denormalization and logging bugs so models produce valid results (completed 2026-03-14)
- [x] **Phase 11: Data Validation & Retraining** - Validate AFLOW/JARVIS fetchers end-to-end and retrain all models on 4-source combined data (completed 2026-03-15)
- [ ] **Phase 12: Project Surfaces** - Update README, dashboard, and GitHub repo to reflect corrected results and full data coverage

## Phase Details

### Phase 10: Model Bug Fixes
**Goal**: M3GNet and TensorNet produce valid, trustworthy predictions on all properties
**Depends on**: Nothing (first phase of v1.1)
**Requirements**: FIX-01, FIX-02
**Success Criteria** (what must be TRUE):
  1. Running predict_structure() for M3GNet/TensorNet returns values in the correct physical range (no double-denormalization)
  2. M3GNet and TensorNet achieve positive R-squared on formation energy (comparable to CGCNN baseline range)
  3. TensorNet training log output shows one entry per epoch with no duplicated lines
**Plans**: 1 plan

Plans:
- [x] 10-01-PLAN.md — Fix double-denormalization bug and TensorNet log duplication

### Phase 11: Data Validation & Retraining
**Goal**: All 4 data sources feed validated data into the pipeline, and all models are retrained on the combined dataset with updated metrics
**Depends on**: Phase 10
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. AFLOW fetcher retrieves lithium cathode records and they pass through clean and graph conversion without errors
  2. JARVIS fetcher retrieves lithium cathode records and they pass through clean and graph conversion without errors
  3. Combined 4-source dataset is larger than the previous 2-source dataset
  4. All models (CGCNN, M3GNet, TensorNet, RF, XGBoost) have fresh metrics on the 4-source dataset stored in results
**Plans**: 2 plans

Plans:
- [x] 11-01-PLAN.md — Validate AFLOW/JARVIS end-to-end through pipeline and fix refresh bug (completed 2026-03-14)
- [x] 11-02-PLAN.md — Retrain all models on 4-source combined dataset (completed 2026-03-15)

### Phase 12: Project Surfaces
**Goal**: Every public-facing artifact accurately reflects the current state of the project with corrected results, all 4 data sources, and current architecture
**Depends on**: Phase 11
**Requirements**: SURF-01, SURF-02, SURF-03
**Success Criteria** (what must be TRUE):
  1. README describes all 4 data sources (MP, OQMD, AFLOW, JARVIS), shows corrected model metrics, and reflects the current architecture
  2. GitHub repo description and topics include references to all data sources and model types
  3. Streamlit dashboard displays corrected metrics for all models, shows AFLOW/JARVIS data in the explorer, and all charts render without errors
  4. No stale v1.0 metrics or 2-source-only references remain in any public surface
**Plans**: TBD

Plans:
- [ ] 12-01: TBD
- [ ] 12-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 10 -> 11 -> 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 10. Model Bug Fixes | 1/1 | Complete    | 2026-03-14 | - |
| 11. Data Validation & Retraining | 2/2 | Complete   | 2026-03-15 | - |
| 12. Project Surfaces | v1.1 | 0/? | Not started | - |

---
*Last updated: 2026-03-15 after 11-02 completion*
