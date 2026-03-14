# Requirements: Lithium-Ion Battery Cathode Performance Prediction

**Defined:** 2026-03-13
**Core Value:** Accurate, reproducible prediction of cathode performance properties from crystal structure, with clear model comparison and publication-quality results

## v1.1 Requirements

Requirements for polish & correctness release. Each maps to roadmap phases.

### Bug Fixes

- [ ] **FIX-01**: M3GNet/TensorNet double-denormalization bug is fixed — models produce valid R² on all properties
- [ ] **FIX-02**: TensorNet training runs without log duplication errors

### Data Validation

- [ ] **DATA-01**: AFLOW fetcher validated end-to-end through the pipeline (fetch → clean → graph)
- [ ] **DATA-02**: JARVIS fetcher validated end-to-end through the pipeline (fetch → clean → graph)
- [ ] **DATA-03**: All models retrained with 4-source combined dataset, metrics updated

### Project Surfaces

- [ ] **SURF-01**: README reflects accurate results, all 4 data sources, and current architecture
- [ ] **SURF-02**: GitHub repo description and topics updated to reflect current project state
- [ ] **SURF-03**: Dashboard displays corrected model results, includes AFLOW/JARVIS data, all views accurate

## Future Requirements

None — v1.1 is a polish milestone.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New model architectures | v1.1 is correctness, not new capabilities |
| New properties to predict | Fix existing predictions first |
| Cloud deployment | Local/academic use per project constraints |
| Generative material design | Prediction only, not inverse design |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-01 | — | Pending |
| FIX-02 | — | Pending |
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| DATA-03 | — | Pending |
| SURF-01 | — | Pending |
| SURF-02 | — | Pending |
| SURF-03 | — | Pending |

**Coverage:**
- v1.1 requirements: 8 total
- Mapped to phases: 0
- Unmapped: 8 ⚠️

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after initial definition*
