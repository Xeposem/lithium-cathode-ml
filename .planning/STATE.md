# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Accurate, reproducible prediction of cathode performance properties from crystal structure, with clear model comparison and publication-quality results
**Current focus:** Phase 1

## Progress

| Phase | Name | Status | Requirements |
|-------|------|--------|-------------|
| 1 | Data Pipeline and Project Foundation | In Progress (1/3 plans) | DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, REPR-01, REPR-02, REPR-03 |
| 2 | Featurization and Baseline Models | Pending | FEAT-01, FEAT-02, FEAT-03, FEAT-04, MODL-03, MODL-04 |
| 3 | CGCNN Implementation | Pending | MODL-01, MODL-05, MODL-06, MODL-07 |
| 4 | MEGNet Implementation | Pending | MODL-02 |
| 5 | Evaluation and Benchmarking | Pending | EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, REPR-04 |
| 6 | Dashboard and Documentation | Pending | DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06, DASH-07, DOCS-01, DOCS-02, DOCS-03, DOCS-04 |

## Current Phase

**Phase 1: Data Pipeline and Project Foundation**
Status: In Progress
Plans: 1/3
Current Plan: 01-02

## Accumulated Context

### Key Decisions
- DataCache.load() returns only data field, not wrapper (clean API for fetchers)
- MD5 hash for cache keys (deterministic, acceptable collision risk at project scale)
- Explicit FileNotFoundError in load_config for clear error messages
- Separate models per property (not multi-output) per research recommendation
- CGCNN before MEGNet (zero dependency conflicts vs matgl/DGL risk)
- Baselines with featurization (fast end-to-end validation)
- Compositional group splitting from day one (prevents leakage)
- Streamlit for dashboard (not Dash -- simpler, fast development, sufficient for dataset scale)
- matgl v1.3.0 for MEGNet (NOT v2.0.0 -- MEGNet not yet ported to PyG in v2)

### Research Flags
- Phase 4 (MEGNet): matgl v1.3.0 + PyTorch compatibility untested; may need separate conda env
- Phase 3 (CGCNN): Transfer learning from full MP (~150K entries) needs strategy research during planning
- OQMD: qmpy_rester unmaintained since 2019; may need direct HTTP fallback
- Battery Data Genome: Not a single API; may need custom access approach

### Blockers
None

### Todos
None

---
*Last updated: 2026-03-06*
*Last session: Completed 01-01-PLAN.md (project foundation, config, cache)*
