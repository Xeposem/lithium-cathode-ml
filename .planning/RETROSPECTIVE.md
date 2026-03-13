# Retrospective

## Milestone: v1.0 — MVP

**Shipped:** 2026-03-13
**Phases:** 9 | **Plans:** 23

### What Was Built
- Data pipeline ingesting cathode materials from Materials Project and OQMD with caching
- CGCNN graph neural network (R²=0.995 on formation energy)
- M3GNet (pretrained fine-tuning) and TensorNet (O(3)-equivariant) via matgl 2.x
- RF/XGBoost baselines with Magpie compositional features
- Interactive Streamlit dashboard (6 pages: overview, comparison, predict, data explorer, materials explorer, crystal viewer)
- End-to-end CLI pipeline with YAML configuration

### What Worked
- Phase-based decomposition kept each unit of work focused and testable
- Compositional group splitting from day one prevented data leakage issues later
- CGCNN-first strategy (Phase 3 before 4) validated GNN infrastructure with zero dependency risk
- Lazy import pattern avoided matgl/DGL version conflicts across the codebase
- Fix phases (7, 8) after milestone audit caught cross-phase wiring issues early

### What Was Inefficient
- MEGNet was built in Phase 4, then entirely replaced by M3GNet/TensorNet in Phase 9 — could have been anticipated
- M3GNet/TensorNet denormalization bug wasn't caught until post-pipeline run — test predictions weren't validated against known values
- Multiple BDG data source implementation that was later removed
- ROADMAP.md plan checkboxes fell out of sync with actual completion state

### Patterns Established
- `_import_matgl()` lazy import helper pattern for optional heavy dependencies
- `build_X_from_config()` pattern for config-driven model construction
- `_render()` function pattern in dashboard pages for safe import outside Streamlit
- Per-property sequential training loop with seed reset before each property
- `compute_metrics` accepting raw arrays (not model objects) for cross-framework compatibility

### Key Lessons
- Always validate test predictions against known values (not just training curves) — near-zero training loss with bad test metrics = normalization bug
- Plan for matgl API instability — the library is evolving rapidly and MEGNet was dropped between v1 and v2
- Windows + CPU-only DGL requires explicit accelerator detection before Lightning trainer init

---
*Last updated: 2026-03-13*
