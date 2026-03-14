---
phase: 6
slug: dashboard-and-documentation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 |
| **Config file** | None (uses defaults) |
| **Quick run command** | `python -m pytest tests/test_dashboard.py -x` |
| **Full suite command** | `python -m pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_dashboard.py -x`
- **After every plan wave:** Run `python -m pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | DASH-01 | unit | `python -m pytest tests/test_dashboard.py::test_load_results -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | DASH-03 | unit | `python -m pytest tests/test_dashboard.py::test_data_explorer_load -x` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | DASH-04 | unit | `python -m pytest tests/test_dashboard.py::test_training_curves -x` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | DASH-05 | unit | `python -m pytest tests/test_dashboard.py::test_materials_filter -x` | ❌ W0 | ⬜ pending |
| 06-01-05 | 01 | 1 | DASH-06 | unit | `python -m pytest tests/test_dashboard.py::test_discovery_ranking -x` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | DASH-02 | unit | `python -m pytest tests/test_dashboard.py::test_predict_composition -x` | ❌ W0 | ⬜ pending |
| 06-02-02 | 02 | 1 | DASH-07 | unit | `python -m pytest tests/test_dashboard.py::test_crystal_render -x` | ❌ W0 | ⬜ pending |
| 06-03-01 | 03 | 2 | DOCS-01 | manual-only | N/A (content review) | N/A | ⬜ pending |
| 06-03-02 | 03 | 2 | DOCS-02 | manual-only | N/A (content review) | N/A | ⬜ pending |
| 06-03-03 | 03 | 2 | DOCS-03 | smoke | `python -m pytest tests/test_dashboard.py::test_readme_exists -x` | ❌ W0 | ⬜ pending |
| 06-03-04 | 03 | 2 | DOCS-04 | manual-only | N/A (content review) | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_dashboard.py` — unit tests for dashboard utility functions (data loading, chart creation, filtering, model loading, crystal rendering)
- [ ] Tests should test utility/helper functions directly, NOT Streamlit widgets (avoids AppTest complexity)

*Note: Dashboard tests focus on the data layer and chart generation functions that can be tested without a running Streamlit server.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README introduction section | DOCS-01 | Content quality review | Read README.md, verify introduction covers project motivation and scope |
| README methodology section | DOCS-02 | Content quality review | Verify methodology covers CGCNN, MEGNet, RF, XGB architectures and key design choices |
| README results summary | DOCS-04 | Content quality review | Verify results table shows best model per property with MAE/R-squared |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
