---
phase: 8
slug: fix-dashboard-cross-phase-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 |
| **Config file** | pyproject.toml (default discovery) |
| **Quick run command** | `python -m pytest tests/test_dashboard.py tests/test_dashboard_predict.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_dashboard.py tests/test_dashboard_predict.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 0 | DASH-02, DASH-07 | unit | `python -m pytest tests/test_dashboard_predict.py -x` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 0 | DASH-03 | unit | `python -m pytest tests/test_dashboard.py::TestGetCachedRecords -x` | ❌ W0 | ⬜ pending |
| 08-01-03 | 01 | 1 | DASH-02, DASH-07 | unit | `python -m pytest tests/test_dashboard_predict.py -x` | ❌ W0 | ⬜ pending |
| 08-01-04 | 01 | 1 | DASH-01, DASH-03, DASH-05, DASH-06 | unit | `python -m pytest tests/test_dashboard.py -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_dashboard.py::TestGetCachedRecords` — update test to use `data/processed/materials.json` path
- [ ] `tests/test_dashboard_predict.py` — add test for `structure_to_graph` import (not `structure_to_pyg_data`)
- [ ] `tests/test_dashboard_predict.py` — add test for MEGNet raw state_dict loading
- [ ] Smoke test: verify predict.py and crystal_viewer.py `main()` called at module level

*Wave 0 stubs ensure feedback sampling starts immediately.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Crystal viewer 3D rendering | DASH-07 | Requires visual confirmation of 3D molecule rendering | Run `streamlit run dashboard/app.py`, navigate to Crystal Viewer, load a structure |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
