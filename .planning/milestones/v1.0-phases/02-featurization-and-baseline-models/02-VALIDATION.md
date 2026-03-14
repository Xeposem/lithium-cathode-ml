---
phase: 2
slug: featurization-and-baseline-models
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 |
| **Config file** | pyproject.toml (pytest section) |
| **Quick run command** | `pytest tests/ -x --timeout=60` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x --timeout=60`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | FEAT-01 | unit | `pytest tests/test_graph.py::test_structure_to_graph -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | FEAT-01 | unit | `pytest tests/test_graph.py::test_no_disconnected_graphs -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | FEAT-02 | unit | `pytest tests/test_graph.py::test_gaussian_expansion -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | FEAT-02 | unit | `pytest tests/test_graph.py::test_configurable_cutoff -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | FEAT-03 | unit | `pytest tests/test_composition.py::test_magpie_features -x` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | FEAT-03 | unit | `pytest tests/test_composition.py::test_nan_handling -x` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | FEAT-04 | unit | `pytest tests/test_split.py::test_group_split_no_leakage -x` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | FEAT-04 | unit | `pytest tests/test_split.py::test_no_formula_overlap -x` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 2 | MODL-03 | integration | `pytest tests/test_baselines.py::test_random_forest -x` | ❌ W0 | ⬜ pending |
| 02-04-02 | 04 | 2 | MODL-04 | integration | `pytest tests/test_baselines.py::test_xgboost -x` | ❌ W0 | ⬜ pending |
| 02-04-03 | 04 | 2 | MODL-03/04 | unit | `pytest tests/test_baselines.py::test_results_json -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_graph.py` — stubs for FEAT-01, FEAT-02
- [ ] `tests/test_composition.py` — stubs for FEAT-03
- [ ] `tests/test_split.py` — stubs for FEAT-04
- [ ] `tests/test_baselines.py` — stubs for MODL-03, MODL-04
- [ ] Update `tests/conftest.py` — add fixtures for sample structures with known neighbor counts

*Existing infrastructure from Phase 1 covers pytest setup and shared fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Graph visualization sanity check | FEAT-01 | Visual inspection of graph structure | Plot a sample graph with networkx, verify atom nodes and bond edges look correct |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
