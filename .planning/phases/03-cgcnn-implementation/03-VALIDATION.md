---
phase: 3
slug: cgcnn-implementation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-05
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 |
| **Config file** | tests/conftest.py |
| **Quick run command** | `pytest tests/test_cgcnn.py tests/test_trainer.py tests/test_model_utils.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_cgcnn.py tests/test_trainer.py tests/test_model_utils.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | MODL-01 | unit | `pytest tests/test_cgcnn.py::test_cgcnn_forward -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | MODL-01 | unit | `pytest tests/test_cgcnn.py::test_cgcnn_architecture -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | MODL-06 | unit | `pytest tests/test_cgcnn.py::test_config_driven -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | MODL-05 | integration | `pytest tests/test_trainer.py::test_per_property_training -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | MODL-07 | integration | `pytest tests/test_trainer.py::test_checkpoint_saving -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | MODL-07 | integration | `pytest tests/test_trainer.py::test_csv_logging -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | MODL-07 | unit | `pytest tests/test_trainer.py::test_results_json_format -x` | ❌ W0 | ⬜ pending |
| 03-00-01 | 00 | 0 | — | unit | `pytest tests/test_model_utils.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_cgcnn.py` — CGCNN model architecture, forward pass, config-driven construction (MODL-01, MODL-06)
- [ ] `tests/test_trainer.py` — GNNTrainer loop, checkpoints, CSV logging, JSON results (MODL-05, MODL-07)
- [ ] `tests/test_model_utils.py` — shared compute_metrics, save_results functions
- [ ] `tests/conftest.py` — add fixtures: sample PyG Data with y target, minimal cgcnn config dict

*Existing infrastructure covers pytest framework and conftest.py base.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training loss converges | MODL-06 | Convergence depends on data quality and hyperparameters | Run full training, inspect loss curves CSV for decreasing trend |
| Cross-validated metrics comparable to baselines | MODL-01 | Requires full pipeline run with real data | Compare cgcnn_results.json vs baseline_results.json |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
