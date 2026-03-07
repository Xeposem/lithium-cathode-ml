---
phase: 7
slug: fix-pipeline-orchestrator-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.4.2 |
| **Config file** | none (default discovery) |
| **Quick run command** | `python -m pytest tests/test_pipeline.py tests/test_evaluation.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_pipeline.py tests/test_evaluation.py tests/test_baselines.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 0 | EVAL-02, DATA-04 | unit | `python -m pytest tests/test_pipeline.py::TestRunTrainStage -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 0 | EVAL-01 | unit | `python -m pytest tests/test_baselines.py -x` | ❌ W0 | ⬜ pending |
| 07-01-03 | 01 | 1 | EVAL-02, DATA-04 | unit | `python -m pytest tests/test_pipeline.py::TestRunTrainStage -x` | ❌ W0 | ⬜ pending |
| 07-01-04 | 01 | 1 | EVAL-01, EVAL-03 | unit | `python -m pytest tests/test_evaluation.py::TestLoadAllResults -x` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pipeline.py::TestRunTrainStage` — new test class verifying config loading uses separate YAML files
- [ ] `tests/test_pipeline.py::TestRunTrainStage::test_loads_processed_records` — verifies data loaded from `data/processed/materials.json`
- [ ] `tests/test_baselines.py` — verify results JSON saved to `baselines/baseline_results.json` path

*Wave 0 stubs ensure feedback sampling starts immediately.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
