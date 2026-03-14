---
phase: 10-model-bug-fixes
verified: 2026-03-13T00:00:00Z
status: gaps_found
score: 4/5 must-haves verified
gaps:
  - truth: "Existing unit tests pass after the fix"
    status: partial
    reason: "TestConvertLightningLogsTensorNet::test_convert_lightning_logs fails — asserts float(rows[0]['train_loss']) == 2.0 but the value is an empty string. This is a pre-existing bug in the test itself (not production code) documented in deferred-items.md. The M3GNet equivalent test correctly expects empty string. The test was NOT fixed as part of this phase."
    artifacts:
      - path: "tests/test_tensornet.py"
        issue: "TestConvertLightningLogsTensorNet::test_convert_lightning_logs asserts incorrect value for rows[0]['train_loss'] — expects 2.0 but convert_lightning_logs correctly returns empty string for epoch-0 train due to Lightning's epoch-shift behavior. Fix: change assertion to rows[0]['train_loss'] == '' matching the M3GNet test pattern."
    missing:
      - "Fix test assertion in tests/test_tensornet.py line 185: replace `assert float(rows[0]['train_loss']) == pytest.approx(2.0)` with `assert rows[0]['train_loss'] == ''`"
---

# Phase 10: Model Bug Fixes Verification Report

**Phase Goal:** M3GNet and TensorNet produce valid, trustworthy predictions on all properties
**Verified:** 2026-03-13
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | predict_with_m3gnet returns values in correct physical range without manual denormalization in training code | VERIFIED | train_m3gnet.py line 267: `predictions = predict_with_m3gnet(model, test_structures)` — no `p * data_std + data_mean` applied. Grep for manual rescaling patterns returns zero matches in both files. |
| 2 | predict_with_tensornet returns values in correct physical range without manual denormalization in training code | VERIFIED | train_tensornet.py line 269: `predictions = predict_with_tensornet(model, test_structures)` — no `p * data_std + data_mean` applied. Comment at line 267-268 explicitly documents the reason. |
| 3 | TensorNet training log output shows one clean entry per epoch with no duplicated lines | VERIFIED | Both trainers: `enable_progress_bar=False`, `enable_model_summary=False`, and `logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)` present. (train_m3gnet.py lines 186-191, train_tensornet.py lines 181-186) |
| 4 | Existing unit tests pass after the fix | FAILED | 33/34 tests pass. `TestConvertLightningLogsTensorNet::test_convert_lightning_logs` fails: asserts `float(rows[0]["train_loss"]) == 2.0` but value is empty string. This is a pre-existing bug in the test, documented in deferred-items.md. |
| 5 | New tests explicitly verify no double-denormalization occurs | VERIFIED | `TestNoDenormalization::test_predictions_not_double_denormalized` exists in both test files. Tests mock `_run_lightning_training` to return `(2.0, 3.0)` (data_mean, data_std), mock `predict_with_m3gnet`/`predict_with_tensornet` to return `[1.5, 2.3, 0.8]`, spy on `compute_metrics`, and assert `y_pred == [1.5, 2.3, 0.8]` (not `[6.5, 8.9, 4.4]`). Both tests pass. |

**Score:** 4/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `cathode_ml/models/train_m3gnet.py` | M3GNet training without double-denormalization | VERIFIED | File exists, substantive (443 lines). `predictions = predict_with_m3gnet(model, test_structures)` at line 267. No manual rescaling. `enable_progress_bar=False` at line 186. |
| `cathode_ml/models/train_tensornet.py` | TensorNet training without double-denormalization | VERIFIED | File exists, substantive (445 lines). `predictions = predict_with_tensornet(model, test_structures)` at line 269. No manual rescaling. `enable_progress_bar=False` at line 181. |
| `tests/test_m3gnet.py` | Test proving denormalization is not applied twice | VERIFIED | `TestNoDenormalization` class at line 462 with `test_predictions_not_double_denormalized`. Spy captures y_pred, asserts raw values. Passes. |
| `tests/test_tensornet.py` | Test proving denormalization is not applied twice | VERIFIED | `TestNoDenormalization` class at line 550 with `test_predictions_not_double_denormalized`. Spy captures y_pred, asserts raw values. Passes. Pre-existing `TestConvertLightningLogsTensorNet` test has incorrect assertion (separate issue). |

**Note on PLAN `contains` field:** The PLAN frontmatter specifies `contains: "raw_predictions = predict_with_m3gnet"` for train_m3gnet.py — this describes the OLD buggy pattern that was supposed to be removed, not the new pattern. The actual file correctly uses `predictions = predict_with_m3gnet(...)` (no `raw_` prefix, no rescaling). The PLAN wording was ambiguous (documenting the code to remove rather than the code to add), but the implementation is correct.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `cathode_ml/models/train_m3gnet.py` | `cathode_ml/models/m3gnet.py` | `predict_with_m3gnet returns already-denormalized values` | WIRED | Line 267: `predictions = predict_with_m3gnet(model, test_structures)` — imported at line 29, result used directly as `y_pred` at line 269 passed to `compute_metrics`. No intermediate transformation. |
| `cathode_ml/models/train_tensornet.py` | `cathode_ml/models/tensornet.py` | `predict_with_tensornet returns already-denormalized values` | WIRED | Line 269: `predictions = predict_with_tensornet(model, test_structures)` — imported at line 30, result used directly as `y_pred` at line 271 passed to `compute_metrics`. No intermediate transformation. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIX-01 | 10-01-PLAN.md | M3GNet/TensorNet double-denormalization bug is fixed — models produce valid R-squared on all properties | SATISFIED | Manual denormalization (`p * data_std + data_mean`) removed from both `train_m3gnet_for_property` and `train_tensornet_for_property`. TestNoDenormalization tests pass in both files. |
| FIX-02 | 10-01-PLAN.md | TensorNet training runs without log duplication errors | SATISFIED | Both trainers set `enable_progress_bar=False`, `enable_model_summary=False`, and Lightning logger to WARNING. Applied to M3GNet as well for consistency. |

Both requirements assigned to Phase 10 in REQUIREMENTS.md are accounted for. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_tensornet.py` | 185 | Incorrect test assertion: `float(rows[0]["train_loss"]) == pytest.approx(2.0)` when actual value is `""` | Warning | Test fails in CI; masks real behavior. Pre-existing bug, documented in deferred-items.md. Not introduced by this phase. |

No TODO/FIXME/HACK/PLACEHOLDER comments found in modified files. No empty implementations. No return null/stub patterns.

### Human Verification Required

None — all critical behaviors are verified programmatically via mocked unit tests. Real end-to-end training verification (actual R-squared improvement) would require GPU and the full matgl stack, which is out of scope for unit verification.

### Gaps Summary

One gap blocks a clean test run: `TestConvertLightningLogsTensorNet::test_convert_lightning_logs` in `tests/test_tensornet.py` line 185 asserts `float(rows[0]["train_loss"]) == pytest.approx(2.0)` but `convert_lightning_logs` correctly returns an empty string for epoch-0 train loss (Lightning's epoch-shift behavior). The M3GNet equivalent test (`test_m3gnet.py` line 175) correctly asserts `rows[0]["train_loss"] == ""`. This is a one-line test fix, not a production code defect. The production fix (double-denormalization removal and log suppression) is correct and complete.

---

_Verified: 2026-03-13_
_Verifier: Claude (gsd-verifier)_
