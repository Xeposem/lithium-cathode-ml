---
phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
plan: 04
subsystem: testing
tags: [m3gnet, tensornet, matgl, testing, migration, cleanup]

requires:
  - phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
    provides: M3GNet and TensorNet wrappers, training orchestrators, configs
  - phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
    provides: Updated evaluation metrics, plots, and dashboard for M3GNet/TensorNet
provides:
  - M3GNet unit tests covering wrapper and training orchestrator
  - TensorNet unit tests covering wrapper and training orchestrator
  - Updated integration tests (pipeline, evaluation, plots, dashboard) for m3gnet/tensornet
  - Deletion of all old MEGNet files (megnet.py, train_megnet.py, megnet.yaml, test_megnet.py)
affects: []

tech-stack:
  added: []
  patterns: [sys-modules-mocking-for-matgl, dynamic-subplot-grid]

key-files:
  created:
    - tests/test_m3gnet.py
    - tests/test_tensornet.py
  modified:
    - tests/test_pipeline.py
    - tests/test_evaluation.py
    - tests/test_plots.py
    - tests/test_dashboard_predict.py
    - cathode_ml/evaluation/plots.py
  deleted:
    - cathode_ml/models/megnet.py
    - cathode_ml/models/train_megnet.py
    - configs/megnet.yaml
    - tests/test_megnet.py

key-decisions:
  - "sys.modules mocking pattern for matgl lazy imports in TensorNet tests (pre-populate matgl.ext.pymatgen)"
  - "Dynamic grid layout (n_cols=3) for parity plots to accommodate 5 models"

patterns-established:
  - "sys.modules pre-population for testing lazy imports from uninstalled packages"

requirements-completed: [MODL-02]

duration: 8min
completed: 2026-03-08
---

# Phase 9 Plan 04: Tests and MEGNet Cleanup Summary

**M3GNet and TensorNet test suites with 32 new tests, updated integration tests, and complete MEGNet file deletion**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-08
- **Completed:** 2026-03-08
- **Tasks:** 5
- **Files modified:** 11 (2 created, 5 modified, 4 deleted)

## Accomplishments
- Created comprehensive M3GNet test suite (16 tests) covering lazy imports, pretrained loading, state dict, prediction, Lightning logs, training, and artifacts
- Created comprehensive TensorNet test suite (16 tests) covering lazy imports, config-driven construction, state dict, prediction, training, and artifacts
- Updated all integration tests to reference m3gnet/tensornet instead of megnet
- Deleted 4 old MEGNet files; zero megnet references remain in active codebase
- Full test suite passes (182 tests, 0 failures)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create M3GNet test suite** - `08767c0` (test)
2. **Task 2: Create TensorNet test suite** - `d69d273` (test)
3. **Task 3: Update existing integration tests** - `cd57bf9` (test)
4. **Task 4: Delete old MEGNet files** - `b91aebf` (refactor)
5. **Task 5: Full test suite verification** - no commit (verification only)

## Files Created/Modified
- `tests/test_m3gnet.py` - 16 tests: lazy imports, pretrained loading, state dict, prediction, Lightning logs, training orchestrator, artifact format
- `tests/test_tensornet.py` - 16 tests: lazy imports, config-driven construction, state dict, prediction, Lightning logs, from-scratch training, artifact format
- `tests/test_pipeline.py` - Updated default models, mock patches, and config assertions for m3gnet/tensornet
- `tests/test_evaluation.py` - Replaced SAMPLE_MEGNET_RESULTS with SAMPLE_M3GNET_RESULTS and SAMPLE_TENSORNET_RESULTS
- `tests/test_plots.py` - Updated gnn_models list and mock CSV fixture for m3gnet/tensornet
- `tests/test_dashboard_predict.py` - Replaced megnet state_dict test with m3gnet/tensornet tests
- `cathode_ml/evaluation/plots.py` - Fixed parity plot grid layout for 5 models (was 2x2, now dynamic)

## Decisions Made
- Used sys.modules pre-population pattern for TensorNet tests because `from matgl.ext.pymatgen import get_element_list` inside `train_tensornet_for_property` requires matgl modules to be mockable before the patch context manager resolves
- Changed parity plot from 2x2 to dynamic 2x3 grid to accommodate 5 models (rf, xgb, cgcnn, m3gnet, tensornet)

## Deviations from Plan

### Auto-fixed Issues

**1. Parity plot grid layout for 5 models**
- **Found during:** Task 3 (Update integration tests)
- **Issue:** plots.py plot_parity used 2x2 grid (4 axes) but MODELS_ORDER now has 5 entries, causing IndexError
- **Fix:** Changed to dynamic grid calculation: n_cols=3, n_rows computed from model count
- **Files modified:** cathode_ml/evaluation/plots.py
- **Verification:** All plot tests pass
- **Committed in:** cd57bf9 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (plot grid layout)
**Impact on plan:** Necessary fix for correctness with 5-model architecture. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 9 migration is complete: all MEGNet references removed
- Full test suite green with 182 passing tests
- No blockers

---
*Phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x*
*Completed: 2026-03-08*
