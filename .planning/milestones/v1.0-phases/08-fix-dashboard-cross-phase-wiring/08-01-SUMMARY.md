---
phase: 08-fix-dashboard-cross-phase-wiring
plan: 01
subsystem: dashboard
tags: [streamlit, wiring, cross-phase, import-fix, data-path]

# Dependency graph
requires:
  - phase: 06-dashboard-and-documentation
    provides: Dashboard pages and utility modules
  - phase: 01-data-pipeline
    provides: data/processed/materials.json output path
  - phase: 03-cgcnn
    provides: structure_to_graph(structure, config) API
  - phase: 04-megnet
    provides: Raw state_dict checkpoint format
provides:
  - Working Predict page with correct GNN graph conversion import
  - Working Crystal Viewer page with module-level main() call
  - Fixed MEGNet checkpoint loading (handles raw and wrapped formats)
  - Correct data path for processed materials in Data Explorer
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level main() call in Streamlit pages (no __name__ guard)"
    - "Dual-format checkpoint loading (raw state_dict + wrapped dict)"
    - "Plain JSON list for materials data (not DataCache wrapper)"

key-files:
  created: []
  modified:
    - dashboard/pages/predict.py
    - dashboard/pages/crystal_viewer.py
    - dashboard/utils/model_loader.py
    - dashboard/utils/data_loader.py
    - tests/test_dashboard.py
    - tests/test_dashboard_predict.py

key-decisions:
  - "Keep backward compatibility for legacy DataCache wrapper format in get_cached_records"
  - "Pass full features config dict to structure_to_graph (not individual graph params)"

patterns-established:
  - "Module-level main() in Streamlit pages: never use if __name__ == '__main__' guard"
  - "Checkpoint loading: always handle both raw state_dict and wrapped dict formats"

requirements-completed: [DASH-01, DASH-02, DASH-03, DASH-05, DASH-06, DASH-07]

# Metrics
duration: 4min
completed: 2026-03-07
---

# Phase 08 Plan 01: Fix Dashboard Cross-Phase Wiring Summary

**Fixed 4 cross-phase wiring bugs: wrong import name, wrong data path, unconditional dict key access, and __name__ guards blocking Streamlit page rendering**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T10:43:58Z
- **Completed:** 2026-03-07T10:47:34Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- All 6 dashboard pages importable without errors
- predict.py and crystal_viewer.py call main() at module level for Streamlit rendering
- model_loader.py uses structure_to_graph(structure, config) matching actual API
- MEGNet checkpoint loading handles both raw state_dict and wrapped format
- get_cached_records reads from data/processed/materials.json (correct pipeline output)
- Full test suite: 175 passed, 0 failed

## Task Commits

Each task was committed atomically:

1. **Task 1: Update tests for correct wiring expectations (TDD RED)** - `424ba43` (test)
2. **Task 2: Fix all 4 dashboard wiring bugs (GREEN)** - `4a35361` (fix)

_Note: TDD task split across two commits (test then fix)_

## Files Created/Modified
- `dashboard/pages/predict.py` - Removed __name__ guard, main() called at module level
- `dashboard/pages/crystal_viewer.py` - Removed __name__ guard, main() called at module level
- `dashboard/utils/model_loader.py` - Fixed import to structure_to_graph, added dual-format checkpoint loading
- `dashboard/utils/data_loader.py` - Changed data path to data/processed/materials.json with legacy fallback
- `tests/test_dashboard.py` - Updated data loader tests for new path/format
- `tests/test_dashboard_predict.py` - Added structure prediction wiring tests and page-level main() tests

## Decisions Made
- Keep backward compatibility for legacy DataCache wrapper format in get_cached_records
- Pass full features config dict to structure_to_graph rather than extracting individual graph params (matches actual API signature)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All dashboard pages now wire correctly to core modules from Phases 1-5, 7
- No further cross-phase wiring bugs known

---
*Phase: 08-fix-dashboard-cross-phase-wiring*
*Completed: 2026-03-07*

## Self-Check: PASSED
