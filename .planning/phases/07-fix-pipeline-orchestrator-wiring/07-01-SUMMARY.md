---
phase: 07-fix-pipeline-orchestrator-wiring
plan: 01
subsystem: pipeline
tags: [pipeline, config, wiring, baselines, json, yaml]

# Dependency graph
requires:
  - phase: 01-data-pipeline
    provides: "data/processed/materials.json, MaterialRecord schema"
  - phase: 02-featurization-baselines
    provides: "baselines.py run_baselines function"
  - phase: 05-evaluation-benchmarking
    provides: "metrics.py load_all_results expecting baselines/baseline_results.json"
provides:
  - "Fixed pipeline.py with separate YAML config loading (features, baselines, cgcnn, megnet)"
  - "Fixed pipeline.py data reading from JSON instead of DataCache"
  - "Fixed baselines.py save path matching metrics.py load path"
affects: [08-train-evaluate-report]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Separate config loading per model type", "JSON data reading with MaterialRecord construction"]

key-files:
  created: []
  modified:
    - cathode_ml/pipeline.py
    - cathode_ml/models/baselines.py
    - tests/test_pipeline.py
    - tests/test_baselines.py

key-decisions:
  - "Import json as _json inside run_train_stage to avoid shadowing module-level names"
  - "Patch cathode_ml.config.load_config (definition site) since pipeline uses lazy imports"

patterns-established:
  - "Separate YAML config per model: features.yaml, baselines.yaml, cgcnn.yaml, megnet.yaml"
  - "JSON data reading pattern: json.load + MaterialRecord(**r) for record construction"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, DATA-04]

# Metrics
duration: 3min
completed: 2026-03-07
---

# Phase 07 Plan 01: Fix Pipeline Orchestrator Wiring Summary

**Fixed three cross-phase wiring bugs: separate YAML config loading, JSON data reading, and baselines save path alignment with evaluation loader**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T10:13:26Z
- **Completed:** 2026-03-07T10:16:30Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- pipeline.py now loads features.yaml, baselines.yaml, cgcnn.yaml, megnet.yaml separately instead of monolithic data.yaml
- pipeline.py reads data/processed/materials.json via json.load + MaterialRecord instead of DataCache
- baselines.py saves results to baselines/baseline_results.json matching metrics.py load_all_results expectation
- Full test suite green: 169 passed, 3 skipped, 0 failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for wiring bugs** - `7c5e52c` (test)
2. **Task 2: Fix pipeline.py and baselines.py** - `3b24b6f` (feat)

_Note: TDD task - test commit (RED) followed by implementation commit (GREEN)_

## Files Created/Modified
- `cathode_ml/pipeline.py` - Fixed run_train_stage: separate config loading, JSON data reading
- `cathode_ml/models/baselines.py` - Fixed results save path to baselines/ subdirectory
- `tests/test_pipeline.py` - Added TestRunTrainStage class with config and data loading tests
- `tests/test_baselines.py` - Added test_results_saved_to_baselines_subdir

## Decisions Made
- Used `import json as _json` inside function to avoid any potential shadowing
- Patching strategy targets cathode_ml.config.load_config (definition site) since pipeline uses lazy imports inside function body

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pipeline orchestrator now correctly wired for end-to-end execution
- Ready for phase 08 (train-evaluate-report) which depends on correct pipeline wiring

---
*Phase: 07-fix-pipeline-orchestrator-wiring*
*Completed: 2026-03-07*

## Self-Check: PASSED
- All 5 files verified present
- Both task commits verified: 7c5e52c, 3b24b6f
- Old patterns (data.yaml, DataCache) confirmed removed from code
