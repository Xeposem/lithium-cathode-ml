---
phase: 05-evaluation-and-benchmarking
plan: 03
subsystem: pipeline
tags: [argparse, cli, orchestration, lazy-imports]

# Dependency graph
requires:
  - phase: 05-evaluation-and-benchmarking
    provides: Evaluation metrics and comparison tables (plan 01)
provides:
  - CLI pipeline orchestrator (python -m cathode_ml)
  - End-to-end workflow: fetch -> featurize -> train -> evaluate
  - Stage skip flags (--skip-fetch, --skip-train)
  - Model selection flag (--models)
affects: [06-dashboard-and-documentation]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy-imports-in-stage-functions, argparse-cli-orchestration, stage-banner-logging]

key-files:
  created:
    - cathode_ml/pipeline.py
    - cathode_ml/__main__.py
    - tests/test_pipeline.py
  modified: []

key-decisions:
  - "Evaluate stage gracefully skips plots module if not yet available (plan 05-02 not yet executed)"
  - "Featurize stage is a pass-through log since featurization happens inline in training orchestrators"

patterns-established:
  - "Stage banner pattern: === Stage N/M: Name === for progress visibility"
  - "Module entry point: cathode_ml/__main__.py delegates to pipeline.main()"

requirements-completed: [REPR-04]

# Metrics
duration: 3min
completed: 2026-03-07
---

# Phase 5 Plan 3: Pipeline Orchestrator Summary

**Argparse CLI orchestrating fetch/featurize/train/evaluate stages with lazy imports, skip flags, and model selection**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-07T07:28:55Z
- **Completed:** 2026-03-07T07:31:25Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- CLI pipeline with `python -m cathode_ml` and `python -m cathode_ml.pipeline` entry points
- Lazy imports in all stage functions for fast `--help` startup (no heavy deps loaded)
- Skip flags (--skip-fetch, --skip-train) and model selection (--models rf cgcnn) working
- 10 unit tests passing with fully mocked dependencies

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for pipeline CLI** - `c3183a8` (test)
2. **Task 1 (GREEN): Pipeline orchestrator implementation** - `f9779f9` (feat)

## Files Created/Modified
- `cathode_ml/pipeline.py` - Full pipeline orchestrator with build_parser, stage functions, run_pipeline, main
- `cathode_ml/__main__.py` - Module entry point enabling `python -m cathode_ml`
- `tests/test_pipeline.py` - 10 tests covering parsing, skip logic, banners, module entry

## Decisions Made
- Evaluate stage uses try/except for plots module import since plan 05-02 (plots) hasn't been executed yet; gracefully logs and continues
- Featurize stage logs an informational message rather than performing work, since featurization is inline in each training orchestrator

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Graceful handling of missing plots module**
- **Found during:** Task 1 (evaluate stage implementation)
- **Issue:** Plan specifies calling apply_nature_style(), plot_bar_comparison(), plot_learning_curves() from evaluation.plots, but this module does not exist (part of plan 05-02 which hasn't executed yet)
- **Fix:** Wrapped plots import in try/except ImportError with informational log message
- **Files modified:** cathode_ml/pipeline.py
- **Verification:** --help runs without error, evaluate stage handles missing module gracefully
- **Committed in:** f9779f9 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Necessary for correctness -- cannot import a module that doesn't exist yet. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Pipeline orchestrator ready for end-to-end runs once data is fetched
- Plots integration will activate automatically when evaluation.plots module is created (plan 05-02)
- Phase 6 (Dashboard and Documentation) can reference pipeline CLI for documentation

---
*Phase: 05-evaluation-and-benchmarking*
*Completed: 2026-03-07*
