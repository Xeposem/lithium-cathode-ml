---
phase: 01-data-pipeline-foundation
plan: 03
subsystem: data
tags: [pymatgen, numpy, csv, cleaning, validation, deduplication, iqr, cli]

# Dependency graph
requires:
  - phase: 01-data-pipeline-foundation/01
    provides: "DataCache, MaterialRecord, FilterRecord, config loader"
provides:
  - "BDGFetcher for Battery Data Genome CSV download and parse"
  - "CleaningPipeline with structure validation, IQR outlier removal, deduplication"
  - "CLI entry point (python -m cathode_ml.data.fetch) orchestrating all fetchers + cleaning"
  - "FilterRecord-based cleaning log saved as JSON"
affects: [02-01, 03-01, 04-01]

# Tech tracking
tech-stack:
  added: [pymatgen, requests]
  patterns: [filter-logging-pipeline, csv-file-fetcher, graceful-degradation, source-priority-dedup]

key-files:
  created:
    - cathode_ml/data/bdg_fetcher.py
    - cathode_ml/data/clean.py
    - cathode_ml/data/fetch.py
    - cathode_ml/data/__main__.py
    - tests/test_bdg_fetcher.py
    - tests/test_clean.py
  modified: []

key-decisions:
  - "BDG fetcher is a CSV file downloader, not an API client -- BDG is not a single API"
  - "BDG returns empty list on failure (graceful degradation -- pipeline works with MP + OQMD alone)"
  - "Deduplication uses source priority: MP > OQMD > BDG"
  - "IQR outlier removal skips when fewer than 4 data points to avoid meaningless statistics"
  - "Fetcher imports in fetch.py are deferred (try/except ImportError) so CLI works even if fetcher modules are unavailable"
  - "Used Optional[float] instead of float|None for Python 3.9 compatibility"

patterns-established:
  - "Filter logging: every filter step produces a FilterRecord for reproducibility"
  - "Graceful degradation: supplementary data sources return empty list on failure"
  - "Source priority dedup: materials_project preferred over oqmd over battery_data_genome"

requirements-completed: [DATA-03, DATA-05, DATA-06]

# Metrics
duration: 6min
completed: 2026-03-06
---

# Phase 1 Plan 03: BDG Fetcher, Cleaning Pipeline, CLI Summary

**BDG CSV fetcher with cache, CleaningPipeline with pymatgen structure validation, IQR outlier removal, source-priority dedup, and CLI entry point orchestrating all sources**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-06T03:16:03Z
- **Completed:** 2026-03-06T03:21:41Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- BDGFetcher downloads CSV from NREL, parses into MaterialRecord, caches results, returns empty list gracefully on failure
- CleaningPipeline validates crystal structures via pymatgen, removes IQR outliers, deduplicates across sources preferring Materials Project
- Every filter step logged as FilterRecord and saved as JSON for full audit trail
- CLI entry point (python -m cathode_ml.data.fetch) orchestrates all fetchers and cleaning with --config and --force-refresh flags
- All 63 tests pass (20 new + 43 existing, zero regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Battery Data Genome fetcher**
   - `0682ca1` (test: failing tests for BDG fetcher)
   - `2e14d5d` (feat: implement BDG fetcher with CSV download and cache)

2. **Task 2: Cleaning pipeline with filter logging and CLI entry point**
   - `063156a` (test: failing tests for cleaning pipeline)
   - `615a228` (feat: implement cleaning pipeline, CLI entry point)

## Files Created/Modified
- `cathode_ml/data/bdg_fetcher.py` - Battery Data Genome CSV downloader and parser
- `cathode_ml/data/clean.py` - CleaningPipeline with validation, outlier removal, dedup, logging
- `cathode_ml/data/fetch.py` - CLI orchestrator running all fetchers + cleaning pipeline
- `cathode_ml/data/__main__.py` - Enables python -m cathode_ml.data.fetch
- `tests/test_bdg_fetcher.py` - 10 tests for BDG fetcher (init, cache, download, parse)
- `tests/test_clean.py` - 10 tests for cleaning pipeline (validate, filter, outlier, dedup, log)

## Decisions Made
- BDG fetcher is a CSV file downloader, not an API client (BDG is not a single API per research)
- BDG returns empty list on failure for graceful degradation
- Deduplication uses source priority: MP > OQMD > BDG
- IQR outlier removal skips when fewer than 4 data points
- Fetcher imports in fetch.py use try/except ImportError for robustness
- Used Optional[float] instead of float|None for Python 3.9 compatibility

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Python 3.9 type hint syntax**
- **Found during:** Task 1 (BDG fetcher implementation)
- **Issue:** Used `float | None` union type syntax which requires Python 3.10+; project uses Python 3.9
- **Fix:** Changed to `Optional[float]` and `Optional[int]` from typing module
- **Files modified:** cathode_ml/data/bdg_fetcher.py
- **Verification:** All tests pass on Python 3.9
- **Committed in:** 2e14d5d (part of Task 1 commit)

**2. [Rule 3 - Blocking] Made fetcher imports resilient in fetch.py**
- **Found during:** Task 2 (CLI entry point)
- **Issue:** MPFetcher and OQMDFetcher may not be available (Plan 01-02 runs independently); hard imports would crash CLI
- **Fix:** Wrapped fetcher imports in try/except ImportError with warning log
- **Files modified:** cathode_ml/data/fetch.py
- **Verification:** python -m cathode_ml.data.fetch --help works without MP/OQMD fetcher modules
- **Committed in:** 615a228 (part of Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correct operation on target Python version and resilient CLI. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Complete data pipeline: 3 fetchers (MP, OQMD, BDG) + cleaning + CLI
- CleaningPipeline ready for use by feature engineering in Phase 2
- Cleaning log provides audit trail for reproducibility
- CLI entry point ready for end-to-end data pipeline runs

## Self-Check: PASSED

All 6 created files verified on disk. All 4 commit hashes verified in git log.

---
*Phase: 01-data-pipeline-foundation*
*Completed: 2026-03-06*
