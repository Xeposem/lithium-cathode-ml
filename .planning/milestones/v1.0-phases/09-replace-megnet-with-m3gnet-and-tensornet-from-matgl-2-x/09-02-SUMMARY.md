---
phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
plan: 02
subsystem: models
tags: [m3gnet, tensornet, matgl, lightning, training, pipeline]

requires:
  - phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
    provides: M3GNet and TensorNet model wrappers (m3gnet.py, tensornet.py)
  - phase: 04-megnet-implementation
    provides: MEGNet training orchestrator pattern (train_megnet.py)
provides:
  - M3GNet Lightning training orchestrator with matgl 2.x APIs
  - TensorNet Lightning training orchestrator (from-scratch training)
  - Pipeline CLI with m3gnet and tensornet model choices (megnet removed)
affects: [09-03-evaluation-dashboard, 09-04-tests]

tech-stack:
  added: []
  patterns: [partial-collate-fn, include-line-graph-flag, torch-dataloader-with-matgl-collate]

key-files:
  created:
    - cathode_ml/models/train_m3gnet.py
    - cathode_ml/models/train_tensornet.py
  modified:
    - cathode_ml/pipeline.py

key-decisions:
  - "Use torch.utils.data.DataLoader (not MGLDataLoader) with matgl collate_fn for pre-split datasets"
  - "M3GNet uses include_line_graph=True + threebody_cutoff=4.0; TensorNet uses include_line_graph=False"
  - "TensorNet derives element_types from all structures (train+val+test) for complete coverage"

patterns-established:
  - "partial(collate_fn_graph, include_line_graph=bool) pattern for matgl 2.x data loading"
  - "Separate training orchestrator per architecture following train_megnet.py template"

requirements-completed: [MODL-02]

duration: 3min
completed: 2026-03-08
---

# Phase 9 Plan 02: Training Orchestrators Summary

**M3GNet and TensorNet Lightning training orchestrators with matgl 2.x APIs, wired into pipeline CLI replacing MEGNet**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-08
- **Completed:** 2026-03-08
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- M3GNet training orchestrator with include_line_graph=True, threebody_cutoff, and pretrained fine-tuning
- TensorNet training orchestrator with include_line_graph=False and from-scratch model construction
- Pipeline CLI updated: megnet removed, m3gnet and tensornet added as model choices

## Task Commits

Each task was committed atomically:

1. **Task 1: Create M3GNet training orchestrator** - `ed5ae53` (feat)
2. **Task 2: Create TensorNet training orchestrator** - `7331915` (feat)
3. **Task 3: Update pipeline.py for m3gnet and tensornet** - `e15607f` (feat)

## Files Created/Modified
- `cathode_ml/models/train_m3gnet.py` - M3GNet Lightning training: _run_lightning_training, train_m3gnet_for_property, train_m3gnet
- `cathode_ml/models/train_tensornet.py` - TensorNet Lightning training: _run_lightning_training, train_tensornet_for_property, train_tensornet
- `cathode_ml/pipeline.py` - Replaced megnet with m3gnet+tensornet in --models choices and train stage

## Decisions Made
- Used torch.utils.data.DataLoader instead of MGLDataLoader since we have pre-split datasets (not using matgl's split_dataset)
- M3GNet uses threebody_cutoff=4.0 and include_line_graph=True for 3-body interactions
- TensorNet derives element_types from all structures (train+val+test) to ensure complete element coverage during model construction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Training orchestrators ready for evaluation/dashboard integration (Plan 09-03)
- Pipeline CLI fully updated with new model choices
- No blockers for next plan

---
*Phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x*
*Completed: 2026-03-08*
