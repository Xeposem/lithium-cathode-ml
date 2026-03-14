---
phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
plan: 01
subsystem: models
tags: [m3gnet, tensornet, matgl, gnn, equivariant]

requires:
  - phase: 04-megnet-implementation
    provides: MEGNet wrapper pattern (lazy imports, predict_structure API)
provides:
  - M3GNet model wrapper with pretrained loading and prediction
  - TensorNet model wrapper with config-driven construction and prediction
  - YAML configs for both architectures
  - matgl 2.x dependency pins
affects: [09-02-training-orchestrators, 09-03-evaluation-dashboard, 09-04-tests]

tech-stack:
  added: [matgl>=2.0.0, lightning>=2.0.0]
  patterns: [config-driven-model-construction, lazy-matgl-imports]

key-files:
  created:
    - cathode_ml/models/m3gnet.py
    - cathode_ml/models/tensornet.py
    - configs/m3gnet.yaml
    - configs/tensornet.yaml
  modified:
    - requirements.txt

key-decisions:
  - "M3GNet default model is M3GNet-MP-2018.6.1-Eform (formation energy, same domain as MEGNet)"
  - "TensorNet uses config-driven build pattern matching CGCNN's build_cgcnn_from_config"
  - "M3GNet config uses CosineAnnealingLR params (decay_steps, decay_alpha) matching matgl 2.x defaults"
  - "lightning>=2.0.0 added as explicit dependency (matgl 2.x uses import lightning as L)"

patterns-established:
  - "build_X_from_config pattern for models without pretrained weights (TensorNet)"
  - "Consistent predict_with_X API across all matgl model wrappers"

requirements-completed: [MODL-02]

duration: 2min
completed: 2026-03-09
---

# Phase 9 Plan 01: Core Model Wrappers Summary

**M3GNet and TensorNet model wrappers with lazy matgl 2.x imports, YAML configs, and updated dependency pins**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-09T03:27:36Z
- **Completed:** 2026-03-09T03:29:28Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments
- M3GNet wrapper with load/list/predict/state_dict matching MEGNet pattern exactly
- TensorNet wrapper with config-driven construction for from-scratch O(3)-equivariant training
- YAML configs with appropriate training hyperparameters for both architectures
- requirements.txt updated to matgl>=2.0.0 with lightning>=2.0.0

## Task Commits

Each task was committed atomically:

1. **Task 1: Create M3GNet model wrapper** - `bb4426b` (feat)
2. **Task 2: Create TensorNet model wrapper** - `4c889c5` (feat)
3. **Task 3: Create M3GNet and TensorNet YAML configs** - `631af38` (feat)
4. **Task 4: Update requirements.txt for matgl 2.x** - `0efc790` (chore)

## Files Created/Modified
- `cathode_ml/models/m3gnet.py` - M3GNet wrapper: load_m3gnet_model, get_available_m3gnet_models, get_m3gnet_state_dict, predict_with_m3gnet
- `cathode_ml/models/tensornet.py` - TensorNet wrapper: build_tensornet_from_config, get_tensornet_state_dict, predict_with_tensornet
- `configs/m3gnet.yaml` - M3GNet config with pretrained model reference and CosineAnnealingLR params
- `configs/tensornet.yaml` - TensorNet config with full architecture params for from-scratch construction
- `requirements.txt` - matgl>=2.0.0, lightning>=2.0.0 added

## Decisions Made
- M3GNet default model is M3GNet-MP-2018.6.1-Eform (formation energy, matching MEGNet's domain)
- TensorNet uses config-driven build pattern (no pretrained models exist for property prediction)
- M3GNet config uses CosineAnnealingLR params (decay_steps=1000, decay_alpha=0.01) matching matgl 2.x defaults
- lightning>=2.0.0 added as explicit dependency since matgl 2.x uses `import lightning as L`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Model wrappers ready for training orchestrators (Plan 09-02)
- YAML configs ready for pipeline integration
- No blockers for next plan

---
*Phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x*
*Completed: 2026-03-09*
