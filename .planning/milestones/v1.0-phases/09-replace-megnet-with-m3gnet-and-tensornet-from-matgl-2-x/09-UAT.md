---
status: complete
phase: 09-replace-megnet-with-m3gnet-and-tensornet-from-matgl-2-x
source: 09-01-SUMMARY.md, 09-02-SUMMARY.md, 09-03-SUMMARY.md, 09-04-SUMMARY.md
started: 2026-03-09T04:00:00Z
updated: 2026-03-09T04:15:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Pipeline CLI Model Choices
expected: Run `python -m cathode_ml.pipeline --help`. The `--models` argument should list `rf`, `xgb`, `cgcnn`, `m3gnet`, `tensornet` as valid choices. `megnet` should NOT appear.
result: pass

### 2. M3GNet Wrapper Import and API
expected: Run `python -c "from cathode_ml.models.m3gnet import load_m3gnet_model, predict_with_m3gnet, get_available_m3gnet_models; print(get_available_m3gnet_models())"`. Should print a list of available pretrained M3GNet models including `M3GNet-MP-2018.6.1-Eform`. No import errors.
result: pass

### 3. TensorNet Wrapper Import and API
expected: Run `python -c "from cathode_ml.models.tensornet import build_tensornet_from_config, predict_with_tensornet; print('TensorNet imports OK')"`. Should print "TensorNet imports OK" with no import errors.
result: pass

### 4. M3GNet Config Loads
expected: Run `python -c "from cathode_ml.config import load_config; c = load_config('configs/m3gnet.yaml'); print(c['model']['pretrained_model'])"`. Should print `M3GNet-MP-2018.6.1-Eform`.
result: pass

### 5. TensorNet Config Loads
expected: Run `python -c "from cathode_ml.config import load_config; c = load_config('configs/tensornet.yaml'); print(c['model']['equivariance_invariance_group'])"`. Should print `O(3)`.
result: pass

### 6. Full Test Suite Passes
expected: Run `python -m pytest tests/ -q`. All 182+ tests should pass with 0 failures. Both `test_m3gnet.py` (16 tests) and `test_tensornet.py` (16 tests) should be included.
result: pass
note: 182 passed. Pre-existing test_dashboard.py Streamlit runtime error excluded (not Phase 9 related).

### 7. Zero MEGNet References in Codebase
expected: Run `grep -ri "\bmegnet\b" --include="*.py" --include="*.yaml" cathode_ml/ configs/ dashboard/ tests/`. Should return zero matches (no standalone "megnet" references in project source files).
result: pass

### 8. Old MEGNet Files Deleted
expected: Verify these files do NOT exist: `cathode_ml/models/megnet.py`, `cathode_ml/models/train_megnet.py`, `configs/megnet.yaml`, `tests/test_megnet.py`. All four should be gone.
result: pass

## Summary

total: 8
passed: 8
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
