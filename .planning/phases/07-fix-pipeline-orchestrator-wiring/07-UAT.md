---
status: complete
phase: 07-fix-pipeline-orchestrator-wiring
source: 07-01-SUMMARY.md
started: 2026-03-07T11:00:00Z
updated: 2026-03-07T11:05:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Pipeline loads separate config files
expected: Run `python -c "from cathode_ml.pipeline import run_train_stage; import argparse; print('import ok')"` — no import errors. Then inspect `cathode_ml/pipeline.py` `run_train_stage` function: it should call `load_config()` for `features.yaml`, `baselines.yaml`, `cgcnn.yaml`, and `megnet.yaml` separately. There should be NO reference to `data.yaml` config extraction (no `config.get("features")` etc.).
result: pass

### 2. Pipeline reads processed data from correct path
expected: In `cathode_ml/pipeline.py`, the `run_train_stage` function reads records via `json.load()` from `data/processed/materials.json` and constructs `MaterialRecord` objects. There should be NO reference to `DataCache` or `cache.load("cleaned_records")`.
result: pass

### 3. Baseline results saved to correct path
expected: In `cathode_ml/models/baselines.py`, the results JSON is saved to `{results_dir}/baselines/baseline_results.json` (not `{results_dir}/baseline_results.json`). This matches the load path in `cathode_ml/evaluation/metrics.py` which expects `base / "baselines" / "baseline_results.json"`.
result: pass

### 4. Test suite passes
expected: Run `python -m pytest tests/test_pipeline.py tests/test_baselines.py tests/test_evaluation.py -x -q` — all tests pass with 0 failures. The new `TestRunTrainStage` tests and `test_results_saved_to_baselines_subdir` should be included.
result: pass

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0

## Gaps

[none]
