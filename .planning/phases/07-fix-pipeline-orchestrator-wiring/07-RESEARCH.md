# Phase 7: Fix Pipeline Orchestrator Wiring - Research

**Researched:** 2026-03-07
**Domain:** Python CLI pipeline integration / config loading / file path consistency
**Confidence:** HIGH

## Summary

Phase 7 addresses three critical cross-phase integration defects identified in the v1.0 milestone audit (findings 1, 2, 3). All three are straightforward wiring bugs where the pipeline orchestrator (`pipeline.py`) uses different paths or config loading strategies than the standalone CLIs that were tested independently. The standalone CLIs work correctly; the unified orchestrator does not.

The fixes are surgical -- each involves changing a small number of lines in one or two files with no architectural changes. The risk is low because the correct patterns already exist in the standalone CLIs (`train_cgcnn.py __main__`, `train_megnet.py __main__`) and the evaluation loader (`metrics.py`). The task is to make `pipeline.py` and `baselines.py` match those patterns.

**Primary recommendation:** Fix all three findings in a single plan with clear before/after for each file. No new libraries or architectural changes needed.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | System evaluates all models with MAE, RMSE, R-squared metrics | Finding 3 fix ensures baseline results are found by `load_all_results`, making evaluation complete across all 4 models |
| EVAL-02 | System uses consistent cross-validation folds across all models | Finding 1 fix ensures pipeline loads `features.yaml` with splitting config, producing identical folds as standalone CLIs |
| EVAL-03 | System generates parity plots for each model and property | Finding 3 fix enables unified result loading needed for plot generation |
| DATA-04 | System caches downloaded data locally to avoid repeated API calls | Finding 2 fix ensures pipeline reads from `data/processed/materials.json` (the cached/processed output of fetch stage), not a nonexistent DataCache key |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyYAML | (already installed) | Config loading | Already used by `config.py` |
| pathlib | stdlib | Path manipulation | Already used throughout |
| json | stdlib | Record deserialization | Already used by standalone CLIs |

### Supporting
No new libraries needed. All changes use existing imports.

## Architecture Patterns

### Current Architecture (Working)
The standalone CLIs follow a correct pattern:
```
configs/features.yaml  --> load_config("configs/features.yaml")
configs/baselines.yaml --> load_config("configs/baselines.yaml")
configs/cgcnn.yaml     --> load_config("configs/cgcnn.yaml")
configs/megnet.yaml    --> load_config("configs/megnet.yaml")
data/processed/materials.json --> json.load() + MaterialRecord(**r)
```

### Current Architecture (Broken -- pipeline.py)
The orchestrator uses a different, incorrect pattern:
```
configs/data.yaml --> load_config("configs/data.yaml")
                  --> config.get("features", {})     # empty dict -- not in data.yaml
                  --> config.get("baselines", {})     # empty dict -- not in data.yaml
                  --> config.get("cgcnn", {})          # empty dict -- not in data.yaml
                  --> config.get("megnet", {})         # empty dict -- not in data.yaml
data/cache/cleaned_records.json --> DataCache.load()  # wrong path
```

### Target Architecture (After Fix)
Pipeline.py should match standalone CLIs exactly:
```python
# In run_train_stage():
features_config = load_config(str(Path(args.config_dir) / "features.yaml"))
baselines_config = load_config(str(Path(args.config_dir) / "baselines.yaml"))
cgcnn_config = load_config(str(Path(args.config_dir) / "cgcnn.yaml"))
megnet_config = load_config(str(Path(args.config_dir) / "megnet.yaml"))

# Data loading -- match standalone CLI pattern:
processed_path = Path("data/processed/materials.json")
with open(processed_path) as f:
    raw_records = json.load(f)
records = [MaterialRecord(**r) for r in raw_records]
```

### Anti-Patterns to Avoid
- **Extracting sub-configs from a monolithic YAML:** Each model has its own YAML file. Do not merge them into data.yaml or extract nonexistent keys.
- **Using DataCache for processed data:** DataCache is for raw API responses. Processed data lives at `data/processed/materials.json` as a plain JSON file.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config loading | Custom section extraction from data.yaml | `load_config()` with separate YAML paths | Each YAML is self-contained; data.yaml has no model/feature sections |
| Record deserialization | DataCache lookup | `json.load()` + `MaterialRecord(**r)` | Matches standalone CLIs; DataCache key "cleaned_records" was never saved |

## Common Pitfalls

### Pitfall 1: Baseline Results Path Inconsistency
**What goes wrong:** `baselines.py` saves JSON to `data/results/baseline_results.json` but `metrics.py` loads from `data/results/baselines/baseline_results.json`.
**Why it happens:** `baselines.py` line 196 uses `Path(results_dir) / "baseline_results.json"` where `results_dir = "data/results"`. But `metrics.py` line 62 expects `base / "baselines" / "baseline_results.json"` where `base = "data/results"`.
**How to avoid:** Fix `baselines.py` to save to `data/results/baselines/baseline_results.json` (matching what the loader expects). The directory already exists because joblib models are saved there.
**Warning signs:** `load_all_results` returns no RF/XGBoost data; comparison tables show only CGCNN and MEGNet.

### Pitfall 2: Forgetting to Import MaterialRecord and json
**What goes wrong:** pipeline.py currently only imports DataCache; the new pattern needs json and MaterialRecord.
**Why it happens:** Copy-paste oversight when changing data loading approach.
**How to avoid:** Add imports inside the lazy-import block of `run_train_stage`.

### Pitfall 3: Config Dir Resolution
**What goes wrong:** `args.config_dir` defaults to `"configs"` (relative). If the working directory is not the project root, paths break.
**Why it happens:** Relative path assumption.
**How to avoid:** This is an existing pattern used by all standalone CLIs. Maintain consistency -- do not change to absolute paths. Document that pipeline must be run from project root (same as existing CLIs).

### Pitfall 4: Breaking Existing Tests
**What goes wrong:** test_pipeline.py mocks stage functions at the module level. Changing function signatures or internal imports could break mocks.
**Why it happens:** Tests mock `run_train_stage` etc. wholesale; they don't test internal config loading.
**How to avoid:** Keep the same function signatures. Add new unit tests that verify the internal loading logic with tmp_path fixtures.

## Code Examples

### Fix 1: Config Loading (pipeline.py run_train_stage)
```python
# BEFORE (broken):
def run_train_stage(args: argparse.Namespace) -> None:
    from cathode_ml.config import load_config
    from cathode_ml.data.cache import DataCache

    config = load_config(str(Path(args.config_dir) / "data.yaml"))
    cache = DataCache(cache_dir="data/cache")
    records = cache.load("cleaned_records")
    features_config = config.get("features", {})
    # ... config.get("baselines", {}), config.get("cgcnn", {}), etc.

# AFTER (fixed):
def run_train_stage(args: argparse.Namespace) -> None:
    import json
    from cathode_ml.config import load_config
    from cathode_ml.data.schemas import MaterialRecord

    config_dir = Path(args.config_dir)
    features_config = load_config(str(config_dir / "features.yaml"))

    # Load processed records (matching standalone CLI pattern)
    processed_path = Path("data/processed/materials.json")
    with open(processed_path) as f:
        raw_records = json.load(f)
    records = [MaterialRecord(**r) for r in raw_records]

    # Baseline models
    baseline_models = [m for m in args.models if m in ("rf", "xgb")]
    if baseline_models:
        from cathode_ml.models.baselines import run_baselines
        baselines_config = load_config(str(config_dir / "baselines.yaml"))
        run_baselines(records, features_config, baselines_config, seed=args.seed)

    # CGCNN
    if "cgcnn" in args.models:
        from cathode_ml.models.train_cgcnn import train_cgcnn
        cgcnn_config = load_config(str(config_dir / "cgcnn.yaml"))
        train_cgcnn(records=records, features_config=features_config,
                    cgcnn_config=cgcnn_config, seed=args.seed)

    # MEGNet
    if "megnet" in args.models:
        from cathode_ml.models.train_megnet import train_megnet
        megnet_config = load_config(str(config_dir / "megnet.yaml"))
        train_megnet(records=records, features_config=features_config,
                     megnet_config=megnet_config, seed=args.seed)
```

### Fix 2: Baseline Results Path (baselines.py)
```python
# BEFORE (broken -- saves to wrong path):
results_path = str(Path(results_dir) / "baseline_results.json")
# results_dir = "data/results" --> saves to data/results/baseline_results.json

# AFTER (fixed -- saves where loader expects):
results_path = str(Path(results_dir) / "baselines" / "baseline_results.json")
# results_dir = "data/results" --> saves to data/results/baselines/baseline_results.json
```

Note: The `baselines_dir` (for joblib files) is already created at `Path(results_dir) / "baselines"`, so the parent directory exists.

### Alternate Fix 2 (change loader instead of saver):
NOT recommended. The loader in `metrics.py` uses a consistent pattern (`base / model_name / model_results.json`) matching CGCNN (`base / "cgcnn" / "cgcnn_results.json"`) and MEGNet (`base / "megnet" / "megnet_results.json"`). Changing the saver maintains consistency.

## State of the Art

No technology changes needed. This is pure bug-fixing of existing code.

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Monolithic data.yaml with all sections | Separate YAML per concern | Phase 2-4 (configs created) | pipeline.py was never updated to match |

## Open Questions

1. **Should baselines.yaml results_dir default be updated?**
   - What we know: `baselines.yaml` has `results_dir: "data/results"`, while cgcnn.yaml has `results_dir: "data/results/cgcnn"` and megnet.yaml has `results_dir: "data/results/megnet"`.
   - What's unclear: Whether to change baselines.yaml to `results_dir: "data/results/baselines"` or keep fixing it in code.
   - Recommendation: Keep baselines.yaml as `results_dir: "data/results"` since it contains both the baselines subdirectory and overall results. Fix only the JSON save path in code to use the `baselines/` subdirectory. This matches how the code already creates `baselines_dir = Path(results_dir) / "baselines"` for joblib files.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | none (default discovery) |
| Quick run command | `python -m pytest tests/test_pipeline.py tests/test_evaluation.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | All models included in evaluation | unit | `python -m pytest tests/test_evaluation.py::TestLoadAllResults -x` | Yes (existing) |
| EVAL-02 | Consistent folds via features.yaml | unit | `python -m pytest tests/test_pipeline.py::TestRunTrainStage -x` | Needs new test class |
| EVAL-03 | Parity plots generated | unit | `python -m pytest tests/test_evaluation.py -x` | Yes (existing covers load) |
| DATA-04 | Pipeline reads processed data | unit | `python -m pytest tests/test_pipeline.py::TestRunTrainStage::test_loads_processed_records -x` | Needs new test |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_pipeline.py tests/test_evaluation.py tests/test_baselines.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline.py::TestRunTrainStage` -- new test class verifying config loading uses separate YAML files
- [ ] `tests/test_pipeline.py::TestRunTrainStage::test_loads_processed_records` -- verifies data loaded from `data/processed/materials.json`
- [ ] `tests/test_baselines.py` -- verify results JSON saved to `baselines/baseline_results.json` path (may need new test or update existing)

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `cathode_ml/pipeline.py` (lines 92-135) -- config loading and data handoff
- Direct code inspection of `cathode_ml/models/baselines.py` (line 196) -- results save path
- Direct code inspection of `cathode_ml/evaluation/metrics.py` (line 62) -- results load path
- Direct code inspection of `cathode_ml/models/train_cgcnn.py` (lines 256-284) -- standalone CLI correct pattern
- Direct code inspection of `cathode_ml/models/train_megnet.py` (lines 406-446) -- standalone CLI correct pattern
- Milestone audit: `.planning/v1.0-MILESTONE-AUDIT.md` -- findings 1, 2, 3

### Secondary (MEDIUM confidence)
- None needed -- all findings verified by code inspection

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new libraries, pure code fixes
- Architecture: HIGH - correct patterns already exist in standalone CLIs; just replicate in pipeline.py
- Pitfalls: HIGH - exact line numbers and paths identified by code inspection

**Research date:** 2026-03-07
**Valid until:** indefinite (code fixes, not library-dependent)
