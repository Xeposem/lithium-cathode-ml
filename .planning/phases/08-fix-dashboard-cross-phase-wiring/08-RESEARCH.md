# Phase 8: Fix Dashboard Cross-Phase Wiring - Research

**Researched:** 2026-03-07
**Domain:** Streamlit dashboard bug fixes (cross-phase integration wiring)
**Confidence:** HIGH

## Summary

Phase 8 is a targeted bug-fix phase closing 4 audit findings (4, 5, 6, 7) that prevent 6 dashboard pages from functioning correctly. All issues are well-characterized wiring defects where Phase 6 (Dashboard) code references APIs, file paths, or patterns that don't match what Phases 1-5 actually produced. No new features are needed -- only correcting mismatches between existing modules.

The fixes are small, surgical, and low-risk. Every bug has a clear root cause documented in the milestone audit, a specific file to change, and a deterministic verification path. The code that needs changing is entirely within the `dashboard/` directory (3 files) with no changes needed to `cathode_ml/` core modules.

**Primary recommendation:** Fix all 4 findings in a single plan since they are independent, small changes (total ~30 lines modified across 3 files) with no interdependencies.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DASH-01 | Web dashboard displays model comparison metrics in tables and charts | Finding 3 (baseline results path) was fixed in Phase 7. Overview and Model Comparison pages already work once baselines load. Verify after Phase 7 fix. |
| DASH-02 | Dashboard allows interactive prediction: user inputs composition/structure, gets predicted properties | Findings 4 (page guard), 5 (wrong import), 6 (checkpoint format) must all be fixed in predict.py and model_loader.py |
| DASH-03 | Dashboard includes data explorer: browse, filter, and visualize dataset distributions | Finding 7: get_cached_records() reads wrong path. Fix data_loader.py to read data/processed/materials.json |
| DASH-05 | Materials explorer: searchable database filterable by properties | Finding 7: Same path fix as DASH-03 -- materials_explorer.py calls get_cached_records() |
| DASH-06 | Materials discovery panel showing top candidates ranked by predicted properties | Finding 7: Same path fix -- discovery panel in materials_explorer.py depends on get_cached_records() |
| DASH-07 | Crystal structure 3D viewer using py3Dmol | Finding 4: crystal_viewer.py has __name__ guard. Remove guard, call main() at module level |
</phase_requirements>

## Standard Stack

### Core (already in project -- no new dependencies)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| streamlit | (pinned in requirements.txt) | Dashboard framework | Already installed |
| py3Dmol | (pinned) | 3D crystal structure rendering | Already installed |
| stmol | (pinned) | Streamlit-py3Dmol bridge | Already installed |
| plotly | (pinned) | Interactive charts | Already installed |
| torch | (pinned) | GNN model loading | Already installed |
| pymatgen | (pinned) | Crystal structure parsing | Already installed |

### No New Libraries Needed
This phase modifies existing code only. No new packages to install.

## Architecture Patterns

### Streamlit Multi-Page Pattern (already established)
The project uses `st.navigation()` with `st.Page()` in `dashboard/app.py`. Pages are loaded as modules. Streamlit executes the entire module file top-to-bottom on each page load.

**Critical pattern:** Pages that use `_render()` at module level work correctly (overview.py, model_comparison.py, data_explorer.py, materials_explorer.py). Pages that gate execution behind `if __name__ == "__main__"` do NOT work because Streamlit never sets `__name__` to `"__main__"` for page modules.

### Working page pattern (used by 4 of 6 pages):
```python
def _render() -> None:
    st.title("Page Title")
    # ... page content ...

_render()  # Called at module level -- Streamlit executes this
```

### Broken page pattern (predict.py, crystal_viewer.py):
```python
def main() -> None:
    st.title("Page Title")
    # ... page content ...

if __name__ == "__main__":  # NEVER true in Streamlit page context
    main()
```

### Fix pattern:
```python
def main() -> None:
    st.title("Page Title")
    # ... page content ...

main()  # Call at module level, matching other pages
```

### Anti-Patterns to Avoid
- **Don't rename main() to _render():** Unnecessary churn. Just remove the guard and call main() directly. Both patterns work.
- **Don't add try/except around the module-level call:** If the page errors, Streamlit should display the error, not silently swallow it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph conversion | Custom structure_to_pyg_data function | Existing `structure_to_graph(structure, config)` | The function already exists in graph.py with full PBC support |
| Data loading path | New data loading function | Fix path in existing `get_cached_records()` | The function logic is correct, only the path is wrong |

## Common Pitfalls

### Pitfall 1: Wrong function signature after import fix
**What goes wrong:** Changing `structure_to_pyg_data` to `structure_to_graph` but not updating the call signature. The old function took `(structure, cutoff, max_neighbors, gaussian_config)` as separate args; the new one takes `(structure, config)` where config is the full features config dict.
**Why it happens:** The function signature differs significantly between the nonexistent function name and the real one.
**How to avoid:** Read `structure_to_graph` signature carefully. It expects `config` dict with a `"graph"` section containing cutoff_radius, max_neighbors, gaussian params, and node_feature_dim.
**Warning signs:** KeyError on 'graph' or unexpected positional arguments.

### Pitfall 2: MEGNet state_dict key mismatch
**What goes wrong:** `model_loader.py` line 141 does `checkpoint["model_state_dict"]` but the training pipeline (train_megnet.py line 205) saves with `torch.save(best_state_dict, ...)` where `best_state_dict` is already the raw state_dict (from `get_megnet_state_dict(model)` which returns `model.model.state_dict()`).
**Why it happens:** CGCNN uses GNNTrainer.save_checkpoint() which wraps in `{"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...}`. MEGNet bypasses GNNTrainer and saves raw state_dict directly.
**How to avoid:** Make the loader handle both formats: try `checkpoint["model_state_dict"]` first, fall back to treating the whole checkpoint as a state_dict.
**Warning signs:** KeyError: 'model_state_dict' when loading MEGNet.

### Pitfall 3: Data path mismatch -- cache vs processed
**What goes wrong:** `get_cached_records()` reads from `data/cache/cleaned_records.json` (DataCache format with timestamp/metadata/data wrapper) but fetch.py saves to `data/processed/materials.json` (plain list of dicts).
**Why it happens:** Phase 1 originally used DataCache but was later changed to write processed JSON directly. Phase 6 dashboard code was written assuming the DataCache path.
**How to avoid:** Change default path to `data/processed/materials.json` and handle the plain-list format (no wrapper).
**Warning signs:** Empty data in Data Explorer and Materials Explorer.

### Pitfall 4: CGCNN checkpoint format vs MEGNet
**What goes wrong:** Assuming both GNN formats are the same. CGCNN checkpoints ARE wrapped with `{"model_state_dict": ...}` (from GNNTrainer). MEGNet checkpoints are NOT wrapped.
**How to avoid:** The fix should ONLY change MEGNet loading logic, not CGCNN. CGCNN loading already works correctly.

## Code Examples

### Fix 1: Remove __name__ guard in predict.py (line 196-197)
```python
# BEFORE (broken):
if __name__ == "__main__":
    main()

# AFTER (working):
main()
```

### Fix 2: Remove __name__ guard in crystal_viewer.py (line 102-103)
```python
# BEFORE (broken):
if __name__ == "__main__":
    main()

# AFTER (working):
main()
```

### Fix 3: Fix structure_to_graph import in model_loader.py (lines 247-259)
```python
# BEFORE (broken -- function doesn't exist):
from cathode_ml.features.graph import structure_to_pyg_data
# ... later:
data = structure_to_pyg_data(
    structure,
    cutoff=graph_cfg["cutoff_radius"],
    max_neighbors=graph_cfg["max_neighbors"],
    gaussian_config=graph_cfg["gaussian"],
)

# AFTER (correct):
from cathode_ml.features.graph import structure_to_graph
import yaml
with open(Path(configs_dir) / "features.yaml") as f:
    feat_cfg = yaml.safe_load(f)
data = structure_to_graph(structure, feat_cfg)
```

### Fix 4: Handle raw state_dict for MEGNet in model_loader.py (lines 131-141)
```python
# BEFORE (broken -- expects wrapper dict):
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.model.load_state_dict(checkpoint["model_state_dict"])

# AFTER (handles both formats):
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# MEGNet saves raw state_dict; CGCNN uses {"model_state_dict": ...} wrapper
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint  # Raw state_dict (MEGNet format)
model.model.load_state_dict(state_dict)
```

### Fix 5: Update get_cached_records() path in data_loader.py (lines 56-84)
```python
# BEFORE (broken path and format assumption):
@st.cache_data
def get_cached_records(cache_dir: str = "data/cache") -> list[dict]:
    cache_path = Path(cache_dir) / "cleaned_records.json"
    # ... reads DataCache wrapper format

# AFTER (correct path and plain JSON list):
@st.cache_data
def get_cached_records(data_dir: str = "data/processed") -> list[dict]:
    data_path = Path(data_dir) / "materials.json"
    if not data_path.exists():
        logger.warning("Processed data not found: %s", data_path)
        return []
    try:
        with open(data_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Handle legacy DataCache wrapper format
        return data.get("data", [data])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load processed data: %s", exc)
        return []
```

## State of the Art

No technology changes needed. All libraries are already in use and stable.

| Finding | Root Cause | Fix Complexity |
|---------|-----------|---------------|
| 4 (page guards) | `__name__` guard incompatible with Streamlit page loading | 2 lines changed, 2 files |
| 5 (wrong import) | Nonexistent function name, wrong signature | ~10 lines changed, 1 file |
| 6 (checkpoint format) | MEGNet saves raw state_dict, loader expects wrapper | ~5 lines changed, 1 file |
| 7 (data path) | Dashboard reads cache path, fetch writes to processed path | ~10 lines changed, 1 file |

## Open Questions

None. All issues are fully characterized by the milestone audit with clear root causes and fixes.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml or pytest defaults |
| Quick run command | `python -m pytest tests/test_dashboard.py tests/test_dashboard_predict.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DASH-01 | Model comparison displays metrics | unit (existing) | `python -m pytest tests/test_dashboard.py::TestGetAllResults -x` | Yes |
| DASH-02 | Predict page renders and runs predictions | unit + import check | `python -m pytest tests/test_dashboard_predict.py -x` | Yes (needs update) |
| DASH-03 | Data explorer loads and displays data | unit | `python -m pytest tests/test_dashboard.py::test_data_explorer_load -x` | Yes (needs path update) |
| DASH-05 | Materials explorer filters work | unit | `python -m pytest tests/test_dashboard.py::test_materials_filter_voltage -x` | Yes |
| DASH-06 | Discovery ranking works | unit | `python -m pytest tests/test_dashboard.py::test_discovery_ranking -x` | Yes |
| DASH-07 | Crystal viewer renders | smoke (import check) | `python -c "from dashboard.pages.crystal_viewer import main"` | No test file |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_dashboard.py tests/test_dashboard_predict.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_dashboard.py::TestGetCachedRecords` -- update test to use `data/processed/materials.json` path (not `data/cache/cleaned_records.json`)
- [ ] `tests/test_dashboard_predict.py` -- add test for structure_to_graph import (not structure_to_pyg_data)
- [ ] `tests/test_dashboard_predict.py` -- add test for MEGNet raw state_dict loading
- [ ] Smoke test: verify predict.py and crystal_viewer.py can be imported and `main` is called at module level

## Sources

### Primary (HIGH confidence)
- **Project source code** -- direct reading of all dashboard files, model training files, and data pipeline files
- **Milestone audit** (`v1.0-MILESTONE-AUDIT.md`) -- documented all 7 findings with root causes and fixes
- **Phase 7 completion** -- confirmed findings 1, 2, 3 are already fixed in pipeline.py

### Secondary (MEDIUM confidence)
- Streamlit multi-page app documentation -- `st.Page()` executes module top-to-bottom, `__name__` is never `"__main__"`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all existing code
- Architecture: HIGH -- patterns already established by working pages
- Pitfalls: HIGH -- every issue has a documented root cause from the audit

**Research date:** 2026-03-07
**Valid until:** Indefinite (bug fixes on stable codebase)
