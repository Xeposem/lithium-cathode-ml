# Phase 6: Dashboard and Documentation - Research

**Researched:** 2026-03-07
**Domain:** Streamlit multi-page dashboard, Plotly charting, py3Dmol crystal visualization, README documentation
**Confidence:** HIGH

## Summary

Phase 6 builds a Streamlit multi-page dashboard with 6 pages (Overview, Data Explorer, Model Comparison, Materials Explorer, Predict, Crystal Viewer) and a comprehensive README. The tech stack is well-established: Streamlit 1.55+ for the app framework, Plotly for interactive charts (via `st.plotly_chart`), and stmol/py3Dmol for 3D crystal rendering. All data sources already exist in the project (`data/results/`, `data/cache/`, model checkpoints).

A critical discovery is that **baseline models (RF, XGBoost) are NOT persisted as model files** -- only their metric results are saved as JSON. GNN models (CGCNN, MEGNet) have `.pt` checkpoints saved by `GNNTrainer.save_checkpoint()`. For the Predict page to offer RF/XGBoost predictions from a user-provided composition, we must add joblib serialization to the baseline training pipeline, or retrain on-the-fly (impractical for dashboard UX).

The project already has `evaluation/metrics.py` with `load_all_results()`, `MODEL_COLORS`, `MODEL_LABELS`, `MODELS_ORDER`, and `PROPERTIES` constants -- these are directly reusable for dashboard data loading and consistent styling across all Plotly charts.

**Primary recommendation:** Use `st.Page`/`st.navigation` API (preferred over pages/ directory) for maximum flexibility, add joblib model persistence for baselines, use `@st.cache_data` for data loading and `@st.cache_resource` for model loading.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Multi-page Streamlit app with sidebar navigation
- 6 pages: Overview, Data Explorer, Model Comparison, Materials Explorer, Predict, Crystal Viewer
- Overview/landing page is a metrics dashboard: model comparison table (reusing metrics.py data), bar chart of MAE across models, key findings summary
- Materials Explorer and Discovery Panel combined on one page: searchable/filterable table (DASH-05) on top, "Top Candidates" panel (DASH-06) below
- Training curves (DASH-04) live on the Model Comparison page as a section alongside metrics table and bar chart
- Two input modes on Predict page: composition string for quick baseline predictions, plus optional CIF file upload for full GNN predictions
- All available models run predictions: RF, XGBoost from composition; CGCNN, MEGNet additionally if CIF provided
- Results displayed as property cards -- one card per predicted property showing values with units
- CIF uploads render inline 3D crystal structure preview (py3Dmol) before predictions
- Plotly for all dashboard charts via st.plotly_chart()
- Matplotlib PNGs from evaluation/plots.py still generated separately for README/paper figures
- Wong colorblind-safe palette: RF=#0072B2, XGB=#D55E00, CGCNN=#009E73, MEGNet=#CC79A7
- Parity plots: interactive Plotly in dashboard (hover to see material ID), static matplotlib PNGs for export
- Data Explorer: histograms for each property distribution, plus interactive scatter matrix
- Academic portfolio tone for README -- written for ML researchers and hiring managers
- Embedded key figures: bar comparison chart, one parity plot, dashboard screenshot (2-3 figures total)
- Methodology section: architecture summaries + key design choices (2-3 paragraphs)
- Results summary table: best model + MAE/R-squared per property
- README sections: Introduction/Motivation, Data Sources, Methodology, Results, Dashboard, How to Run/Reproduce

### Claude's Discretion
- Exact Streamlit page file naming and directory structure (pages/ convention)
- Plotly chart styling details (axis formatting, legend placement)
- Crystal viewer integration specifics (py3Dmol vs alternative if compatibility issues)
- README section ordering and transition prose
- Dashboard theme/styling (Streamlit config)
- Materials Explorer filter widget design (sliders, dropdowns, multiselect)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DASH-01 | Web dashboard displays model comparison metrics in tables and charts | `load_all_results()` provides unified data; Plotly grouped bar + `st.dataframe` for tables |
| DASH-02 | Dashboard allows interactive prediction: user inputs composition/structure, gets predicted properties | Requires joblib persistence for baselines + torch checkpoint loading for GNNs; `featurize_compositions()` for composition input |
| DASH-03 | Dashboard includes data explorer: browse, filter, visualize dataset distributions | `DataCache.load("cleaned_records")` provides raw data; Plotly histograms + scatter matrix |
| DASH-04 | Dashboard displays training curves per model | CSV metrics files at `data/results/{model}/{property}_metrics.csv`; Plotly line charts |
| DASH-05 | Materials explorer: searchable database filterable by voltage, formation energy, capacity, elements, stability | Same cache data; Streamlit widgets (sliders, multiselect) for filtering |
| DASH-06 | Materials discovery panel showing top candidate materials | Sort/rank cached materials by predicted properties; display top-N table |
| DASH-07 | Crystal structure 3D viewer using py3Dmol | stmol + py3Dmol; pymatgen Structure.to(fmt="cif") for conversion |
| DOCS-01 | README includes project introduction and motivation | Academic portfolio tone; lithium cathode prediction context |
| DOCS-02 | README includes methodology section | Architecture summaries for RF, XGBoost, CGCNN, MEGNet; compositional splitting |
| DOCS-03 | README includes pipeline implementation details | Document `pipeline.py` stages, CLI flags, config system |
| DOCS-04 | README includes results summary with key findings | Embed bar chart PNG, parity plot, results summary table |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | >=1.55.0 | Multi-page web app framework | De facto standard for Python data apps; chosen in project decisions |
| plotly | >=5.18.0 | Interactive charts (bar, scatter, line, histogram) | st.plotly_chart() integration; hover/zoom/pan for data exploration |
| stmol | >=0.0.9 | Streamlit component wrapping py3Dmol | Only maintained Streamlit crystal/molecular 3D viewer |
| py3Dmol | >=2.1.0 | WebGL-based 3D molecular/crystal viewer | Standard for CIF/crystal visualization; renders in browser |
| joblib | >=1.3.0 | Serialize fitted sklearn/xgboost models | Standard sklearn model persistence; already a transitive dep of scikit-learn |

### Supporting (already in project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pymatgen | ==2025.10.7 | Structure.from_str() for CIF parsing, Structure.to() for CIF output | Crystal viewer + Predict page CIF upload |
| pandas | ==2.2.3 | DataFrame for table display and filtering | Data Explorer, Materials Explorer |
| numpy | ==1.26.4 | Array operations for predictions | Predict page inference |
| matplotlib | (existing) | Static figure generation for README | README embedded figures only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| stmol | crystal_toolkit | crystal_toolkit requires Dash, not Streamlit-compatible |
| stmol | streamlit_3dmol | Less maintained fork; stmol is the published/cited package |
| Plotly | Altair/Vega-Lite | Plotly has better 3D support and richer hover; locked decision anyway |

**Installation:**
```bash
pip install streamlit>=1.55.0 plotly>=5.18.0 stmol>=0.0.9 py3Dmol>=2.1.0
```

Note: `joblib` is already installed as a dependency of scikit-learn. No separate install needed.

## Architecture Patterns

### Recommended Project Structure
```
dashboard/
    app.py              # Entrypoint: st.navigation + st.Page definitions
    pages/
        overview.py     # DASH-01: metrics dashboard landing
        data_explorer.py    # DASH-03: dataset browser + distributions
        model_comparison.py # DASH-01 + DASH-04: metrics tables + training curves
        materials_explorer.py  # DASH-05 + DASH-06: filterable table + discovery panel
        predict.py      # DASH-02: composition/CIF input -> predictions
        crystal_viewer.py   # DASH-07: 3D structure viewer
    utils/
        __init__.py
        data_loader.py  # Cached data loading functions
        model_loader.py # Model loading (joblib for baselines, torch for GNNs)
        charts.py       # Plotly chart factory functions with consistent styling
```

### Pattern 1: st.Page + st.navigation Entrypoint
**What:** Use the modern Streamlit API for multi-page apps (preferred over pages/ directory convention).
**When to use:** Always -- this is the recommended approach since Streamlit 1.36+.
**Example:**
```python
# dashboard/app.py
import streamlit as st

st.set_page_config(
    page_title="Cathode ML Dashboard",
    page_icon=":material/battery_charging_full:",
    layout="wide",
)

overview = st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True)
data_explorer = st.Page("pages/data_explorer.py", title="Data Explorer", icon=":material/bar_chart:")
model_comparison = st.Page("pages/model_comparison.py", title="Model Comparison", icon=":material/compare:")
materials_explorer = st.Page("pages/materials_explorer.py", title="Materials Explorer", icon=":material/science:")
predict = st.Page("pages/predict.py", title="Predict", icon=":material/psychology:")
crystal_viewer = st.Page("pages/crystal_viewer.py", title="Crystal Viewer", icon=":material/view_in_ar:")

pg = st.navigation({
    "Results": [overview, model_comparison],
    "Explore": [data_explorer, materials_explorer],
    "Tools": [predict, crystal_viewer],
})
pg.run()
```

### Pattern 2: Cached Data Loading
**What:** Use `@st.cache_data` for data (JSON, CSV, DataFrames) and `@st.cache_resource` for models.
**When to use:** Every page that loads data or models -- prevents reloading on every Streamlit rerun.
**Example:**
```python
# dashboard/utils/data_loader.py
import streamlit as st
from cathode_ml.evaluation.metrics import load_all_results, PROPERTIES, MODELS_ORDER

@st.cache_data
def get_all_results(results_base: str = "data/results") -> dict:
    """Load unified model results with caching."""
    return load_all_results(results_base)

@st.cache_data
def get_cached_records() -> list[dict]:
    """Load cleaned material records from data cache."""
    from cathode_ml.data.cache import DataCache
    cache = DataCache("data/cache")
    return cache.load("cleaned_records")
```

### Pattern 3: Consistent Plotly Color Mapping
**What:** Reuse `MODEL_COLORS` from metrics.py for all Plotly charts.
**When to use:** Every chart that shows model-keyed data.
**Example:**
```python
import plotly.graph_objects as go
from cathode_ml.evaluation.metrics import MODEL_COLORS, MODEL_LABELS, MODELS_ORDER

def make_bar_comparison(all_results: dict, property_names: list[str]) -> go.Figure:
    fig = go.Figure()
    for model_key in MODELS_ORDER:
        if model_key not in MODEL_COLORS:
            continue
        mae_values = [
            all_results.get(p, {}).get(model_key, {}).get("mae", 0)
            for p in property_names
        ]
        fig.add_trace(go.Bar(
            name=MODEL_LABELS[model_key],
            x=property_names,
            y=mae_values,
            marker_color=MODEL_COLORS[model_key],
        ))
    fig.update_layout(barmode="group", yaxis_title="MAE")
    return fig
```

### Pattern 4: CIF Upload + py3Dmol Rendering
**What:** Parse uploaded CIF with pymatgen, render with stmol/py3Dmol.
**When to use:** Predict page (CIF upload preview) and Crystal Viewer page.
**Example:**
```python
import py3Dmol
from stmol import showmol
from pymatgen.core import Structure

def render_structure(structure: Structure, width: int = 600, height: int = 400):
    """Render a pymatgen Structure as interactive 3D viewer."""
    cif_str = structure.to(fmt="cif")
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(cif_str, "cif")
    viewer.setStyle({"sphere": {"radius": 0.4}, "stick": {"radius": 0.15}})
    viewer.zoomTo()
    showmol(viewer, height=height, width=width)

# In Streamlit page:
uploaded = st.file_uploader("Upload CIF file", type=["cif"])
if uploaded:
    cif_content = uploaded.read().decode("utf-8")
    structure = Structure.from_str(cif_content, fmt="cif")
    render_structure(structure)
```

### Pattern 5: Model Loading for Predictions
**What:** Load persisted models for inference on user input.
**When to use:** Predict page.
**Example:**
```python
import streamlit as st
import joblib
import torch

@st.cache_resource
def load_baseline_model(model_type: str, property_name: str):
    """Load a persisted sklearn/xgboost model."""
    path = f"data/results/baselines/{model_type}_{property_name}.joblib"
    return joblib.load(path)

@st.cache_resource
def load_gnn_model(model_name: str, property_name: str):
    """Load a GNN checkpoint and return model in eval mode."""
    path = f"data/results/{model_name}/{model_name}_{property_name}_best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Reconstruct model architecture, load state_dict
    # ...
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
```

### Anti-Patterns to Avoid
- **Loading data inside widget callbacks:** Data should be cached at module level with `@st.cache_data`, not reloaded in button handlers.
- **Storing large DataFrames in session_state:** Use `@st.cache_data` instead -- session_state is per-session, cache_data is shared across sessions.
- **Importing heavy ML libraries at page top level:** Use lazy imports inside prediction functions to keep non-prediction pages fast.
- **Using `use_container_width=True` in st.plotly_chart:** This parameter is deprecated as of late 2025. Use `width="stretch"` instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 3D crystal rendering | Custom WebGL/Three.js viewer | stmol + py3Dmol | Handles CIF parsing, atom coloring, interactive controls out of the box |
| Multi-page navigation | Custom session_state router | st.Page + st.navigation | Built-in sidebar nav, URL routing, page isolation |
| Interactive charts | Custom JS/D3 | Plotly via st.plotly_chart | Hover tooltips, zoom, pan, export -- all free |
| Data table with filtering | Custom HTML tables | st.dataframe with column_config | Built-in sorting, searching, column formatting |
| Model persistence (baselines) | Custom pickle wrapper | joblib.dump / joblib.load | Standard sklearn pattern, handles numpy arrays efficiently |

**Key insight:** Streamlit's built-in widgets (st.slider, st.multiselect, st.dataframe) handle 90% of the Materials Explorer filtering UX. The remaining 10% is just composing filters into a pandas query.

## Common Pitfalls

### Pitfall 1: Baseline Models Not Persisted
**What goes wrong:** The Predict page needs trained RF/XGBoost models but the current `baselines.py` only saves metrics JSON, not the fitted model objects.
**Why it happens:** Phase 2 focused on evaluation metrics, not inference reuse.
**How to avoid:** Add `joblib.dump(model, path)` calls in `run_baselines()` after training each model. Save to `data/results/baselines/{model_type}_{property}.joblib`.
**Warning signs:** `FileNotFoundError` when Predict page tries to load a `.joblib` file.

### Pitfall 2: GNN Model Reconstruction for Inference
**What goes wrong:** Loading a `.pt` checkpoint gives a `state_dict` but you need the model architecture instantiated first.
**Why it happens:** `GNNTrainer.save_checkpoint()` saves `model_state_dict` but not the model class/config needed to reconstruct it.
**How to avoid:** The model loader must: (1) read the config from the checkpoint or from `configs/cgcnn.yaml`/`configs/megnet.yaml`, (2) build the model with `build_cgcnn_from_config()` or equivalent, (3) load the state_dict.
**Warning signs:** `RuntimeError: Error(s) in loading state_dict` from shape mismatches.

### Pitfall 3: Streamlit Reruns on Every Widget Interaction
**What goes wrong:** Expensive computations (model inference, data loading) run on every slider/button change.
**Why it happens:** Streamlit reruns the entire page script on any widget interaction.
**How to avoid:** Use `@st.cache_data` for data loading, `@st.cache_resource` for model loading, and `st.session_state` for prediction results that should persist across reruns.
**Warning signs:** Dashboard feels sluggish, repeated "Loading..." spinners.

### Pitfall 4: py3Dmol Rendering in Streamlit
**What goes wrong:** py3Dmol viewer doesn't render or shows blank iframe.
**Why it happens:** stmol uses `streamlit.components.v1.html()` internally; viewer dimensions must be explicitly set.
**How to avoid:** Always pass explicit `width` and `height` to `showmol()`. Test with a known-good CIF first (e.g., LiCoO2).
**Warning signs:** White/blank space where viewer should be.

### Pitfall 5: Large DataFrame Rendering Kills Performance
**What goes wrong:** Materials Explorer with thousands of rows causes browser lag.
**Why it happens:** `st.dataframe` renders all rows; Plotly scatter with 10K+ points slows browser.
**How to avoid:** Apply filters BEFORE display. Use `st.dataframe` with pagination (built-in). For scatter plots, downsample if >5K points.
**Warning signs:** Browser tab consuming >1GB RAM, frozen UI.

### Pitfall 6: CIF String Encoding Issues
**What goes wrong:** `Structure.from_str()` fails on uploaded CIF files.
**Why it happens:** `st.file_uploader` returns bytes; forgetting `.decode("utf-8")` or encountering non-UTF-8 CIF files.
**How to avoid:** Always decode with `uploaded.read().decode("utf-8")`, wrap in try/except with user-friendly error message.
**Warning signs:** `UnicodeDecodeError` or `ValueError: Invalid CIF`.

## Code Examples

### Loading Training Curves from CSV
```python
# Source: Existing pattern from evaluation/plots.py
import pandas as pd
import plotly.graph_objects as go
from cathode_ml.evaluation.metrics import MODEL_COLORS, MODEL_LABELS

def make_training_curves(results_base: str, model: str, prop: str) -> go.Figure | None:
    """Create interactive Plotly training curve from CSV metrics."""
    csv_path = f"{results_base}/{model}/{prop}_metrics.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None

    fig = go.Figure()
    if "train_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["train_loss"],
            name="Train Loss", line=dict(color="#333333"),
        ))
    if "val_loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df["val_loss"],
            name="Val Loss", line=dict(color=MODEL_COLORS[model]),
        ))
    fig.update_layout(
        title=f"{MODEL_LABELS[model]} - {prop.replace('_', ' ').title()}",
        xaxis_title="Epoch", yaxis_title="Loss",
    )
    return fig
```

### Materials Explorer Filtering Pattern
```python
# Source: Standard Streamlit widget pattern
import streamlit as st
import pandas as pd

def materials_filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Apply user-selected filters to materials DataFrame."""
    filtered = df.copy()

    # Voltage range slider
    if "voltage" in df.columns:
        v_min, v_max = float(df["voltage"].min()), float(df["voltage"].max())
        v_range = st.slider("Voltage (V)", v_min, v_max, (v_min, v_max))
        filtered = filtered[filtered["voltage"].between(*v_range)]

    # Element filter
    all_elements = sorted(set(e for f in df["formula"] for e in _extract_elements(f)))
    selected = st.multiselect("Must contain elements", all_elements)
    if selected:
        filtered = filtered[filtered["formula"].apply(
            lambda f: all(e in f for e in selected)
        )]

    # Stability filter
    if "energy_above_hull" in df.columns:
        stable_only = st.checkbox("Stable only (E_hull = 0)")
        if stable_only:
            filtered = filtered[filtered["energy_above_hull"] <= 0.0]

    return filtered
```

### Baseline Model Persistence (new code needed)
```python
# Must be added to cathode_ml/models/baselines.py
import joblib
from pathlib import Path

def save_model(model, model_type: str, property_name: str, results_dir: str) -> str:
    """Save fitted baseline model with joblib."""
    path = Path(results_dir) / f"{model_type}_{property_name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    return str(path)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pages/ directory auto-detection | st.Page + st.navigation | Streamlit 1.36 (2024) | Full control over page order, grouping, icons |
| st.cache (deprecated) | st.cache_data / st.cache_resource | Streamlit 1.18 (2023) | Separate data vs resource caching semantics |
| use_container_width=True | width="stretch" | Streamlit late 2025 | Deprecation warning if using old param |
| Static matplotlib in Streamlit | Plotly via st.plotly_chart | N/A (always available) | Interactive hover, zoom, pan |

**Deprecated/outdated:**
- `st.cache`: Replaced by `st.cache_data` and `st.cache_resource`
- `use_container_width=True` in `st.plotly_chart`: Deprecated late 2025, use `width="stretch"`
- pages/ directory convention: Still works but `st.Page`/`st.navigation` is preferred

## Open Questions

1. **Baseline model persistence -- retrofit or forward-only?**
   - What we know: RF/XGBoost models are not saved. Predict page needs them.
   - What's unclear: Should we modify `baselines.py` to always save models (breaking existing pipeline flow), or add a separate "export models" step?
   - Recommendation: Modify `run_baselines()` to also save models via `joblib.dump()`. This is a small, backward-compatible change (existing JSON saving still works). Users who have already trained will need to retrain once.

2. **GNN inference without full training stack**
   - What we know: CGCNN checkpoints contain `model_state_dict`. MEGNet uses Lightning checkpoints converted to `.pt`.
   - What's unclear: Whether MEGNet model reconstruction requires matgl (heavy dependency) or just torch.
   - Recommendation: Use lazy imports for matgl only when MEGNet prediction is requested. Add clear error message if matgl is not installed.

3. **Materials Explorer data source**
   - What we know: `DataCache.load("cleaned_records")` returns the cleaned dataset as a list of dicts.
   - What's unclear: Exact number of records and whether all have structure_dict populated (OQMD records have empty structure_dict).
   - Recommendation: Handle gracefully -- show "N/A" for missing fields, disable crystal viewer link for records without structures.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | None (uses defaults) |
| Quick run command | `python -m pytest tests/test_dashboard.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DASH-01 | Dashboard loads results and displays metrics table | unit | `python -m pytest tests/test_dashboard.py::test_load_results -x` | No - Wave 0 |
| DASH-02 | Predict page returns predictions from composition input | unit | `python -m pytest tests/test_dashboard.py::test_predict_composition -x` | No - Wave 0 |
| DASH-03 | Data explorer loads cached records and creates histograms | unit | `python -m pytest tests/test_dashboard.py::test_data_explorer_load -x` | No - Wave 0 |
| DASH-04 | Training curves load from CSV and render as Plotly figures | unit | `python -m pytest tests/test_dashboard.py::test_training_curves -x` | No - Wave 0 |
| DASH-05 | Materials explorer filters by voltage, elements, stability | unit | `python -m pytest tests/test_dashboard.py::test_materials_filter -x` | No - Wave 0 |
| DASH-06 | Discovery panel ranks materials by predicted properties | unit | `python -m pytest tests/test_dashboard.py::test_discovery_ranking -x` | No - Wave 0 |
| DASH-07 | Crystal viewer renders structure from CIF string | unit | `python -m pytest tests/test_dashboard.py::test_crystal_render -x` | No - Wave 0 |
| DOCS-01 | README contains introduction section | manual-only | N/A (content review) | N/A |
| DOCS-02 | README contains methodology section | manual-only | N/A (content review) | N/A |
| DOCS-03 | README contains pipeline details and CLI instructions | smoke | `python -m pytest tests/test_dashboard.py::test_readme_exists -x` | No - Wave 0 |
| DOCS-04 | README contains results summary table | manual-only | N/A (content review) | N/A |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_dashboard.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_dashboard.py` -- covers DASH-01 through DASH-07 (unit tests for data loading, chart creation, filtering logic, model loading)
- [ ] Dashboard tests should test utility functions (data_loader, charts, model_loader) NOT Streamlit widgets directly (Streamlit widget testing requires AppTest which adds complexity)

## Sources

### Primary (HIGH confidence)
- [Streamlit official docs - multipage apps](https://docs.streamlit.io/develop/concepts/multipage-apps) - st.Page/st.navigation API
- [Streamlit official docs - st.plotly_chart](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart) - Plotly integration, width parameter
- [Streamlit official docs - caching](https://docs.streamlit.io/develop/concepts/architecture/caching) - cache_data vs cache_resource
- [stmol GitHub](https://github.com/napoles-uach/stmol) - Crystal rendering with showmol()
- [pymatgen.org](https://pymatgen.org/) - Structure.from_str(), Structure.to(fmt="cif")

### Secondary (MEDIUM confidence)
- [Streamlit PyPI](https://pypi.org/project/streamlit/) - Version 1.55.0 latest (March 2026)
- [stmol on Libraries.io](https://libraries.io/pypi/stmol) - Version 0.0.9 latest

### Tertiary (LOW confidence)
- None -- all findings verified against official sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Streamlit, Plotly, stmol all verified via official docs/PyPI
- Architecture: HIGH - st.Page/st.navigation is documented official API; patterns from official tutorials
- Pitfalls: HIGH - Baseline model persistence gap verified by reading source code; Streamlit rerun behavior is well-documented
- Crystal viewer: MEDIUM - stmol 0.0.9 is small/niche; CIF rendering tested in published papers but not extensively for periodic crystals

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (Streamlit releases monthly but API is stable)
