# Phase 5: Evaluation and Benchmarking - Research

**Researched:** 2026-03-06
**Domain:** matplotlib publication plotting, unified evaluation, CLI pipeline orchestration
**Confidence:** HIGH

## Summary

Phase 5 builds three interconnected components on top of the existing model training infrastructure: (1) a unified evaluation module that loads JSON result artifacts from all four models (RF, XGBoost, CGCNN, MEGNet) and produces comparison tables, (2) a matplotlib-based plotting module that generates publication-quality parity plots, bar charts, and learning curves, and (3) a CLI pipeline that orchestrates the entire workflow (fetch, clean, featurize, train, evaluate) with a single command.

The project already has strong conventions: JSON result artifacts with consistent keys (mae, rmse, r2, n_train, n_test), CSV per-epoch metrics for GNNs (columns: epoch, train_loss, val_loss, val_mae, lr), compositional splitting via `compositional_split()`, and `compute_metrics()` in `models/utils.py`. The evaluation module reads these existing artifacts -- it does NOT retrain models. The CLI pipeline calls existing training functions (`run_baselines`, `train_cgcnn`, `train_megnet`) and the new evaluation module.

**Primary recommendation:** Build `cathode_ml/evaluation/` as a new subpackage with `metrics.py` (JSON loading + table generation), `plots.py` (all matplotlib figures), and `__main__.py` (CLI entry point). Build `cathode_ml/pipeline.py` as the top-level pipeline orchestrator with `cathode_ml/__main__.py` as its module entry point.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Nature/Science minimal style: no grid, high contrast, frameless legends, clean spines
- Wong colorblind-safe palette: RF=#0072B2 (blue), XGB=#D55E00 (orange), CGCNN=#009E73 (green), MEGNet=#CC79A7 (pink)
- 300 DPI PNG output format
- matplotlib only (already a dependency, no new deps)
- One markdown table per property (formation energy, voltage, capacity, energy_above_hull) with models as rows and MAE/RMSE/R-squared as columns
- Bold best value per column (lowest MAE/RMSE, highest R-squared)
- Metrics: MAE, RMSE, R-squared only
- Output as both markdown (.md) and JSON artifacts
- 2x2 subplot grid per property (one panel per model: RF, XGB, CGCNN, MEGNet)
- R-squared and MAE displayed as text annotations in upper-left corner of each panel
- 4 figures total (one per property)
- Asterisk/dagger convention: "MEGNet+" with footnote "+ Fine-tuned from pretrained MEGNet-MP-2019.4.1"
- CGCNN, RF, XGB labeled without annotation (all trained from scratch)
- Both single pipeline command AND individual subcommands
- Pipeline entry point: `python -m cathode_ml.pipeline` runs fetch -> clean -> featurize -> train -> evaluate
- Individual subcommands for each stage (fetch, featurize, train, evaluate)
- `--models` flag to select which models to train/evaluate (default: all)
- `--skip-fetch` and `--skip-train` flags for skipping stages
- Stage banners + Python logging for progress reporting (=== Stage 1/5: Fetching Data ===)
- argparse for CLI argument parsing

### Claude's Discretion
- Exact font sizes and axis label formatting within Nature style
- Figure file naming convention within data/results/figures/
- Learning curve smoothing (if any)
- Bar chart grouping for multi-property comparison figure
- Subcommand naming and help text

### Deferred Ideas (OUT OF SCOPE)
- Kafka, Kubernetes -- enterprise infrastructure tools
- MLflow experiment tracking (INFR-01) -- deferred to v2
- Docker containerized pipeline (INFR-03) -- deferred to v2
- Optuna hyperparameter tuning (INFR-02) -- deferred to v2
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | System evaluates all models with MAE, RMSE, and R-squared metrics | Unified evaluation loader reads existing JSON artifacts from data/results/{baselines,cgcnn,megnet}/ and consolidates into comparison tables |
| EVAL-02 | System uses consistent cross-validation folds across all models for fair comparison | Already guaranteed by compositional_split() with same seed/config; evaluation just loads pre-computed results |
| EVAL-03 | System generates parity plots (predicted vs actual) for each model and property | matplotlib 2x2 subplot parity plots with Wong palette and Nature style; requires saving y_true/y_pred during training |
| EVAL-04 | System generates bar chart comparisons of model performance across properties | Grouped bar chart from consolidated metrics JSON; one bar group per property, one bar per model |
| EVAL-05 | System generates learning/training curves (loss, validation MAE per epoch) | Read CSV per-epoch metrics from data/results/cgcnn/ and data/results/megnet/; plot with dual y-axis or separate panels |
| REPR-04 | System provides CLI entry points to run full pipeline | cathode_ml/pipeline.py with argparse; python -m cathode_ml.pipeline runs all stages end-to-end |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.8+ | All publication figures | Already a dependency; Nature/Science style achievable with rcParams |
| numpy | 1.26+ | Array operations for metrics aggregation | Already a dependency |
| pandas | 2.1+ | CSV reading for training logs | Already a dependency (via matminer) |
| argparse | stdlib | CLI argument parsing | Standard library; consistent with existing __main__.py modules |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json | stdlib | Load/save result artifacts | Reading model results, writing comparison JSON |
| pathlib | stdlib | Path manipulation | All file I/O throughout evaluation |
| logging | stdlib | Progress reporting | Stage banners and status updates |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | seaborn | Adds dependency; Nature style easier with raw matplotlib rcParams |
| argparse | click | Adds dependency; argparse sufficient for this scope |
| pandas CSV | csv stdlib | pandas handles NaN/missing columns in Lightning logs cleanly |

**Installation:**
No new packages needed. All dependencies already installed.

## Architecture Patterns

### Recommended Project Structure
```
cathode_ml/
├── evaluation/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # python -m cathode_ml.evaluation
│   ├── metrics.py           # Load results, build comparison tables
│   └── plots.py             # All matplotlib figure generation
├── pipeline.py              # Full pipeline orchestrator
└── __main__.py              # python -m cathode_ml.pipeline entry
```

Output structure:
```
data/results/
├── baselines/
│   └── baseline_results.json
├── cgcnn/
│   ├── cgcnn_results.json
│   ├── formation_energy_per_atom_metrics.csv
│   └── ... (per-property CSV)
├── megnet/
│   ├── megnet_results.json
│   ├── formation_energy_per_atom_metrics.csv
│   └── ... (per-property CSV)
├── figures/
│   ├── parity_formation_energy_per_atom.png
│   ├── parity_voltage.png
│   ├── parity_capacity.png
│   ├── parity_energy_above_hull.png
│   ├── model_comparison_bar.png
│   └── learning_curves.png
└── comparison/
    ├── comparison.md
    └── comparison.json
```

### Pattern 1: Unified Result Loading
**What:** A single function loads all model results from their respective directories and normalizes into a unified dict structure.
**When to use:** Any time evaluation needs to compare models.
**Example:**
```python
def load_all_results(results_base: str = "data/results") -> dict:
    """Load results from all model directories.

    Returns:
        Dict: {property: {model_name: {mae, rmse, r2, n_train, n_test}}}
    """
    base = Path(results_base)
    all_results = {}

    # Baselines: nested as {property: {rf: {...}, xgb: {...}}}
    baseline_path = base / "baselines" / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baselines = json.load(f)
        for prop, models in baselines.items():
            all_results.setdefault(prop, {}).update(models)

    # CGCNN: nested as {property: {cgcnn: {...}}}
    cgcnn_path = base / "cgcnn" / "cgcnn_results.json"
    if cgcnn_path.exists():
        with open(cgcnn_path) as f:
            cgcnn = json.load(f)
        for prop, models in cgcnn.items():
            all_results.setdefault(prop, {}).update(models)

    # MEGNet: nested as {property: {megnet: {...}}}
    megnet_path = base / "megnet" / "megnet_results.json"
    if megnet_path.exists():
        with open(megnet_path) as f:
            megnet = json.load(f)
        for prop, models in megnet.items():
            all_results.setdefault(prop, {}).update(models)

    return all_results
```

### Pattern 2: Matplotlib Nature Style via rcParams
**What:** Set global matplotlib style once at module level for all figures.
**When to use:** Before any figure generation.
**Example:**
```python
# Wong colorblind-safe palette
MODEL_COLORS = {
    "rf": "#0072B2",
    "xgb": "#D55E00",
    "cgcnn": "#009E73",
    "megnet": "#CC79A7",
}

MODEL_LABELS = {
    "rf": "RF",
    "xgb": "XGBoost",
    "cgcnn": "CGCNN",
    "megnet": "MEGNet\u2020",  # dagger for pretrained
}

NATURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

def apply_nature_style():
    import matplotlib as mpl
    mpl.rcParams.update(NATURE_STYLE)
```

### Pattern 3: Pipeline Stage Orchestration
**What:** Sequential stage execution with skip flags, stage banners, and error handling.
**When to use:** The main pipeline entry point.
**Example:**
```python
def run_pipeline(args):
    stages = [
        ("Fetching Data", run_fetch, not args.skip_fetch),
        ("Featurizing", run_featurize, True),
        ("Training Models", run_train, not args.skip_train),
        ("Evaluating", run_evaluate, True),
    ]
    for i, (name, fn, enabled) in enumerate(stages, 1):
        if enabled:
            logger.info("=== Stage %d/%d: %s ===", i, len(stages), name)
            fn(args)
        else:
            logger.info("=== Stage %d/%d: %s (SKIPPED) ===", i, len(stages), name)
```

### Pattern 4: Parity Plot with 2x2 Subplots
**What:** One figure per property with 4 panels (one per model), diagonal reference line, text annotations.
**When to use:** EVAL-03 parity plots.
**Example:**
```python
def plot_parity(property_name, model_predictions, output_path):
    """Generate 2x2 parity plot for one property.

    Args:
        property_name: Target property name.
        model_predictions: Dict {model_name: {"y_true": [...], "y_pred": [...], "mae": float, "r2": float}}
        output_path: Path to save PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    models_order = ["rf", "xgb", "cgcnn", "megnet"]

    for ax, model_key in zip(axes.flat, models_order):
        data = model_predictions.get(model_key)
        if data is None:
            ax.set_visible(False)
            continue

        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])

        ax.scatter(y_true, y_pred, s=10, alpha=0.6,
                   color=MODEL_COLORS[model_key], edgecolors="none")

        # Diagonal reference
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)

        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        # Annotation in upper-left
        ax.text(0.05, 0.95, f"R$^2$ = {data['r2']:.3f}\nMAE = {data['mae']:.3f}",
                transform=ax.transAxes, va="top", fontsize=7)

    fig.suptitle(property_name.replace("_", " ").title(), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
```

### Anti-Patterns to Avoid
- **Retraining during evaluation:** The evaluation module should ONLY load existing JSON/CSV artifacts. Training is done by the pipeline or individual model commands.
- **Hardcoded file paths:** Use config or function parameters for results_dir, not string literals scattered through code.
- **plt.show() in scripts:** Use plt.savefig() and plt.close() only. Interactive display breaks headless execution.
- **Single monolithic plotting function:** Separate parity plots, bar charts, and learning curves into distinct functions for testability.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Metric computation | Custom MAE/RMSE/R2 formulas | `compute_metrics()` from `models/utils.py` | Already validated, consistent across all models |
| Compositional splitting | Custom split logic | `compositional_split()` from `features/split.py` | Same function used by all training orchestrators |
| JSON result format | New schema | Existing `{mae, rmse, r2, n_train, n_test}` dict | Consistency with Phases 2-4 |
| Colorblind palette | Manual hex picking | Wong palette constants (locked decision) | Scientifically validated, user-specified |

**Key insight:** The evaluation phase is primarily a *consumer* of existing artifacts and conventions. The infrastructure is already built; this phase connects and visualizes.

## Common Pitfalls

### Pitfall 1: Missing Predictions Data for Parity Plots
**What goes wrong:** The existing JSON results only contain aggregated metrics (mae, rmse, r2), not individual predictions (y_true, y_pred arrays). Parity plots need per-sample predictions.
**Why it happens:** Phases 2-4 focused on saving summary metrics, not prediction arrays.
**How to avoid:** The evaluation module needs to either (a) re-run inference on test data to get predictions, or (b) modify the pipeline to save predictions during training. Option (a) is cleaner since it avoids modifying completed phases -- the evaluation script loads models/data, recomputes splits, and runs inference.
**Warning signs:** Empty or missing y_true/y_pred when trying to generate parity plots.

### Pitfall 2: Inconsistent Result JSON Nesting
**What goes wrong:** Baselines save as `{property: {rf: {...}, xgb: {...}}}` while CGCNN saves as `{property: {cgcnn: {...}}}` and MEGNet as `{property: {megnet: {...}}}`. The loader must handle this nesting correctly.
**Why it happens:** Different phases developed independently with slightly different structures.
**How to avoid:** The `load_all_results()` function must normalize all formats into `{property: {model: metrics}}`.
**Warning signs:** KeyError when accessing model results by name.

### Pitfall 3: Missing Properties for Some Models
**What goes wrong:** Some models may have skipped a property (< 5 records). The comparison table and bar chart must handle missing entries gracefully.
**Why it happens:** All training orchestrators skip properties with fewer than 5 valid records.
**How to avoid:** Use `.get()` with fallback, mark missing cells as "N/A" in tables, skip missing bars in charts.
**Warning signs:** KeyError in table generation or bar chart with uneven number of bars.

### Pitfall 4: Font Availability on Different Systems
**What goes wrong:** Arial may not be available on Linux systems. Matplotlib falls back to DejaVu Sans which looks different.
**Why it happens:** Arial is a proprietary Microsoft font not installed by default on Linux.
**How to avoid:** Specify fallback chain: `["Arial", "Helvetica", "DejaVu Sans"]`. DejaVu Sans is always available with matplotlib.
**Warning signs:** Matplotlib font warnings in logs.

### Pitfall 5: Pipeline Import Overhead
**What goes wrong:** Importing matgl/DGL/PyTorch at pipeline start slows down even simple operations like `--help`.
**Why it happens:** Top-level imports of heavy science libraries.
**How to avoid:** Use lazy imports in the pipeline -- only import heavy modules inside the stage functions that need them, consistent with existing project patterns.
**Warning signs:** 10+ second startup time for `python -m cathode_ml.pipeline --help`.

### Pitfall 6: CSV Column Mismatch Between CGCNN and MEGNet
**What goes wrong:** CGCNN metrics CSV uses columns `epoch,train_loss,val_loss,val_mae,lr` from GNNTrainer. MEGNet uses Lightning CSVLogger which gets converted by `convert_lightning_logs()` to `epoch,train_loss,val_loss,val_mae,train_mae`. The `lr` column may be missing for MEGNet.
**Why it happens:** Different training backends (custom GNNTrainer vs Lightning).
**How to avoid:** Learning curve plotting must handle optional columns. Use `df.get("lr", None)` patterns.
**Warning signs:** KeyError or NaN values when plotting MEGNet learning curves.

## Code Examples

### Markdown Table Generation
```python
def generate_comparison_table(all_results: dict, property_name: str) -> str:
    """Generate markdown comparison table for one property.

    Args:
        all_results: {property: {model: {mae, rmse, r2, ...}}}
        property_name: Which property to tabulate.

    Returns:
        Markdown string with bolded best values.
    """
    prop_results = all_results.get(property_name, {})
    models_order = ["rf", "xgb", "cgcnn", "megnet"]
    metrics = ["mae", "rmse", "r2"]

    # Find best values
    best = {}
    for metric in metrics:
        values = {m: prop_results[m][metric] for m in models_order if m in prop_results}
        if values:
            if metric == "r2":
                best[metric] = max(values, key=values.get)
            else:
                best[metric] = min(values, key=values.get)

    # Build table
    header = f"| Model | MAE | RMSE | R\u00b2 |"
    sep = "|-------|-----|------|-----|"
    rows = [header, sep]

    for model in models_order:
        if model not in prop_results:
            continue
        m = prop_results[model]
        label = MODEL_LABELS[model]
        cells = []
        for metric in metrics:
            val = f"{m[metric]:.4f}"
            if best.get(metric) == model:
                val = f"**{val}**"
            cells.append(val)
        rows.append(f"| {label} | {' | '.join(cells)} |")

    # Add footnote for MEGNet
    rows.append("")
    rows.append("\u2020 Fine-tuned from pretrained MEGNet-MP-2019.4.1")

    return "\n".join(rows)
```

### Learning Curve Plotting
```python
def plot_learning_curves(results_base: str, output_path: str):
    """Plot training/learning curves for CGCNN and MEGNet.

    Reads per-epoch CSV files and plots train_loss and val_mae.
    """
    import pandas as pd

    properties = ["formation_energy_per_atom", "voltage", "capacity", "energy_above_hull"]
    gnn_models = {"cgcnn": "CGCNN", "megnet": "MEGNet\u2020"}

    fig, axes = plt.subplots(len(properties), 2, figsize=(10, 3 * len(properties)))

    for row, prop in enumerate(properties):
        for col, (model_key, model_label) in enumerate(gnn_models.items()):
            ax = axes[row, col]
            csv_path = Path(results_base) / model_key / f"{prop}_metrics.csv"

            if not csv_path.exists():
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            df = pd.read_csv(csv_path)

            # Plot loss
            ax.plot(df["epoch"], df["train_loss"], color="#333333",
                    linewidth=0.8, label="Train Loss")
            if "val_loss" in df.columns:
                ax.plot(df["epoch"], df["val_loss"], color=MODEL_COLORS[model_key],
                        linewidth=0.8, label="Val Loss")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{model_label} - {prop.replace('_', ' ').title()}")
            ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
```

### Pipeline CLI with argparse
```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cathode-ml-pipeline",
        description="Run the full cathode ML pipeline end-to-end",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rf", "xgb", "cgcnn", "megnet"],
        default=["rf", "xgb", "cgcnn", "megnet"],
        help="Models to train/evaluate (default: all)",
    )
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetching (use cached data)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use saved checkpoints/results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--config-dir", default="configs",
                        help="Directory containing YAML configs (default: configs)")
    return parser
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| plt.style.use("seaborn") | Custom rcParams dict | matplotlib 3.6+ deprecated seaborn style | Must define style manually |
| Single script with all logic | Package with submodules | Project convention | evaluation/ as subpackage |
| Interactive plt.show() | plt.savefig() + plt.close() | Standard for headless/CI | All figures saved as files |

**Deprecated/outdated:**
- `plt.style.use("seaborn")`: Deprecated in matplotlib 3.6, removed in 3.8+. Use `seaborn-v0_8` or custom rcParams.
- `matplotlib.cm.get_cmap()`: Deprecated in 3.7+. Use `matplotlib.colormaps["name"]` instead. Not relevant here since we use explicit hex colors.

## Open Questions

1. **Prediction Arrays for Parity Plots**
   - What we know: Existing JSON results contain only aggregated metrics, not per-sample predictions.
   - What's unclear: Whether to save predictions during training (modifies Phase 2-4 code) or re-run inference during evaluation.
   - Recommendation: Re-run inference during evaluation. The evaluation script loads processed data, recomputes splits (same seed = same splits), loads trained models, and generates predictions. This avoids modifying completed phases. For baselines, models can be retrained quickly (seconds). For GNNs, load checkpoint weights and run inference only.

2. **Baseline Model Persistence**
   - What we know: RF and XGBoost models are not saved as artifacts (only JSON metrics are saved).
   - What's unclear: Whether to add joblib model saving to baselines or retrain during evaluation.
   - Recommendation: For parity plots, retrain baselines (fast, < 30 seconds) since they are not persisted. Alternatively, add prediction saving to the evaluation flow. Since compositional_split with the same seed produces identical splits, retraining yields identical results.

3. **MEGNet Predictions Without Full Lightning Stack**
   - What we know: `predict_with_megnet()` already exists in `models/megnet.py`.
   - What's unclear: Whether saved .pt checkpoint can be loaded without full Lightning setup.
   - Recommendation: Use `predict_with_megnet()` which uses matgl's native inference. Load the model via `load_megnet_model()` then load saved state_dict. This avoids Lightning dependency for inference-only.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pyproject.toml) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/test_evaluation.py -x` |
| Full suite command | `pytest tests/ -v --tb=short` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Load all model results and compute unified metrics | unit | `pytest tests/test_evaluation.py::test_load_all_results -x` | No - Wave 0 |
| EVAL-02 | Consistent cross-validation folds | unit | `pytest tests/test_evaluation.py::test_consistent_splits -x` | No - Wave 0 |
| EVAL-03 | Parity plot generation | unit | `pytest tests/test_evaluation.py::test_parity_plot -x` | No - Wave 0 |
| EVAL-04 | Bar chart comparison | unit | `pytest tests/test_evaluation.py::test_bar_chart -x` | No - Wave 0 |
| EVAL-05 | Learning curve plotting | unit | `pytest tests/test_evaluation.py::test_learning_curves -x` | No - Wave 0 |
| REPR-04 | CLI pipeline runs end-to-end | unit | `pytest tests/test_pipeline.py::test_pipeline_cli -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_evaluation.py tests/test_pipeline.py -x`
- **Per wave merge:** `pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_evaluation.py` -- covers EVAL-01 through EVAL-05
- [ ] `tests/test_pipeline.py` -- covers REPR-04
- [ ] No new framework install needed (pytest already configured)

## Sources

### Primary (HIGH confidence)
- Project codebase: `cathode_ml/models/utils.py` -- compute_metrics() API and save_results() API
- Project codebase: `cathode_ml/models/baselines.py` -- baseline result JSON format
- Project codebase: `cathode_ml/models/train_cgcnn.py` -- CGCNN result JSON and CSV format
- Project codebase: `cathode_ml/models/train_megnet.py` -- MEGNet result JSON and CSV format
- Project codebase: `cathode_ml/data/__main__.py` -- existing CLI entry point pattern
- Project codebase: `cathode_ml/models/trainer.py` -- GNNTrainer CSV column format
- Project codebase: `cathode_ml/config.py` -- load_config() and set_seeds() utilities
- Project codebase: `cathode_ml/data/schemas.py` -- MaterialRecord dataclass
- Project codebase: `configs/features.yaml` -- target_properties list and splitting config

### Secondary (MEDIUM confidence)
- matplotlib documentation: rcParams customization for Nature/Science style
- Wong 2011 colorblind-safe palette: published in Nature Methods

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in project, no new dependencies
- Architecture: HIGH - follows established project patterns (subpackage modules, JSON artifacts, argparse CLI)
- Pitfalls: HIGH - identified from direct code inspection of existing result formats and training pipelines
- Plotting: MEDIUM - Nature style rcParams are well-documented but exact font rendering depends on system

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no fast-moving dependencies)
