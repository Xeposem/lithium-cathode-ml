# Phase 5: Evaluation and Benchmarking - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Rigorous cross-model comparison (RF, XGBoost, CGCNN, MEGNet) with publication-quality figures, markdown comparison tables, and a CLI pipeline that runs the full workflow (fetch, clean, featurize, train, evaluate) end-to-end. Covers requirements EVAL-01 through EVAL-05 and REPR-04.

</domain>

<decisions>
## Implementation Decisions

### Figure Style & Aesthetics
- Nature/Science minimal style: no grid, high contrast, frameless legends, clean spines
- Wong colorblind-safe palette: RF=#0072B2 (blue), XGB=#D55E00 (orange), CGCNN=#009E73 (green), MEGNet=#CC79A7 (pink)
- 300 DPI PNG output format
- matplotlib only (already a dependency, no new deps)

### Comparison Presentation
- One markdown table per property (formation energy, voltage, capacity, energy_above_hull) with models as rows and MAE/RMSE/R-squared as columns
- Bold best value per column (lowest MAE/RMSE, highest R-squared) -- standard academic convention
- Metrics: MAE, RMSE, R-squared only (consistent with compute_metrics() from Phases 2-4)
- Output as both markdown (.md) and JSON artifacts

### Parity Plots
- 2x2 subplot grid per property (one panel per model: RF, XGB, CGCNN, MEGNet)
- R-squared and MAE displayed as text annotations in upper-left corner of each panel
- 4 figures total (one per property)

### Fairness Annotations
- Asterisk/dagger convention: "MEGNet+" with footnote "+ Fine-tuned from pretrained MEGNet-MP-2019.4.1"
- Applied consistently in both tables and figure panel titles
- CGCNN, RF, XGB labeled without annotation (all trained from scratch)

### CLI Pipeline Design
- Both single pipeline command AND individual subcommands
- Pipeline entry point: `python -m cathode_ml.pipeline` runs fetch -> clean -> featurize -> train -> evaluate
- Individual subcommands for each stage (fetch, featurize, train, evaluate)
- `--models` flag to select which models to train/evaluate (default: all)
- `--skip-fetch` and `--skip-train` flags for skipping stages (use cached data / saved checkpoints)
- Stage banners + Python logging for progress reporting (=== Stage 1/5: Fetching Data ===)
- argparse for CLI argument parsing (standard library, consistent with existing __main__.py modules)

### Claude's Discretion
- Exact font sizes and axis label formatting within Nature style
- Figure file naming convention within data/results/figures/
- Learning curve smoothing (if any)
- Bar chart grouping for multi-property comparison figure
- Subcommand naming and help text

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `models/utils.py`: compute_metrics(y_true, y_pred, n_train) and save_results(results, path) -- shared across all models, directly reusable for unified evaluation
- `models/train_cgcnn.py` and `models/train_megnet.py`: Training orchestrators that save JSON results and CSV per-epoch metrics
- `models/baselines.py`: Baseline training with same JSON result format
- `features/split.py`: compositional_split() ensures identical folds across all models
- `config.py`: load_config(), set_seeds(), get_project_root() for pipeline config
- `data/__main__.py`: Existing CLI entry point pattern for fetch stage

### Established Patterns
- JSON results in data/results/{model_name}/ with keys: mae, rmse, r2, n_train, n_test
- CSV per-epoch metrics for GNNs (columns: epoch, train_loss, val_loss, val_mae, lr)
- YAML config files in configs/ directory
- Lazy imports for heavy deps (matgl, xgboost)
- Per-property sequential training with skip if <5 records

### Integration Points
- Input: data/results/baselines/, data/results/cgcnn/, data/results/megnet/ (JSON artifacts from Phases 2-4)
- Input: data/results/cgcnn/ and data/results/megnet/ (CSV training logs for learning curves)
- Output: data/results/figures/ (PNG plots), data/results/comparison/ (markdown + JSON tables)
- New files: cathode_ml/evaluation/ module, cathode_ml/pipeline.py, cathode_ml/__main__.py

</code_context>

<specifics>
## Specific Ideas

- Nature/Science journal aesthetic for all figures -- minimal, high-contrast, publication-ready
- Wong colorblind-safe palette specifically chosen for accessibility
- Table layout inspired by the preview: one table per property with dagger annotation for MEGNet pretrained status

</specifics>

<deferred>
## Deferred Ideas

- Kafka, Kubernetes -- enterprise infrastructure tools; don't fit single-machine ML pipeline scope
- MLflow experiment tracking (INFR-01) -- deferred to v2
- Docker containerized pipeline (INFR-03) -- deferred to v2
- Optuna hyperparameter tuning (INFR-02) -- deferred to v2

</deferred>

---

*Phase: 05-evaluation-and-benchmarking*
*Context gathered: 2026-03-06*
