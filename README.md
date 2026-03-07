# Lithium-Ion Battery Cathode Performance Prediction

**Predicting cathode material properties from crystal structure using traditional ML and graph neural networks to accelerate battery materials discovery.**

## Introduction

Lithium-ion batteries are central to the global transition toward renewable energy, yet identifying optimal cathode materials remains a bottleneck. Key performance indicators -- voltage, capacity, formation energy, and thermodynamic stability -- depend on complex structure-property relationships that are expensive to evaluate through first-principles computation alone. Machine learning offers a path to rapid, accurate screening of candidate cathode materials, potentially accelerating the discovery cycle by orders of magnitude.

This project benchmarks two families of models on a curated multi-source dataset of lithium cathode materials. Traditional ML baselines (Random Forest, XGBoost) operate on 132-dimensional Magpie composition descriptors, capturing elemental statistics without explicit structural information. Graph neural networks (CGCNN, MEGNet) operate directly on crystal graphs, encoding atomic environments and bond geometries. By comparing these approaches under identical evaluation conditions -- compositional group splitting, consistent metrics, shared data -- we quantify the value of structural information for each target property.

The result is a reproducible, end-to-end pipeline from raw data ingestion through model training, evaluation, and interactive exploration via a Streamlit dashboard.

## Data Sources

The dataset is assembled from three complementary sources, each contributing different aspects of cathode characterization:

- **Materials Project** (mp-api): Primary source providing lithium cathode electrode entries with full crystal structures, formation energies, and computed voltages. Accessed via the official Python client.
- **OQMD** (REST API): Open Quantum Materials Database supplying supplementary DFT-computed properties for cross-validation and expanded coverage of the lithium oxide chemical space.
- **Battery Data Genome** (CSV): Experimental battery cycling data providing measured capacity and voltage values that ground-truth computational predictions.

Records are deduplicated using source priority (Materials Project > OQMD > Battery Data Genome) to resolve conflicts when the same material appears in multiple databases. Compositional group splitting ensures that polymorphs (different structures of the same composition) never appear in both training and test sets, preventing data leakage.

## Methodology

### Model Architectures

**Random Forest (RF):** An ensemble of 100 decision trees trained on 132-dimensional Magpie composition descriptors. Magpie features encode elemental statistics (mean, deviation, range, etc.) of properties like electronegativity, atomic radius, and valence electrons. The ensemble approach provides robust predictions with built-in feature importance via Gini impurity.

**XGBoost (XGB):** Gradient-boosted trees with L1/L2 regularization operating on the same Magpie feature space. XGBoost applies sequential boosting with early stopping on a validation set, typically achieving lower bias than Random Forest at the cost of additional hyperparameter sensitivity.

**CGCNN (Crystal Graph Convolutional Neural Network):** Following Xie & Grossman (2018), crystal structures are represented as graphs where nodes are atoms and edges connect neighbors within a cutoff radius. Node features are one-hot element embeddings; edge features use Gaussian distance expansion. PyTorch Geometric CGConv layers propagate information through the graph, and a global mean pooling layer produces a fixed-length representation for property regression.

**MEGNet (Materials Graph Network):** A pre-trained MEGNet model (MEGNet-MP-2018.6.1-Eform) fine-tuned via the matgl library. MEGNet augments the atom-bond graph with global state features capturing bulk material properties. Fine-tuning from a model pre-trained on 60,000+ Materials Project entries provides strong inductive bias, especially beneficial when target datasets are small.

### Design Choices

- **Separate models per property:** Each target property (formation energy, voltage, capacity, energy above hull) is predicted by an independent model, following standard practice in materials property prediction.
- **Compositional group splitting:** Train/validation/test splits are performed at the composition level using GroupShuffleSplit, ensuring no polymorph leakage between splits.
- **Config-driven hyperparameters:** All model settings are specified in YAML configuration files (`configs/`), enabling reproducible experiments without code changes.

### Evaluation

All models are evaluated on identical held-out test sets using three metrics:
- **MAE** (Mean Absolute Error): Primary metric for interpretable error magnitude
- **RMSE** (Root Mean Squared Error): Penalizes large prediction errors
- **R-squared**: Proportion of variance explained, for comparing across properties with different scales

## Results

### Summary Table

| Property | Best Model | MAE | R-squared |
|---|---|---|---|
| formation_energy_per_atom | -- | -- | -- |
| voltage | -- | -- | -- |
| capacity | -- | -- | -- |
| energy_above_hull | -- | -- | -- |

> **Note:** Values above are placeholders. Run the full pipeline (`python -m cathode_ml`) to populate results from your data. Actual metrics will be written to `data/results/` as JSON files.

If the evaluation pipeline has been executed, a visual comparison is available:

![Model Comparison](data/results/figures/bar_comparison.png)

**Interpretation:** Graph neural networks (CGCNN, MEGNet) tend to outperform composition-only baselines on properties that depend strongly on crystal structure (e.g., formation energy, stability), where bond geometries and atomic environments carry information not captured by elemental statistics alone. For composition-dominated properties like voltage, traditional baselines remain competitive, suggesting that elemental chemistry is the primary driver.

## Dashboard

The project includes a 6-page interactive Streamlit dashboard for exploring data, models, and predictions.

<!-- TODO: Add dashboard screenshot after first run -->

### Pages

1. **Overview** -- Landing page with a best-model-per-property summary table and an MAE bar chart comparing all models at a glance.
2. **Model Comparison** -- Detailed per-property metrics tables and GNN training/validation loss curves for monitoring convergence.
3. **Data Explorer** -- Interactive histograms of property distributions and scatter matrices for exploring correlations across the dataset.
4. **Materials Explorer** -- Searchable, filterable table of all materials with property-range sliders, plus a "Top Candidates" panel highlighting promising compositions.
5. **Predict** -- Enter a composition string (e.g., "LiFePO4") for quick baseline predictions, or upload a CIF file for full GNN predictions including structural features.
6. **Crystal Viewer** -- 3D interactive crystal structure visualization (py3Dmol) for uploaded CIF files, with rotatable, zoomable rendering.

### Launch

```bash
streamlit run dashboard/app.py
```

## How to Run

### Prerequisites

- Python 3.9+
- conda or pip (conda recommended for PyTorch/PyG dependencies)

### Installation

```bash
git clone https://github.com/your-username/lithium-cathode-ml.git
cd lithium-cathode-ml
pip install -r requirements.txt
```

### Configuration

All experiment settings are managed through YAML files in the `configs/` directory:

| File | Purpose |
|---|---|
| `data.yaml` | Data source URLs, cache settings, cleaning thresholds |
| `features.yaml` | Composition descriptor settings, graph construction parameters |
| `baselines.yaml` | Random Forest and XGBoost hyperparameters |
| `cgcnn.yaml` | CGCNN architecture, training schedule, and cutoff radius |
| `megnet.yaml` | MEGNet fine-tuning settings and pre-trained model selection |

### Running the Pipeline

The full pipeline runs all stages sequentially:

```bash
python -m cathode_ml
```

**Pipeline stages:**

1. **Fetch** -- Download and cache data from Materials Project, OQMD, and Battery Data Genome
2. **Featurize** -- Composition descriptors and crystal graphs are computed inline during training
3. **Train** -- Train all four model architectures on each target property
4. **Evaluate** -- Generate comparison tables, bar charts, and learning curves

**Selective execution:**

```bash
python -m cathode_ml --skip-fetch          # Use cached data (skip download)
python -m cathode_ml --skip-train          # Use saved models (skip training)
python -m cathode_ml --models rf xgb       # Train only baseline models
python -m cathode_ml --models cgcnn megnet # Train only GNN models
python -m cathode_ml --seed 123            # Set random seed for reproducibility
```

### Materials Project API Key

Data fetching from the Materials Project requires an API key. Set it as an environment variable:

```bash
export MP_API_KEY="your-api-key-here"
```

Alternatively, create a `.env` file in the project root:

```
MP_API_KEY=your-api-key-here
```

You can obtain a free API key at [materialsproject.org](https://materialsproject.org).

### Dashboard

After running the pipeline (or with pre-existing results), launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
lithium-cathode-ml/
├── cathode_ml/                  # Core ML package
│   ├── __main__.py              # Module entry point
│   ├── pipeline.py              # CLI pipeline orchestrator
│   ├── config.py                # YAML configuration loader
│   ├── data/                    # Data ingestion and cleaning
│   │   ├── fetch.py             # Multi-source fetch orchestrator
│   │   ├── mp_fetcher.py        # Materials Project client
│   │   ├── oqmd_fetcher.py      # OQMD REST client
│   │   ├── bdg_fetcher.py       # Battery Data Genome downloader
│   │   ├── clean.py             # Deduplication and outlier removal
│   │   ├── cache.py             # Disk cache with hash-based keys
│   │   └── schemas.py           # Data record schema definitions
│   ├── features/                # Feature engineering
│   │   ├── composition.py       # Magpie composition descriptors
│   │   ├── graph.py             # Crystal graph construction
│   │   └── split.py             # Compositional group splitting
│   ├── models/                  # Model implementations
│   │   ├── baselines.py         # RF and XGBoost training
│   │   ├── cgcnn.py             # CGCNN architecture (PyG)
│   │   ├── train_cgcnn.py       # CGCNN training orchestrator
│   │   ├── megnet.py            # MEGNet wrapper (matgl)
│   │   ├── train_megnet.py      # MEGNet fine-tuning orchestrator
│   │   ├── trainer.py           # Model-agnostic GNN trainer
│   │   └── utils.py             # Shared utilities (compute_metrics)
│   └── evaluation/              # Benchmarking and visualization
│       ├── metrics.py           # Metric computation and tables
│       └── plots.py             # Matplotlib figure generation
├── dashboard/                   # Streamlit interactive dashboard
│   ├── app.py                   # Multi-page app entrypoint
│   ├── pages/                   # Dashboard pages
│   │   ├── overview.py          # Metrics summary and bar chart
│   │   ├── model_comparison.py  # Per-property comparison
│   │   ├── data_explorer.py     # Property distributions
│   │   ├── materials_explorer.py # Material search and filtering
│   │   ├── predict.py           # Interactive predictions
│   │   └── crystal_viewer.py    # 3D structure viewer
│   └── utils/                   # Dashboard utilities
│       ├── data_loader.py       # Cached data loading
│       ├── charts.py            # Plotly chart factories
│       └── model_loader.py      # Model loading for predictions
├── configs/                     # YAML configuration files
│   ├── data.yaml
│   ├── features.yaml
│   ├── baselines.yaml
│   ├── cgcnn.yaml
│   └── megnet.yaml
├── data/                        # Data directory (gitignored contents)
│   └── results/                 # Model outputs, metrics, figures
├── tests/                       # Test suite
│   ├── conftest.py              # Shared fixtures
│   ├── test_*.py                # Unit tests per module
│   └── ...
└── requirements.txt             # Pinned dependencies
```

## License

Academic project -- not currently released under a formal license.

## Citation

If you use this work in your research, please cite:

```
@software{lithium_cathode_ml,
  title={Lithium-Ion Battery Cathode Performance Prediction},
  year={2026},
  url={https://github.com/your-username/lithium-cathode-ml}
}
```
