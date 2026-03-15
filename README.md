# Lithium-Ion Battery Cathode Performance Prediction

**Predicting cathode material properties from crystal structure using traditional ML and graph neural networks to accelerate battery materials discovery.**

## Introduction

Lithium-ion batteries are central to the global transition toward renewable energy, yet identifying optimal cathode materials remains a bottleneck. Key performance indicators -- voltage, capacity, formation energy, and thermodynamic stability -- depend on complex structure-property relationships that are expensive to evaluate through first-principles computation alone. Machine learning offers a path to rapid, accurate screening of candidate cathode materials, potentially accelerating the discovery cycle by orders of magnitude.

This project benchmarks two families of models on a curated multi-source dataset of lithium cathode materials. Traditional ML baselines (Random Forest, XGBoost) operate on 132-dimensional Magpie composition descriptors, capturing elemental statistics without explicit structural information. Graph neural networks (CGCNN, M3GNet, TensorNet) operate directly on crystal graphs, encoding atomic environments and bond geometries. By comparing these approaches under identical evaluation conditions -- compositional group splitting, consistent metrics, shared data -- we quantify the value of structural information for each target property.

The result is a reproducible, end-to-end pipeline from raw data ingestion through model training, evaluation, and interactive exploration via a Streamlit dashboard.

## Data Sources

The dataset is assembled from four complementary sources, each contributing different aspects of cathode characterization:

- **Materials Project** (mp-api): Primary source providing lithium cathode electrode entries with full crystal structures, formation energies, and computed voltages. Accessed via the official Python client.
- **OQMD** (REST API): Open Quantum Materials Database supplying supplementary DFT-computed properties for cross-validation and expanded coverage of the lithium oxide chemical space.
- **AFLOW** (REST API): Automatic Flow for Materials Discovery database providing DFT-computed formation energies and structural data for lithium-containing compounds. Accessed via the AFLOW REST API.
- **JARVIS** (REST API): Joint Automated Repository for Various Integrated Simulations providing DFT properties computed with OptB88vdW functional. Accessed via the JARVIS-Tools API.

Records are deduplicated using source priority (Materials Project > OQMD > AFLOW > JARVIS) to resolve conflicts when the same material appears in multiple databases. Compositional group splitting ensures that polymorphs (different structures of the same composition) never appear in both training and test sets, preventing data leakage.

## Methodology

### Model Architectures

**Random Forest (RF):** An ensemble of 100 decision trees trained on 132-dimensional Magpie composition descriptors. Magpie features encode elemental statistics (mean, deviation, range, etc.) of properties like electronegativity, atomic radius, and valence electrons. The ensemble approach provides robust predictions with built-in feature importance via Gini impurity.

**XGBoost (XGB):** Gradient-boosted trees with L1/L2 regularization operating on the same Magpie feature space. XGBoost applies sequential boosting with early stopping on a validation set, typically achieving lower bias than Random Forest at the cost of additional hyperparameter sensitivity.

**CGCNN (Crystal Graph Convolutional Neural Network):** Following Xie & Grossman (2018), crystal structures are represented as graphs where nodes are atoms and edges connect neighbors within a cutoff radius. Node features are one-hot element embeddings; edge features use Gaussian distance expansion. PyTorch Geometric CGConv layers propagate information through the graph, and a global mean pooling layer produces a fixed-length representation for property regression.

**M3GNet (Materials 3-body Graph Network):** An invariant GNN with 3-body interactions, fine-tuned from the pretrained M3GNet-MP-2018.6.1-Eform model via the matgl 2.x library. M3GNet captures many-body interactions through explicit three-body terms (bond angles), providing richer structural encoding than pairwise-only models. Fine-tuning from a model pre-trained on 60,000+ Materials Project entries provides strong inductive bias, especially beneficial when target datasets are small.

**TensorNet:** An O(3)-equivariant tensor network trained from scratch using the matgl 2.x library. TensorNet represents atomic interactions as Cartesian tensors, maintaining rotational equivariance without spherical harmonics. This architecture is particularly effective for properties sensitive to directional bonding environments.

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

### Best Model per Property

| Property | Best Model | MAE | R-squared |
|---|---|---|---|
| formation_energy_per_atom | CGCNN | 0.0341 | 0.9952 |
| voltage | XGBoost | 0.4336 | 0.6791 |
| capacity | CGCNN | 48.78 | 0.4652 |
| energy_above_hull | CGCNN | 0.0211 | 0.6903 |

### Full Model Comparison

#### Formation Energy per Atom

Train: 36,962 | Test: 4,731

| Model | MAE | RMSE | R-squared |
|---|---|---|---|
| RF | 0.0746 | 0.1393 | 0.9810 |
| XGBoost | 0.0713 | 0.1224 | 0.9853 |
| **CGCNN** | **0.0341** | **0.0697** | **0.9952** |
| M3GNet\* | 0.3210 | 0.4096 | 0.8358 |
| TensorNet | 5.5991 | 7.5781 | -55.2231 |

_\* Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

#### Voltage

Train: 2,934 | Test: 368

| Model | MAE | RMSE | R-squared |
|---|---|---|---|
| RF | 0.4514 | 0.6380 | 0.6529 |
| **XGBoost** | **0.4336** | **0.6135** | **0.6791** |
| CGCNN | 0.4921 | 0.7280 | 0.5482 |
| M3GNet\* | 6.2511 | 6.3554 | -33.4367 |
| TensorNet | 34.4207 | 39.2295 | -1311.0687 |

_\* Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

#### Capacity

Train: 2,934 | Test: 368

| Model | MAE | RMSE | R-squared |
|---|---|---|---|
| RF | 50.2189 | 68.2151 | 0.4302 |
| XGBoost | 49.2205 | 67.9244 | 0.4351 |
| **CGCNN** | **48.7821** | **66.0855** | **0.4652** |
| M3GNet\* | 162.6733 | 186.0789 | -3.2398 |
| TensorNet | 169.8472 | 192.2349 | -3.5250 |

_\* Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

#### Energy Above Hull

Train: 34,583 | Test: 4,405

| Model | MAE | RMSE | R-squared |
|---|---|---|---|
| RF | 0.0278 | 0.0683 | 0.3826 |
| XGBoost | 0.0296 | 0.0681 | 0.3858 |
| **CGCNN** | **0.0211** | **0.0484** | **0.6903** |
| M3GNet\* | 3.4989 | 3.6105 | -1724.6533 |
| TensorNet | 29.7300 | 42.5254 | -239399.4541 |

_\* Fine-tuned from pretrained M3GNet-MP-2018.6.1-Eform_

If the evaluation pipeline has been executed, a visual comparison is available:

![Model Comparison](data/results/figures/bar_comparison.png)

### Interpretation

Trained on 46,389 records from 4 data sources (Materials Project, OQMD, AFLOW, JARVIS), CGCNN is the strongest overall model, winning 3 of 4 target properties. Its advantage on formation energy (R-squared 0.9952 vs XGBoost's 0.9853) demonstrates that crystal structure information provides measurable lift for structure-sensitive properties, where bond geometries and atomic environments carry information not captured by elemental statistics alone.

For composition-dominated properties such as voltage and capacity, traditional ML baselines remain competitive or superior. XGBoost wins on voltage (R-squared 0.6791), confirming that elemental chemistry is the primary predictor for these properties.

**M3GNet underperformance:** M3GNet is fine-tuned from a pretrained formation-energy model (M3GNet-MP-2018.6.1-Eform). While this provides a reasonable starting point for formation energy prediction (R-squared 0.8358), it causes domain mismatch when applied to voltage, capacity, and stability targets. The pretrained weights are biased toward formation energy patterns, and the limited fine-tuning epochs typical of transfer learning are insufficient to overcome this bias for dissimilar target properties, resulting in predictions that are worse than predicting the mean (negative R-squared).

**TensorNet underperformance:** TensorNet is trained from scratch with no pretraining. This architecture requires substantially more training data and epochs to converge than are available in the current training budget. Without pretrained representations to bootstrap learning, the model fails to learn meaningful structure-property mappings, producing predictions with negative R-squared across all properties -- indicating outputs that are worse than a constant mean prediction.

## Dashboard

The project includes a 6-page interactive Streamlit dashboard for exploring data, models, and predictions.

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
| `m3gnet.yaml` | M3GNet fine-tuning settings and pre-trained model selection |
| `tensornet.yaml` | TensorNet architecture and training configuration |

### Running the Pipeline

The full pipeline runs all stages sequentially:

```bash
python -m cathode_ml
```

**Pipeline stages:**

1. **Fetch** -- Download and cache data from Materials Project, OQMD, AFLOW, and JARVIS
2. **Featurize** -- Composition descriptors and crystal graphs are computed inline during training
3. **Train** -- Train all five model architectures on each target property
4. **Evaluate** -- Generate comparison tables, bar charts, and learning curves

**Selective execution:**

```bash
python -m cathode_ml --skip-fetch          # Use cached data (skip download)
python -m cathode_ml --skip-train          # Use saved models (skip training)
python -m cathode_ml --models rf xgb       # Train only baseline models
python -m cathode_ml --models cgcnn m3gnet tensornet  # Train only GNN models
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
│   │   ├── aflow_fetcher.py     # AFLOW REST client
│   │   ├── jarvis_fetcher.py    # JARVIS-DFT client
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
│   │   ├── m3gnet.py            # M3GNet wrapper (matgl 2.x)
│   │   ├── train_m3gnet.py      # M3GNet fine-tuning orchestrator
│   │   ├── tensornet.py         # TensorNet wrapper (matgl 2.x)
│   │   ├── train_tensornet.py   # TensorNet training orchestrator
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
│   ├── m3gnet.yaml
│   └── tensornet.yaml
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
  url={https://github.com/Xeposem/lithium-cathode-ml}
}
```
