# Architecture Research

**Domain:** ML-based battery cathode performance prediction
**Researched:** 2026-03-05
**Confidence:** HIGH (core patterns well-established in materials informatics literature)

## Standard Architecture

### System Overview

```
+------------------------------------------------------------------+
|                        WEB DASHBOARD (Dash)                       |
|  [Model Metrics] [Material Explorer] [Predictions] [Training Viz] |
+---------------------------+--------------------------------------+
                            |
                    reads artifacts
                            |
+---------------------------v--------------------------------------+
|                     ARTIFACTS STORE                               |
|  models/   results/   figures/   processed_data/   configs/      |
+--------^-----------^-----------^---------------------------------+
         |           |           |
   +-----+    +------+    +-----+
   |          |           |
+--+---+ +---+----+ +----+----+
|TRAIN | |EVALUATE| |VISUALIZE|
|      | |        | |         |
|PyTorch| |MAE,RMSE| |Plotly   |
|Light- | |R2,parity| |MatPlotl.|
|ning   | |plots   | |py3Dmol  |
+--^---+ +---^----+ +----^----+
   |          |           |
   +-----+----+-----+----+
         |          |
+--------+----------+------------------------------------------+
|                  DATASET LAYER                                |
|  PyG Dataset classes (InMemoryDataset)                        |
|  Crystal graphs: atoms=nodes, bonds=edges                     |
|  Matminer feature vectors (for sklearn baselines)             |
+---------------------------^----------------------------------+
                            |
                   graph construction
                   + featurization
                            |
+---------------------------+----------------------------------+
|                  DATA PIPELINE                                |
|  API ingestion -> cleaning -> structure validation ->         |
|  graph construction (pymatgen Structure -> PyG Data)          |
|  + tabular featurization (matminer -> pandas DataFrame)       |
+---------------------------^----------------------------------+
                            |
                    API calls + caching
                            |
+---------------------------+----------------------------------+
|                  DATA SOURCES                                 |
|  Materials Project (mp-api)  |  OQMD (qmpy_rester)           |
|  Battery Data Genome         |  Local cache (JSON/SQLite)     |
+--------------------------------------------------------------+
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Data Sources | Raw crystal structure + property data | mp-api MPRester, qmpy_rester, HTTP requests |
| Data Pipeline | Ingest, validate, clean, cache raw data | Python scripts with pymatgen Structure objects, JSON/SQLite cache |
| Featurization (GNN) | Crystal structure -> graph (atoms=nodes, bonds=edges) | matgl Structure2Graph or PyG from pymatgen, Gaussian bond expansion |
| Featurization (Tabular) | Crystal structure -> feature vectors for sklearn | matminer ElementProperty, composition/structure featurizers |
| Dataset Layer | Serve batched data to models | PyG InMemoryDataset for GNNs, pandas DataFrame for sklearn |
| Training | Train models with hyperparameter configs | PyTorch Lightning Trainer, sklearn fit() |
| Evaluation | Compute metrics, generate parity plots | MAE, RMSE, R2 per property; parity plots; learning curves |
| Artifacts Store | Persist models, metrics, figures | File system: models/, results/, figures/ |
| Web Dashboard | Interactive exploration of results and materials | Dash (Plotly) multi-page app |

## Recommended Project Structure

```
lithium-cathode-ml/
├── configs/                    # Experiment configuration files
│   ├── data.yaml               # API keys, data source params, filtering criteria
│   ├── cgcnn.yaml              # CGCNN hyperparameters
│   ├── megnet.yaml             # MEGNet hyperparameters
│   └── baseline.yaml           # sklearn model configs
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                   # Data pipeline
│   │   ├── __init__.py
│   │   ├── fetch.py            # API clients (MP, OQMD, BDG) with caching
│   │   ├── clean.py            # Validation, deduplication, filtering
│   │   ├── featurize.py        # Matminer-based tabular features
│   │   ├── graph.py            # Crystal structure -> PyG graph construction
│   │   └── dataset.py          # PyG InMemoryDataset + sklearn-ready DataFrames
│   │
│   ├── models/                 # Model definitions
│   │   ├── __init__.py
│   │   ├── cgcnn.py            # CGCNN (using PyG CGConv layers)
│   │   ├── megnet.py           # MEGNet (via matgl or custom PyG)
│   │   └── baselines.py        # RandomForest, GradientBoosting, etc.
│   │
│   ├── training/               # Training + evaluation logic
│   │   ├── __init__.py
│   │   ├── trainer.py          # PyTorch Lightning module wrapping GNNs
│   │   ├── evaluate.py         # Metrics computation (MAE, RMSE, R2)
│   │   └── utils.py            # Seeds, device setup, early stopping config
│   │
│   └── visualization/          # Plot generation
│       ├── __init__.py
│       ├── plots.py            # Parity plots, learning curves, bar charts
│       └── structure.py        # py3Dmol crystal structure rendering
│
├── dashboard/                  # Web dashboard (Dash app)
│   ├── app.py                  # Dash app entry point
│   ├── pages/
│   │   ├── overview.py         # Model comparison metrics
│   │   ├── explorer.py         # Materials database browser
│   │   ├── predict.py          # Interactive prediction panel
│   │   ├── training.py         # Training curves visualization
│   │   └── discovery.py        # Top candidate materials
│   ├── components/             # Reusable Dash components
│   └── assets/                 # CSS, static files
│
├── scripts/                    # CLI entry points
│   ├── fetch_data.py           # Run data pipeline
│   ├── train.py                # Train a model from config
│   ├── evaluate.py             # Evaluate saved model
│   └── run_dashboard.py        # Launch dashboard
│
├── data/                       # Data directory (gitignored except README)
│   ├── raw/                    # API responses (cached JSON)
│   ├── processed/              # Cleaned structures
│   └── graphs/                 # Serialized PyG datasets
│
├── artifacts/                  # Training outputs (gitignored except README)
│   ├── models/                 # Saved model checkpoints
│   ├── results/                # Metrics JSON/CSV
│   └── figures/                # Generated plots
│
├── notebooks/                  # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_comparison.ipynb
│
├── tests/                      # Unit + integration tests
│   ├── test_data/
│   ├── test_models/
│   └── test_pipeline/
│
├── pyproject.toml              # Project metadata + dependencies
├── requirements.txt            # Pinned dependencies for reproducibility
└── .env.example                # API key template (MP_API_KEY, etc.)
```

## Architectural Patterns

### Pattern 1: Two-Track Featurization

The system needs two parallel featurization paths because GNNs and sklearn baselines consume fundamentally different input formats.

**GNN Track:** pymatgen Structure -> graph (nodes = atoms with element embeddings, edges = bonds within cutoff radius with Gaussian distance expansion) -> PyG Data objects -> PyG DataLoader batches.

**Tabular Track:** pymatgen Structure -> matminer featurizers (ElementProperty, composition stats, structural descriptors) -> pandas DataFrame -> sklearn Pipeline with StandardScaler.

Both tracks share the same cleaned pymatgen Structure objects from the data pipeline, diverging only at featurization. This avoids duplication and ensures identical data splits across model types.

```python
# Shared clean structures
structures, targets = load_cleaned_data()
train_idx, val_idx, test_idx = split_indices(len(structures), seed=42)

# GNN track
graph_dataset = CathodeGraphDataset(structures, targets)
train_loader = DataLoader(graph_dataset[train_idx], batch_size=64)

# Tabular track
featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie"), ...])
X = featurizer.featurize_many(structures)
X_train, y_train = X[train_idx], targets[train_idx]
```

### Pattern 2: Config-Driven Experiments

Use YAML config files for all experiment parameters. This is essential for reproducibility in an academic project.

```yaml
# configs/cgcnn.yaml
model:
  type: cgcnn
  atom_fea_len: 64
  n_conv: 3
  n_h: 128
  classification: false
training:
  epochs: 300
  lr: 0.001
  batch_size: 64
  weight_decay: 0.0001
  scheduler: reduce_on_plateau
data:
  target_property: formation_energy_per_atom
  cutoff_radius: 8.0
  max_neighbors: 12
  train_ratio: 0.8
  val_ratio: 0.1
  seed: 42
```

### Pattern 3: PyTorch Lightning for Training

Wrap GNN models in a LightningModule to get logging, checkpointing, early stopping, and GPU management for free. MatGL already uses this pattern natively.

```python
class CathodePredictor(pl.LightningModule):
    def __init__(self, model, lr, scheduler_config):
        self.model = model  # CGCNN or MEGNet
        self.loss_fn = nn.L1Loss()  # MAE for regression

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss_fn(pred, batch.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        self.log("val_mae", F.l1_loss(pred, batch.y))
```

### Pattern 4: Artifact-Based Dashboard Decoupling

The dashboard reads saved artifacts (JSON metrics, CSV results, saved model checkpoints) rather than importing training code. This decouples the dashboard from the ML pipeline entirely.

**Why:** The dashboard can be developed and run independently. No GPU needed to view results. Multiple experiment results can be compared without re-running training.

```
Training pipeline writes:
  artifacts/results/cgcnn_formation_energy.json
  artifacts/results/megnet_formation_energy.json
  artifacts/results/baseline_rf_formation_energy.json
  artifacts/models/cgcnn_best.pt
  artifacts/figures/parity_cgcnn.png

Dashboard reads:
  glob("artifacts/results/*.json") -> comparison table
  load_model("artifacts/models/cgcnn_best.pt") -> interactive prediction
```

## Data Flow

### Stage 1: Data Acquisition

```
Materials Project API ----> raw/mp_cathodes.json
       (mp-api MPRester)         |
                                 |
OQMD API -----------------> raw/oqmd_cathodes.json
       (qmpy_rester)             |
                                 |
Battery Data Genome -------> raw/bdg_data.json
       (HTTP/download)           |
                                 v
                         MERGE + DEDUPLICATE
                                 |
                                 v
                    processed/cathode_structures.pkl
                    (list of pymatgen Structure + target dict)
```

**Key decisions at this stage:**
- Cache all API responses to avoid re-fetching (rate limits, reproducibility)
- Filter for lithium-containing cathode materials (element_set must contain Li)
- Merge on composition + space group to deduplicate across databases
- Store as pymatgen Structure objects (the lingua franca of materials informatics)

### Stage 2: Featurization + Graph Construction

```
cathode_structures.pkl
        |
        +---------> Graph Construction (for GNNs)
        |              pymatgen Structure -> PyG Data
        |              - Node features: element embedding (atomic number,
        |                electronegativity, radius, etc.)
        |              - Edge features: Gaussian-expanded interatomic distances
        |              - Cutoff radius: 8.0 A (standard for CGCNN)
        |              - Max neighbors: 12
        |              Output: data/graphs/cathode_dataset.pt
        |
        +---------> Tabular Features (for sklearn)
                       matminer featurizers -> DataFrame
                       - ElementProperty (magpie preset)
                       - Composition-based features
                       - Optional: structural descriptors
                       Output: data/processed/features.csv
```

### Stage 3: Training + Evaluation

```
For each (model_type, target_property, config):
    Load dataset -> split (train/val/test with fixed seed)
    Train model -> save best checkpoint
    Evaluate on test set -> compute MAE, RMSE, R2
    Generate parity plot
    Save all to artifacts/
```

### Stage 4: Dashboard Consumption

```
artifacts/results/*.json  -----> Comparison tables, bar charts
artifacts/figures/*.png   -----> Embedded plots
artifacts/models/*.pt     -----> Interactive prediction (load model, run inference)
data/processed/           -----> Materials explorer (browse, filter, search)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Pipeline Script

**What:** Single `main.py` that fetches data, featurizes, trains, evaluates, and plots in one long script.

**Why bad:** Cannot re-run evaluation without re-training. Cannot change one data source without re-fetching everything. Impossible to debug or test individual stages.

**Instead:** Separate CLI scripts for each stage (`fetch_data.py`, `train.py`, `evaluate.py`) that read from and write to the artifacts store. Each stage is independently runnable and testable.

### Anti-Pattern 2: Dashboard Importing Model Training Code

**What:** `from src.training.trainer import CathodePredictor` inside the dashboard.

**Why bad:** Dashboard now requires PyTorch, CUDA, and the full ML stack just to display results. Breaks if training code changes. Cannot display results from experiments run on different machines.

**Instead:** Dashboard reads JSON/CSV artifacts and loads saved model checkpoints with a thin inference wrapper. The dashboard only needs torch (CPU) for the prediction panel, not the full training apparatus.

### Anti-Pattern 3: Hardcoded Material Filters

**What:** Filtering logic like `if "Li" in composition and "O" in composition` scattered through code.

**Why bad:** Changing the scope of the study (e.g., adding Na-ion cathodes or sulfide cathodes) requires editing code in multiple places.

**Instead:** Define material filtering criteria in `configs/data.yaml`:
```yaml
filters:
  must_contain: [Li]
  property_range:
    formation_energy_per_atom: [-5.0, 0.5]
  exclude_elements: [He, Ne, Ar]
```

### Anti-Pattern 4: Mixing PyG and DGL

**What:** Using matgl (DGL-based, though migrating to PyG) for MEGNet and raw PyG for CGCNN.

**Why bad:** Two different graph data formats, two different DataLoader patterns, two sets of dependencies. DGL is no longer actively maintained.

**Instead:** Standardize on PyTorch Geometric. Implement CGCNN using PyG's native `CGConv` layer. For MEGNet, either use matgl v2.0+ (which defaults to PyG backend) or implement MEGNet's update scheme directly in PyG. This ensures one graph format, one DataLoader, and one set of batching utilities.

### Anti-Pattern 5: Per-Property Separate Pipelines

**What:** Completely separate data pipelines and training scripts for capacity, voltage, stability, and cycle life.

**Why bad:** Massive code duplication. Data processing bugs must be fixed in multiple places.

**Instead:** One unified pipeline parameterized by `target_property` in the config. The dataset class stores all properties; the training script selects which property to predict based on config. Multi-output models (predict all properties simultaneously) can be explored as an optimization later.

## Integration Points

### pymatgen as the Central Data Object

Every component communicates through pymatgen `Structure` objects. This is the standard in materials informatics and ensures interoperability:
- API clients return Structures (mp-api natively, OQMD via conversion)
- Graph constructors consume Structures
- Matminer featurizers consume Structures
- py3Dmol visualization consumes Structures (via CIF export)

### PyTorch Geometric as the Graph Framework

PyG `Data` objects are the standard graph representation:
- `data.x`: node features (atom embeddings) [num_atoms, atom_fea_len]
- `data.edge_index`: bond connectivity [2, num_bonds]
- `data.edge_attr`: bond features (Gaussian-expanded distances) [num_bonds, bond_fea_len]
- `data.y`: target property [1]

Both CGCNN and MEGNet models consume this format. PyG's `DataLoader` handles batching with automatic graph indexing.

### Dashboard Integration via Artifacts

The dashboard integrates with the ML pipeline exclusively through the filesystem:

| Artifact | Format | Dashboard Consumer |
|----------|--------|-------------------|
| Model metrics | JSON | Comparison tables, bar charts |
| Parity plot data | CSV | Interactive Plotly scatter plots |
| Training curves | CSV (epoch, train_loss, val_loss) | Line charts |
| Saved models | .pt checkpoint | Interactive prediction panel |
| Crystal structures | pymatgen Structure (pickle/JSON) | py3Dmol 3D viewer |
| Material database | CSV/Parquet | Explorer with filtering/search |

### Dashboard Framework: Dash (Plotly)

**Recommendation: Dash over Streamlit.** Rationale:

1. **Multi-page apps native:** Dash has first-class multi-page support matching the required dashboard panels (overview, explorer, predict, training, discovery).
2. **Callback system:** Cross-filtering between materials explorer and property plots requires Dash's callback architecture. Streamlit's re-run-everything model is clunky for linked views.
3. **Publication quality:** Dash IS Plotly -- direct access to all Plotly figure customization for paper-ready charts.
4. **py3Dmol embedding:** Both frameworks can embed py3Dmol via HTML iframes, but Dash's `dash_bio` and HTML component system handles this more cleanly.
5. **Academic use case fit:** The dashboard serves multiple linked panels with cross-filtering, which is Dash's strength. Streamlit is better for simpler single-page ML demos.

## Build Order (Dependency Chain)

The architecture has clear dependency layers. Build bottom-up:

```
Phase 1: Data Pipeline
  configs/data.yaml + src/data/fetch.py + src/data/clean.py
  WHY FIRST: Everything depends on data. Validate API access, caching,
  and data quality before any modeling.

Phase 2: Featurization + Dataset
  src/data/graph.py + src/data/featurize.py + src/data/dataset.py
  DEPENDS ON: Phase 1 (clean structures)
  WHY SECOND: Models cannot be trained without graph/feature data.

Phase 3: Baseline Models
  src/models/baselines.py + basic training/evaluation
  DEPENDS ON: Phase 2 (tabular features)
  WHY THIRD: Fast iteration, establishes performance floor. Sklearn
  models train in seconds, validating the entire pipeline end-to-end.

Phase 4: GNN Models (CGCNN + MEGNet)
  src/models/cgcnn.py + src/models/megnet.py + src/training/trainer.py
  DEPENDS ON: Phase 2 (graph datasets)
  WHY FOURTH: Core research contribution. Requires validated data pipeline.

Phase 5: Evaluation + Visualization
  src/training/evaluate.py + src/visualization/plots.py
  DEPENDS ON: Phase 3 + 4 (trained models)
  WHY FIFTH: Cross-model comparison, parity plots, publication figures.

Phase 6: Web Dashboard
  dashboard/*
  DEPENDS ON: Phase 5 (artifacts to display)
  WHY LAST: Reads artifacts from all prior phases. Can be developed
  in parallel with Phase 4/5 using mock data, but full integration
  requires completed artifacts.
```

**Parallelism opportunity:** Phase 3 (baselines) and Phase 4 (GNNs) can be developed in parallel once Phase 2 is complete, since they consume different featurization tracks.

## Scalability Considerations

| Concern | This Project (100s materials) | Scaled (10K+ materials) |
|---------|-------------------------------|------------------------|
| Data fetching | Sequential API calls, JSON cache | Paginated batch fetching, SQLite cache |
| Graph construction | In-memory, seconds | Pre-compute and serialize to disk |
| Training | Single GPU, minutes per model | Multi-GPU with Lightning, hours |
| Dashboard | Local Dash server | Same (reads static artifacts) |
| Storage | ~100MB total | ~10GB, use Parquet over CSV |

For this academic project, the 100s-materials scale is the realistic scope. The architecture supports scaling without restructuring because each stage reads/writes to the artifacts store independently.

## Key Technology Versions (as of 2026-03)

| Package | Recommended Version | Notes |
|---------|-------------------|-------|
| PyTorch | 2.x (latest stable) | CUDA 12.x compatible |
| PyTorch Geometric | 2.5+ | Native CGConv layer |
| matgl | 1.3+ (or 2.0+ when stable) | PyG backend for MEGNet |
| pymatgen | 2025.x+ | Core structure handling |
| mp-api | 0.41+ | Materials Project client |
| matminer | 0.9+ | Tabular featurization |
| scikit-learn | 1.4+ | Baseline models |
| pytorch-lightning | 2.x | Training orchestration |
| dash | 2.x | Web dashboard |
| plotly | 5.x | Visualization |
| py3Dmol | 2.x | Crystal structure viewer |

## Sources

- [CGCNN original repository (txie-93/cgcnn)](https://github.com/txie-93/cgcnn)
- [PyG CGConv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CGConv.html)
- [MatGL - Materials Graph Library](https://matgl.ai/)
- [MatGL GitHub (materialsvirtuallab/matgl)](https://github.com/materialsvirtuallab/matgl)
- [MatGL paper (npj Computational Materials 2025)](https://www.nature.com/articles/s41524-025-01742-y)
- [Materials Project API Getting Started](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started)
- [OQMD RESTful API documentation](https://static.oqmd.org/static/docs/restful.html)
- [qmpy_rester GitHub](https://github.com/mohanliu/qmpy_rester)
- [Matminer documentation](https://hackingmaterials.lbl.gov/matminer/)
- [Matminer featurizers](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.html)
- [Streamlit vs Dash comparison (2026)](https://docs.kanaries.net/topics/Streamlit/streamlit-vs-dash)
- [Dash vs Streamlit comparison (dasroot.net)](https://dasroot.net/posts/2025/12/building-python-dashboards-streamlit-vs/)

---
*Architecture research for: ML-based lithium-ion battery cathode performance prediction*
*Researched: 2026-03-05*
