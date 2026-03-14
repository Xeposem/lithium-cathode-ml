# Phase 2: Featurization and Baseline Models - Research

**Researched:** 2026-03-05
**Domain:** Crystal structure featurization, composition descriptors, tabular ML baselines, group-based data splitting
**Confidence:** HIGH

## Summary

Phase 2 converts cleaned MaterialRecord objects (from Phase 1) into two parallel feature representations: (1) PyTorch Geometric graph objects where atoms are nodes and bonds are edges with Gaussian distance expansion features, and (2) Magpie composition descriptor vectors via matminer for tabular ML baselines. It also implements compositional group splitting to prevent polymorph data leakage, then trains Random Forest and XGBoost baseline models to establish a performance floor.

The critical technical decisions are the graph construction from pymatgen Structures (using `get_all_neighbors` with a cutoff radius), the Gaussian basis expansion for edge features (matching CGCNN conventions), and using `reduced_formula` from pymatgen's Composition class as group keys for sklearn's `GroupShuffleSplit`. These graph objects will be consumed directly by CGCNN in Phase 3.

**Primary recommendation:** Build the graph converter using pymatgen's `Structure.get_all_neighbors(r=8.0)` for neighbor finding, one-hot atomic number encoding for node features, and Gaussian distance expansion (0-5 Angstrom, 80 filters) for edge features. Use matminer's `ElementProperty.from_preset("magpie")` for composition features. Split using `GroupShuffleSplit` on reduced formula. Train `RandomForestRegressor` and `XGBRegressor` per-property with results saved as JSON.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FEAT-01 | Crystal structures to PyG graph (atoms=nodes, bonds=edges) | Graph construction via pymatgen `get_all_neighbors` + PyG `Data` object; one-hot atomic number for node features |
| FEAT-02 | Gaussian distance expansion for edge features with configurable cutoff | Gaussian basis expansion class with configurable dmin/dmax/num_gaussians/cutoff_radius; standard CGCNN parameters |
| FEAT-03 | Magpie composition descriptors via matminer | `ElementProperty.from_preset("magpie")` producing 132-feature vectors; handle NaN columns |
| FEAT-04 | Compositional group splitting (not random) to prevent leakage | `GroupShuffleSplit` with `reduced_formula` as group key; train/val/test 80/10/10 |
| MODL-03 | Random Forest baseline with scikit-learn | `RandomForestRegressor` on Magpie features; per-property models |
| MODL-04 | XGBoost/GBM baseline | `XGBRegressor` on Magpie features; per-property models |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch-geometric | 2.7.0 | Graph neural network data structures (`Data`, `DataLoader`) | Industry-standard GNN library; provides `CGConv` layer for Phase 3 |
| torch | 2.2+ | Tensor operations, GPU support | Required by torch-geometric |
| matminer | 0.9.3 | Magpie composition descriptor generation | Standard materials informatics featurization toolkit |
| scikit-learn | 1.4+ | Random Forest, GroupShuffleSplit, metrics | Standard ML library; already widely used |
| xgboost | 2.1+ | XGBRegressor gradient boosting baseline | Standard tabular ML; sklearn-compatible API |
| pymatgen | 2025.10.7 | Structure parsing, neighbor finding, Composition | Already installed from Phase 1 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.26.4 | Array operations | Already installed |
| pandas | 2.2.3 | DataFrame for matminer featurizers | Already installed |
| joblib | (bundled) | Model serialization | Saving sklearn models |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pymatgen `get_all_neighbors` | PyG `radius_graph` | radius_graph does not handle periodic boundary conditions natively; pymatgen handles PBC correctly |
| matminer Magpie | Manual elemental stats | Magpie covers 132 descriptors with tested implementations; hand-rolling is error-prone |
| GroupShuffleSplit | Custom splitting | sklearn's implementation is well-tested and handles edge cases |

**Installation:**
```bash
pip install torch-geometric matminer xgboost
# torch-geometric requires companion packages:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

## Architecture Patterns

### Recommended Project Structure
```
cathode_ml/
  features/
    __init__.py
    graph.py          # FEAT-01, FEAT-02: Structure -> PyG Data
    composition.py    # FEAT-03: Formula -> Magpie vector
    split.py          # FEAT-04: Compositional group splitting
  models/
    __init__.py
    baselines.py      # MODL-03, MODL-04: RF and XGBoost
configs/
  features.yaml       # Cutoff radius, Gaussian params, split ratios
  baselines.yaml       # RF and XGBoost hyperparameters
data/
  processed/           # Saved PyG graphs, feature matrices
  results/             # Baseline JSON results
tests/
  test_graph.py
  test_composition.py
  test_split.py
  test_baselines.py
```

### Pattern 1: Crystal Structure to PyG Graph
**What:** Convert a pymatgen Structure to a torch_geometric Data object with atom nodes, bond edges, and Gaussian-expanded distance features.
**When to use:** Every crystal structure must pass through this before GNN training (Phase 3+) or graph validation.
**Example:**
```python
# Source: CGCNN paper + PyG Data docs
import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure

def structure_to_graph(structure: Structure, cutoff: float = 8.0,
                       max_neighbors: int = 12,
                       dmin: float = 0.0, dmax: float = 5.0,
                       num_gaussians: int = 80) -> Data:
    """Convert pymatgen Structure to PyG Data object."""
    # Node features: one-hot encode atomic number (up to 100 elements)
    atom_numbers = [site.specie.Z for site in structure]
    x = torch.zeros(len(atom_numbers), 100)
    for i, z in enumerate(atom_numbers):
        x[i, z] = 1.0

    # Find neighbors within cutoff using pymatgen (handles PBC)
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    src, dst, distances = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        # Sort by distance and take max_neighbors closest
        neighbors_sorted = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for neighbor in neighbors_sorted:
            src.append(i)
            dst.append(neighbor[2])  # index
            distances.append(neighbor[1])  # distance

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    distances = torch.tensor(distances, dtype=torch.float)

    # Gaussian distance expansion
    edge_attr = gaussian_expansion(distances, dmin, dmax, num_gaussians)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def gaussian_expansion(distances: torch.Tensor, dmin: float, dmax: float,
                       num_gaussians: int) -> torch.Tensor:
    """Expand distances into Gaussian basis functions."""
    centers = torch.linspace(dmin, dmax, num_gaussians)
    gamma = 1.0 / (centers[1] - centers[0])  # width parameter
    # Shape: (num_edges, num_gaussians)
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)
```

### Pattern 2: Magpie Composition Featurization
**What:** Convert a chemical formula string to a 132-dimensional Magpie descriptor vector.
**When to use:** For tabular ML baselines (Random Forest, XGBoost).
**Example:**
```python
# Source: matminer docs
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
import pandas as pd
import numpy as np

def featurize_compositions(formulas: list[str]) -> tuple[np.ndarray, list[str]]:
    """Generate Magpie feature matrix from formula strings."""
    featurizer = ElementProperty.from_preset("magpie")

    df = pd.DataFrame({"formula": formulas})
    df["composition"] = df["formula"].apply(Composition)

    # featurize returns DataFrame with 132 columns
    features_df = featurizer.featurize_dataframe(
        df, col_id="composition", ignore_errors=True
    )

    # Drop the formula/composition columns, keep only features
    feature_cols = featurizer.feature_labels()
    X = features_df[feature_cols].values

    # Handle NaN: fill with column median
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j]

    return X, feature_cols
```

### Pattern 3: Compositional Group Splitting
**What:** Split data by reduced formula groups so polymorphs stay together.
**When to use:** All train/val/test splits in this project.
**Example:**
```python
# Source: sklearn GroupShuffleSplit docs
from sklearn.model_selection import GroupShuffleSplit
from pymatgen.core import Composition

def get_group_keys(formulas: list[str]) -> list[str]:
    """Get reduced formula as group key for each entry."""
    return [Composition(f).reduced_formula for f in formulas]

def compositional_split(X, y, groups, test_size=0.1, val_size=0.1, seed=42):
    """Split into train/val/test with compositional grouping."""
    # First split: separate test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    # Second split: separate val from train
    val_frac = val_size / (1 - test_size)  # adjust fraction for remaining data
    groups_trainval = [groups[i] for i in trainval_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss_val.split(
        X[trainval_idx], y[trainval_idx], groups_trainval
    ))

    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    return train_idx, val_idx, test_idx
```

### Anti-Patterns to Avoid
- **Random splitting with crystal data:** Polymorphs of the same composition will leak between train/test, inflating metrics. Always use group-based splitting.
- **Building graphs without periodic boundary conditions:** Using simple Euclidean distance or PyG's `radius_graph` ignores periodic images. Must use pymatgen's neighbor finding which handles PBC.
- **Hardcoding Gaussian parameters:** dmin, dmax, num_gaussians, and cutoff_radius must come from YAML config. Phase 3 CGCNN training may need to tune these.
- **One model for all properties:** Per project decision (STATE.md), use separate models per property -- not multi-output regressors.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Composition descriptors | Manual elemental stats calculator | `matminer.ElementProperty.from_preset("magpie")` | 132 descriptors covering 6 statistics over 22 elemental properties; edge cases handled |
| Periodic neighbor finding | Euclidean distance matrix | `pymatgen.Structure.get_all_neighbors()` | Handles periodic boundary conditions, image atoms, lattice symmetry |
| Group-based splitting | Custom group assignment + split | `sklearn.model_selection.GroupShuffleSplit` | Well-tested, handles edge cases with small groups, deterministic with seed |
| Reduced formula normalization | String parsing of formulas | `pymatgen.Composition.reduced_formula` | Handles fractional compositions, ordering, normalization |
| Gaussian basis expansion | Custom math functions | Standalone utility class (small enough to implement) | Simple enough to write; matches CGCNN convention exactly |

**Key insight:** The featurization pipeline bridges materials science (pymatgen structures, matminer descriptors) and ML frameworks (PyG, sklearn). Using established materials science libraries for the science parts prevents subtle errors in periodic boundary handling, formula normalization, and descriptor computation.

## Common Pitfalls

### Pitfall 1: Disconnected Graphs from Too-Small Cutoff Radius
**What goes wrong:** If the cutoff radius is too small (e.g., 3 Angstrom), some structures produce graphs with isolated nodes (disconnected components), causing NaN gradients during GNN training.
**Why it happens:** Large unit cells with widely spaced atoms may have no neighbors within a small radius.
**How to avoid:** Use cutoff of 8.0 Angstrom (CGCNN default). Add a validation check: assert every node has at least one edge. Log warnings for structures that still produce disconnected graphs even at 8A.
**Warning signs:** `Data.num_edges == 0` or isolated nodes in the graph.

### Pitfall 2: Matminer NaN Features
**What goes wrong:** Some compositions produce NaN values in certain Magpie descriptors, especially for single-element materials or exotic elements.
**Why it happens:** Statistical features like "range" or "std" are undefined for single-element compositions; some elements lack data for certain properties.
**How to avoid:** Use `ignore_errors=True` in `featurize_dataframe`. After featurization, impute NaN with column medians. Drop any columns that are entirely NaN.
**Warning signs:** NaN values in feature matrix; XGBoost may handle NaN but Random Forest will fail silently or raise errors.

### Pitfall 3: Group Leakage via Non-Reduced Formulas
**What goes wrong:** Using `formula_pretty` or raw formula instead of `reduced_formula` as group keys. "Li2Co2O4" and "LiCoO2" would be treated as different groups despite being the same composition.
**Why it happens:** Different data sources may report the same material with different formula conventions.
**How to avoid:** Always normalize formulas through `Composition(formula).reduced_formula` before grouping.
**Warning signs:** Suspiciously high test metrics compared to literature; same compositions appearing in both train and test.

### Pitfall 4: PyG Data Object Missing Required Attributes
**What goes wrong:** PyG DataLoader or batching fails because Data objects lack consistent attributes across the dataset.
**Why it happens:** Different structures produce different numbers of nodes/edges, and if any attribute is missing on one Data object, batching breaks.
**How to avoid:** Ensure every Data object has `x`, `edge_index`, `edge_attr`, and `y` attributes. Use consistent dtypes. Add a `validate_graph` function that checks these.
**Warning signs:** Runtime errors during DataLoader iteration; shape mismatches in batching.

### Pitfall 5: Target Property Missing for Some Materials
**What goes wrong:** Not all materials have all target properties (voltage, capacity may be missing for non-battery materials).
**Why it happens:** OQMD materials have formation energy but no voltage/capacity. Phase 1 data has heterogeneous property coverage.
**How to avoid:** Train per-property models only on materials that have that property. Create separate train/val/test splits per property if needed, or use one split and filter out missing-property entries at training time.
**Warning signs:** NaN in target arrays; models trained on tiny subsets.

## Code Examples

### Loading Cleaned Data from Phase 1
```python
# Phase 1 produces MaterialRecord objects; load from cache
from cathode_ml.config import load_config
from cathode_ml.data.cache import DataCache
from cathode_ml.data.schemas import MaterialRecord

config = load_config("configs/data.yaml")
cache = DataCache(config["cache"]["directory"])

# Load cleaned records (produced by Phase 1 cleaning pipeline)
records = cache.load("cleaned_materials")
```

### Full Baseline Training Pipeline
```python
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def train_baseline(X_train, y_train, X_test, y_test, model_type="rf", seed=42):
    """Train a baseline model and return metrics dict."""
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1
        )
    elif model_type == "xgb":
        model = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return model, metrics

def save_results(results: dict, path: str):
    """Save baseline results as JSON artifact."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
```

### Graph Connectivity Validation
```python
def validate_graph(data) -> tuple[bool, str]:
    """Validate a PyG Data object for CGCNN readiness."""
    if data.x is None or data.x.shape[0] == 0:
        return False, "No node features"
    if data.edge_index is None or data.edge_index.shape[1] == 0:
        return False, "No edges (disconnected graph)"

    num_nodes = data.x.shape[0]
    # Check all nodes have at least one edge
    unique_nodes = torch.unique(data.edge_index)
    if len(unique_nodes) < num_nodes:
        return False, f"Disconnected: {num_nodes - len(unique_nodes)} isolated nodes"

    # Check edge_attr dimensions match edge count
    if data.edge_attr is not None:
        if data.edge_attr.shape[0] != data.edge_index.shape[1]:
            return False, "Edge attr count != edge count"

    return True, ""
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CIF file parsing per CGCNN original | pymatgen Structure objects (already in memory from Phase 1) | Project design | No file I/O overhead; structures already validated |
| One-hot for 92 elements | One-hot for atomic number (up to 100) | Common practice | Covers all expected elements including rare earth |
| Fixed 12 neighbors per atom | Configurable `max_neighbors` with cutoff | Refinement | Flexibility for different structure types |
| Manual train/test random split | GroupShuffleSplit on reduced formula | Best practice since ~2020 | Prevents polymorph leakage; more realistic generalization estimate |

**Deprecated/outdated:**
- CGCNN original code uses CIF file loading and custom PyTorch Dataset -- we use pymatgen structures from Phase 1 cache instead
- matminer 0.7.x had different API for some featurizers -- 0.9.3 is current stable

## Open Questions

1. **Target property availability per source**
   - What we know: MP has formation_energy and energy_above_hull; BDG has voltage and capacity; OQMD has formation energy but empty structure_dicts
   - What's unclear: Exact count of materials with each target property after Phase 1 cleaning
   - Recommendation: At featurization time, count available targets per property and log. Only materials with valid structures AND the target property can be used for that property's model.

2. **OQMD structures**
   - What we know: STATE.md notes "OQMD structure_dict is empty dict (REST API does not return full crystal structure)"
   - What's unclear: Whether OQMD materials can be used for graph features at all
   - Recommendation: OQMD materials can only be used for Magpie composition features (FEAT-03), not graph features (FEAT-01). Flag this in the splitting logic.

3. **Optimal max_neighbors setting**
   - What we know: CGCNN paper uses 12 neighbors; some implementations use all within cutoff
   - What's unclear: Whether capping at 12 vs. using all neighbors matters for cathode materials specifically
   - Recommendation: Default to 12 (matching CGCNN) but make configurable in YAML. Can be tuned in Phase 3.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | exists (implicit via pytest discovery) |
| Quick run command | `pytest tests/ -x --timeout=30` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-01 | Structure produces PyG Data with nodes, edges | unit | `pytest tests/test_graph.py::test_structure_to_graph -x` | No -- Wave 0 |
| FEAT-01 | Zero disconnected graphs (validation) | unit | `pytest tests/test_graph.py::test_no_disconnected_graphs -x` | No -- Wave 0 |
| FEAT-02 | Gaussian expansion produces correct shape | unit | `pytest tests/test_graph.py::test_gaussian_expansion -x` | No -- Wave 0 |
| FEAT-02 | Cutoff radius is configurable | unit | `pytest tests/test_graph.py::test_configurable_cutoff -x` | No -- Wave 0 |
| FEAT-03 | Magpie descriptors produce 132-dim vectors | unit | `pytest tests/test_composition.py::test_magpie_features -x` | No -- Wave 0 |
| FEAT-03 | NaN handling in Magpie output | unit | `pytest tests/test_composition.py::test_nan_handling -x` | No -- Wave 0 |
| FEAT-04 | Splits group by reduced formula | unit | `pytest tests/test_split.py::test_group_split_no_leakage -x` | No -- Wave 0 |
| FEAT-04 | No formula appears in both train and test | unit | `pytest tests/test_split.py::test_no_formula_overlap -x` | No -- Wave 0 |
| MODL-03 | RF produces MAE, RMSE, R2 on test data | integration | `pytest tests/test_baselines.py::test_random_forest -x` | No -- Wave 0 |
| MODL-04 | XGBoost produces MAE, RMSE, R2 on test data | integration | `pytest tests/test_baselines.py::test_xgboost -x` | No -- Wave 0 |
| MODL-03/04 | Results saved as JSON artifacts | unit | `pytest tests/test_baselines.py::test_results_json -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x --timeout=60`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_graph.py` -- covers FEAT-01, FEAT-02
- [ ] `tests/test_composition.py` -- covers FEAT-03
- [ ] `tests/test_split.py` -- covers FEAT-04
- [ ] `tests/test_baselines.py` -- covers MODL-03, MODL-04
- [ ] Update `tests/conftest.py` -- add fixtures for sample structures with known neighbor counts

## Sources

### Primary (HIGH confidence)
- [PyG Data object docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html) - Data structure, attributes, shapes
- [PyG CGConv docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CGConv.html) - Layer API, forward signature, edge feature requirements
- [matminer composition featurizers](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.composition.html) - ElementProperty, Magpie preset, feature labels
- [sklearn GroupKFold docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) - Group-based splitting API
- [pymatgen core docs](https://pymatgen.org/pymatgen.core.html) - Structure.get_all_neighbors, Composition.reduced_formula
- [CGCNN paper (arXiv:1710.10324)](https://arxiv.org/abs/1710.10324) - Original architecture, Gaussian expansion, graph construction
- [CGCNN reference implementation](https://github.com/txie-93/cgcnn) - atom_init.json, CIFData construction

### Secondary (MEDIUM confidence)
- [xgboost PyPI](https://pypi.org/project/xgboost/) - Version 2.1+, XGBRegressor API
- [matminer PyPI](https://pypi.org/project/matminer/) - Version 0.9.3 latest stable
- [torch-geometric PyPI](https://pypi.org/project/torch-geometric/) - Version 2.7.0, installation instructions
- [PyG radius_graph issue #1862](https://github.com/pyg-team/pytorch_geometric/issues/1862) - PBC limitations of radius_graph

### Tertiary (LOW confidence)
- None -- all key claims verified against official documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified on PyPI/official docs with current versions
- Architecture: HIGH - graph construction pattern well-established in CGCNN literature; composition featurization is matminer's primary use case
- Pitfalls: HIGH - disconnected graphs, NaN features, and polymorph leakage are well-documented issues in materials ML literature
- Splitting: HIGH - GroupShuffleSplit is the standard sklearn tool for this exact problem

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain; libraries mature)
