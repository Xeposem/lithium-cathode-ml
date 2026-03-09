# Phase 9 Research: Replace MEGNet with M3GNet and TensorNet from matgl 2.x

**Completed:** 2026-03-08
**Sources:** matgl GitHub repo (materialsvirtuallab/matgl main branch), matgl examples, debug investigation, existing codebase analysis

---

## 1. Context and Motivation

matgl v2.x (released Nov 2025) defaults to PyG backend and reorganized its model lineup. The project currently uses matgl v1.x MEGNet (DGL-only), but `requirements.txt` specifies `matgl>=1.1.0` which resolves to v2.x. MEGNet pretrained models still exist in matgl 2.x (`MEGNet-MP-2018.6.1-Eform`), and MEGNet is still exported when the DGL backend is active. However, the project should migrate to newer architectures (M3GNet, TensorNet) that represent the current state of the art and have active development.

**Key fact discovered during research:** MEGNet is NOT dropped in matgl 2.x. It is DGL-only but still available when `MATGL_BACKEND=DGL`. The debug investigation's premise that "matgl 2.0.6 dropped MEGNet entirely" is incorrect. However, migrating to M3GNet and TensorNet is still the right move because:
1. M3GNet is the successor architecture with 3-body interactions and better accuracy
2. TensorNet is O(3)-equivariant and faster
3. Both have active development; MEGNet does not
4. Adds model diversity for comparison (invariant vs equivariant GNNs)

---

## 2. matgl 2.x Architecture Overview

### Backend System
- Default backend: **PyG** (set via `MATGL_BACKEND` env var or `matgl.config.BACKEND`)
- DGL backend: Set `MATGL_BACKEND=DGL` before importing matgl
- Backend selection controls which model implementations are available:
  - **DGL only:** CHGNet, M3GNet, MEGNet, SO3Net, QET
  - **PyG only:** TensorNetWarp
  - **Both:** TensorNet (separate implementations: `_tensornet_dgl.py`, `_tensornet_pyg.py`)

### Model Availability by Backend
| Model | DGL | PyG | Pretrained Property Models |
|-------|-----|-----|---------------------------|
| M3GNet | Yes | No | M3GNet-MP-2018.6.1-Eform |
| TensorNet | Yes (TensorNetDGL) | Yes | PES only (no property prediction) |
| MEGNet | Yes | No | MEGNet-MP-2018.6.1-Eform |

### Critical Implication for This Project
- **M3GNet requires DGL backend** (`MATGL_BACKEND=DGL`)
- **TensorNet works with either backend** but PyG is default/preferred
- When loading M3GNet pretrained models with PyG backend configured, matgl **automatically switches to DGL** and warns. This is handled inside `load_model()`.
- Both DGL and PyG can coexist in the same Python environment

---

## 3. M3GNet: Detailed API and Usage

### Loading Pretrained Model
```python
import matgl
# load_model auto-detects DGL requirement for M3GNet
model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
```
The `load_model()` function reads `model.json` from the pretrained directory, identifies the model class, and calls its `.load()` method. For M3GNet, it automatically sets the DGL backend.

### Constructor Parameters (for training from scratch)
```python
from matgl.models import M3GNet

model = M3GNet(
    element_types=elem_list,       # List[str] from get_element_list()
    is_intensive=True,             # Per-atom property (True for formation energy)
    readout_type="set2set",        # "set2set", "weighted_atom", or "reduce_atom"
    # Defaults:
    dim_node_embedding=64,
    dim_edge_embedding=64,
    nblocks=3,
    units=64,
    ntargets=1,
    cutoff=5.0,
    threebody_cutoff=4.0,
    max_n=3, max_l=3,
    rbf_type="SphericalBessel",
    activation_type="swish",
    task_type="regression",
)
```

### Prediction (Inference)
```python
prediction = model.predict_structure(structure)  # pymatgen Structure -> float
```
Identical API to MEGNet. Drop-in replacement.

### Training Pipeline (from matgl examples)
```python
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph, split_dataset
from matgl.utils.training import ModelLightningModule
from functools import partial
import lightning as L

# 1. Prepare converter
elem_list = get_element_list(structures)
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

# 2. Create dataset (M3GNet needs include_line_graph=True for 3-body)
dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels={"eform": targets},
    include_line_graph=True,       # REQUIRED for M3GNet (3-body interactions)
)

# 3. Split and create loaders
train_data, val_data, test_data = split_dataset(dataset, frac_list=[0.8, 0.1, 0.1])
collate_fn = partial(collate_fn_graph, include_line_graph=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn,
    batch_size=128,
    num_workers=0,
)

# 4. Lightning module
lit_module = ModelLightningModule(model=model, include_line_graph=True)

# 5. Train
trainer = L.Trainer(max_epochs=1000, accelerator="auto", logger=CSVLogger(...))
trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

### Key Differences from Current MEGNet Code
| Aspect | MEGNet (current) | M3GNet (new) |
|--------|-----------------|--------------|
| `include_line_graph` | Not used | **Required = True** |
| `MGLDataset` | No `threebody_cutoff` | Needs `threebody_cutoff=4.0` |
| `collate_fn_graph` | No args | Needs `include_line_graph=True` via `partial` |
| `ModelLightningModule` | `model=model, lr=lr` | `model=model, include_line_graph=True` |
| Model construction | Always from pretrained | Can be from scratch or pretrained |
| `predict_structure()` | Same | Same |

### Pretrained Fine-Tuning vs Training from Scratch
The pretrained model `M3GNet-MP-2018.6.1-Eform` is trained on Materials Project formation energies. For the project:
- **Formation energy:** Fine-tune from pretrained (same target property)
- **Voltage, capacity, energy_above_hull:** Can still fine-tune from pretrained (transfer learning) or train from scratch

---

## 4. TensorNet: Detailed API and Usage

### No Pretrained Property Prediction Models
Available TensorNet pretrained models are all PES (potential energy surface):
- `TensorNet-MatPES-PBE-v2025.1-PES`
- `TensorNet-MatPES-r2SCAN-v2025.1-PES`
- `TensorNet-ANI-1x-Subset-PES`
- `TensorNetDGL-MatPES-PBE-v2025.1-PES`
- `TensorNetDGL-MatPES-r2SCAN-v2025.1-PES`

PES models predict energy/forces/stresses, not scalar properties like formation energy. TensorNet must be trained from scratch for property prediction.

### Constructor Parameters (PyG version)
```python
from matgl.models import TensorNet

model = TensorNet(
    element_types=elem_list,
    is_intensive=True,
    # Defaults:
    units=64,
    nblocks=2,
    num_rbf=32,
    rbf_type="Gaussian",
    cutoff=5.0,
    max_n=3, max_l=3,
    readout_type="weighted_atom",  # or "reduce_atom"
    task_type="regression",
    activation_type="swish",
    equivariance_invariance_group="O(3)",  # or "SO(3)"
    ntargets=1,
)
```

### Training Pipeline
TensorNet uses the same matgl training infrastructure but does NOT need line graphs (no 3-body interactions):

```python
# Converter
converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

# Dataset (no threebody_cutoff, no include_line_graph)
dataset = MGLDataset(
    structures=structures,
    converter=converter,
    labels={"eform": targets},
    include_line_graph=False,
)

# Loaders (no include_line_graph in collate_fn)
collate_fn = partial(collate_fn_graph, include_line_graph=False)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data, val_data=val_data, test_data=test_data,
    collate_fn=collate_fn, batch_size=128,
)

# Lightning (no include_line_graph)
lit_module = ModelLightningModule(model=model, include_line_graph=False)
trainer = L.Trainer(max_epochs=1000, accelerator="auto")
trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

### TensorNet PyG Data Requirements
The PyG TensorNet `forward()` expects:
- `graph.node_type`: Node type indices
- `graph.edge_index`: Edge connectivity
- `graph.pos`: Cartesian coordinates
- `graph.pbc_offset` / `graph.pbc_offshift`: Periodic boundary condition handling
- `graph.batch`: Batch assignment tensor

These are handled by `Structure2Graph` converter + `MGLDataset`; no manual preparation needed.

### Prediction
```python
prediction = model.predict_structure(structure)  # Same API as M3GNet
```

---

## 5. ModelLightningModule Details

The `ModelLightningModule` is backend-aware (separate implementations in `_training_dgl.py` and `_training_pyg.py`) but shares the same interface:

### Constructor
```python
ModelLightningModule(
    model=model,
    include_line_graph=False,    # True for M3GNet, False for TensorNet
    data_mean=0.0,               # For target normalization
    data_std=1.0,                # For target normalization
    loss="mse_loss",             # "mse_loss", "huber_loss", "smooth_l1_loss", "l1_loss"
    lr=0.001,                    # Learning rate
    decay_steps=1000,            # CosineAnnealingLR T_max
    decay_alpha=0.01,            # Minimum LR fraction
    sync_dist=False,             # Multi-GPU sync
)
```

### Loss and Metrics
- Training/validation steps log: `Total_Loss`, `MAE`, `RMSE`
- Loss function denormalizes predictions before computing (multiplies by `data_std`, adds `data_mean`)
- Default optimizer: Adam with CosineAnnealingLR scheduler

### Key Difference from Current Code
Current MEGNet code passes `lr=lr` to `ModelLightningModule`. In matgl 2.x, the parameter is still `lr` but scheduler is CosineAnnealingLR by default (not ReduceOnPlateau). The current project uses ReduceOnPlateau via Lightning callbacks, but matgl 2.x's `ModelLightningModule` has its own `configure_optimizers()` with CosineAnnealing built in.

**Decision needed:** Use matgl's built-in optimizer/scheduler, or override with project's ReduceOnPlateau pattern.

---

## 6. MGLDataLoader API (matgl 2.x)

The `MGLDataLoader` function signature changed in v2.x:
```python
# v2.x (keyword args, returns tuple)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn,
    batch_size=128,
    num_workers=0,
)
```

### vs Current Code (v1.x style):
```python
# v1.x (positional, separate calls)
train_loader = MGLDataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_graph)
val_loader = MGLDataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_graph)
```

**This is a breaking API change** that must be addressed in the migration.

---

## 7. Backend Coexistence Strategy

### How DGL + PyG Coexist
- Both packages can be installed simultaneously (`pip install dgl torch-geometric`)
- matgl uses `MATGL_BACKEND` env var (default: `"PYG"`)
- For M3GNet: matgl auto-switches to DGL when loading M3GNet pretrained model
- For TensorNet: Uses default PyG backend

### Recommended Approach
1. Keep both `dgl>=2.0.0` and `torch-geometric>=2.6.0` in `requirements.txt`
2. Set `MATGL_BACKEND` dynamically per-model:
   - For M3GNet training: Set `os.environ["MATGL_BACKEND"] = "DGL"` before matgl import
   - For TensorNet training: Use default PyG backend
3. Alternative: Let `load_model()` handle DGL switching automatically for M3GNet pretrained models (it already does this)

### Risk: Import Order
If matgl is imported once with PyG backend, then M3GNet models are loaded, matgl auto-switches. But if both M3GNet and TensorNet are trained in the same process, the backend switching may cause issues. **Mitigation:** Train M3GNet first (triggers DGL switch), then TensorNet (which has a DGL implementation too, so it still works).

---

## 8. Dependency Changes

### Current `requirements.txt`
```
matgl>=1.1.0    # Needs update
dgl>=2.0.0      # Keep (M3GNet needs it)
torch-geometric>=2.6.0  # Keep (CGCNN + TensorNet PyG)
```

### Required Changes
```
matgl>=2.0.0    # Pin to v2.x
dgl>=2.0.0      # Keep
torch-geometric>=2.6.0  # Keep
lightning>=2.0.0  # May need explicit pin (matgl uses it)
```

### Lightning Dependency
matgl 2.x uses `import lightning as L` (not `pytorch_lightning`). The current MEGNet code already uses this pattern. Confirm `lightning` package is installed (it's a dependency of matgl but should be explicit in `requirements.txt`).

---

## 9. Impact on Existing GNNTrainer

The existing `GNNTrainer` class (`cathode_ml/models/trainer.py`) is a PyG-native training loop used by CGCNN. It:
- Accepts any `nn.Module` + PyG `DataLoader`
- Handles early stopping, checkpointing, CSV logging
- Uses `ReduceLROnPlateau` scheduler

**M3GNet and TensorNet use matgl's Lightning-based training**, not `GNNTrainer`. This is the same pattern as the current MEGNet code (which also uses Lightning, not GNNTrainer).

**Decision:** Keep `GNNTrainer` for CGCNN. Use `ModelLightningModule` + Lightning `Trainer` for both M3GNet and TensorNet (matching current MEGNet pattern). The training orchestrators (`train_m3gnet.py`, `train_tensornet.py`) will follow the `train_megnet.py` pattern, not the `train_cgcnn.py` pattern.

---

## 10. File Change Summary

### Files to Create (8)
| File | Purpose |
|------|---------|
| `cathode_ml/models/m3gnet.py` | M3GNet model wrapper (load, predict, list models) |
| `cathode_ml/models/tensornet.py` | TensorNet model wrapper (construct, predict) |
| `cathode_ml/models/train_m3gnet.py` | M3GNet Lightning training orchestrator |
| `cathode_ml/models/train_tensornet.py` | TensorNet Lightning training orchestrator |
| `configs/m3gnet.yaml` | M3GNet config (pretrained model, training params) |
| `configs/tensornet.yaml` | TensorNet config (architecture params, training params) |
| `tests/test_m3gnet.py` | M3GNet unit tests |
| `tests/test_tensornet.py` | TensorNet unit tests |

### Files to Update (17)
| File | Change |
|------|--------|
| `cathode_ml/pipeline.py` | Add `"m3gnet"`, `"tensornet"` to `--models` choices; add train stages |
| `cathode_ml/evaluation/metrics.py` | Add m3gnet/tensornet to colors, labels, order, load_all_results; remove megnet |
| `cathode_ml/evaluation/plots.py` | Update GNN model list for learning curves grid |
| `cathode_ml/evaluation/__init__.py` | Update docstring |
| `dashboard/pages/overview.py` | Update model list text |
| `dashboard/pages/model_comparison.py` | Update gnn_models list |
| `dashboard/pages/predict.py` | Update MODEL_LABELS |
| `dashboard/utils/model_loader.py` | Add m3gnet/tensornet branches; remove megnet |
| `cathode_ml/features/graph.py` | Update docstring |
| `cathode_ml/models/utils.py` | Update docstring |
| `cathode_ml/models/trainer.py` | Update docstring |
| `tests/test_pipeline.py` | Update model choices, add m3gnet/tensornet mocks |
| `tests/test_evaluation.py` | Replace megnet samples with m3gnet/tensornet |
| `tests/test_plots.py` | Update gnn_models references |
| `tests/test_dashboard_predict.py` | Replace megnet loader test |
| `README.md` | Update model descriptions, CLI examples, file tree |
| `requirements.txt` | Pin `matgl>=2.0.0`, add `lightning>=2.0.0` |

### Files to Delete (4)
| File | Reason |
|------|--------|
| `cathode_ml/models/megnet.py` | Replaced by m3gnet.py + tensornet.py |
| `cathode_ml/models/train_megnet.py` | Replaced by train_m3gnet.py + train_tensornet.py |
| `configs/megnet.yaml` | Replaced by m3gnet.yaml + tensornet.yaml |
| `tests/test_megnet.py` | Replaced by test_m3gnet.py + test_tensornet.py |

---

## 11. Config Structure Design

### `configs/m3gnet.yaml`
```yaml
model:
  pretrained_model: "M3GNet-MP-2018.6.1-Eform"
  # Architecture params inherited from pretrained; override only if training from scratch:
  # dim_node_embedding: 64
  # dim_edge_embedding: 64
  # nblocks: 3
  # units: 64
  # cutoff: 5.0
  # threebody_cutoff: 4.0
  # readout_type: "set2set"
  # is_intensive: true

training:
  learning_rate: 0.001         # matgl default (higher than MEGNet's 0.0001)
  batch_size: 128
  n_epochs: 1000
  early_stopping_patience: 100
  loss: "mse_loss"             # "mse_loss", "huber_loss", "l1_loss"
  decay_steps: 1000            # CosineAnnealingLR T_max
  decay_alpha: 0.01            # Min LR fraction

results_dir: "data/results/m3gnet"
```

### `configs/tensornet.yaml`
```yaml
model:
  # No pretrained model -- train from scratch
  units: 64
  nblocks: 2
  num_rbf: 32
  cutoff: 5.0
  rbf_type: "Gaussian"
  readout_type: "weighted_atom"
  activation_type: "swish"
  equivariance_invariance_group: "O(3)"
  is_intensive: true
  ntargets: 1

training:
  learning_rate: 0.001
  batch_size: 128
  n_epochs: 1000
  early_stopping_patience: 100
  loss: "mse_loss"
  decay_steps: 1000
  decay_alpha: 0.01

results_dir: "data/results/tensornet"
```

---

## 12. Key Design Decisions Required

### D1: Backend Strategy
**Options:**
- (a) Always set `MATGL_BACKEND=DGL` for the entire project (both M3GNet and TensorNet use DGL versions)
- (b) Let matgl auto-switch backends per model (default PyG, auto-DGL for M3GNet)
- (c) Explicitly set backend before each model's training

**Recommendation:** Option (a) -- Always DGL. Simplifies code, avoids backend switching mid-process. TensorNet has a DGL implementation (`TensorNetDGL`) that works fine. CGCNN uses PyG DataLoader directly (doesn't go through matgl), so no conflict.

### D2: ModelLightningModule Optimizer
**Options:**
- (a) Use matgl's built-in CosineAnnealingLR (simpler, matches matgl examples)
- (b) Override with ReduceOnPlateau via Lightning callbacks (matches current project pattern)

**Recommendation:** Option (a) for both M3GNet and TensorNet. CosineAnnealing is the matgl default and well-tested. Reduces custom code.

### D3: Footnote/Label Convention
**Options:**
- (a) M3GNet gets dagger (pretrained), TensorNet does not
- (b) Both get distinct markers
- (c) No special markers

**Recommendation:** Option (a). M3GNet is fine-tuned from pretrained (like MEGNet was), so dagger is appropriate. TensorNet trains from scratch.

### D4: Pipeline Default Models
**Options:**
- (a) `["rf", "xgb", "cgcnn", "m3gnet", "tensornet"]` -- all five by default
- (b) `["rf", "xgb", "cgcnn", "m3gnet"]` -- TensorNet optional (slow from-scratch training)

**Recommendation:** Option (a). All five models for complete comparison. Users can `--models rf cgcnn m3gnet` to skip any.

### D5: Results Directory Structure
Keep parallel to current pattern:
```
data/results/
  baselines/
  cgcnn/
  m3gnet/       # was megnet/
  tensornet/    # new
  comparison/
  figures/
```

---

## 13. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| matgl 2.x MGLDataLoader API changed from v1.x | High | Use v2.x keyword-arg API per examples (researched above) |
| DGL/PyG backend switching within same process | Medium | Use DGL-only strategy (Decision D1 option a) |
| TensorNet from-scratch training needs more epochs/tuning | Medium | Start with matgl defaults, document expected longer training |
| `include_line_graph=True` forgotten for M3GNet | High | Enforce in code; test explicitly |
| `lightning` package version mismatch | Low | Pin in requirements.txt |
| `collate_fn_graph` requires `partial` wrapping in v2.x | Medium | Follow matgl example pattern exactly |
| Windows-specific DGL issues | Medium | Already using DGL in current project; test on Windows |

---

## 14. Implementation Order

Recommended plan structure:

1. **Plan 01: Core Model Wrappers + Configs** -- Create m3gnet.py, tensornet.py, m3gnet.yaml, tensornet.yaml. Delete megnet.py, megnet.yaml. Update requirements.txt.
2. **Plan 02: Training Orchestrators** -- Create train_m3gnet.py, train_tensornet.py. Delete train_megnet.py. Update pipeline.py.
3. **Plan 03: Evaluation + Dashboard + Docs** -- Update metrics.py, plots.py, all dashboard files, README.md. Remove all MEGNet references.
4. **Plan 04: Tests** -- Create test_m3gnet.py, test_tensornet.py. Delete test_megnet.py. Update test_pipeline.py, test_evaluation.py, test_plots.py, test_dashboard_predict.py.

---

## 15. Patterns to Preserve

From the existing codebase, these patterns must be maintained:
- **Lazy imports** for matgl/DGL (inside function bodies, not module-level)
- **Centralized `_import_matgl()` helper** with helpful error message
- **`predict_with_X(model, structures)` function** returning `List[float]`
- **Per-property sequential training loop** with `_set_seeds()` reset
- **Compositional group splitting** via `compositional_split()` / `get_group_keys()`
- **Output format:** `{property: {model_key: {mae, rmse, r2, n_train, n_test}}}`
- **Lightning CSV log conversion** to project-standard format
- **Checkpoint naming:** `{model}_{property}_best.pt`, `{model}_{property}_final.pt`
- **Config-driven construction:** `build_X_from_config()` pattern
- **Minimum 5 valid records per property** to train (skip otherwise)

---

*Research complete. Ready for planning.*
