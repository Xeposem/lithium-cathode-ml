# Phase 3: CGCNN Implementation - Research

**Researched:** 2026-03-05
**Domain:** Crystal Graph Convolutional Neural Networks (PyTorch Geometric)
**Confidence:** HIGH

## Summary

Phase 3 implements CGCNN for cathode property prediction using PyTorch Geometric's `CGConv` layer, building a reusable GNN training infrastructure that MEGNet (Phase 4) will share. The existing codebase already provides graph construction (`features/graph.py` with `structure_to_graph()`), compositional splitting (`features/split.py`), and evaluation metrics (`models/baselines.py` with `evaluate_model()` and `save_results()`). The primary work is: (1) a CGCNN model class wrapping CGConv layers with pooling and FC head, (2) a generic GNNTrainer class for training/evaluation/artifact saving, (3) a YAML config for CGCNN hyperparameters, and (4) wiring it all together with the existing graph and splitting infrastructure.

The original CGCNN paper (Xie & Grossman, 2018) uses SGD with LR=0.01 and MultiStepLR, but the user has decided on Adam with LR~1e-3 and ReduceLROnPlateau, which is the more modern standard. PyG's CGConv layer has a built-in `batch_norm` parameter, which aligns with the user's decision for batch normalization after each conv layer.

**Primary recommendation:** Build a modular CGCNN model class using `torch_geometric.nn.conv.CGConv` with `global_mean_pool` readout, wrapped by a generic `GNNTrainer` that handles the train/val/eval loop, early stopping, LR scheduling, checkpoint saving, and CSV metric logging. Extract shared evaluation functions to `models/utils.py`.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Early stopping: patience-based on validation loss (e.g. patience=50)
- Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience~30)
- Device: auto-detect CUDA, fall back to CPU transparently (set once at start)
- Number of CGConv layers: configurable via YAML (default 3 per original paper)
- Hidden dimension: 128
- Batch normalization after each conv layer
- Loss function: MSE (standard for regression)
- Per-epoch metrics saved as CSV per property (columns: epoch, train_loss, val_loss, val_mae, lr)
- Checkpoints: save both best (lowest val loss) and final epoch model
- Checkpoint naming: cgcnn_{property}_best.pt, cgcnn_{property}_final.pt
- Final evaluation results: same JSON format as baselines (mae, rmse, r2, n_train, n_test)
- All artifacts stored under data/results/cgcnn/
- Generic GNNTrainer class that accepts any PyTorch model + DataLoader
- Trainer expects pre-built DataLoaders (not creating them internally)
- Separate config files per model: configs/cgcnn.yaml and configs/megnet.yaml
- Extract evaluate_model from baselines.py into shared models/utils.py

### Claude's Discretion
- Graph-level readout/pooling strategy (mean, mean+max, etc.)
- Sequential vs parallel training of per-property models
- Exact early stopping patience and scheduler patience values
- Batch size selection
- Number of fully-connected layers after pooling

### Deferred Ideas (OUT OF SCOPE)
None

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-01 | System implements CGCNN using PyTorch Geometric's CGConv | CGConv API fully documented; channels, dim, batch_norm params verified; original paper architecture mapped |
| MODL-05 | System predicts capacity, voltage, stability, and formation energy (separate models per property) | Existing pattern from baselines.py loops over target_properties; CGCNN uses same 4 properties from features.yaml |
| MODL-06 | System trains each model with architecture-appropriate hyperparameters | Original CGCNN paper defaults documented; user locked LR~1e-3, ReduceLROnPlateau, ~400 epochs; YAML config pattern established |
| MODL-07 | System stores model checkpoints and training artifacts as JSON/CSV | Checkpoint format (state_dict + metadata), CSV logging columns, and JSON evaluation format all specified in CONTEXT.md |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.1.0 | Deep learning framework | Already in requirements.txt |
| torch-geometric | >=2.6.0 | Graph neural network layers, DataLoader, pooling | Already in requirements.txt; provides CGConv |
| PyYAML | 6.0.2 | Config loading | Already in requirements.txt; established pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | >=1.3.0 | MAE, RMSE, R2 metrics | Already used in baselines; reuse in shared utils |
| numpy | 1.26.4 | Array operations | Already in requirements.txt |

### No New Dependencies Needed
All required libraries are already in `requirements.txt`. No new packages to install.

## Architecture Patterns

### Recommended Project Structure
```
cathode_ml/
  models/
    __init__.py          # existing
    baselines.py         # existing (remove evaluate_model, save_results -> import from utils)
    cgcnn.py             # NEW: CGCNN model class
    trainer.py           # NEW: Generic GNNTrainer class
    utils.py             # NEW: Shared evaluate_model, save_results, compute_metrics
configs/
  cgcnn.yaml             # NEW: CGCNN hyperparameters
  features.yaml          # existing (graph config consumed by CGCNN)
data/
  results/
    cgcnn/               # NEW: CGCNN artifacts directory
      {property}_metrics.csv
      cgcnn_{property}_best.pt
      cgcnn_{property}_final.pt
      cgcnn_results.json
```

### Pattern 1: CGCNN Model Class
**What:** PyTorch nn.Module wrapping CGConv layers, pooling, and FC head
**When to use:** For crystal graph -> property prediction
**Example:**
```python
# Source: PyG CGConv docs + original CGCNN paper architecture
import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool

class CGCNNModel(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128,
                 n_conv=3, n_fc=1, batch_norm=True):
        super().__init__()
        # Project input features to hidden dim
        self.embedding = nn.Linear(node_feature_dim, hidden_dim)

        # CGConv layers (channels=hidden_dim, dim=edge_feature_dim)
        self.convs = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_feature_dim,
                   batch_norm=batch_norm)
            for _ in range(n_conv)
        ])

        # FC head after pooling
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus()

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)  # CGConv has internal residual
        x = global_mean_pool(x, batch)  # Graph-level readout
        x = self.activation(self.fc(x))
        return self.out(x).squeeze(-1)
```

### Pattern 2: Generic GNNTrainer
**What:** Train loop with early stopping, LR scheduling, checkpointing, CSV logging
**When to use:** Any GNN model (CGCNN now, MEGNet later)
**Key design:**
```python
class GNNTrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 patience=50, results_dir="data/results/cgcnn"):
        # Trainer does NOT create DataLoaders (MEGNet may need different batching)
        ...

    def train_epoch(self, train_loader) -> float:
        # Returns train_loss
        ...

    def validate(self, val_loader) -> dict:
        # Returns {val_loss, val_mae}
        ...

    def fit(self, train_loader, val_loader, n_epochs) -> dict:
        # Main loop: train, validate, check early stop, save checkpoints
        # Returns training history
        ...

    def save_checkpoint(self, path, metadata):
        # Save model state_dict + optimizer state + metadata
        ...
```

### Pattern 3: Data Preparation Pipeline
**What:** Convert MaterialRecords to PyG DataLoader
**When to use:** Before training, transforms structures to graphs with targets
**Key detail:** The existing `structure_to_graph()` returns Data objects WITHOUT targets. Target values (y) must be attached per property before creating DataLoader.
```python
from torch_geometric.loader import DataLoader

# For each property, create Data objects with y attribute
graphs = []
for record in records:
    structure = Structure.from_dict(record.structure_dict)
    data = structure_to_graph(structure, features_config)
    data.y = torch.tensor(getattr(record, property_name), dtype=torch.float32)
    graphs.append(data)

loader = DataLoader(graphs, batch_size=64, shuffle=True)
```

### Pattern 4: Config-Driven Training
**What:** YAML config specifies all hyperparameters
**Follows:** Established pattern from configs/baselines.yaml, configs/features.yaml
```yaml
# configs/cgcnn.yaml
model:
  hidden_dim: 128
  n_conv: 3
  n_fc: 1
  batch_norm: true

training:
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  batch_size: 64
  n_epochs: 400
  early_stopping_patience: 50
  scheduler:
    type: reduce_on_plateau
    factor: 0.5
    patience: 30
    min_lr: 1.0e-6

results_dir: "data/results/cgcnn"
```

### Anti-Patterns to Avoid
- **Multi-output model:** User decided separate models per property, matching baselines pattern. Do NOT build a shared encoder with multiple heads.
- **DataLoader inside Trainer:** Trainer accepts pre-built DataLoaders. MEGNet (Phase 4) may need matgl-specific batching, so Trainer must not assume PyG DataLoader construction.
- **Hardcoded hyperparameters:** All values from YAML config, never magic numbers in code.
- **Creating graphs during training:** Graph construction is expensive. Pre-compute all graphs once, then index into them per property/split.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Crystal graph convolution | Custom message passing | `torch_geometric.nn.CGConv` | Handles residual connections, batch norm, aggregation correctly |
| Graph batching | Manual padding/collation | `torch_geometric.loader.DataLoader` | Automatically creates batch vector for variable-size graphs |
| Graph-level pooling | Manual scatter operations | `torch_geometric.nn.global_mean_pool` | Handles batch vector correctly, numerically stable |
| Evaluation metrics | Custom MAE/RMSE/R2 | `sklearn.metrics` (via shared utils.py) | Consistent with baselines, handles edge cases |
| LR scheduling | Custom decay logic | `torch.optim.lr_scheduler.ReduceLROnPlateau` | Well-tested, configurable, monitors validation loss |

**Key insight:** PyG's CGConv already implements the full CGCNN convolution including sigmoid/softplus gating, residual connections, and optional batch normalization. The model class just stacks these layers with pooling and an FC head.

## Common Pitfalls

### Pitfall 1: Forgetting `data.batch` in Forward Pass
**What goes wrong:** `global_mean_pool(x, batch)` requires the `batch` tensor that maps each node to its graph. If `batch` is None (single graph without DataLoader), pooling fails.
**Why it happens:** PyG DataLoader auto-creates `batch` attribute when batching, but a single Data object has no `batch`.
**How to avoid:** Always use PyG DataLoader (even with batch_size=1). The DataLoader creates the batch tensor automatically.
**Warning signs:** `RuntimeError` or incorrect output shapes during inference on single graphs.

### Pitfall 2: CGConv Channel Mismatch
**What goes wrong:** CGConv expects input node features to match `channels` parameter. If the embedding linear layer output doesn't match, silent shape errors or crashes.
**Why it happens:** Node features from `structure_to_graph()` are one-hot (dim=100), but CGConv channels should be the hidden dim (128).
**How to avoid:** Always project through `nn.Linear(node_feature_dim, hidden_dim)` before the first CGConv layer.
**Warning signs:** Shape mismatch errors on first forward pass.

### Pitfall 3: Not Setting `model.eval()` During Validation
**What goes wrong:** Batch normalization uses running statistics in eval mode, training statistics in train mode. Metrics will be noisy if model stays in train mode during validation.
**Why it happens:** Forgetting `model.eval()` / `model.train()` toggle.
**How to avoid:** Trainer.validate() must call `model.eval()` with `torch.no_grad()`, then `model.train()` after.

### Pitfall 4: GPU/CPU Tensor Mixing
**What goes wrong:** Data on CPU, model on GPU (or vice versa) causes RuntimeError.
**Why it happens:** Forgetting to move batch data to device in training loop.
**How to avoid:** In training loop, `batch = batch.to(device)` before forward pass. Set device once at start.
**Warning signs:** "Expected all tensors to be on the same device" error.

### Pitfall 5: Structures Without Valid Graphs
**What goes wrong:** Some crystal structures may produce empty graphs (no edges within cutoff radius), causing NaN losses.
**Why it happens:** Unusual structures, very small unit cells, or extreme lattice parameters.
**How to avoid:** Filter out invalid graphs using existing `validate_graph()` before creating DataLoader. Log how many are skipped.
**Warning signs:** NaN loss values, sudden spikes in training loss.

### Pitfall 6: Checkpoint Saving Without Metadata
**What goes wrong:** Model checkpoint loads but can't be used because architecture hyperparameters are unknown.
**Why it happens:** Saving only `state_dict` without recording the config that created the model.
**How to avoid:** Save `{model_state_dict, optimizer_state_dict, epoch, val_loss, config}` in checkpoint.

## Code Examples

### CGConv Layer Usage (Verified from PyG Docs)
```python
# Source: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CGConv.html
from torch_geometric.nn import CGConv

# channels: hidden dim of node features
# dim: dimensionality of edge features (80 Gaussians in this project)
# batch_norm: True per user decision
conv = CGConv(channels=128, dim=80, batch_norm=True)

# Forward: x=(N, 128), edge_index=(2, E), edge_attr=(E, 80)
# Output: x_updated=(N, 128) -- CGConv includes residual connection
x_out = conv(x, edge_index, edge_attr)
```

### PyG DataLoader Batching (Verified from PyG Docs)
```python
# Source: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
from torch_geometric.loader import DataLoader

# List of Data objects, each with x, edge_index, edge_attr, y
data_list = [data1, data2, data3, ...]
loader = DataLoader(data_list, batch_size=64, shuffle=True)

for batch in loader:
    # batch.x: all nodes concatenated
    # batch.edge_index: all edges (indices offset per graph)
    # batch.edge_attr: all edge features
    # batch.batch: maps each node to its graph index
    # batch.y: concatenated targets
    out = model(batch)
    loss = F.mse_loss(out, batch.y)
```

### ReduceLROnPlateau Usage
```python
# Source: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6
)

# After each epoch's validation
scheduler.step(val_loss)  # Must pass the monitored metric
current_lr = optimizer.param_groups[0]['lr']
```

### Extracting evaluate_model to Shared Utils
```python
# models/utils.py - extracted from baselines.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
from pathlib import Path

def compute_metrics(y_true, y_pred, n_train):
    """Compute MAE, RMSE, R2 metrics -- used by both baselines and GNNs."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
    }

def save_results(results, path):
    """Save results dict to JSON. Creates parent dirs."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
```

## Discretion Recommendations

### Graph-level readout: `global_mean_pool` (RECOMMENDED)
**Rationale:** The original CGCNN paper uses mean pooling. Mean pooling is size-invariant (crystal unit cells vary in atom count) and is the standard for material property prediction. Concatenating mean+max adds parameters with unclear benefit for this dataset size.

### Training order: Sequential per-property (RECOMMENDED)
**Rationale:** Sequential training is simpler, uses less memory, and matches the baselines pattern (loop over target_properties). Parallel would require multi-GPU or complex scheduling with no real benefit at this scale.

### Early stopping patience: 50 epochs (RECOMMENDED)
**Rationale:** With 400 max epochs and ReduceLROnPlateau patience=30, the LR will be reduced first, giving the model a chance to improve at lower LR before stopping. Patience=50 gives roughly one LR reduction cycle before stopping.

### Scheduler patience: 30 epochs (RECOMMENDED)
**Rationale:** Balances between reducing too aggressively (losing convergence) and waiting too long (wasting compute). Standard value for ReduceLROnPlateau in materials science GNNs.

### Batch size: 64 (RECOMMENDED)
**Rationale:** Original CGCNN used 256 on ~40K structures from full Materials Project. This project has a smaller cathode-specific dataset (likely hundreds to low thousands). Batch size 64 is appropriate for smaller datasets and fits comfortably in GPU memory.

### FC layers after pooling: 1 hidden layer (RECOMMENDED)
**Rationale:** Original CGCNN uses 1 FC hidden layer (n_h=1) with h_fea_len=128 followed by softplus, then output layer. This is sufficient for regression; deeper FC heads risk overfitting on small datasets.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom CGCNN message passing | PyG CGConv layer | PyG 1.x+ (2019+) | No need to implement custom convolution |
| `torch_geometric.data.DataLoader` | `torch_geometric.loader.DataLoader` | PyG 2.0 (2022) | Import path changed |
| SGD + MultiStepLR (original paper) | Adam + ReduceLROnPlateau | Community practice ~2020+ | Faster convergence, adaptive LR |
| atom_fea_len=64 (original paper) | hidden_dim=128 | User decision | More expressive representation |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | tests/conftest.py (exists) |
| Quick run command | `pytest tests/test_cgcnn.py tests/test_trainer.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-01 | CGCNN model forward pass produces correct output shape | unit | `pytest tests/test_cgcnn.py::test_cgcnn_forward -x` | No - Wave 0 |
| MODL-01 | CGConv layers used with correct channels/dim | unit | `pytest tests/test_cgcnn.py::test_cgcnn_architecture -x` | No - Wave 0 |
| MODL-05 | Separate model trained per property | integration | `pytest tests/test_trainer.py::test_per_property_training -x` | No - Wave 0 |
| MODL-06 | Config loaded from YAML, not hardcoded | unit | `pytest tests/test_cgcnn.py::test_config_driven -x` | No - Wave 0 |
| MODL-07 | Checkpoints saved (best + final) | integration | `pytest tests/test_trainer.py::test_checkpoint_saving -x` | No - Wave 0 |
| MODL-07 | CSV metrics logged per epoch | integration | `pytest tests/test_trainer.py::test_csv_logging -x` | No - Wave 0 |
| MODL-07 | JSON results in baseline-compatible format | unit | `pytest tests/test_trainer.py::test_results_json_format -x` | No - Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_cgcnn.py tests/test_trainer.py tests/test_model_utils.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_cgcnn.py` -- covers MODL-01, MODL-06 (model architecture, forward pass, config)
- [ ] `tests/test_trainer.py` -- covers MODL-05, MODL-07 (training loop, checkpoints, CSV, JSON)
- [ ] `tests/test_model_utils.py` -- covers shared utils (compute_metrics, save_results)
- [ ] Fixtures in `tests/conftest.py` -- add sample graph Data with y target, minimal cgcnn config dict

## Open Questions

1. **Dataset size for cathode-specific data**
   - What we know: Full Materials Project has ~150K structures. Cathode-specific subset is filtered.
   - What's unclear: Exact number of valid records with structures after Phase 1/2 cleaning.
   - Recommendation: Log dataset sizes per property. If fewer than 100, consider reducing batch size to 32 and max epochs.

2. **Transfer learning from full MP**
   - What we know: STATE.md flags "Transfer learning from full MP (~150K entries) needs strategy research."
   - What's unclear: Whether pre-training on full MP then fine-tuning on cathodes would help.
   - Recommendation: OUT OF SCOPE for Phase 3 (also listed as v2/ADV scope). Build clean architecture that could support it later.

## Sources

### Primary (HIGH confidence)
- [PyG CGConv API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CGConv.html) - constructor params, forward signature, batch_norm
- [PyG CGConv source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cg_conv.html) - internal implementation, residual connection confirmed
- [PyG DataLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html) - batching mechanism for variable-size graphs
- [Original CGCNN GitHub](https://github.com/txie-93/cgcnn) - reference architecture (3 conv, mean pool, 1 FC, softplus)
- [Original CGCNN paper](https://arxiv.org/abs/1710.10324) - Xie & Grossman 2018, architecture and hyperparameters
- [PyTorch ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) - scheduler API
- Existing codebase: `features/graph.py`, `models/baselines.py`, `features/split.py`, `config.py` - established patterns

### Secondary (MEDIUM confidence)
- [PyG global_mean_pool](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html) - pooling API

### Tertiary (LOW confidence)
- None -- all findings verified against official sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in requirements.txt, APIs verified from official docs
- Architecture: HIGH - CGConv API verified, original paper architecture confirmed from reference implementation
- Pitfalls: HIGH - based on established PyG patterns and verified API behavior
- Discretion recommendations: MEDIUM - based on original paper defaults and community practice, but optimal values are dataset-dependent

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain, PyG CGConv API unlikely to change)
