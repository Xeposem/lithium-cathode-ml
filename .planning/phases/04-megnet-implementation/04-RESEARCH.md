# Phase 4: MEGNet Implementation - Research

**Researched:** 2026-03-06
**Domain:** MEGNet graph neural network via matgl with DGL backend
**Confidence:** MEDIUM

## Summary

Phase 4 implements MEGNet via matgl for cathode property prediction (formation energy, voltage, stability, capacity) as separate per-property models. The core approach is to fine-tune pretrained MEGNet-MP-2019.4.1 weights using matgl's built-in Lightning trainer, then extract training logs into the project's standard artifact format (CSV metrics, JSON results, .pt checkpoints) matching CGCNN output.

The main technical challenge is environment isolation: matgl v1.3.0 requires DGL (which is no longer actively maintained), and this must coexist with or be isolated from the existing PyG-based CGCNN pipeline. The user has decided to try same-environment first, with Docker as fallback. matgl uses its own Structure2Graph converter and MEGNetDataset/MGLDataLoader stack -- none of the existing PyG DataLoaders or GNNTrainer can be used directly. The wrapper pattern (extracting Lightning CSVLogger output into project artifact format) is the key integration point.

**Primary recommendation:** Use matgl v1.3.0's native Lightning training stack (MEGNetDataset, MGLDataLoader, ModelLightningModule, CSVLogger) for all training, then write a thin post-training wrapper that reads Lightning's CSV logs and converts predictions to the project's JSON/CSV artifact format via the existing `compute_metrics()` and `save_results()` utilities.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use matgl's built-in Lightning trainer (not our GNNTrainer) -- matgl handles DGL batching and MEGNet-specific forward pass natively
- Write a wrapper to extract Lightning's training logs into our standard artifact format
- Produce same artifact types as CGCNN: per-epoch CSV metrics, JSON results, .pt checkpoints
- Checkpoint naming: megnet_{property}_best.pt, megnet_{property}_final.pt in data/results/megnet/
- CLI entry point: `python -m cathode_ml.models.train_megnet` with --seed flag (same pattern as CGCNN)
- Fine-tune from matgl's pretrained MEGNet-MP-2019.4.1 (not training from scratch)
- Full fine-tuning -- all layers unfrozen, trained with lower LR (~1e-4)
- Phase 5 comparison must clearly note that MEGNet uses pretrained weights while CGCNN trains from scratch
- Try installing matgl+DGL in the same environment as PyG first; fallback to Docker container
- Lazy imports for matgl/DGL inside train_megnet functions (consistent with existing xgboost pattern)
- Use matgl's own Structure2Graph converters for DGL graph construction
- Use matgl's default cutoff radius (from pretrained model), not CGCNN's 8.0 A
- Load input data from data/processed/materials.json (same source as CGCNN)
- Compositional splitting uses same folds as CGCNN and baselines

### Claude's Discretion
- matgl Lightning trainer configuration details (callbacks, logging backend)
- Exact wrapper implementation for extracting Lightning logs to CSV/JSON
- Batch size tuning for MEGNet
- How to handle matgl's state features (global attributes)
- Docker container configuration if needed for fallback

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-02 | System implements MEGNet via matgl with proper architecture matching | matgl v1.3.0 provides MEGNet class with pretrained weights (MEGNet-MP-2019.4.1), Structure2Graph converter, MEGNetDataset, MGLDataLoader, ModelLightningModule for training, and CSVLogger for metrics. Fine-tuning from pretrained weights with all layers unfrozen at LR ~1e-4 is the standard approach. |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matgl | 1.3.0 | MEGNet implementation + pretrained weights | Official graph deep learning library for materials; provides MEGNet, pretrained models, Lightning integration. v1.3.0 is last DGL-based release before v2 PyG migration. |
| dgl | 2.2.0 | Graph neural network backend for matgl | Required DGL version for matgl v1.3.0. DGL is no longer actively maintained but stable at this version. |
| pytorch-lightning / lightning | >=2.0 | Training framework | matgl's ModelLightningModule and training tutorials use Lightning natively. |
| torch | 2.3.0 | Deep learning framework | Recommended PyTorch version for DGL 2.2.0 compatibility. Must match existing project torch version. |
| numpy | <2.0 | Numerical ops | matgl+DGL require numpy<2 for compatibility. Project already pins 1.26.4. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pymatgen | (existing) | Structure objects for Structure2Graph input | Already installed; matgl's converter takes pymatgen Structure directly |
| torchdata | <=0.8.0 | Data loading utilities for DGL | Required by matgl/DGL for data pipeline compatibility |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matgl v1.3.0 | matgl v2.0.x | v2 defaults to PyG but MEGNet is NOT yet ported to PyG in v2 -- would still need DGL backend with extra config. v1.3.0 is simpler. |
| Lightning trainer | GNNTrainer | GNNTrainer expects PyG DataLoaders, not DGL. Would require rewriting batch handling. matgl's Lightning integration is tested and correct. |

**Installation (attempt same-env first):**
```bash
pip install "numpy<2"
pip install dgl==2.2.0
pip install torch==2.3.0
pip install "torchdata<=0.8.0"
pip install matgl==1.3.0
```

**If conflicts arise (Docker fallback):**
```dockerfile
FROM python:3.11-slim
RUN pip install matgl==1.3.0 dgl==2.2.0 torch==2.3.0 "numpy<2" "torchdata<=0.8.0" pymatgen
COPY cathode_ml/ /app/cathode_ml/
COPY configs/ /app/configs/
COPY data/processed/materials.json /app/data/processed/
WORKDIR /app
ENTRYPOINT ["python", "-m", "cathode_ml.models.train_megnet"]
```

## Architecture Patterns

### Recommended Project Structure
```
cathode_ml/
  models/
    megnet.py           # MEGNet wrapper: load pretrained, build model, predict
    train_megnet.py     # Training orchestrator (mirrors train_cgcnn.py pattern)
configs/
  megnet.yaml           # MEGNet-specific hyperparameters
data/
  results/
    megnet/             # Output: checkpoints, CSV metrics, JSON results
```

### Pattern 1: matgl Lightning Training Pipeline
**What:** Use matgl's native MEGNetDataset + MGLDataLoader + ModelLightningModule + Lightning Trainer for training, then post-process logs.
**When to use:** Always for MEGNet -- matgl handles DGL batching, graph construction, and MEGNet-specific forward pass natively.
**Example:**
```python
# Source: matgl.ai MEGNet tutorial
import matgl
from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list
from matgl.graph._data_dgl import MGLDataLoader, MGLDataset, collate_fn_graph
from matgl.utils.training import ModelLightningModule
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Load pretrained model
model = matgl.load_model("MEGNet-MP-2019.4.1-Eform")  # or relevant variant

# Setup converter -- use element list from dataset
elem_list = get_element_list(structures)
converter = Structure2Graph(element_types=elem_list, cutoff=model.cutoff)

# Build dataset
dataset = MGLDataset(
    structures=structures,
    labels={"Eform": targets},
    converter=converter,
)

# Split and create data loaders
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_graph,
    batch_size=128,
    num_workers=0,
)

# Wrap in Lightning module
lit_module = ModelLightningModule(model=model)

# Configure trainer with logging
logger = CSVLogger("data/results/megnet", name=f"megnet_{property_name}")
checkpoint_cb = ModelCheckpoint(
    dirpath="data/results/megnet",
    filename=f"megnet_{property_name}_best",
    monitor="val_MAE",
    mode="min",
    save_top_k=1,
)
trainer = L.Trainer(
    max_epochs=1000,
    accelerator="auto",
    logger=logger,
    callbacks=[checkpoint_cb],
)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

### Pattern 2: Post-Training Artifact Conversion
**What:** After Lightning training completes, read its CSV logs and convert to the project's standard format.
**When to use:** After every property training run, to produce artifacts matching CGCNN output.
**Example:**
```python
import pandas as pd
from cathode_ml.models.utils import compute_metrics, save_results

def convert_lightning_logs(log_dir: str, output_csv: str) -> None:
    """Convert Lightning CSVLogger output to project standard CSV format."""
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    # Lightning logs train and val on separate rows; merge by epoch
    train_metrics = metrics[["epoch", "train_loss", "train_MAE"]].dropna()
    val_metrics = metrics[["epoch", "val_loss", "val_MAE"]].dropna()
    merged = pd.merge(train_metrics, val_metrics, on="epoch", how="outer")

    # Rename to match project standard columns
    merged = merged.rename(columns={
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        "val_MAE": "val_mae",
        "train_MAE": "train_mae",
    })
    merged.to_csv(output_csv, index=False)
```

### Pattern 3: Lazy Import Isolation
**What:** Import matgl/DGL only inside MEGNet-specific functions, never at module top level.
**When to use:** In megnet.py and train_megnet.py to keep the rest of the package functional without matgl installed.
**Example:**
```python
# Source: existing pattern from cathode_ml/models/baselines.py (xgboost)
def load_megnet_model(model_name: str):
    """Load pretrained MEGNet model with lazy matgl import."""
    try:
        import matgl
    except ImportError:
        raise ImportError(
            "matgl is required for MEGNet training. "
            "Install with: pip install matgl==1.3.0 dgl==2.2.0"
        )
    return matgl.load_model(model_name)
```

### Pattern 4: Same Compositional Splits
**What:** Use identical split logic (same seed, same compositional_split function) to ensure MEGNet and CGCNN train/val/test on the exact same data partitions.
**When to use:** Always -- this is a hard requirement for fair comparison.
**Example:**
```python
# Reuse existing split infrastructure
from cathode_ml.features.split import compositional_split, get_group_keys

formulas = [r.formula for r in valid_records]
groups = get_group_keys(formulas)
train_idx, val_idx, test_idx = compositional_split(
    n_samples=len(valid_records),
    groups=groups,
    test_size=0.1,
    val_size=0.1,
    seed=seed,  # Same seed as CGCNN
)
# Then use these indices to partition structures and targets
# before constructing MEGNetDataset
```

### Anti-Patterns to Avoid
- **Using GNNTrainer with MEGNet:** GNNTrainer expects PyG DataLoader batches (`batch.x`, `batch.edge_index`). MEGNet uses DGL graphs with different batch semantics. Do not attempt to adapt GNNTrainer.
- **Changing pretrained model cutoff:** MEGNet-MP-2019.4.1 was trained with a specific cutoff radius. Using a different cutoff invalidates the pretrained weights and will produce garbage results.
- **Importing matgl at module level:** Would break `import cathode_ml` for users without matgl/DGL installed. Use lazy imports only.
- **Building MEGNet from scratch:** The pretrained weights provide massive advantage for small datasets. Always fine-tune, never train from random initialization.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DGL graph construction from crystal structures | Custom DGL graph builder | `matgl.ext._pymatgen_dgl.Structure2Graph` | matgl knows exactly what atom/bond/state features MEGNet expects; custom builders will miss state features or use wrong feature encodings |
| DGL data batching | Custom collation | `matgl.graph._data_dgl.collate_fn_graph` + `MGLDataLoader` | DGL graph batching is non-trivial (state attributes, heterogeneous graphs); matgl handles edge cases |
| MEGNet training loop | Custom training loop | `ModelLightningModule` + Lightning Trainer | matgl's module handles loss computation, metric logging, and DGL-specific tensor ops correctly |
| Element type encoding | Manual one-hot or embedding | `matgl.ext._pymatgen_dgl.get_element_list` | Ensures element ordering matches pretrained model expectations |

**Key insight:** matgl provides a complete vertical stack from pymatgen Structure to trained MEGNet. Using any component outside this stack (e.g., own graph builder + matgl model) risks feature mismatches that silently produce bad results.

## Common Pitfalls

### Pitfall 1: DGL and PyG Version Conflicts
**What goes wrong:** Installing matgl+DGL alongside PyG causes import errors or silent numerical bugs due to conflicting CUDA/PyTorch operator registrations.
**Why it happens:** DGL and PyG both register custom C++ operators for sparse matrix operations. Different torch versions or CUDA toolkit mismatches cause load failures.
**How to avoid:** Install DGL first, then PyG, both pinned to versions tested against the same torch. On Windows, use CPU-only DGL to avoid CUDA conflicts. Test with `python -c "import dgl; import torch_geometric; print('OK')"` after installation.
**Warning signs:** `RuntimeError: operator not found`, `ImportError: DLL load failed`, segfaults during training.

### Pitfall 2: Pretrained Model Name Mismatch
**What goes wrong:** `matgl.load_model("MEGNet-MP-2019.4.1")` fails because the exact model string doesn't match available pretrained models.
**Why it happens:** Pretrained model names include suffixes like `-Eform`, `-BandGap-mfi`. The formation energy model is `MEGNet-MP-2018.6.1-Eform` (NOT 2019.4.1 for Eform).
**How to avoid:** Call `matgl.get_available_pretrained_models()` to list exact model names at runtime. The 2019.4.1 version may only exist for band gap, not formation energy. If no exact match exists, use the closest available pretrained model (MEGNet-MP-2018.6.1-Eform for formation energy tasks).
**Warning signs:** `ValueError` or `FileNotFoundError` from `load_model()`.

### Pitfall 3: Lightning CSVLogger Column Names Differ from Project Format
**What goes wrong:** Wrapper assumes columns like `train_loss`, `val_loss`, `val_mae` but Lightning logs use `train_MAE`, `val_MAE`, `train_Total_Loss`.
**Why it happens:** matgl's ModelLightningModule logs metrics with its own naming convention (capital MAE, Total_Loss prefix).
**How to avoid:** Inspect actual CSVLogger output column names after first training run. Build wrapper with column name mapping, not hardcoded assumptions.
**Warning signs:** KeyError when reading metrics CSV, empty columns in converted output.

### Pitfall 4: State Features (Global Attributes) Misconfiguration
**What goes wrong:** MEGNet expects a state feature vector (global graph attributes) but it's not provided or is wrong dimension.
**Why it happens:** MEGNet architecture includes Set2Set readout that incorporates global state. The pretrained model expects specific state dimensionality.
**How to avoid:** When using pretrained model, its `dim_state_embedding` is already set. For the MEGNetDataset, pass state attributes as zeros of the correct dimension if no meaningful global features exist. The pretrained model's configuration specifies the expected state dimension.
**Warning signs:** Shape mismatch errors during forward pass, NaN losses in early epochs.

### Pitfall 5: Checkpoint Format Mismatch
**What goes wrong:** Lightning ModelCheckpoint saves `.ckpt` files (Lightning format), not `.pt` files (raw torch state_dict) as required by project artifact format.
**Why it happens:** Lightning checkpoints include optimizer state, scheduler state, and other training metadata in a Lightning-specific format.
**How to avoid:** After training, load the best `.ckpt` checkpoint and extract the model state_dict to save as `.pt` file in the project format. Alternatively, add a custom callback that saves `.pt` format alongside Lightning checkpoints.
**Warning signs:** Wrong file extension, inability to load with `torch.load()` directly.

### Pitfall 6: NumPy 2.0 Incompatibility
**What goes wrong:** matgl or DGL crashes with attribute errors or dtype issues.
**Why it happens:** DGL 2.2.0 uses deprecated NumPy APIs removed in NumPy 2.0.
**How to avoid:** Project already pins numpy==1.26.4 which is <2.0. Ensure this constraint is maintained when installing matgl.
**Warning signs:** `AttributeError: module 'numpy' has no attribute 'float_'`.

## Code Examples

### Loading and Inspecting Pretrained MEGNet
```python
# Source: matgl.ai Property Predictions tutorial
import matgl

# List all available pretrained models
available = matgl.get_available_pretrained_models()
megnet_models = [m for m in available if "MEGNet" in m]
print("Available MEGNet models:", megnet_models)

# Load formation energy model
model = matgl.load_model("MEGNet-MP-2018.6.1-Eform")

# Inspect model configuration
print(f"Cutoff: {model.cutoff}")
print(f"Model type: {type(model.model)}")

# Quick prediction on a structure
from pymatgen.core import Structure
struct = Structure.from_dict(structure_dict)
eform = model.predict_structure(struct)
print(f"Predicted Eform: {eform:.4f} eV/atom")
```

### Complete Training Orchestrator Pattern
```python
# Source: Adapted from matgl tutorial + project train_cgcnn.py pattern
def train_megnet_for_property(
    structures, targets, property_name, megnet_config, seed
):
    """Train MEGNet for a single property (mirrors CGCNN per-property loop)."""
    import matgl
    from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list
    from matgl.graph._data_dgl import MGLDataLoader, MGLDataset, collate_fn_graph
    from matgl.utils.training import ModelLightningModule
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger

    # Load pretrained model
    model_name = megnet_config.get("pretrained_model", "MEGNet-MP-2018.6.1-Eform")
    pretrained = matgl.load_model(model_name)

    # Setup converter with pretrained model's cutoff
    elem_list = get_element_list(structures)
    converter = Structure2Graph(element_types=elem_list, cutoff=pretrained.cutoff)

    # Build dataset
    dataset = MGLDataset(
        structures=structures,
        labels={property_name: targets},
        converter=converter,
    )

    # Already split by compositional_split -- partition dataset
    # (train_data, val_data, test_data are subsets of dataset)

    # Create data loaders
    batch_size = megnet_config["training"].get("batch_size", 128)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=batch_size,
        num_workers=0,
    )

    # Lightning module with pretrained model
    lit_module = ModelLightningModule(model=pretrained.model, lr=1e-4)

    # Configure trainer
    results_dir = megnet_config.get("results_dir", "data/results/megnet")
    logger = CSVLogger(results_dir, name=f"{property_name}")
    trainer = L.Trainer(
        max_epochs=megnet_config["training"].get("n_epochs", 1000),
        accelerator="auto",
        logger=logger,
    )

    # Train
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate on test set and produce standard metrics
    # ... (use compute_metrics + save_results from utils.py)
```

### megnet.yaml Configuration Template
```yaml
# MEGNet model configuration for cathode property prediction
# Fine-tunes pretrained MEGNet-MP-2019.4.1 (or closest available)

model:
  pretrained_model: "MEGNet-MP-2018.6.1-Eform"  # Verify with get_available_pretrained_models()
  # Architecture params inherited from pretrained -- do not override

training:
  optimizer: adam
  learning_rate: 0.0001       # Lower LR for fine-tuning (vs CGCNN's 0.001)
  weight_decay: 0.0
  batch_size: 128             # Larger batches for MEGNet (vs CGCNN's 64)
  n_epochs: 1000              # More epochs with lower LR (vs CGCNN's 400)
  early_stopping_patience: 100  # More patience due to lower LR
  scheduler:
    type: reduce_on_plateau
    factor: 0.5
    patience: 50
    min_lr: 1.0e-7

results_dir: "data/results/megnet"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| megnet Python package (TF/Keras) | matgl (PyTorch/DGL) | 2022-2023 | matgl is the official PyTorch reimplementation by the same group (Materials Virtual Lab) |
| DGL backend (matgl v1.x) | PyG backend (matgl v2.x) | Nov 2025 | MEGNet NOT yet ported to PyG in v2; must use v1.3.0 or v2 with DGL backend |
| Train from scratch | Fine-tune from pretrained | 2023+ | Pretrained models on Materials Project (~130K structures) provide better starting point for small datasets |

**Deprecated/outdated:**
- Original `megnet` package (TensorFlow): Replaced by matgl. Do not use.
- matgl v2.0+ for MEGNet: MEGNet has not been ported to PyG yet. Use v1.3.0 or v2 with explicit DGL backend.
- DGL library: No longer actively maintained (last release 2024). Stable at v2.2.0 but no future updates expected.

## Open Questions

1. **Exact pretrained model name for formation energy**
   - What we know: Tutorial shows `MEGNet-MP-2018.6.1-Eform`. CONTEXT.md references `MEGNet-MP-2019.4.1`.
   - What's unclear: Whether a 2019.4.1 formation energy variant exists, or only the band gap (mfi) variant is 2019.4.1.
   - Recommendation: At runtime, call `matgl.get_available_pretrained_models()` and select the best match. Log the actual model used. Use `MEGNet-MP-2018.6.1-Eform` as default if 2019.4.1-Eform doesn't exist.

2. **DGL + PyG coexistence on Windows**
   - What we know: Both install via pip. DGL 2.2.0 requires torch 2.3.0. Project uses torch>=2.1.0 with PyG.
   - What's unclear: Whether DGL's Windows CPU wheel works alongside PyG without conflicts.
   - Recommendation: Test installation first. If fails, document Docker fallback. The lazy import pattern ensures the rest of the package works regardless.

3. **ModelLightningModule learning rate configuration**
   - What we know: Tutorial shows `ModelLightningModule(model=model)` with default LR.
   - What's unclear: How to pass custom LR (1e-4) to the Lightning module, and whether it supports ReduceLROnPlateau.
   - Recommendation: Inspect `ModelLightningModule.__init__` signature at implementation time. May need to pass `lr` parameter or create a custom LightningModule subclass.

4. **MEGNetDataset subsetting for compositional splits**
   - What we know: MGLDataLoader takes train_data, val_data, test_data as separate objects.
   - What's unclear: Whether MGLDataset supports integer indexing for splitting, or if structures must be pre-partitioned before dataset creation.
   - Recommendation: Pre-partition structures and targets into train/val/test lists using compositional_split indices, then create three separate MGLDataset instances.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 |
| Config file | tests/ directory (no pytest.ini, uses defaults) |
| Quick run command | `pytest tests/test_megnet.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODL-02 | MEGNet loads pretrained model and produces predictions | unit | `pytest tests/test_megnet.py::test_load_pretrained -x` | Wave 0 |
| MODL-02 | MEGNet trains on sample data via Lightning | integration | `pytest tests/test_megnet.py::test_megnet_training -x` | Wave 0 |
| MODL-02 | train_megnet produces standard artifact format (CSV, JSON) | integration | `pytest tests/test_megnet.py::test_artifact_format -x` | Wave 0 |
| MODL-02 | MEGNet uses same compositional splits as CGCNN | unit | `pytest tests/test_megnet.py::test_same_splits -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_megnet.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_megnet.py` -- covers MODL-02 (MEGNet model loading, training, artifact format)
- [ ] Note: Tests requiring matgl/DGL should be marked with `@pytest.mark.skipif` when matgl is not installed, consistent with the lazy import pattern

## Sources

### Primary (HIGH confidence)
- [matgl.ai MEGNet Training Tutorial](https://matgl.ai/tutorials/Training%20a%20MEGNet%20Formation%20Energy%20Model%20with%20PyTorch%20Lightning.html) - Full training pipeline code with Structure2Graph, MEGNetDataset, MGLDataLoader, ModelLightningModule, CSVLogger
- [matgl.ai Property Predictions Tutorial](https://matgl.ai/tutorials/Property%20Predictions%20using%20MEGNet%20or%20M3GNet%20Models.html) - load_model() API, predict_structure(), available pretrained models
- [matgl PyPI](https://pypi.org/project/matgl/) - Version info, dependencies, Python requirements
- [matgl GitHub](https://github.com/materialsvirtuallab/matgl) - README, releases, DGL backend setup

### Secondary (MEDIUM confidence)
- [matgl Changelog](https://matgl.ai/changes.html) - v2.0.0 PyG migration status, MEGNet not ported to PyG
- [matgl GitHub Releases](https://github.com/materialsvirtuallab/matgl/releases) - v2.0.0 release notes (Nov 2025)

### Tertiary (LOW confidence)
- Exact DGL 2.2.0 + PyG coexistence on Windows -- not verified, needs installation testing
- ModelLightningModule exact constructor signature for LR and scheduler -- needs runtime inspection
- Pretrained model name `MEGNet-MP-2019.4.1-Eform` existence -- only `MEGNet-MP-2018.6.1-Eform` confirmed in tutorials

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - matgl v1.3.0 API confirmed from official tutorial, but exact DGL version pinning and Windows compatibility are unverified
- Architecture: HIGH - Training pipeline pattern well-documented in official matgl tutorials
- Pitfalls: MEDIUM - DGL+PyG coexistence and Lightning log format based on experience + docs, not direct testing

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (matgl stable at v1.3.0; DGL no longer updated)
