# Phase 3: CGCNN Implementation - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement CGCNN using PyG CGConv layers for cathode property prediction (formation energy, voltage, stability, capacity) as separate per-property models. Build a reusable GNN training infrastructure that MEGNet (Phase 4) will share. Save model checkpoints, training curves, and evaluation metrics as artifacts.

</domain>

<decisions>
## Implementation Decisions

### Training Infrastructure
- Early stopping: patience-based on validation loss (e.g. patience=50)
- Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience~30)
- Device: auto-detect CUDA, fall back to CPU transparently (set once at start)

### Model Architecture
- Number of CGConv layers: configurable via YAML (default 3 per original paper)
- Hidden dimension: 128
- Batch normalization after each conv layer
- Loss function: MSE (standard for regression)

### Artifact & Logging Format
- Per-epoch metrics saved as CSV per property (columns: epoch, train_loss, val_loss, val_mae, lr)
- Checkpoints: save both best (lowest val loss) and final epoch model
- Checkpoint naming: cgcnn_{property}_best.pt, cgcnn_{property}_final.pt
- Final evaluation results: same JSON format as baselines (mae, rmse, r2, n_train, n_test)
- All artifacts stored under data/results/cgcnn/

### Reusability for MEGNet
- Generic GNNTrainer class that accepts any PyTorch model + DataLoader
- Trainer expects pre-built DataLoaders (not creating them internally) — MEGNet may need different batching via matgl
- Separate config files per model: configs/cgcnn.yaml and configs/megnet.yaml
- Extract evaluate_model from baselines.py into shared models/utils.py — both baselines and GNNs use identical evaluation metrics

### Claude's Discretion
- Graph-level readout/pooling strategy (mean, mean+max, etc.)
- Sequential vs parallel training of per-property models
- Exact early stopping patience and scheduler patience values
- Batch size selection
- Number of fully-connected layers after pooling

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `features/graph.py`: structure_to_graph() already converts pymatgen Structure to PyG Data with one-hot atomic numbers and Gaussian edge features — CGCNN consumes these directly
- `features/split.py`: compositional_split() with GroupShuffleSplit — reuse for train/val/test splitting
- `features/composition.py`: featurize_compositions() for Magpie features (not needed for CGCNN but available)
- `models/baselines.py`: evaluate_model() computes MAE/RMSE/R2, save_results() writes JSON — extract to shared utils
- `config.py`: load_config(), set_seeds(), get_project_root() — reuse for CGCNN config loading

### Established Patterns
- YAML config files in configs/ directory (data.yaml, features.yaml, baselines.yaml)
- Separate models per property (not multi-output) — baselines loop over target_properties
- Results saved as JSON to data/results/
- Lazy imports for heavy dependencies (xgboost pattern in baselines.py)
- Fixed random seeds via config["random_seeds"]

### Integration Points
- Graph config already in configs/features.yaml (cutoff_radius=8.0, max_neighbors=12, 80 Gaussians, node_feature_dim=100)
- Target properties defined in features.yaml: formation_energy_per_atom, voltage, capacity, energy_above_hull
- Splitting params in features.yaml: 80/10/10 train/val/test
- New files: configs/cgcnn.yaml, cathode_ml/models/cgcnn.py, cathode_ml/models/trainer.py, cathode_ml/models/utils.py

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches following the original CGCNN paper patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-cgcnn-implementation*
*Context gathered: 2026-03-05*
