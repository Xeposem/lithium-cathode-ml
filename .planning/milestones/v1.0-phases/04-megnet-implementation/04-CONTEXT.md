# Phase 4: MEGNet Implementation - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement MEGNet via matgl v1.3.0 for cathode property prediction (formation energy, voltage, stability, capacity) as separate per-property models. Fine-tune from pretrained MEGNet-MP-2019.4.1 weights. Produce results on identical data splits and in the same artifact format as CGCNN for fair head-to-head comparison. Isolate DGL dependency to avoid breaking the existing PyG-based pipeline.

</domain>

<decisions>
## Implementation Decisions

### Training Approach
- Use matgl's built-in Lightning trainer (not our GNNTrainer) — matgl handles DGL batching and MEGNet-specific forward pass natively
- Write a wrapper to extract Lightning's training logs into our standard artifact format
- Produce same artifact types as CGCNN: per-epoch CSV metrics, JSON results, .pt checkpoints
- Checkpoint naming: megnet_{property}_best.pt, megnet_{property}_final.pt in data/results/megnet/
- CLI entry point: `python -m cathode_ml.models.train_megnet` with --seed flag (same pattern as CGCNN)

### Pretrained Weights
- Fine-tune from matgl's pretrained MEGNet-MP-2019.4.1 (not training from scratch)
- Full fine-tuning — all layers unfrozen, trained with lower LR (~1e-4)
- Phase 5 comparison must clearly note that MEGNet uses pretrained weights while CGCNN trains from scratch — academically transparent

### Environment Isolation
- Try installing matgl+DGL in the same environment as PyG first
- If conflicts arise, fallback to Docker container for MEGNet training
- Lazy imports for matgl/DGL inside train_megnet functions (consistent with existing xgboost lazy import pattern) — rest of the package works without them installed

### Graph Construction
- Use matgl's own Structure2Graph converters to build DGL graphs from pymatgen Structures — matgl knows what features MEGNet expects (atom features, bond features, state features)
- Use matgl's default cutoff radius (from pretrained model), not CGCNN's 8.0 A — changing cutoff would invalidate pretrained weights
- Load input data from data/processed/materials.json (same source as CGCNN) to guarantee identical input dataset
- Compositional splitting uses same folds as CGCNN and baselines

### Claude's Discretion
- matgl Lightning trainer configuration details (callbacks, logging backend)
- Exact wrapper implementation for extracting Lightning logs to CSV/JSON
- Batch size tuning for MEGNet
- How to handle matgl's state features (global attributes)
- Docker container configuration if needed for fallback

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `models/trainer.py`: GNNTrainer — NOT directly reusable for MEGNet (expects PyG DataLoaders), but its artifact format (CSV columns, JSON structure) is the target output format
- `models/utils.py`: compute_metrics() and save_results() — directly reusable for MEGNet evaluation
- `models/train_cgcnn.py`: Pattern template for train_megnet.py — same orchestration flow (load records, split, train per-property, save results)
- `features/split.py`: compositional_split() and get_group_keys() — reusable for identical train/val/test splits
- `data/schemas.py`: MaterialRecord — same input data schema
- `config.py`: load_config(), set_seeds() — reuse for MEGNet config loading

### Established Patterns
- YAML config files in configs/ directory — will add configs/megnet.yaml
- Per-property sequential training loop with skip if <5 records
- Seeds reset before each property model init
- Results saved as JSON to data/results/{model_name}/
- Lazy imports for optional heavy dependencies (xgboost pattern in baselines.py)

### Integration Points
- Input: data/processed/materials.json (same as CGCNN)
- Config: configs/megnet.yaml (new, MEGNet-specific hyperparameters)
- Output: data/results/megnet/ (checkpoints, CSV metrics, JSON results)
- New files: cathode_ml/models/megnet.py, cathode_ml/models/train_megnet.py, configs/megnet.yaml

</code_context>

<specifics>
## Specific Ideas

- MEGNet-MP-2019.4.1 as the specific pretrained model (trained on Materials Project formation energies — most relevant to cathode targets)
- Comparison fairness: Phase 5 results tables/plots must annotate that MEGNet uses pretrained weights vs CGCNN from scratch

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-megnet-implementation*
*Context gathered: 2026-03-06*
