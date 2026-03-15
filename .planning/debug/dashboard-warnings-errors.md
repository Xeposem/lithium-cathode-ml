---
status: awaiting_human_verify
trigger: "Investigate and fix dashboard warnings and errors in the lithium-cathode-ml project"
created: 2026-03-15T00:00:00Z
updated: 2026-03-15T00:00:00Z
---

## Current Focus

hypothesis: TensorNet load failure is caused by model_loader.py calling model.model.load_state_dict() on a plain nn.Module (TensorNet has no .model wrapper). The .pt files are raw state dicts saved directly from get_tensornet_state_dict() which does getattr(model, 'model', model) — meaning TensorNet IS the inner module.
test: Confirmed by inspecting checkpoint keys (raw TensorNet keys like bond_expansion.rbf.centers) and TensorNet class structure
expecting: Fix by removing the .model wrapper call in load_gnn_model() for tensornet branch
next_action: Apply all 4 categories of fixes

## Symptoms

expected: Dashboard runs cleanly without deprecation warnings or errors
actual: Multiple warnings and 3 categories of issues appear
errors:
  1. Streamlit deprecation: use_container_width deprecated after 2025-12-31, replace with width='stretch' or width='content'
  2. sklearn InconsistentVersionWarning: models pickled with 1.6.1, loaded with 1.8.0
  3. XGBoost version warning: old serialized model needs re-export
  4. TensorNet load failures: 'TensorNet' object has no attribute 'model'
  5. pymatgen warnings: No Pauling electronegativity for Ar, He, Ne
  6. matminer warning: MagpieData impute_nan default will change
reproduction: Run the Streamlit dashboard
started: After package upgrades

## Eliminated

- hypothesis: TensorNet checkpoints are missing or corrupted
  evidence: .pt files exist and contain valid state dict keys (bond_expansion.rbf.centers etc)
  timestamp: 2026-03-15T00:00:00Z

- hypothesis: TensorNet model is wrapped in a container with .model attribute
  evidence: TensorNet is a plain nn.Module in matgl 2.x. get_tensornet_state_dict() uses getattr(model, 'model', model) which returns model itself. The saved .pt is therefore a direct state_dict of TensorNet.
  timestamp: 2026-03-15T00:00:00Z

## Evidence

- timestamp: 2026-03-15T00:00:00Z
  checked: data/results/tensornet/tensornet_formation_energy_per_atom_best.pt
  found: Raw state dict with keys like bond_expansion.rbf.centers, tensor_embedding.*, layers.*, readout.* — no wrapper
  implication: model_loader.py must call model.load_state_dict() not model.model.load_state_dict()

- timestamp: 2026-03-15T00:00:00Z
  checked: dashboard/utils/model_loader.py lines 222-226
  found: Code calls model.model.load_state_dict(state_dict) then model.model.eval()
  implication: Direct bug causing 'TensorNet' object has no attribute 'model'

- timestamp: 2026-03-15T00:00:00Z
  checked: cathode_ml/models/tensornet.py get_tensornet_state_dict()
  found: Uses getattr(model, 'model', model) — returns model itself since TensorNet has no .model attr
  implication: Confirms .pt files are direct TensorNet state dicts

- timestamp: 2026-03-15T00:00:00Z
  checked: use_container_width occurrences
  found: 4 occurrences in dashboard/pages/ (data_explorer.py:134,147; model_comparison.py:86; overview.py:68)
  implication: All 4 must be replaced with width='stretch'

- timestamp: 2026-03-15T00:00:00Z
  checked: data/results/baselines/ - sklearn/xgboost model files
  found: joblib files for rf and xgb, 4 properties each
  implication: Need warning suppression or retraining; check if training scripts available

## Resolution

root_cause: |
  1. TensorNet: model_loader.py calls model.model.load_state_dict() but TensorNet is a plain nn.Module
  2. use_container_width: deprecated Streamlit parameter in 4 dashboard files
  3. sklearn/xgboost: version mismatch between training and current env
  4. pymatgen/matminer: cosmetic warnings about noble gas electronegativities and impute_nan

fix: |
  1. Fix model_loader.py tensornet branch: model.load_state_dict(state_dict) and model.eval()
  2. Replace use_container_width=True with width='stretch' in all 4 files
  3. Add warning suppression in dashboard app.py or model_loader.py for sklearn/xgboost
  4. Add warning filters for pymatgen/matminer cosmetic warnings in app.py

verification: |
  1. TensorNet: Tested all 4 properties with load_gnn_model() - all return model (not None). Root cause was model.model.load_state_dict() on plain nn.Module; fixed to model.load_state_dict() with element list inferred from emb weight shape.
  2. use_container_width: grep confirms 0 occurrences in project Python files (only in .venv library code). 4 occurrences replaced with width='stretch'.
  3. sklearn/xgboost: 8 joblib files retrained with sklearn 1.8.0 + xgboost 3.2.0. Load without InconsistentVersionWarning.
  4. pymatgen/matminer: Warning filters added to dashboard/app.py entrypoint.
files_changed:
  - dashboard/utils/model_loader.py
  - dashboard/pages/data_explorer.py
  - dashboard/pages/model_comparison.py
  - dashboard/pages/overview.py
  - dashboard/app.py
