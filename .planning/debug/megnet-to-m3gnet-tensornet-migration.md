# Debug: MEGNet to M3GNet + TensorNet Migration

**Issue:** matgl 2.0.6 dropped MEGNet entirely. Need to replace MEGNet with M3GNet and TensorNet from matgl 2.x.

**Status:** Investigation complete. Changes NOT yet applied.

---

## 1. Root Cause

- The project was built against **matgl v1.3.0** (DGL backend), which provided `MEGNet` classes.
- **matgl v2.x** migrated to a PyG backend and **removed MEGNet entirely**.
- `requirements.txt` specifies `matgl>=1.1.0`, which will resolve to v2.x and break all MEGNet imports.
- The install message in `megnet.py` still says `pip install matgl==1.3.0 dgl==2.2.0`.

## 2. Complete File Inventory (All MEGNet References)

### Core Model Files (MUST REWRITE)
| File | Lines | Impact |
|------|-------|--------|
| `cathode_ml/models/megnet.py` | Entire file | Model wrapper: `load_megnet_model`, `get_available_megnet_models`, `get_megnet_state_dict`, `predict_with_megnet`. All assume matgl v1.x MEGNet classes. |
| `cathode_ml/models/train_megnet.py` | Entire file | Training orchestrator: imports from `megnet.py`, uses `matgl.ext.pymatgen.Structure2Graph`, `MGLDataset`, `MGLDataLoader`, `ModelLightningModule`. All matgl v1.x APIs. |
| `configs/megnet.yaml` | Entire file | Config references `MEGNet-MP-2018.6.1-Eform` pretrained model (does not exist in matgl 2.x). |

### Pipeline Integration (MUST UPDATE)
| File | Lines | Impact |
|------|-------|--------|
| `cathode_ml/pipeline.py` | 42-43, 131-140 | CLI `--models` choices include `"megnet"`. Train stage imports `train_megnet` and loads `megnet.yaml`. |
| `cathode_ml/__main__.py` | — | Delegates to pipeline (indirect reference). |

### Evaluation / Metrics (MUST UPDATE)
| File | Lines | Impact |
|------|-------|--------|
| `cathode_ml/evaluation/metrics.py` | 21, 24-29, 32, 41-42, 87-98, 158-162 | `MODEL_COLORS["megnet"]`, `MODEL_LABELS["megnet"]` (with dagger), `MODELS_ORDER` includes `"megnet"`, `_MEGNET_FOOTNOTE`, `load_all_results()` reads `megnet/megnet_results.json`. |
| `cathode_ml/evaluation/plots.py` | 66, 197, 205 | Parity plot layout assumes "MEGNet bottom-right". Learning curves grid includes `"megnet"` column. |
| `cathode_ml/evaluation/__init__.py` | 4 | Docstring mentions MEGNet. |

### Dashboard (MUST UPDATE)
| File | Lines | Impact |
|------|-------|--------|
| `dashboard/pages/overview.py` | 27 | Prose text: "Random Forest, XGBoost, CGCNN, and MEGNet". |
| `dashboard/pages/model_comparison.py` | 77 | `gnn_models = ["cgcnn", "megnet"]` for training curves. |
| `dashboard/pages/predict.py` | 35 | `MODEL_LABELS` dict: `"megnet": "MEGNet"`. |
| `dashboard/utils/model_loader.py` | 74, 115-151, 239, 262-263 | `load_gnn_model()` has full MEGNet branch loading pretrained + state dict. `predict_from_structure()` iterates `["cgcnn", "megnet"]` with `predict_structure` call. |
| `dashboard/utils/data_loader.py` | — | Imports from `metrics.py` (indirect via `MODELS_ORDER`, `MODEL_LABELS`). |
| `dashboard/utils/charts.py` | — | Uses `MODELS_ORDER`, `MODEL_LABELS`, `MODEL_COLORS` from metrics (indirect). |

### Features (MINOR UPDATE)
| File | Lines | Impact |
|------|-------|--------|
| `cathode_ml/features/graph.py` | 4 | Docstring: "suitable for graph neural networks (CGCNN, MEGNet)". |
| `cathode_ml/models/utils.py` | 4 | Docstring: "deep learning (CGCNN, MEGNet) model pipelines". |
| `cathode_ml/models/trainer.py` | 4 | Docstring: "Used by CGCNN (Phase 3) and MEGNet (Phase 4)". |

### Tests (MUST REWRITE)
| File | Lines | Impact |
|------|-------|--------|
| `tests/test_megnet.py` | Entire file (335 lines) | All 9 test classes test MEGNet-specific behavior: lazy imports, pretrained loading, state dict, Lightning logs, artifact format, split consistency, checkpoint naming, per-property loop. |
| `tests/test_pipeline.py` | 32, 160-206, 213-239 | Default models include `"megnet"`. Integration tests mock `train_megnet` and assert `megnet.yaml` is loaded. |
| `tests/test_evaluation.py` | 48-53, 62-76, 104-116, 139, 162, 167-177, 200, 229-234 | `SAMPLE_MEGNET_RESULTS`, `_create_results_tree(megnet=...)`, tests for megnet results loading, dagger label, footnote. |
| `tests/test_plots.py` | 33, 48, 60, 228-232 | Learning curves grid includes `"megnet"` column. Test for MEGNet CSV without lr column. |
| `tests/test_dashboard_predict.py` | 201-221 | `test_megnet_raw_state_dict_loading()` tests MEGNet loader handles raw state_dict. |

### Documentation (MUST UPDATE)
| File | Lines | Impact |
|------|-------|--------|
| `README.md` | 9, 33, 65, 113, 136, 188-189, 213 | Multiple sections describe MEGNet architecture, CLI usage, file tree. |

### Dependencies
| File | Lines | Impact |
|------|-------|--------|
| `requirements.txt` | 28 | `matgl>=1.1.0` -- needs version constraint update. Also `dgl>=2.0.0` (line 7). |

---

## 3. Current Architecture (matgl v1.x MEGNet)

### `cathode_ml/models/megnet.py`
- `_import_matgl()` -- lazy import of matgl
- `load_megnet_model(model_name)` -- calls `matgl.load_model("MEGNet-MP-2018.6.1-Eform")`
- `get_available_megnet_models()` -- filters `matgl.get_available_pretrained_models()` for "MEGNet"
- `get_megnet_state_dict(model)` -- extracts `model.model.state_dict()`
- `predict_with_megnet(model, structures)` -- calls `model.predict_structure(struct)`

### `cathode_ml/models/train_megnet.py`
- Uses matgl v1.x APIs: `Structure2Graph`, `MGLDataset`, `MGLDataLoader`, `collate_fn_graph`, `ModelLightningModule`
- Fine-tunes pretrained MEGNet with Lightning trainer
- Saves `.pt` checkpoints + converts Lightning CSV logs
- Identical compositional splitting as CGCNN
- Output format: `{property: {"megnet": {mae, rmse, r2, n_train, n_test}}}`

### `configs/megnet.yaml`
- References `MEGNet-MP-2018.6.1-Eform` pretrained model
- Training: lr=0.0001, batch_size=128, n_epochs=1000, patience=100
- Results dir: `data/results/megnet`

---

## 4. Migration Plan

### Strategy: Replace "megnet" with TWO model options: "m3gnet" and "tensornet"

#### M3GNet (DGL backend)
- **Pretrained model available**: `M3GNet-MP-2018.6.1-Eform` (formation energy)
- Invariant GNN with 3-body interactions
- Uses DGL backend in matgl 2.x
- Fine-tuning approach similar to current MEGNet workflow
- ~0.020 eV/atom MAE on formation energy benchmarks

#### TensorNet (PyG backend)
- **No pretrained property models** -- train from scratch
- O(3)-equivariant GNN, faster training
- Uses PyG backend in matgl 2.x
- Need to construct model architecture from config
- ~0.020 eV/atom MAE on formation energy benchmarks

### Files to Create
1. `cathode_ml/models/m3gnet.py` -- M3GNet model wrapper (replaces megnet.py)
2. `cathode_ml/models/tensornet.py` -- TensorNet model wrapper (new)
3. `cathode_ml/models/train_m3gnet.py` -- M3GNet training orchestrator
4. `cathode_ml/models/train_tensornet.py` -- TensorNet training orchestrator
5. `configs/m3gnet.yaml` -- M3GNet config (replaces megnet.yaml)
6. `configs/tensornet.yaml` -- TensorNet config (new)
7. `tests/test_m3gnet.py` -- M3GNet tests (replaces test_megnet.py)
8. `tests/test_tensornet.py` -- TensorNet tests (new)

### Files to Update
1. `cathode_ml/pipeline.py` -- Add "m3gnet" and "tensornet" to `--models` choices, add train stages
2. `cathode_ml/evaluation/metrics.py` -- Add m3gnet/tensornet to `MODEL_COLORS`, `MODEL_LABELS`, `MODELS_ORDER`, `load_all_results()`; remove megnet; update footnote for M3GNet pretrained
3. `cathode_ml/evaluation/plots.py` -- Update gnn_models list, parity grid, learning curves
4. `cathode_ml/evaluation/__init__.py` -- Update docstring
5. `dashboard/pages/overview.py` -- Update model list text
6. `dashboard/pages/model_comparison.py` -- Update gnn_models list
7. `dashboard/pages/predict.py` -- Update MODEL_LABELS
8. `dashboard/utils/model_loader.py` -- Add m3gnet/tensornet branches to `load_gnn_model()` and `predict_from_structure()`
9. `cathode_ml/features/graph.py` -- Update docstring
10. `cathode_ml/models/utils.py` -- Update docstring
11. `cathode_ml/models/trainer.py` -- Update docstring
12. `tests/test_pipeline.py` -- Update default models, add m3gnet/tensornet mocks
13. `tests/test_evaluation.py` -- Replace megnet samples with m3gnet/tensornet
14. `tests/test_plots.py` -- Update gnn_models references
15. `tests/test_dashboard_predict.py` -- Replace megnet loader test
16. `README.md` -- Rewrite model descriptions, CLI examples, file tree
17. `requirements.txt` -- Pin `matgl>=2.0.0`, keep `dgl>=2.0.0` for M3GNet, keep `torch-geometric>=2.6.0` for TensorNet

### Files to Delete
1. `cathode_ml/models/megnet.py`
2. `cathode_ml/models/train_megnet.py`
3. `configs/megnet.yaml`
4. `tests/test_megnet.py`

---

## 5. Key matgl 2.x API Differences to Account For

### M3GNet in matgl 2.x
- Load pretrained: `matgl.load_model("M3GNet-MP-2018.6.1-Eform")`
- `matgl.ext.pymatgen.Structure2Graph` still exists
- `MGLDataset`, `MGLDataLoader` may have changed -- need to verify
- `ModelLightningModule` -- need to verify
- DGL must be installed: `pip install dgl`

### TensorNet in matgl 2.x
- `from matgl.models import TensorNet`
- Construct from scratch with architecture params
- Uses PyG data structures, not DGL
- Need `torch-geometric` installed
- May need `matgl.ext.pymatgen.Structure2Graph` with PyG backend

### Common
- `matgl.get_available_pretrained_models()` still works
- Pretrained models include M3GNet variants but NOT MEGNet
- Both models support `predict_structure()` after loading

---

## 6. Risk Assessment

- **High risk:** matgl 2.x API changes for `MGLDataset`, `MGLDataLoader`, `ModelLightningModule` may differ from v1.x. Need to test against actual matgl 2.x installation.
- **Medium risk:** TensorNet training from scratch may require different hyperparameter tuning than fine-tuning pretrained models.
- **Low risk:** Evaluation/dashboard changes are mostly string replacements.
- **Dependency conflict:** M3GNet needs DGL, TensorNet needs PyG. Both are already in requirements.txt but may conflict. Need to verify they coexist.

---

## 7. Recommended Implementation Order

1. **Create model wrappers** (`m3gnet.py`, `tensornet.py`) -- test with actual matgl 2.x
2. **Create training orchestrators** (`train_m3gnet.py`, `train_tensornet.py`)
3. **Create config files** (`m3gnet.yaml`, `tensornet.yaml`)
4. **Update pipeline.py** -- add new model options
5. **Update evaluation** (metrics.py, plots.py, __init__.py)
6. **Update dashboard** (all 4 files)
7. **Update tests** (rewrite test_megnet.py -> test_m3gnet.py + test_tensornet.py, update integration tests)
8. **Update docs** (README.md, docstrings)
9. **Delete old files** (megnet.py, train_megnet.py, megnet.yaml, test_megnet.py)
10. **Update requirements.txt** -- pin matgl>=2.0.0
