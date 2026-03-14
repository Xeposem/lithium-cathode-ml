# Lithium-Ion Battery Cathode Performance Prediction

## What This Is

A machine learning platform that predicts lithium-ion battery cathode properties — capacity, voltage, stability, and formation energy — using crystal structure data from Materials Project, OQMD, AFLOW, and JARVIS. It benchmarks CGCNN, M3GNet, and TensorNet graph neural networks alongside RF/XGBoost baselines, and presents results through an interactive Streamlit dashboard. Built as an academic/portfolio project with paper-ready reproducibility.

## Core Value

Accurate, reproducible prediction of cathode performance properties from crystal structure, with clear model comparison and publication-quality results.

## Requirements

### Validated

- ✓ API-based data ingestion from Materials Project and OQMD — v1.0
- ✓ Crystal structure to PyG graph representation (atoms as nodes, bonds as edges) — v1.0
- ✓ CGCNN model for property prediction (R²=0.995 formation energy) — v1.0
- ✓ M3GNet and TensorNet via matgl 2.x (replaced MEGNet) — v1.0
- ✓ RF and XGBoost baseline models for comparison — v1.0
- ✓ Multi-property prediction (capacity, voltage, stability, formation energy) — v1.0
- ✓ Model evaluation with MAE, RMSE, R² metrics — v1.0
- ✓ Publication-quality benchmark comparison plots — v1.0
- ✓ Streamlit dashboard with model metrics, comparison charts, predictions, data/materials explorer, crystal viewer — v1.0
- ✓ End-to-end CLI pipeline (fetch → clean → train → evaluate) — v1.0
- ✓ README with introduction, methodology, pipeline details, and results — v1.0
- ✓ YAML-driven configuration with fixed seeds for reproducibility — v1.0

### Active

#### Current Milestone: v1.1 Polish & Correctness

**Goal:** Fix known bugs, validate all data sources, retrain models, and update all project surfaces to reflect accurate state.

**Target features:**
- Fix M3GNet/TensorNet double-denormalization bug
- Fix TensorNet training log duplication error
- Validate AFLOW and JARVIS fetchers end-to-end
- Retrain all models with 4-source combined data
- Update README with accurate results, all 4 data sources, current architecture
- Update GitHub repo description/topics
- Update dashboard with corrected model results, AFLOW/JARVIS data, accurate displays

### Out of Scope

- Mobile app — web dashboard is sufficient
- Real-time model retraining — batch training with saved checkpoints
- Deployment to cloud production — local/academic use
- Generative material design — prediction only, not inverse design

## Context

- Shipped v1.0 with 11,266 LOC Python across 150 commits
- Tech stack: PyTorch, PyG (CGCNN), matgl 2.x + DGL (M3GNet/TensorNet), scikit-learn, Streamlit
- 4 data sources configured: Materials Project, OQMD, AFLOW, JARVIS (latter 2 newly added)
- ~41K records for formation energy, ~3.3K for voltage/capacity (from MP electrodes)
- Separate models per property (not multi-output) per research recommendation
- Windows CPU-only environment (DGL without CUDA)
- CGCNN is the current best-performing GNN; M3GNet/TensorNet results invalid due to denorm bug

## Constraints

- **Tech stack**: PyTorch, PyG, matgl 2.x + DGL, scikit-learn
- **Data access**: All data via public APIs (reproducible acquisition)
- **Reproducibility**: Fixed random seeds, versioned dependencies, documented pipeline
- **Output quality**: Publication-quality figures and tables
- **Platform**: Must work on Windows with CPU-only DGL

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Separate models per property | Research shows better results than multi-output for small datasets | ✓ Good |
| CGCNN via PyG | Zero dependency conflicts, strong performance | ✓ Good |
| M3GNet + TensorNet replacing MEGNet | matgl 2.x dropped MEGNet; M3GNet (pretrained) + TensorNet (equivariant) | ⚠️ Revisit (denorm bug) |
| API-based data collection | Ensures reproducibility and automation | ✓ Good |
| Streamlit dashboard | Simpler than Dash, fast development, sufficient for dataset scale | ✓ Good |
| Compositional group splitting | Prevents polymorph leakage between train/test | ✓ Good |
| scipy KDTree for neighbors | Workaround for pymatgen Cython dtype bug on Windows | ✓ Good |

---
*Last updated: 2026-03-13 after v1.1 milestone start*
