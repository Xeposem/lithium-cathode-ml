# Stack Research

**Domain:** ML-based battery cathode property prediction
**Researched:** 2026-03-05
**Confidence:** HIGH (core stack), MEDIUM (MEGNet/matgl compatibility)

## Recommended Stack

### Core ML Framework

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| PyTorch | 2.10.0 | Deep learning framework | Industry standard, required by PyG and matgl. Latest stable release (Jan 2026). Requires Python >=3.10 | HIGH |
| PyTorch Lightning | 2.6.1 | Training orchestration | Used by matgl for MEGNet/M3GNet training. Handles logging, checkpointing, GPU management out of the box. Reduces boilerplate significantly | HIGH |
| scikit-learn | 1.8.0 | Baseline ML models | Random Forest, Gradient Boosting, SVR baselines. v1.8 adds native Array API support for GPU arrays. Supports Python 3.11-3.14 | HIGH |

### Graph Neural Networks

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| PyTorch Geometric (PyG) | 2.7.0 | GNN framework for CGCNN | The standard GNN library for PyTorch. Ships CGConv layer (CGCNN operator) natively. Actively maintained, conda no longer supported (use pip). Minimal install works without external libs from v2.3+ | HIGH |
| matgl | 1.3.0 | MEGNet implementation | Official MEGNet/M3GNet library from Materials Virtual Lab. v1.3.0 (Aug 2025) has pretrained models and PyG framework support. **Use v1.3.0, not v2.0.0** -- see rationale below | MEDIUM |
| DGL | 2.2.0 | Backend for matgl MEGNet | Required by matgl for MEGNet models. matgl v2.0.0 defaults to PyG but only TensorNet is ported -- MEGNet still needs DGL. Pin to 2.2.0 per matgl compatibility | MEDIUM |

**Critical matgl version decision:** Use matgl v1.3.0 (not v2.0.0). In v2.0.0, MEGNet has NOT been ported to PyG yet -- only TensorNet has. v2.0.0 also pins PyTorch to 2.3.0 and numpy<2 for DGL compatibility, which conflicts with the rest of the stack. v1.3.0 is more stable for MEGNet training with PyTorch Lightning.

**CGCNN approach:** Implement CGCNN using PyG's built-in `torch_geometric.nn.conv.CGConv` layer rather than the original txie-93/cgcnn repo (which targets PyTorch 1.x and has no active maintenance). PyG's implementation is current and integrates with standard PyG data pipelines.

### Materials Science Libraries

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| pymatgen | 2025.10.7 | Crystal structure processing | Powers the Materials Project. CIF parsing, structure manipulation, symmetry analysis, Voronoi neighbor detection for graph construction. Required by matgl | HIGH |
| ASE | 3.27.0 | Atomic simulation toolkit | Complementary to pymatgen for structure I/O, visualization backends, and format conversion. Useful for OQMD data processing | HIGH |
| mp-api | 0.45.x | Materials Project API client | Official Python client for MP. Provides MPRester for querying structures, battery data, formation energies. Requires free API key | HIGH |
| qmpy-rester | 0.2.0 | OQMD API client | Python wrapper for OQMD REST API. No API key required. Last updated 2019 but the API is stable and functional. Supplement with direct HTTP requests if needed | LOW |

### Data & Numerical

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| numpy | >=1.26,<2.0 | Numerical computing | Pin to numpy<2 for DGL/matgl compatibility. This is a known constraint | HIGH |
| pandas | >=2.1 | Data manipulation | Tabular data handling for datasets, feature engineering, results aggregation | HIGH |
| scipy | >=1.12 | Scientific computing | Distance calculations for graph construction, statistical analysis | HIGH |

### Visualization & Dashboard

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Streamlit | 1.55.0 | Web dashboard framework | Best fit for academic/portfolio ML dashboards. Simpler than Dash, more dashboard-capable than Gradio. Native support for plots, tables, interactive widgets. Python-only (no JS needed) | HIGH |
| plotly | 6.5.2 | Interactive charts | Publication-quality interactive plots. Native Streamlit integration. 30+ chart types including 3D, statistical, and scientific charts | HIGH |
| matplotlib | >=3.8 | Static publication plots | Required for paper-quality figures. Export to PDF/SVG. Seaborn for statistical plots | HIGH |
| py3Dmol | 2.5.4 | 3D crystal structure viewer | Interactive 3D visualization in web/notebook. Crystal structure support via pymatgen CIF export. Embeddable in Streamlit via stmol | HIGH |
| stmol | >=0.0.9 | Streamlit + py3Dmol bridge | Wraps py3Dmol for Streamlit apps. Supports crystal structure rendering with unit cell repetition. Published in Frontiers in Molecular Biosciences | MEDIUM |

### Development & Reproducibility

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| Python | 3.11 | Runtime | Best compatibility across all packages. 3.12+ may have issues with DGL/older packages |
| conda/mamba | latest | Environment management | Preferred over bare pip for scientific Python. Handles CUDA toolkit dependencies |
| wandb or mlflow | latest | Experiment tracking | Track training runs, hyperparameters, metrics. wandb for cloud, mlflow for local-only |
| jupyter | latest | Notebooks | Exploration, prototyping, paper figures |
| pytest | latest | Testing | Unit tests for data pipeline, model I/O |
| black + ruff | latest | Code formatting/linting | Consistent code style |

## Installation

```bash
# Create environment
conda create -n cathode-ml python=3.11
conda activate cathode-ml

# PyTorch with CUDA (adjust cu version for your GPU)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu124

# PyTorch Geometric (for CGCNN)
pip install torch-geometric==2.7.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.10.0+cu124.html

# matgl for MEGNet (pins its own DGL dependency)
pip install matgl==1.3.0

# Materials science
pip install pymatgen mp-api ase qmpy-rester

# ML baselines
pip install scikit-learn==1.8.0

# Training orchestration
pip install pytorch-lightning==2.6.1

# Visualization & dashboard
pip install streamlit plotly matplotlib seaborn
pip install py3Dmol stmol

# Data & utilities
pip install pandas scipy requests tqdm

# Development
pip install pytest black ruff jupyter wandb
```

**Important:** The install order matters. Install PyTorch first, then PyG, then matgl. matgl may attempt to install its own DGL version -- verify compatibility after install.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| PyG for CGCNN | Original txie-93/cgcnn repo | Never -- original targets PyTorch 1.x, unmaintained. PyG has CGConv built-in |
| PyG for GNN framework | DGL standalone | Only if you need DGL-specific models. DGL maintenance is declining; MatGL is migrating away from it |
| matgl for MEGNet | Original megnet (TensorFlow/Keras) | Never -- original MEGNet was TF-based and deprecated. matgl is the official PyTorch successor |
| Streamlit for dashboard | Dash (Plotly) | If you need enterprise-grade multi-user dashboards or embedding in existing Flask apps. Overkill for academic/portfolio use |
| Streamlit for dashboard | Gradio | If dashboard is purely model inference demo. Gradio lacks data exploration and custom layout capabilities needed here |
| py3Dmol for 3D viz | Crystal Toolkit (by MP team) | If building a full materials explorer app. Crystal Toolkit is heavier but has native pymatgen integration. py3Dmol is lighter and sufficient for this project |
| wandb for tracking | MLflow | If you want fully local experiment tracking with no cloud dependency. MLflow runs a local server |
| pymatgen for structures | ASE only | Never for this project -- pymatgen is required by matgl and mp-api. ASE is complementary, not a replacement |
| Python 3.11 | Python 3.12 | Only if DGL drops 3.11 support. 3.12 has potential compatibility issues with DGL 2.2.0 |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| TensorFlow/Keras | Project specifies PyTorch. Original MEGNet was TF-based but matgl replaces it with PyTorch | PyTorch + matgl |
| Original megnet package | Deprecated TF implementation, no longer maintained by Materials Virtual Lab | matgl (PyTorch reimplementation) |
| Original txie-93/cgcnn | Targets PyTorch 1.x, no pip package, manual data pipeline | PyG CGConv layer |
| DGL as primary GNN framework | Declining maintenance, MatGL migrating away. Only use as matgl backend dependency | PyTorch Geometric |
| matgl v2.0.0 | MEGNet not yet ported to PyG. Pins PyTorch 2.3.0 and numpy<2 which conflicts with modern stack | matgl v1.3.0 |
| numpy >= 2.0 | Breaks DGL 2.2.0 compatibility which matgl requires for MEGNet | numpy <2.0 (>=1.26) |
| Flask/Django for dashboard | Massive overkill for academic ML dashboard. Requires separate frontend | Streamlit |
| Panel/Voila | Less community support, fewer components than Streamlit for ML dashboards | Streamlit |
| qmpy (full package) | Requires PostgreSQL database setup, designed for hosting OQMD locally. Extremely heavy | qmpy-rester (REST API client) |

## Version Compatibility Matrix

This is the critical compatibility chain. Versions must be coordinated.

| Package | Compatible With | Constraint Source | Notes |
|---------|-----------------|-------------------|-------|
| PyTorch 2.10.0 | PyG 2.7.0 | PyG docs | PyG 2.7 supports PyTorch 2.5-2.8+ |
| matgl 1.3.0 | DGL 2.2.0, PyTorch ~2.3.0 | matgl pinned deps | **Conflict risk:** matgl pins older PyTorch. May need to test with 2.10.0 or use separate env |
| DGL 2.2.0 | numpy <2.0 | DGL requirement | Hard constraint on numpy version |
| PyTorch Lightning 2.6.1 | PyTorch 2.10.0 | Lightning docs | Full support |
| scikit-learn 1.8.0 | Python 3.11-3.14 | sklearn docs | No conflicts |
| pymatgen 2025.10.7 | mp-api 0.45.x | Co-developed | Both from Materials Project team |
| Streamlit 1.55.0 | plotly 6.x, py3Dmol 2.x | Streamlit ecosystem | Native plotly support |
| stmol | py3Dmol, Streamlit | stmol deps | Bridge component |

### Known Compatibility Risk

**matgl/DGL vs modern PyTorch:** matgl v1.3.0 pins PyTorch to ~2.3.0 for DGL compatibility. Running MEGNet training with PyTorch 2.10.0 is untested and may fail. Mitigation strategies:

1. **Preferred:** Use a separate conda environment for MEGNet training (PyTorch 2.3.0 + matgl 1.3.0 + DGL 2.2.0) and export trained models for inference in the main environment
2. **Alternative:** Test matgl 1.3.0 with PyTorch 2.10.0 -- it may work if DGL 2.2.0 loads successfully
3. **Fallback:** Implement MEGNet architecture from scratch in pure PyG, using matgl as reference only

The CGCNN side (PyG) has no such conflict and runs cleanly on PyTorch 2.10.0.

## Data Source Access Summary

| Source | API | Client Library | Auth | Data Available |
|--------|-----|----------------|------|----------------|
| Materials Project | REST API v3 | mp-api (MPRester) | Free API key (register at materialsproject.org) | Crystal structures, formation energies, band gaps, battery data, ~150K materials |
| OQMD | REST API v1.4 | qmpy-rester | None required | ~700K DFT calculations, formation energies, structures |
| Battery Data Genome | Mixed (download + API) | Direct HTTP / requests | None | Battery cycling data, cathode/anode performance data |
| Battery Archive | REST API | Direct HTTP / requests | None | Cycling data, capacity fade, publicly contributed datasets |

**Note on Battery Data Genome:** This is not a single API endpoint but a collection of datasets and tools. Primary battery cathode data will come from Materials Project's battery explorer (accessible via mp-api). OQMD supplements with additional DFT-computed properties. Battery Archive provides experimental cycling data which is complementary but secondary for crystal-structure-based property prediction.

## Sources

- [PyTorch Releases](https://github.com/pytorch/pytorch/releases) - Version history and compatibility
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html) - Installation and version compatibility
- [PyG CGConv Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CGConv.html) - CGCNN operator in PyG
- [MatGL Home](https://matgl.ai/) - MEGNet/M3GNet PyTorch implementation
- [MatGL PyPI](https://pypi.org/project/matgl/) - Version history, v2.0.0 changelog
- [MatGL Change Log](https://matgl.ai/changes.html) - Backend migration details
- [MatGL Nature Paper](https://www.nature.com/articles/s41524-025-01742-y) - Architecture and design decisions
- [MatGL MEGNet Training Tutorial](https://matgl.ai/tutorials/Training%20a%20MEGNet%20Formation%20Energy%20Model%20with%20PyTorch%20Lightning.html)
- [pymatgen](https://pymatgen.org/) - Crystal structure processing
- [mp-api PyPI](https://pypi.org/project/mp-api/) - Materials Project Python client
- [Materials Project API Documentation](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started)
- [OQMD REST API](http://oqmd.org/static/docs/restful.html) - OQMD data access
- [qmpy_rester GitHub](https://github.com/mohanliu/qmpy_rester) - OQMD Python wrapper
- [ASE PyPI](https://pypi.org/project/ase/) - Atomic Simulation Environment
- [scikit-learn 1.8 Release](https://scikit-learn.org/stable/whats_new.html) - Array API support
- [Streamlit PyPI](https://pypi.org/project/streamlit/) - Dashboard framework
- [Plotly PyPI](https://pypi.org/project/plotly/) - Interactive visualization
- [py3Dmol PyPI](https://pypi.org/project/py3Dmol/) - 3D molecular visualization
- [stmol GitHub](https://github.com/napoles-uach/stmol) - Streamlit py3Dmol component
- [stmol Publication](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2022.990846/full) - Frontiers in Molecular Biosciences
- [PyTorch Lightning PyPI](https://pypi.org/project/pytorch-lightning/) - Training orchestration
- [DGL GitHub](https://github.com/dmlc/dgl) - Deep Graph Library status

---
*Stack research for: ML-based lithium-ion battery cathode property prediction*
*Researched: 2026-03-05*
