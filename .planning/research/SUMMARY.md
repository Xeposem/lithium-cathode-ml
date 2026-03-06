# Project Research Summary

**Project:** Lithium-Ion Battery Cathode Performance Prediction
**Domain:** Materials informatics / ML-based crystal property prediction
**Researched:** 2026-03-05
**Confidence:** HIGH (core ML pipeline), MEDIUM (MEGNet/matgl compatibility)

## Executive Summary

This project is a crystal-structure-based ML benchmarking platform: it predicts cathode material properties (formation energy, voltage, stability) from crystal structures using graph neural networks (CGCNN and MEGNet) and compares them against traditional ML baselines. This is a well-established problem class in materials informatics with clear precedents from Matbench, the Materials Project, and dozens of published GNN benchmarks. Experts build these systems bottom-up: data pipeline first (API ingestion, cleaning, graph construction), then models (baselines before GNNs), then evaluation, then visualization. The recommended approach follows this exact sequence with PyTorch Geometric as the unified GNN framework, matgl v1.3.0 for MEGNet, and Streamlit for the interactive dashboard.

The primary technical risk is the matgl/DGL compatibility chain. matgl v1.3.0 pins PyTorch to ~2.3.0 for DGL compatibility, while the rest of the stack targets PyTorch 2.10.0. The recommended mitigation is a separate conda environment for MEGNet training, exporting trained models for inference in the main environment. The CGCNN side (pure PyG) has no such conflict. A secondary risk is overfitting: cathode-specific datasets from Materials Project contain only 2,400-5,700 entries, which is small for GNNs with tens of thousands of parameters. Transfer learning (pre-train on full MP, fine-tune on cathodes) is essential, not optional.

The most dangerous pitfall is data leakage from polymorphic materials. Random train/test splits place polymorphs (same composition, different structures) into both sets, inflating metrics dramatically. Compositional group splitting must be designed into the data pipeline from day one -- retrofitting it invalidates all prior results. The project should also be explicit that predictions are DFT-level approximations, not experimental measurements, and should not claim to predict cycle life from crystal structure alone.

## Key Findings

### Recommended Stack

The stack centers on PyTorch + PyTorch Geometric for GNNs, with matgl providing the MEGNet implementation and pymatgen as the universal crystal structure interchange format. Key constraint: numpy must be pinned to <2.0 for DGL/matgl compatibility.

**Core technologies:**
- **PyTorch 2.10.0 + PyTorch Geometric 2.7.0**: Unified GNN framework. PyG provides native CGConv layer for CGCNN, eliminating the need for the unmaintained original CGCNN repo
- **matgl 1.3.0 (NOT v2.0.0)**: Official MEGNet implementation from Materials Virtual Lab. v2.0.0 has not ported MEGNet to PyG yet and pins conflicting dependency versions
- **DGL 2.2.0**: Required backend for matgl MEGNet. Declining maintenance; use only as matgl dependency, not as primary framework
- **pymatgen + mp-api**: Crystal structure processing and Materials Project API access. The lingua franca of materials informatics
- **scikit-learn 1.8.0**: Baseline models (Random Forest, Gradient Boosting) with Magpie descriptors via matminer
- **PyTorch Lightning 2.6.1**: Training orchestration. Used natively by matgl; handles logging, checkpointing, GPU management
- **Streamlit + Plotly + py3Dmol**: Dashboard stack. Streamlit for rapid development, Plotly for interactive charts, py3Dmol/stmol for crystal structure visualization
- **Python 3.11**: Best compatibility across all packages. 3.12+ risks DGL breakage

**Critical version constraint:** matgl v1.3.0 pins PyTorch ~2.3.0. MEGNet training likely needs a separate conda environment. CGCNN (pure PyG) runs cleanly on PyTorch 2.10.0.

### Expected Features

**Must have (table stakes):**
- API-based data ingestion from Materials Project (primary) with caching and rate limit handling
- Crystal structure to graph conversion (pymatgen Structure -> PyG Data) with configurable cutoff radius
- CGCNN and MEGNet implementations trained and compared on identical data splits
- Traditional ML baselines (Random Forest with Magpie descriptors) to establish performance floor
- Multi-property prediction: formation energy, voltage, stability (separate models per property)
- Standard evaluation: MAE, RMSE, R-squared with 5-fold cross-validation on compositional splits
- Publication-quality parity plots and model comparison bar charts
- Reproducibility: fixed seeds, pinned dependencies, YAML configuration for all hyperparameters

**Should have (differentiators):**
- Interactive Streamlit dashboard with model comparison, materials explorer, and crystal structure viewer
- Training curves visualization comparing convergence across models
- Additional data sources (OQMD, Battery Data Genome) beyond Materials Project
- Experiment tracking via MLflow
- Hyperparameter tuning with Optuna

**Defer (v2+):**
- Materials discovery panel with Pareto-optimal candidate ranking
- SHAP/GNN interpretability (attention visualization, node importance)
- Matbench leaderboard compatibility
- Docker containerization
- Multi-task learning (shared encoder, property-specific heads)

### Architecture Approach

The system follows a layered, artifact-based architecture where each stage (ingest, featurize, train, evaluate, visualize) reads from and writes to a shared artifacts store. The dashboard is fully decoupled from the training pipeline -- it reads JSON metrics, CSV results, and saved model checkpoints rather than importing training code. Two parallel featurization tracks diverge from shared clean pymatgen Structures: a GNN track (crystal graphs via PyG) and a tabular track (Magpie descriptors via matminer).

**Major components:**
1. **Data Pipeline** (src/data/) -- API ingestion, cleaning, validation, caching. Outputs clean pymatgen Structures
2. **Featurization** (src/data/graph.py + featurize.py) -- Two-track: crystal graph construction for GNNs, Magpie descriptors for sklearn
3. **Dataset Layer** (src/data/dataset.py) -- PyG InMemoryDataset for GNNs, pandas DataFrame for baselines. Shared train/val/test indices
4. **Models** (src/models/) -- CGCNN (PyG CGConv), MEGNet (matgl), baselines (sklearn)
5. **Training** (src/training/) -- PyTorch Lightning module, config-driven, per-model hyperparameters
6. **Evaluation** (src/training/evaluate.py) -- Metrics computation, parity plots, cross-model comparison
7. **Dashboard** (dashboard/) -- Streamlit multi-page app reading artifacts. Materials explorer, model comparison, crystal viewer

### Critical Pitfalls

1. **Data leakage from polymorphs** -- Use compositional group splitting (all entries with same reduced formula in same fold). Report both random-split and compositional-split metrics. Must be designed into data pipeline from day one.

2. **DFT vs experimental confusion** -- Materials Project properties are DFT approximations, not ground truth. Formation energy and voltage are in scope (DFT-derived). Cycle life and experimental capacity are NOT predictable from crystal structure alone. Be explicit about this in all outputs.

3. **GNN overfitting on small cathode datasets** -- Only 2,400-5,700 cathode entries vs tens of thousands of model parameters. Transfer learning (pre-train on full MP, fine-tune on cathodes) is essential. Also reduce model capacity and use aggressive regularization.

4. **Wrong graph construction parameters** -- Radius cutoff and max_neighbors dramatically affect quality. Start with CGCNN defaults (radius=8.0, max_neighbors=12), validate that zero graphs are disconnected, sweep parameters early.

5. **MEGNet/CGCNN training dynamics mismatch** -- MEGNet needs ~1000 epochs, lower LR (1e-4), larger batches. CGCNN needs ~400 epochs, standard LR (1e-3). Independent hyperparameter tuning per model is required for a fair comparison.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Project Foundation and Data Pipeline
**Rationale:** Everything depends on data. The data pipeline has the most pitfalls (leakage, API versioning, DFT confusion) and all downstream phases require clean, validated data. Starting here validates API access, caching, and data quality before any modeling investment.
**Delivers:** Reproducible data ingestion from Materials Project, cleaned/filtered cathode dataset as pymatgen Structures, cached raw data with version tracking, compositional group splitting strategy, YAML configuration system, project package structure (src/ layout).
**Addresses features:** API-based data ingestion, data preprocessing pipeline, train/val/test splitting, config management, dependency pinning.
**Avoids pitfalls:** P1 (data leakage -- compositional splits designed in), P2 (DFT confusion -- target properties defined), P6 (API rate limits -- caching), P12 (non-reproducibility -- seeds, versions, package structure).

### Phase 2: Featurization and Baseline Models
**Rationale:** Two-track featurization (graphs + tabular) must be built before any model training. Baselines train in seconds, validating the entire pipeline end-to-end and establishing a performance floor. If Random Forest with Magpie features is already strong, that is itself a finding.
**Delivers:** Crystal-to-graph conversion (PyG Data objects), Magpie descriptor generation via matminer, PyG InMemoryDataset, sklearn baselines (Random Forest, Gradient Boosting) with evaluation metrics, initial parity plots.
**Addresses features:** Crystal structure to graph conversion, RF baseline, composition-based features, evaluation metrics.
**Avoids pitfalls:** P3 (wrong graph params -- validate connectivity), P10 (poor feature engineering -- use Magpie preset with feature selection), P5 (wrong metrics -- establish evaluation framework here).

### Phase 3: CGCNN Implementation and Training
**Rationale:** CGCNN is the simpler GNN architecture (no global state, no DGL dependency). Runs cleanly on PyTorch 2.10.0 + PyG. Build the PyTorch Lightning training infrastructure here, which MEGNet will reuse. Transfer learning from full MP should be implemented at this stage.
**Delivers:** CGCNN model using PyG CGConv, Lightning training module with logging/checkpointing/early stopping, transfer learning pipeline (pre-train on full MP, fine-tune on cathodes), cross-validated evaluation results, parity plots.
**Addresses features:** CGCNN implementation, multi-property prediction, standard evaluation.
**Avoids pitfalls:** P4 (overfitting -- transfer learning + regularization), P7 (training dynamics -- CGCNN-specific hyperparameters).

### Phase 4: MEGNet Implementation and Training
**Rationale:** MEGNet depends on the training infrastructure from Phase 3 but has its own dependency chain (matgl v1.3.0, DGL 2.2.0, potentially separate conda environment). Isolating it in its own phase contains the compatibility risk.
**Delivers:** MEGNet model via matgl, MEGNet-specific training config (longer training, lower LR), cross-validated results on identical folds as CGCNN, fair head-to-head comparison.
**Addresses features:** MEGNet implementation, model comparison.
**Avoids pitfalls:** P7 (training mismatch -- independent hyperparameters), P4 (overfitting -- transfer learning), matgl/DGL compatibility risk contained in this phase.

### Phase 5: Evaluation, Visualization, and Benchmarking
**Rationale:** All models are trained; this phase produces the core deliverable: a rigorous cross-model comparison with publication-quality figures. Parity plots, bar charts, per-chemistry-family analysis, learning curves.
**Delivers:** Comprehensive evaluation across all models (CGCNN, MEGNet, RF, GBM), publication-quality figures, comparison tables, per-property and per-chemistry breakdowns, artifacts (JSON metrics, CSV results, saved figures) ready for dashboard consumption.
**Addresses features:** Benchmark comparison plots, publication-quality figures, multi-property evaluation.
**Avoids pitfalls:** P5 (misleading metrics -- MAE in physical units, identical folds, parity plots with systematic bias checking).

### Phase 6: Interactive Web Dashboard
**Rationale:** The dashboard reads artifacts from all prior phases. It can be stubbed with mock data during earlier phases, but full integration requires completed artifacts. This is the portfolio differentiator -- most academic projects stop at notebooks.
**Delivers:** Streamlit multi-page app with: model comparison tab (metrics tables, bar charts), materials explorer (filterable dataset browser), crystal structure 3D viewer (py3Dmol), training curves tab, prediction panel (input structure, get predictions).
**Addresses features:** All dashboard requirements from PROJECT.md.
**Avoids pitfalls:** P11 (performance -- paginate materials explorer, lazy-load structures, pre-compute aggregates).

### Phase Ordering Rationale

- **Strict bottom-up dependency chain:** Data -> featurization -> baselines -> GNNs -> evaluation -> dashboard. Each phase produces artifacts consumed by the next.
- **Baselines before GNNs:** Validates the pipeline end-to-end in minutes rather than hours. Establishes performance floor. If baselines are strong, the GNN comparison is more interesting, not less.
- **CGCNN before MEGNet:** CGCNN has zero dependency conflicts (pure PyG). Build training infrastructure on the clean path first, then tackle matgl/DGL complexity.
- **Evaluation as a separate phase:** Forces rigorous, cross-model comparison rather than ad-hoc per-model evaluation. Produces all artifacts the dashboard needs.
- **Dashboard last:** Maximum impact with minimum re-work. All data and results are finalized before building the presentation layer.
- **Parallelism opportunity:** Phases 3 and 4 (CGCNN and MEGNet) can be developed in parallel after Phase 2, since they share the same dataset but use different frameworks.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (MEGNet):** matgl v1.3.0 compatibility with PyTorch 2.10.0 is untested. May need separate conda environment. DGL installation on Windows adds complexity. Research the exact working version combination before implementation.
- **Phase 3 (CGCNN Transfer Learning):** Pre-training on full MP dataset (~150K entries) has data download and training time implications. Research optimal pre-training strategy (which properties, how many epochs, frozen layers during fine-tuning).

Phases with standard patterns (skip research-phase):
- **Phase 1 (Data Pipeline):** Well-documented MP API with official Python client. Standard pymatgen patterns. Caching and splitting are solved problems.
- **Phase 2 (Featurization + Baselines):** PyG graph construction and matminer Magpie features are thoroughly documented with tutorials.
- **Phase 5 (Evaluation):** Matbench provides standardized evaluation methodology. Matplotlib/Plotly plotting is routine.
- **Phase 6 (Dashboard):** Streamlit is well-documented with many ML dashboard examples. stmol/py3Dmol integration has published tutorials.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH (core), MEDIUM (matgl compat) | PyTorch + PyG + pymatgen are proven. matgl v1.3.0 + DGL + PyTorch version pinning is the main uncertainty. |
| Features | HIGH | Well-established domain with clear precedents from Matbench, published GNN benchmarks, and Materials Project tools. Feature landscape is well-mapped. |
| Architecture | HIGH | Layered artifact-based architecture is standard for ML benchmarking projects. Two-track featurization is the established pattern. |
| Pitfalls | HIGH | Extensively documented in literature. Data leakage, overfitting, and DFT-vs-experimental confusion are the top three risks, all with known mitigations. |

**Overall confidence:** HIGH. This is a well-trodden domain with clear patterns. The only MEDIUM-confidence area is the matgl/DGL dependency chain, which is contained in Phase 4 with known fallback strategies.

### Gaps to Address

- **matgl v1.3.0 + PyTorch 2.10.0 compatibility:** Untested combination. Validate early in Phase 4 or accept the separate-environment approach. Fallback: implement MEGNet from scratch in pure PyG using matgl as reference.
- **Cathode dataset size:** Exact count depends on MP filtering criteria (element set, property availability). Run a quick API query in Phase 1 to determine actual dataset size and whether transfer learning is strictly necessary or merely beneficial.
- **OQMD data quality:** qmpy-rester is unmaintained (last update 2019). The REST API works but may need direct HTTP requests as fallback. OQMD is supplementary, not primary.
- **Dashboard framework disagreement:** STACK.md recommends Streamlit; ARCHITECTURE.md recommends Dash. Recommendation: **use Streamlit**. The dataset is small (thousands, not hundreds of thousands of entries), Streamlit's caching (`@st.cache_data`) handles this scale, and Streamlit is significantly faster to develop. Dash's callback complexity is not warranted for this project scope.
- **Battery Data Genome access:** Not a single API endpoint but a collection of datasets. May require custom scraping. Defer to Phase 6 or v1.x scope.

## Sources

### Primary (HIGH confidence)
- [Materials Project API Documentation](https://docs.materialsproject.org/) -- data access, battery data, API versioning
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) -- CGConv layer, installation, version compatibility
- [MatGL Documentation](https://matgl.ai/) -- MEGNet implementation, training tutorials, version migration
- [Matbench](https://matbench.materialsproject.org/) -- evaluation methodology, train/test splitting, benchmarks
- [pymatgen Documentation](https://pymatgen.org/) -- crystal structure processing, neighbor finding

### Secondary (MEDIUM confidence)
- [MatGL Nature Paper](https://www.nature.com/articles/s41524-025-01742-y) -- architecture decisions, DGL-to-PyG migration rationale
- [Leakage and the reproducibility crisis (Patterns, 2023)](https://www.sciencedirect.com/science/article/pii/S2666389923001599) -- data leakage in ML for science
- [Examining GNNs for crystal structures (Science Advances)](https://www.science.org/doi/10.1126/sciadv.adi3245) -- GNN limitations, degenerate representations
- [Application-oriented ML for battery science (npj Comp. Mater., 2025)](https://www.nature.com/articles/s41524-025-01575-9) -- domain-specific pitfalls
- [State-of-the-Art ML for Lithium Battery Cathode Design (Adv. Energy Mater. 2025)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aenm.202405300)

### Tertiary (LOW confidence)
- [qmpy_rester GitHub](https://github.com/mohanliu/qmpy_rester) -- OQMD access; unmaintained since 2019, API still functional
- [Battery Data Genome](https://www.nist.gov/programs-projects/battery-data-genome) -- collection of datasets, access method unclear

---
*Research completed: 2026-03-05*
*Ready for roadmap: yes*
