# Feature Landscape

**Domain:** ML-based battery cathode performance prediction platform
**Researched:** 2026-03-05
**Confidence:** HIGH (well-established domain with clear precedents from Materials Project, Matbench, and published GNN benchmarks)

## Table Stakes (Users Expect These)

Features that any serious ML materials prediction project must have. Without these, the project lacks academic credibility or basic usability.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **API-based data ingestion** (Materials Project, OQMD) | Reproducibility is non-negotiable in academic work; manual data downloads are not reproducible | Medium | MP v2 API uses mp-api Python client. OQMD uses REST. Battery Data Genome may require custom scraping. Rate limiting and caching needed. |
| **Crystal structure to graph conversion** | Core input representation for GNN models; atoms as nodes, bonds as edges with distance-based cutoffs | Medium | Use pymatgen's `Structure` object as intermediate. Gaussian distance expansion for edge features. Configurable cutoff radius (typically 8 angstroms). |
| **CGCNN implementation** | The foundational crystal GNN; expected baseline in any crystal property prediction study | High | PyTorch Geometric preferred over raw PyTorch. Original paper uses specific architecture (64-dim node features, 2 conv layers, pooling). |
| **MEGNet implementation** | Second leading crystal GNN; its global state features improve over CGCNN. Comparing both is the stated project goal | High | MEGNet adds edge and global state updates beyond node features. Can use DGL or PyG backend. Must match original architecture for credible comparison. |
| **Traditional ML baselines** (RF, GBM, XGBoost) | Establishes whether GNNs actually outperform simpler approaches. Reviewers expect this comparison | Low | Use scikit-learn with composition-based features (Magpie descriptors via matminer). Fast to implement. |
| **Multi-property prediction** (voltage, capacity, stability, formation energy) | Single-property prediction is too narrow for a cathode platform | Medium | Research suggests separate models per property outperform multi-output for crystal GNNs. Start with separate, optionally explore multi-task. |
| **Standard evaluation metrics** (MAE, RMSE, R-squared) | Matbench standard; required for any comparison to published results | Low | Use nested cross-validation (5-fold, seed 18012019) per Matbench convention. Report per-property metrics. |
| **Train/val/test split with fixed seeds** | Basic reproducibility requirement | Low | Stratify by crystal system or composition space to avoid data leakage. Document split strategy. |
| **Publication-quality benchmark plots** | The primary output artifact for an academic project | Medium | Parity plots (predicted vs actual), bar charts comparing model MAE, learning curves. Use matplotlib with consistent styling. |
| **Data preprocessing pipeline** | Filtering invalid structures, handling missing values, outlier removal, feature normalization | Medium | Remove structures with negative formation energy artifacts, filter by energy above hull, handle partial occupancy. Document every filter. |
| **Dependency pinning and environment file** | Minimum reproducibility standard | Low | `requirements.txt` or `environment.yml` with exact versions. Pin PyTorch, PyG, pymatgen versions. |
| **Configuration management** | Hyperparameters, paths, model settings in config files not hardcoded | Low | YAML config files. Separate configs per model. Override via CLI args. |

## Differentiators (Competitive Advantage)

Features that elevate the project from "another GNN benchmark" to a standout portfolio piece or paper-worthy contribution.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Interactive web dashboard** (model comparison) | Most academic projects stop at Jupyter notebooks. A live dashboard demonstrates engineering skill and makes results accessible | High | Streamlit is the right choice: fast to build, good py3Dmol integration via stmol, built-in caching. Dash is overkill for this scope. |
| **Materials explorer panel** | Browse the dataset interactively with filtering by voltage, capacity, formation energy, element composition, stability | Medium | Searchable/filterable table with column sorting. Link rows to crystal structure viewer. Inspired by Materials Project explorer. |
| **Materials discovery panel** | Show top predicted candidates ranked by desirable properties (high voltage + high capacity + low energy above hull) | Medium | Pareto front visualization. Multi-objective ranking. Highlight materials not yet experimentally validated. |
| **Crystal structure 3D viewer** | Interactive visualization of crystal structures in the browser; connects predictions to physical intuition | Medium | py3Dmol via stmol (Streamlit component). Support ball-and-stick, polyhedral views. Show unit cell boundaries. Color by element. |
| **Training curves dashboard** | Live or stored loss curves, learning rate schedules, convergence comparison across models | Low | Log metrics per epoch during training. Plot loss, validation MAE, LR schedule. Compare convergence speed across CGCNN/MEGNet. |
| **Experiment tracking integration** | Systematic logging of all training runs with hyperparameters, metrics, artifacts | Medium | Use MLflow (open-source, self-hosted) over W&B for an academic project. Log configs, metrics per epoch, model checkpoints, and dataset versions. |
| **Feature importance / model interpretability** | Explain which atomic features or structural motifs drive predictions | High | For traditional ML: SHAP values on Magpie descriptors. For GNNs: attention weight visualization, GradCAM on graph nodes. Academic differentiator. |
| **Matbench-compatible evaluation** | Run on standard Matbench tasks to place results on the public leaderboard | Medium | Matbench provides standardized datasets and splits. Even partial compatibility (e.g., formation energy task) adds credibility. |
| **Automated data pipeline** (fetch, clean, featurize, split) | End-to-end automation from raw API data to model-ready tensors with a single command | Medium | CLI entry point: `python pipeline.py --fetch --clean --featurize`. Cache intermediate results. Idempotent reruns. |
| **Hyperparameter tuning framework** | Systematic search rather than manual tuning; critical for fair model comparison | Medium | Use Optuna for Bayesian optimization. Define search spaces per model. Log all trials to MLflow. |
| **Composition-based feature engineering** (Magpie descriptors) | Bridges traditional ML and GNN comparison; uses matminer featurizers | Low | matminer's `ElementProperty` featurizer generates ~132 Magpie descriptors per composition. Standard approach for ML baselines. |

## Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Generative / inverse material design** | Sounds impressive; "AI discovers new batteries" | Completely different problem domain (VAE/GAN architectures, validity checking, synthesizability). Scope explosion for a prediction project. Out of scope per PROJECT.md. | Focus on discovery panel ranking known/predicted materials by desirable properties. |
| **Real-time model retraining in dashboard** | Seems interactive and modern | Training GNNs takes hours. Real-time retraining in a web app is impractical and adds massive complexity (job queues, GPU management). Out of scope per PROJECT.md. | Show pre-trained results. Allow users to select model checkpoints. |
| **Cloud deployment / production API** | "Deploy to AWS/GCP for others to use" | Adds DevOps complexity (containers, CI/CD, scaling, auth) with no academic value. Out of scope per PROJECT.md. | Local Streamlit app. Docker container for reproducible local deployment if desired. |
| **Mobile-responsive dashboard** | "Works on phones" | Crystal structure viewers and data tables are fundamentally desktop experiences. Mobile optimization wastes time. Out of scope per PROJECT.md. | Desktop-first Streamlit layout. Acceptable for academic/portfolio use. |
| **Custom GNN architecture invention** | Tempting to propose a "novel" architecture | Without significant theoretical contribution, a slightly modified CGCNN/MEGNet adds noise, not signal. Reviewers see through this. | Faithfully implement and fairly compare established architectures. Novelty comes from the battery-specific application and analysis. |
| **Comprehensive database of all battery types** | "Cover all chemistries" | Lithium cathodes alone have thousands of entries. Adding anodes, electrolytes, sodium-ion, etc. dilutes focus and multiplies data cleaning work. | Scope to lithium-ion cathode materials specifically. Mention extensibility in documentation. |
| **Transfer learning from pre-trained foundation models** | Hot topic in materials ML (2025-2026) | Pre-trained materials models (e.g., from JARVIS or Matbench Discovery) require careful domain adaptation. Adds complexity without guaranteed benefit for this focused domain. | Mention as future work. Train from scratch on curated cathode dataset. |

## Feature Dependencies

```
Data Ingestion (API clients)
  --> Data Preprocessing (cleaning, filtering)
    --> Crystal-to-Graph Conversion
      --> CGCNN Model
      --> MEGNet Model
    --> Composition Feature Engineering (Magpie)
      --> Traditional ML Baselines (RF, XGBoost)
    --> Train/Val/Test Splitting
      --> All model training

All Model Training
  --> Evaluation Metrics (MAE, RMSE, R2)
    --> Benchmark Comparison Plots
    --> Dashboard: Model Comparison Tab

Data Preprocessing
  --> Materials Explorer (needs clean dataset)
  --> Materials Discovery Panel (needs predictions + dataset)

Crystal-to-Graph Conversion
  --> Crystal Structure Viewer (needs pymatgen Structure objects)

Experiment Tracking (MLflow)
  --> Training Curves Dashboard (reads from MLflow logs)
  --> Hyperparameter Tuning (logs to MLflow)
```

Key dependency chains:
1. **Data pipeline must come first**: Nothing works without ingested, cleaned data
2. **Graph conversion before GNNs**: CGCNN and MEGNet need graph representations
3. **All models trained before dashboard**: Dashboard displays results, not generates them
4. **Evaluation before visualization**: Can't plot what hasn't been measured

## MVP Definition

### Launch With (v1) -- Core ML Pipeline

The minimum credible academic ML project for cathode property prediction:

1. **Data ingestion** from Materials Project API (largest, best-documented source)
2. **Data preprocessing** with documented filters and cleaning steps
3. **Crystal-to-graph conversion** using pymatgen + PyG
4. **CGCNN model** implementation and training
5. **MEGNet model** implementation and training
6. **Random Forest baseline** with Magpie descriptors
7. **Evaluation** with MAE, RMSE, R-squared on proper cross-validation splits
8. **Benchmark comparison plots** (parity plots, model comparison bars)
9. **Fixed seeds and pinned dependencies** for reproducibility
10. **YAML configuration** for all hyperparameters

### Add After Validation (v1.x) -- Dashboard and Exploration

Once the ML pipeline produces reliable results:

1. **Streamlit dashboard** with model comparison tab
2. **Training curves** visualization
3. **Materials explorer** with filtering and search
4. **Crystal structure viewer** (py3Dmol via stmol)
5. **Additional data sources** (OQMD, Battery Data Genome)
6. **XGBoost/GBM baselines** alongside Random Forest
7. **MLflow experiment tracking** integration
8. **Hyperparameter tuning** with Optuna

### Future Consideration (v2+) -- Discovery and Interpretability

Polish and differentiation features:

1. **Materials discovery panel** with Pareto-optimal candidate ranking
2. **Feature importance / SHAP analysis** for traditional ML models
3. **GNN interpretability** (attention visualization, node importance)
4. **Matbench-compatible evaluation** on standard tasks
5. **Multi-task learning** experiments (shared encoder, property-specific heads)
6. **Docker container** for one-command reproducible execution

## Feature Prioritization Matrix

| Feature | Impact | Effort | Priority | Phase |
|---------|--------|--------|----------|-------|
| MP API data ingestion | Critical | Medium | P0 | v1 |
| Data preprocessing pipeline | Critical | Medium | P0 | v1 |
| Crystal-to-graph conversion | Critical | Medium | P0 | v1 |
| CGCNN implementation | Critical | High | P0 | v1 |
| MEGNet implementation | Critical | High | P0 | v1 |
| RF baseline (Magpie features) | High | Low | P0 | v1 |
| Evaluation metrics + CV | Critical | Low | P0 | v1 |
| Benchmark plots | High | Medium | P0 | v1 |
| Config management (YAML) | High | Low | P0 | v1 |
| Reproducibility (seeds, deps) | Critical | Low | P0 | v1 |
| Streamlit dashboard | High | High | P1 | v1.x |
| Materials explorer | High | Medium | P1 | v1.x |
| Crystal structure viewer | Medium | Medium | P1 | v1.x |
| Training curves tab | Medium | Low | P1 | v1.x |
| Additional data sources | Medium | Medium | P1 | v1.x |
| MLflow integration | Medium | Medium | P1 | v1.x |
| Optuna hyperparameter tuning | Medium | Medium | P1 | v1.x |
| Discovery panel | Medium | Medium | P2 | v2+ |
| SHAP / interpretability | Medium | High | P2 | v2+ |
| GNN attention visualization | Low | High | P2 | v2+ |
| Matbench compatibility | Medium | Medium | P2 | v2+ |
| Docker reproducibility | Low | Low | P2 | v2+ |

## Sources

- [Matbench: Benchmarks for materials science property prediction](https://matbench.materialsproject.org/)
- [Matbench Discovery: ML crystal stability predictions (Nature Machine Intelligence, 2025)](https://matbench-discovery.materialsproject.org/)
- [Materials Project Documentation](https://docs.materialsproject.org/)
- [Scalable deeper GNNs for materials property prediction (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9122959/)
- [Stmol: Streamlit molecular visualization component](https://github.com/napoles-uach/stmol)
- [State-of-the-Art ML for Lithium Battery Cathode Design (Adv. Energy Mater. 2025)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aenm.202405300)
- [Examining GNNs for crystal structures (Science Advances)](https://www.science.org/doi/10.1126/sciadv.adi3245)
- [Application-oriented ML for battery science (npj Computational Materials 2025)](https://www.nature.com/articles/s41524-025-01575-9)
- [ML Experiment Tracking Tools Comparison](https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/)

---
*Feature research for: ML-based lithium-ion battery cathode performance prediction*
*Researched: 2026-03-05*
