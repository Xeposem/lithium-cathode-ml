# Pitfalls Research

**Domain:** ML-based lithium-ion battery cathode performance prediction (CGCNN/MEGNet on Materials Project data)
**Researched:** 2026-03-05
**Confidence:** HIGH (well-documented domain with extensive literature on failure modes)

---

## Critical Pitfalls

These cause fundamentally wrong results, wasted months, or full rewrites.

### Pitfall 1: Data Leakage from Structurally Similar / Polymorphic Materials

**What goes wrong:** Random train/test splits place polymorphs (same composition, different crystal structures) or near-identical compositions into both train and test sets. The model memorizes compositional identity rather than learning structure-property relationships. Test MAE looks excellent but the model fails on genuinely novel compositions.

**Why it happens:** Materials Project contains multiple entries for the same chemical formula (e.g., several CaMn2O4 polymorphs -- marokite, spinel, CaFe2O4 types). Random splitting guarantees some polymorphs leak across the boundary. Additionally, substitutional variants (e.g., LiNi0.5Mn0.5O2 vs LiNi0.6Mn0.4O2) share nearly identical graph topology, creating a subtler form of leakage.

**How to avoid:**
- Use **compositional grouping** for train/test splits: all entries sharing the same reduced formula must land in the same fold. Follow the Matbench Discovery approach (OOD-Composition split).
- Additionally filter by **structure prototype**: remove entries with matching prototype labels across splits, following the WBM/MP prototype filtering methodology.
- Report both random-split and compositional-split metrics to quantify the leakage effect. A large gap between them is itself a finding worth reporting.

**Warning signs:**
- Test R-squared above 0.95 on the first attempt with a simple model
- Performance degrades dramatically when predicting on a held-out composition family (e.g., train on oxides, test on phosphates)
- Model predicts nearly identical values for all polymorphs of a composition

**Phase to address:** Data pipeline / preprocessing phase. This must be designed into the splitting strategy from day one -- retrofitting it invalidates all prior evaluation.

---

### Pitfall 2: Treating Computed DFT Properties as Ground Truth

**What goes wrong:** Materials Project properties (formation energy, voltage, band gap) are DFT-computed approximations, not experimental measurements. Models trained on these values learn to replicate DFT errors, not physical reality. Capacity and cycle life are not directly available from DFT and must be derived or sourced separately.

**Why it happens:** The MP API returns clean numerical values that feel like ground truth. Researchers forget that GGA-level DFT systematically underestimates band gaps by ~40%, overestimates voltages for certain chemistries, and cannot capture kinetic properties like cycle life at all.

**How to avoid:**
- Clearly separate DFT-derived targets (voltage, formation energy, stability) from experimentally-derived targets (capacity, cycle life, Coulombic efficiency). The Battery Data Genome is more appropriate for experimental properties.
- Report that predictions are "DFT-level" not "experimental-level" in all outputs.
- For voltage prediction specifically, use the InsertionElectrode data from MP which provides computed intercalation voltages (approximately 3,014 voltage pairs available), but acknowledge systematic DFT bias.
- Do NOT attempt to predict cycle life from crystal structure alone -- it depends on electrode engineering, electrolyte, and cycling protocol, not just the cathode material.

**Warning signs:**
- Claiming to predict "battery performance" when actually predicting DFT formation energy
- Combining DFT targets with experimental targets in the same multi-output model without accounting for different error distributions
- Validation against experimental literature shows systematic offsets

**Phase to address:** Data collection and target definition phase. Decide early which properties are in scope and which data source provides each.

---

### Pitfall 3: Crystal Graph Construction with Wrong Radius/Neighbor Parameters

**What goes wrong:** The radius cutoff and max_neighbors parameters for building crystal graphs dramatically affect model quality but have no single correct value. Too small a radius (e.g., 4 Angstrom) produces disconnected graphs for some structures. Too large (e.g., 12 Angstrom) creates dense graphs that are slow to process and include physically meaningless long-range edges. Wrong max_neighbors truncates coordination environments.

**Why it happens:** Different CGCNN implementations use different defaults (radius: 4-8 Angstrom, max_neighbors: 8-12). Researchers copy defaults without verifying they work for their specific material class. Cathode materials with large unit cells (layered oxides, olivines) have different optimal parameters than simple binary compounds.

**How to avoid:**
- Use CGCNN defaults as starting point: radius=8.0, max_neighbors=12, step=0.2 for Gaussian distance expansion.
- **Validate graph connectivity**: after construction, check that zero graphs are disconnected. Log any structures that produce isolated nodes.
- Run a hyperparameter sweep on radius (6, 8, 10) and max_neighbors (8, 12, 16) early in development.
- Use pymatgen's `Structure.get_neighbors()` for consistent neighbor finding.

**Warning signs:**
- NaN losses during training (often caused by disconnected graphs producing undefined pooling)
- Certain material families (e.g., large-cell layered structures) consistently have high prediction error
- Graph construction produces wildly different numbers of edges per node across the dataset

**Phase to address:** Feature engineering / graph construction phase. Must be validated before any model training begins.

---

### Pitfall 4: Overfitting GNNs on Small Battery Cathode Datasets

**What goes wrong:** Battery-specific subsets of Materials Project are small (approximately 2,400-5,700 cathode entries depending on filtering). CGCNN and MEGNet have tens of thousands of parameters. The model memorizes the training set and produces meaningless predictions on new materials.

**Why it happens:** Researchers train deep GNNs designed for the full MP dataset (~150k entries) on a filtered cathode-only subset without adjusting model capacity. MEGNet's global state vector and attention mechanisms exacerbate this with additional parameters. More message-passing layers cause more overfitting on small datasets due to information oversquashing.

**How to avoid:**
- **Transfer learning is essential**: Pre-train CGCNN/MEGNet on the full Materials Project dataset for formation energy prediction, then fine-tune on the cathode-specific target. This is well-established practice and dramatically improves performance on small datasets.
- Reduce model capacity for direct training: fewer convolutional layers (2-3 instead of 5+), smaller hidden dimensions.
- Use aggressive regularization: dropout (0.2-0.3), weight decay, early stopping with patience of 50-100 epochs.
- Track train vs. validation loss curves religiously. A growing gap after epoch 50-100 means overfitting.
- Consider the simpler baseline first: if RF or XGBoost with Magpie/matminer features matches GNN performance on your small dataset, report that honestly.

**Warning signs:**
- Training loss reaches near-zero while validation loss plateaus or increases
- Train R-squared > 0.99 but test R-squared < 0.7
- Adding more GNN layers or parameters does not improve validation performance
- The traditional ML baseline performs comparably to the GNN

**Phase to address:** Model training phase. But the decision to use transfer learning must be made during architecture design.

---

### Pitfall 5: Wrong or Misleading Evaluation Metrics

**What goes wrong:** Reporting only R-squared on a random split gives an inflated sense of model quality. MAE in eV/atom means nothing to a battery scientist without conversion to practically relevant units. Comparing CGCNN vs MEGNet vs baseline on different splits or different subsets produces meaningless rankings.

**Why it happens:** R-squared is sensitive to outliers and range of target values -- a wide-range dataset can show high R-squared even with poor absolute accuracy. Researchers report whichever metric looks best rather than the one that matters for the application.

**How to avoid:**
- **Primary metric**: MAE in physically meaningful units (mV for voltage, mAh/g for capacity, eV/atom for formation energy).
- **Always report**: MAE, RMSE, and R-squared together. RMSE penalizes large errors which matter for materials screening.
- Use **identical cross-validation folds** (fixed seed, same splitting strategy) across all models. Matbench uses KFold with 5 splits, seed 18012019.
- Report **per-chemistry-family** metrics (layered oxides vs spinels vs olivines) to reveal if the model fails on specific subgroups.
- Include a **parity plot** (predicted vs actual) -- it reveals systematic bias that summary statistics hide.

**Warning signs:**
- R-squared looks great but MAE is larger than the chemically meaningful difference between materials
- Model consistently over/under-predicts for one chemistry family (visible on parity plot)
- Different papers report different "best models" because they used different evaluation setups

**Phase to address:** Evaluation framework phase, but must be decided before training begins so all models are compared fairly.

---

## Moderate Pitfalls

### Pitfall 6: Materials Project API Rate Limits and Versioning Instability

**What goes wrong:** Data collection scripts hit the 25 requests/second rate limit, crash partway through, and produce incomplete datasets. Worse: MP database versions change, so results collected at different times are not comparable. A March 2025 patch fixed bugs in dielectric and piezoelectric collections; a February 2025 fix changed elastic tensor data for 2,484 compounds.

**How to avoid:**
- Use `mp_api` client (not the legacy API) with built-in rate limiting and retry logic.
- **Cache all downloaded data locally** with the database version recorded. Pin to a specific MP database version (e.g., 2024.11.1) and document it.
- Download in bulk using the tips from MP docs for large downloads rather than per-material queries.
- For OQMD: its REST API is slower and less reliable -- download the full dataset dump if possible.
- Save raw API responses as JSON/pickle alongside processed DataFrames for reproducibility.

**Warning signs:**
- HTTP 429 errors in data collection logs
- Dataset size changes between runs without code changes
- Unexplained property value changes for specific materials between collection dates

**Phase to address:** Data collection phase.

---

### Pitfall 7: MEGNet vs CGCNN Training Dynamics Mismatch

**What goes wrong:** Researchers apply the same training configuration to both models and conclude one is better when actually it just needed different hyperparameters. MEGNet requires approximately 1000 epochs to converge while CGCNN converges in approximately 400. MEGNet's global state computation adds training time and memory. Using the same learning rate, batch size, and scheduler for both produces misleading comparisons.

**How to avoid:**
- Treat each architecture as a separate experiment with its own hyperparameter budget.
- MEGNet: lower learning rate (1e-4 to 3e-4), longer training (800-1200 epochs), larger batch sizes to stabilize global state updates.
- CGCNN: standard learning rate (1e-3), 300-500 epochs, cosine annealing or step LR scheduler.
- Use early stopping independently for each model.
- Report wall-clock training time alongside accuracy -- MEGNet may win on accuracy but lose on compute efficiency.
- Note: MatGL (MEGNet's maintained implementation) is migrating from DGL to PyTorch Geometric from v2.0.0. **Use PyG backend** -- DGL is no longer actively maintained.

**Warning signs:**
- MEGNet loss still decreasing when training stops
- CGCNN loss plateaued 200 epochs ago but training continues
- One model has 3x the training time but only marginal accuracy improvement

**Phase to address:** Model training and benchmarking phase.

---

### Pitfall 8: Ignoring Degenerate Graph Representations

**What goes wrong:** CGCNN using only pairwise distances can map physically distinct crystal structures to identical graph representations. Two different structures with the same set of interatomic distances but different angular arrangements produce the same graph, making them indistinguishable to the model.

**How to avoid:**
- Include angular information: use three-body interactions or angular features. Models like ALIGNN and DimeNet address this explicitly.
- At minimum, include bond angles as edge features or use Voronoi-based tessellation for neighbor finding (captures coordination geometry).
- If sticking with vanilla CGCNN for simplicity (reasonable for a benchmark project), acknowledge this limitation explicitly and note which structure pairs in the dataset are affected.

**Warning signs:**
- Two materials with different properties but similar distances get nearly identical predictions
- Model struggles specifically with polymorphs that differ in angular arrangement but not bond lengths

**Phase to address:** Feature engineering / architecture selection phase.

---

### Pitfall 9: Multi-Property Prediction Architecture Mistakes

**What goes wrong:** Using a single multi-output model for voltage, capacity, formation energy, and stability when these properties have different scales, different data availability, and different physical origins. The model compromises on all properties rather than excelling at any.

**How to avoid:**
- **Start with separate single-property models** for each target. This is the baseline.
- Only attempt multi-task learning if: (a) you have evidence of shared representations (e.g., formation energy and voltage are physically related), and (b) you have enough data.
- If multi-task: use task-specific output heads with shared graph convolution backbone. Use uncertainty-weighted loss (Kendall et al.) to handle different scales automatically.
- Never mix DFT-derived and experimental targets in the same multi-output model.

**Warning signs:**
- One property's loss dominates training, others barely improve
- Adding a property to the multi-task model makes other properties worse
- Different properties have wildly different amounts of training data (imbalanced multi-task)

**Phase to address:** Architecture design phase.

---

### Pitfall 10: Feature Engineering Mistakes with Matminer/Magpie for Baselines

**What goes wrong:** For scikit-learn baselines, researchers use raw element fractions as features instead of physically meaningful descriptors. Or they include too many correlated matminer features (200+) without selection, causing multicollinearity and poor generalization.

**How to avoid:**
- Use matminer's `ElementProperty` featurizer with the "magpie" preset -- it generates approximately 132 composition-based features that are well-tested for materials property prediction.
- For structure-based baselines, use `SineCoulombMatrix` or `OrbitalFieldMatrix` from matminer.
- Apply feature selection: remove features with near-zero variance, then use mutual information or recursive feature elimination.
- Always include a composition-only baseline (no structure) to quantify how much structural information the GNN actually captures.

**Warning signs:**
- Baseline model has 200+ features but only 2000 training samples (p >> n territory)
- Feature importance shows only 5-10 features matter -- the rest are noise
- Baseline outperforms GNN (indicates the GNN is not leveraging structural information)

**Phase to address:** Baseline model phase.

---

## Performance Traps

### Pitfall 11: Dashboard Performance Collapse with Full Dataset

**What goes wrong:** Loading the full Materials Project dataset (150k+ structures) into a Streamlit dashboard causes the app to become unresponsive. Crystal structure visualization with py3Dmol is slow for large unit cells. Interactive filtering recomputes on every widget change.

**How to avoid:**
- **Use Dash (Plotly) instead of Streamlit** for this project. Streamlit re-executes the entire script on every interaction, which is fatal for large datasets. Dash callbacks update only what changed.
- Pre-compute and cache all aggregate statistics (distribution plots, summary tables) at build time, not runtime.
- For the materials explorer: use server-side pagination (show 50 materials at a time, not 5000).
- For py3Dmol: lazy-load structures only when a specific material is selected, never pre-render all structures.
- Store processed data in Parquet format for fast loading (not CSV).

**Warning signs:**
- Dashboard takes >5 seconds to load or respond to filter changes
- Browser tab memory exceeds 1GB
- Crystal viewer crashes on structures with >100 atoms in the unit cell

**Phase to address:** Dashboard development phase.

---

### Pitfall 12: Non-Reproducible Data Pipeline

**What goes wrong:** The project claims reproducibility but the data pipeline produces different results on different runs because API data changed, random seeds were not set everywhere, or library versions drifted.

**How to avoid:**
- Pin all dependency versions in `requirements.txt` (exact versions, not ranges).
- Set random seeds in: Python (`random.seed`), NumPy (`np.random.seed`), PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`), scikit-learn (pass `random_state` to every function).
- Cache raw API data with a timestamp and database version. Make the pipeline work from cached data by default, with an explicit flag to re-download.
- Use `torch.use_deterministic_algorithms(True)` during training (with the CUBLAS_WORKSPACE_CONFIG environment variable set if using CUDA).

**Warning signs:**
- Re-running training produces different metrics
- Colleague cannot reproduce results from the same code
- `pip install -r requirements.txt` installs different versions than what was used

**Phase to address:** Project setup / infrastructure phase (before any data or modeling work).

---

## Technical Debt Patterns

### Debt 1: Monolithic Jupyter Notebook

The project starts as a single notebook that does data loading, preprocessing, model training, and evaluation. By the time the dashboard needs model artifacts, the notebook must be refactored into modules. Prevention: start with a `src/` package structure from day one with separate modules for data, features, models, evaluation, and visualization.

### Debt 2: Hardcoded Paths and API Keys

API keys embedded in code, absolute paths to data directories. Prevention: use environment variables or `.env` files for API keys (never commit them), and use relative paths or a config file for data directories.

### Debt 3: No Model Registry

Model checkpoints saved as `model_v2_final_FINAL.pt` with no metadata about hyperparameters, dataset version, or training configuration. Prevention: use MLflow or at minimum save a JSON sidecar with every checkpoint containing full training config, dataset hash, and performance metrics.

---

## Integration Gotchas

### Gotcha 1: Pymatgen / MP API Version Coupling

The `mp_api` client version must match the MP API version. Pymatgen structure objects change between versions -- a Structure saved with pymatgen 2023 may not deserialize correctly with pymatgen 2025. Always serialize structures as CIF or POSCAR (standard formats) alongside pymatgen objects.

### Gotcha 2: PyTorch Geometric Version vs CUDA Version

PyG installation is notoriously fragile. The PyG version must match both the PyTorch version and the CUDA version exactly. Use the official installation matrix at `https://pytorch-geometric.readthedocs.io/`. On Windows (this project's platform), CUDA compatibility is even more constrained. Consider using conda over pip for PyG installation.

### Gotcha 3: MatGL Migration from DGL to PyG

MatGL (the maintained MEGNet implementation) is actively migrating from DGL to PyG. Starting a project with DGL now means migration pain later. Use the PyG backend from the start (MatGL >= 2.0.0).

---

## "Looks Done But Isn't" Checklist

- [ ] **Model "works" but evaluation is on random split** -- compositional split not implemented, metrics are inflated
- [ ] **Dashboard shows metrics but from a single run** -- no error bars, no cross-validation, results could be noise
- [ ] **GNN outperforms baseline** -- but baseline used raw features instead of proper Magpie descriptors
- [ ] **"Multi-property prediction"** -- but only formation energy actually has enough data to learn; other targets are decorative
- [ ] **"Reproducible pipeline"** -- but random seeds not set in all locations, or API data not cached
- [ ] **Crystal structure viewer works** -- but only tested on small unit cells; crashes on layered structures with 50+ atoms
- [ ] **Training curves plotted** -- but only loss, not validation loss; overfitting not visible
- [ ] **"Publication-quality figures"** -- but axis labels missing units, font sizes too small, no error bars

---

## Pitfall-to-Phase Mapping

| Phase | Pitfall | Severity | Mitigation |
|-------|---------|----------|------------|
| Project setup | P12: Non-reproducible pipeline | Critical | Pin versions, set seeds, create package structure |
| Data collection | P2: DFT vs experimental confusion | Critical | Define targets and data sources explicitly |
| Data collection | P6: MP API rate limits/versioning | Moderate | Cache data, pin DB version, use bulk download |
| Data preprocessing | P1: Structural/compositional leakage | Critical | Compositional group splitting, prototype filtering |
| Feature engineering | P3: Wrong graph construction params | Critical | Validate connectivity, sweep radius/neighbors |
| Feature engineering | P8: Degenerate graph representations | Moderate | Add angular features or acknowledge limitation |
| Baseline modeling | P10: Poor feature engineering | Moderate | Use Magpie preset, apply feature selection |
| Architecture design | P9: Multi-property mistakes | Moderate | Start single-property, add multi-task carefully |
| Model training | P4: GNN overfitting | Critical | Transfer learning, reduce capacity, regularize |
| Model training | P7: MEGNet/CGCNN config mismatch | Moderate | Independent hyperparameter tuning per model |
| Evaluation | P5: Wrong/misleading metrics | Critical | MAE in physical units, identical folds, parity plots |
| Dashboard | P11: Performance collapse | Moderate | Use Dash, paginate, lazy-load structures |

---

## Sources

- [Matbench Benchmark](https://matbench.materialsproject.org/) -- train/test splitting methodology, OOD-Composition splits
- [Matbench Discovery](https://matbench-discovery.materialsproject.org/data) -- prototype filtering, leakage prevention
- [Leakage and the reproducibility crisis in ML-based science (Patterns, 2023)](https://www.sciencedirect.com/science/article/pii/S2666389923001599) -- data leakage across 290+ papers
- [CGCNN repository (txie-93)](https://github.com/txie-93/cgcnn) -- default parameters, graph construction
- [Examining GNNs for crystal structures: Limitations (Science Advances)](https://www.science.org/doi/10.1126/sciadv.adi3245) -- degenerate representations
- [MatGL](https://matgl.ai/) -- MEGNet implementation, DGL to PyG migration
- [Materials Project API documentation](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started) -- rate limits, bulk download tips
- [MP Database Versions](https://docs.materialsproject.org/changes/database-versions) -- versioning and data quality patches
- [DenseGNN (npj Computational Materials)](https://www.nature.com/articles/s41524-024-01444-x) -- overfitting solutions for small crystal datasets
- [Application-oriented design of ML paradigms for battery science (npj Computational Materials, 2025)](https://www.nature.com/articles/s41524-025-01575-9) -- domain-specific ML pitfalls
- [Voltage Mining for Cathodes (ACS AMI)](https://pubs.acs.org/doi/10.1021/acsami.4c15742) -- cathode dataset construction, polymorph handling
- [Utilizing ML to Advance Battery Materials Design (Chem. Mater., 2024)](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c03486) -- ML challenges in battery materials

---
*Pitfalls research for: ML-based lithium-ion battery cathode performance prediction*
*Researched: 2026-03-05*
