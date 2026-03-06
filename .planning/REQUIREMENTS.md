# Requirements: Lithium-Ion Battery Cathode Performance Prediction

**Defined:** 2026-03-05
**Core Value:** Accurate, reproducible prediction of cathode performance properties from crystal structure, with clear model comparison and publication-quality results

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: System ingests cathode material data from Materials Project via mp-api programmatically
- [x] **DATA-02**: System ingests supplementary data from OQMD via REST API
- [x] **DATA-03**: System ingests data from Battery Data Genome datasets
- [x] **DATA-04**: System caches downloaded data locally to avoid repeated API calls
- [x] **DATA-05**: System preprocesses data: filters invalid structures, removes outliers, handles missing values
- [x] **DATA-06**: System documents every preprocessing filter applied with rationale

### Featurization

- [x] **FEAT-01**: System converts crystal structures to PyG graph representations (atoms as nodes, bonds as edges)
- [x] **FEAT-02**: System uses Gaussian distance expansion for edge features with configurable cutoff radius
- [x] **FEAT-03**: System generates Magpie composition-based descriptors via matminer for baseline models
- [x] **FEAT-04**: System implements compositional group splitting (not random) to prevent data leakage

### Models

- [ ] **MODL-01**: System implements CGCNN using PyTorch Geometric's CGConv
- [ ] **MODL-02**: System implements MEGNet via matgl with proper architecture matching
- [x] **MODL-03**: System implements Random Forest baseline with scikit-learn on Magpie features
- [x] **MODL-04**: System implements XGBoost/GBM baseline for additional comparison
- [ ] **MODL-05**: System predicts capacity, voltage, stability, and formation energy (separate models per property)
- [ ] **MODL-06**: System trains each model with architecture-appropriate hyperparameters (not identical configs)
- [ ] **MODL-07**: System stores model checkpoints and training artifacts as JSON/CSV

### Evaluation

- [ ] **EVAL-01**: System evaluates all models with MAE, RMSE, and R-squared metrics
- [ ] **EVAL-02**: System uses consistent cross-validation folds across all models for fair comparison
- [ ] **EVAL-03**: System generates parity plots (predicted vs actual) for each model and property
- [ ] **EVAL-04**: System generates bar chart comparisons of model performance across properties
- [ ] **EVAL-05**: System generates learning/training curves (loss, validation MAE per epoch)

### Dashboard

- [ ] **DASH-01**: Web dashboard displays model comparison metrics in tables and charts
- [ ] **DASH-02**: Dashboard allows interactive prediction: user inputs composition/structure, gets predicted properties
- [ ] **DASH-03**: Dashboard includes data explorer: browse, filter, and visualize dataset distributions
- [ ] **DASH-04**: Dashboard displays training curves (loss, learning rate, convergence) per model
- [ ] **DASH-05**: Dashboard includes materials explorer: searchable database filterable by voltage, formation energy, capacity, elements, stability threshold
- [ ] **DASH-06**: Dashboard includes materials discovery panel showing top candidate materials ranked by predicted properties
- [ ] **DASH-07**: Dashboard includes crystal structure 3D viewer using py3Dmol or equivalent

### Reproducibility

- [x] **REPR-01**: System uses fixed random seeds across all experiments
- [x] **REPR-02**: System provides pinned dependency file (requirements.txt or environment.yml)
- [x] **REPR-03**: System uses YAML configuration files for all hyperparameters and settings
- [ ] **REPR-04**: System provides CLI entry points to run full pipeline (fetch, clean, featurize, train, evaluate)

### Documentation

- [ ] **DOCS-01**: README includes project introduction and motivation
- [ ] **DOCS-02**: README includes methodology section covering model architectures and evaluation approach
- [ ] **DOCS-03**: README includes pipeline implementation details (data flow, how to run)
- [ ] **DOCS-04**: README includes results summary with key findings

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Analysis

- **ADV-01**: Feature importance / SHAP analysis for traditional ML models
- **ADV-02**: GNN interpretability (attention weight visualization, node importance)
- **ADV-03**: Matbench-compatible evaluation on standard benchmark tasks
- **ADV-04**: Multi-task learning experiments (shared encoder, property-specific heads)

### Infrastructure

- **INFR-01**: MLflow experiment tracking integration
- **INFR-02**: Optuna hyperparameter tuning framework
- **INFR-03**: Docker container for one-command reproducible execution

## Out of Scope

| Feature | Reason |
|---------|--------|
| Generative / inverse material design | Different problem domain entirely (VAE/GAN); scope explosion |
| Real-time model retraining in dashboard | GNN training takes hours; impractical in web app |
| Cloud deployment / production API | DevOps complexity with no academic value |
| Mobile-responsive dashboard | Crystal viewers and data tables are desktop experiences |
| Custom novel GNN architecture | Without theoretical contribution, adds noise not signal |
| All battery chemistries | Lithium cathodes alone have thousands of entries; dilutes focus |
| Transfer learning from foundation models | Adds complexity without guaranteed benefit; future work |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| FEAT-01 | Phase 2 | Complete |
| FEAT-02 | Phase 2 | Complete |
| FEAT-03 | Phase 2 | Complete |
| FEAT-04 | Phase 2 | Complete |
| MODL-01 | Phase 3 | Pending |
| MODL-02 | Phase 4 | Pending |
| MODL-03 | Phase 2 | Complete |
| MODL-04 | Phase 2 | Complete |
| MODL-05 | Phase 3 | Pending |
| MODL-06 | Phase 3 | Pending |
| MODL-07 | Phase 3 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| EVAL-05 | Phase 5 | Pending |
| DASH-01 | Phase 6 | Pending |
| DASH-02 | Phase 6 | Pending |
| DASH-03 | Phase 6 | Pending |
| DASH-04 | Phase 6 | Pending |
| DASH-05 | Phase 6 | Pending |
| DASH-06 | Phase 6 | Pending |
| DASH-07 | Phase 6 | Pending |
| REPR-01 | Phase 1 | Complete |
| REPR-02 | Phase 1 | Complete |
| REPR-03 | Phase 1 | Complete |
| REPR-04 | Phase 5 | Pending |
| DOCS-01 | Phase 6 | Pending |
| DOCS-02 | Phase 6 | Pending |
| DOCS-03 | Phase 6 | Pending |
| DOCS-04 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after roadmap creation*
