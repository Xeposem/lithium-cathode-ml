# Lithium-Ion Battery Cathode Performance Prediction

## What This Is

A machine learning platform that predicts lithium-ion battery cathode properties — capacity, voltage, stability, and cycle life — using crystal structure data from Materials Project, OQMD, and Battery Data Genome. It benchmarks CGCNN vs MEGNet graph neural networks alongside traditional ML baselines, and presents results through an interactive web dashboard. Built as an academic/portfolio project with paper-ready reproducibility.

## Core Value

Accurate, reproducible prediction of cathode performance properties from crystal structure, with clear model comparison and publication-quality results.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] API-based data ingestion from Materials Project, OQMD, and Battery Data Genome
- [ ] Crystal structure to graph representation (atoms as nodes, bonds as edges)
- [ ] CGCNN model for property prediction
- [ ] MEGNet model for property prediction
- [ ] Scikit-learn baseline models for comparison
- [ ] Multi-property prediction (capacity, voltage, stability, cycle life)
- [ ] Model evaluation with MAE, RMSE, R² metrics
- [ ] Publication-quality benchmark comparison plots
- [ ] Web dashboard with model metrics and comparison tables/charts
- [ ] Dashboard: interactive prediction (input composition/structure → predicted properties)
- [ ] Dashboard: data explorer with dataset browsing, filtering, and distribution visualization
- [ ] Dashboard: training curves (loss, learning rate, convergence)
- [ ] Dashboard: materials explorer — searchable database filterable by voltage, formation energy, capacity, elements, stability threshold
- [ ] Dashboard: materials discovery panel showing top candidate materials
- [ ] Dashboard: crystal structure viewer (py3Dmol or equivalent)
- [ ] README with introduction, methodology, and pipeline implementation details

### Out of Scope

- Mobile app — web dashboard is sufficient
- Real-time model retraining — batch training with saved checkpoints
- Deployment to cloud production — local/academic use
- Generative material design — prediction only, not inverse design

## Context

- Academic/portfolio project targeting paper-ready reproducibility
- Datasets: Materials Project (MP API), OQMD (REST API), Battery Data Genome
- GNN architectures: CGCNN (Crystal Graph Convolutional Neural Network) and MEGNet (MatErials Graph Network), both representing crystal structures as graphs
- Research should inform whether to use multi-output models or separate models per property
- Crystal structure visualization via py3Dmol or similar 3D viewer
- Performance metrics must be clearly comparable across all model types

## Constraints

- **Tech stack**: PyTorch for deep learning, scikit-learn for baselines, GNN framework (PyG or DGL) for graph networks
- **Data access**: All data via public APIs (reproducible acquisition)
- **Reproducibility**: Fixed random seeds, versioned dependencies, documented pipeline
- **Output quality**: Publication-quality figures and tables

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Benchmark CGCNN vs MEGNet | Compare two leading crystal GNN architectures | — Pending |
| API-based data collection | Ensures reproducibility and automation | — Pending |
| Web dashboard for visualization | Interactive exploration of results and materials | — Pending |
| Research-driven model architecture | Let domain research inform multi-output vs separate models | — Pending |

---
*Last updated: 2026-03-05 after initialization*
