# Milestones

## v1.0 MVP (Shipped: 2026-03-13)

**Phases completed:** 9 phases, 23 plans, 0 tasks

**Key accomplishments:**
- Full data pipeline ingesting cathode materials from Materials Project and OQMD with local caching and automated cleaning
- CGCNN graph neural network achieving R²=0.995 on formation energy prediction from crystal structure
- M3GNet (pretrained fine-tuning) and TensorNet (O(3)-equivariant, from scratch) via matgl 2.x
- RF and XGBoost baselines with Magpie compositional features for comparison
- Interactive Streamlit dashboard with model comparison, structure-based prediction, materials explorer, and crystal viewer
- End-to-end CLI pipeline: fetch → clean → train → evaluate with YAML-driven configuration

**Known issues carried to v1.1:**
- M3GNet/TensorNet double-denormalization bug: `predict_structure()` already denormalizes, training code applies it again → negative R² on all properties
- TensorNet log duplication error during training
- AFLOW and JARVIS fetchers implemented but not yet validated end-to-end

**Stats:** 150 commits | 11,266 LOC Python | 5 days (2026-03-05 → 2026-03-09)

---

