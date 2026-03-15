---
phase: quick
plan: 1
type: execute
wave: 1
depends_on: []
files_modified: [README.md]
autonomous: true
requirements: [quick-readme-update]

must_haves:
  truths:
    - "README Results section shows full 5-model comparison table (not just best-per-property)"
    - "README explains M3GNet and TensorNet poor performance with specific reasoning"
    - "README summary table reflects latest comparison data from data/results/comparison/"
    - "All changes are committed to git"
  artifacts:
    - path: "README.md"
      provides: "Updated results section with full model comparison and explanations"
  key_links: []
---

<objective>
Update the README.md Results section with comprehensive model results, full 5-model comparison tables, and detailed explanations of model performance including why M3GNet and TensorNet underperformed.

Purpose: The current README shows only a best-model-per-property summary table and a brief interpretation. It needs expanded results showing all 5 models and honest explanations of the GNN performance variations, particularly the negative R-squared values for M3GNet and TensorNet.

Output: Updated README.md committed to git.
</objective>

<execution_context>
@C:/Users/regis/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/regis/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@README.md
@data/results/comparison/comparison.md
@data/results/comparison/model_comparison.csv
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update README Results section with full model comparison and explanations</name>
  <files>README.md</files>
  <action>
Update the `## Results` section of README.md with the following changes:

1. **Replace the Summary Table** with a full per-property comparison showing all 5 models. Use the data from `data/results/comparison/model_comparison.csv`. Format as 4 property tables (one per property: formation_energy_per_atom, voltage, capacity, energy_above_hull), each showing all 5 models with MAE, RMSE, R-squared columns. Bold the best model in each table. Include train/test counts from the CSV (n_train, n_test). Add the M3GNet footnote marker for the pretrained model.

2. **Add a "Best Model per Property" summary table** above the detailed tables as a quick-reference:
   - formation_energy_per_atom: CGCNN (MAE 0.0341, R2 0.9952)
   - voltage: XGBoost (MAE 0.4336, R2 0.6791)
   - capacity: CGCNN (MAE 48.78, R2 0.4652)
   - energy_above_hull: CGCNN (MAE 0.0211, R2 0.6903)

3. **Update the Interpretation paragraph** to include:
   - Total dataset size: 46,389 records from 4 sources
   - CGCNN is the strongest overall GNN, winning 3 of 4 properties
   - M3GNet underperformance explanation: fine-tuning from a pretrained formation-energy model causes domain mismatch when applied to voltage/capacity/stability targets. The pretrained weights bias the model toward formation energy patterns, and limited fine-tuning epochs (typical of transfer learning) are insufficient to overcome this bias for dissimilar target properties
   - TensorNet underperformance explanation: trained from scratch with no pretraining, TensorNet requires substantially more training data and epochs to converge. With the current dataset size and training budget, the model fails to learn meaningful representations, producing predictions that are worse than predicting the mean (negative R-squared)
   - Note that for composition-dominated properties (voltage, capacity), traditional ML baselines remain competitive or superior because elemental chemistry is the primary predictor
   - Mention that CGCNN's advantage on formation energy (R2 0.9952 vs XGBoost 0.9853) demonstrates that crystal structure information provides measurable lift for structure-sensitive properties

4. **Keep all other README sections unchanged** (Introduction, Data Sources, Methodology, Dashboard, How to Run, Project Structure, License, Citation).

5. Also commit the pending change in `cathode_ml/features/composition.py` (matminer tqdm warning suppression fix) since it is a small staged improvement.
  </action>
  <verify>
    <automated>python -c "import re; r=open('README.md').read(); assert 'M3GNet' in r and 'TensorNet' in r and 'negative R' in r.lower() or 'negative r' in r.lower() or 'worse than' in r.lower(); assert r.count('|') > 50; print('README has full model tables and explanations')"</automated>
  </verify>
  <done>README.md contains full 5-model comparison tables for all 4 properties, best-model summary table, and detailed interpretation explaining M3GNet/TensorNet underperformance with specific technical reasoning. Changes committed to git.</done>
</task>

</tasks>

<verification>
- README contains per-property tables with all 5 models
- Interpretation paragraph explains M3GNet domain mismatch and TensorNet convergence issues
- Numbers match data/results/comparison/model_comparison.csv exactly
- All changes committed
</verification>

<success_criteria>
- Full 5-model results tables present for all 4 target properties
- Best-model summary table present
- M3GNet and TensorNet performance explained with technical reasoning
- Committed to git on main branch
</success_criteria>

<output>
After completion, the updated README.md is committed.
</output>
