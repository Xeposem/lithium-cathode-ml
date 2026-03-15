---
phase: quick-2
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [README.md]
autonomous: true
must_haves:
  truths:
    - "Python version prerequisite matches pyproject.toml"
    - "Project structure tree includes all significant files present on disk"
    - "All CLI commands are correct and runnable"
    - "Config file table matches actual configs/ directory"
    - "Dashboard page descriptions match actual pages/"
  artifacts:
    - path: "README.md"
      provides: "Accurate project documentation"
  key_links: []
---

<objective>
Audit and fix README.md so that every factual claim (Python version, project tree, CLI commands, config list, dashboard pages) matches the actual project state on disk.

Purpose: Ensure README is trustworthy for anyone cloning the repo.
Output: Corrected README.md
</objective>

<context>
@README.md
</context>

<audit_findings>
## Pre-computed Audit Results

The planner has already compared README.md against the actual filesystem. Here are the exact discrepancies to fix:

### 1. Python Version (WRONG)
- README line 155 says: "Python 3.10+"
- pyproject.toml line 9 says: `requires-python = ">=3.11"`
- Fix: Change "Python 3.10+" to "Python 3.11+"

### 2. Project Structure Tree (INCOMPLETE)
Files present on disk but missing from the README tree:

**Should add:**
- `pyproject.toml` -- project metadata and build config (significant project file, at root level)
- `environment.yml` -- conda environment file (significant, alternative to requirements.txt)
- `docs/figures/` -- directory containing bar_comparison.png (referenced by README itself on line 120)
- `.env.example` -- example environment variables file

**Acceptable omissions (do NOT add):**
- `__init__.py` files in every package -- standard boilerplate, clutters tree
- `__main__.py` in `cathode_ml/data/` and `cathode_ml/evaluation/` -- minor entry points
- `tests/__init__.py` -- boilerplate

### 3. CLI Commands (CORRECT - no changes needed)
- `python -m cathode_ml` -- correct
- `streamlit run dashboard/app.py` -- correct (`streamlit run` is valid)
- `--skip-fetch`, `--skip-train`, `--models`, `--seed` flags -- all exist in pipeline.py

### 4. Config Files Table (CORRECT - no changes needed)
All 6 YAML files listed match `configs/` directory exactly.

### 5. Dashboard Pages (CORRECT - no changes needed)
All 6 pages listed match `dashboard/pages/` directory exactly.
</audit_findings>

<tasks>

<task type="auto">
  <name>Task 1: Fix Python version and update project structure tree</name>
  <files>README.md</files>
  <action>
Make these specific edits to README.md:

1. **Line 155** (Prerequisites section): Change "Python 3.10+" to "Python 3.11+" to match pyproject.toml's `requires-python = ">=3.11"`.

2. **Project Structure tree** (line 228-289): Add the following entries in their correct alphabetical/logical positions:

   At the root level of the tree, add these after `├── data/` and before `├── tests/`:
   ```
   ├── docs/                        # Documentation assets
   │   └── figures/                 # Generated plots and charts
   │       └── bar_comparison.png   # Model comparison bar chart
   ```

   At the root level, after `├── tests/` block and before `└── requirements.txt`:
   ```
   ├── .env.example                 # Example environment variables
   ├── environment.yml              # Conda environment specification
   ├── pyproject.toml               # Project metadata and build config
   ```

   Change the final `└── requirements.txt` line to keep it as the last entry with `└──` prefix.

   The final root-level entries should be ordered:
   ```
   ├── data/                        # Data directory (gitignored contents)
   │   └── results/                 # Model outputs, metrics, figures
   ├── docs/                        # Documentation assets
   │   └── figures/                 # Generated plots and charts
   │       └── bar_comparison.png   # Model comparison bar chart
   ├── tests/                       # Test suite
   │   ├── conftest.py              # Shared fixtures
   │   ├── test_*.py                # Unit tests per module
   │   └── ...
   ├── .env.example                 # Example environment variables
   ├── environment.yml              # Conda environment specification
   ├── pyproject.toml               # Project metadata and build config
   └── requirements.txt             # Pinned dependencies
   ```

Do NOT add `__init__.py` files to the tree -- they are standard boilerplate and would clutter it.
  </action>
  <verify>
    <automated>python -c "r=open('README.md').read(); assert '3.11+' in r, 'Python version not updated'; assert '3.10+' not in r, 'Old Python version still present'; assert 'pyproject.toml' in r, 'pyproject.toml missing from tree'; assert 'environment.yml' in r, 'environment.yml missing from tree'; assert '.env.example' in r, '.env.example missing from tree'; assert 'docs/' in r, 'docs/ missing from tree'; print('All checks passed')"</automated>
  </verify>
  <done>README.md accurately reflects Python 3.11+ requirement and project structure tree includes all significant root-level files (pyproject.toml, environment.yml, .env.example, docs/)</done>
</task>

</tasks>

<verification>
- `python -c "..."` verification script confirms all changes
- Manual scan: no other factual claims found to be incorrect (CLI commands, config table, dashboard pages all verified accurate)
</verification>

<success_criteria>
- Python version says "3.11+" matching pyproject.toml
- Project tree includes pyproject.toml, environment.yml, .env.example, docs/figures/
- No previously-correct content has been broken
</success_criteria>

<output>
After completion, create `.planning/quick/2-audit-readme-accuracy-verify-project-str/2-SUMMARY.md`
</output>
