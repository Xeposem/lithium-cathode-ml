# Phase 1: Data Pipeline and Project Foundation - Research

**Researched:** 2026-03-05
**Domain:** Materials science data ingestion, crystal structure validation, reproducible ML project setup
**Confidence:** HIGH

## Summary

Phase 1 establishes the entire project foundation: dependency management, configuration system, and a three-source data pipeline that fetches, cleans, validates, and caches lithium cathode material data. The primary data source is the Materials Project (mp-api) which provides crystal structures with computed properties (formation energy, voltage via insertion electrodes, energy above hull for stability). OQMD supplements with additional DFT-computed formation energies via its REST API (qmpy_rester or direct HTTP). The "Battery Data Genome" is not a single API but a collection of NREL datasets -- for crystal-structure-based property prediction, the relevant battery data comes primarily from Materials Project's insertion electrode data, supplemented by literature-mined datasets like the auto-generated battery materials database (Nature Scientific Data).

The key technical challenges are: (1) mp-api's insertion_electrodes endpoint returns voltage/capacity data per electrode rather than per material, requiring joining with summary data; (2) qmpy_rester is unmaintained since 2019 and may need direct HTTP fallback; (3) Battery Data Genome requires custom download/parsing rather than a standard API client; (4) deduplication across sources requires matching on composition + space group; (5) all random operations must use fixed seeds from config.

**Primary recommendation:** Build a modular `cathode_ml` package from day one with YAML-driven configuration, three separate fetcher classes (one per source) sharing a common cache interface, and pymatgen Structure as the canonical data object throughout.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Ingest cathode data from Materials Project via mp-api | MPRester summary.search() for structures + properties; insertion_electrodes.search() for voltage data. Requires free API key. |
| DATA-02 | Ingest supplementary data from OQMD via REST API | qmpy_rester get_oqmd_phases() with element_set filter. No API key. Unmaintained -- need HTTP fallback. |
| DATA-03 | Ingest data from Battery Data Genome datasets | Not a single API. Use NREL battery data tools / direct download. Literature-mined datasets via nature.com/articles/s41597-020-00602-2. |
| DATA-04 | Cache downloaded data locally to avoid repeated API calls | JSON file cache with metadata (timestamp, db version, query params). Check cache before API call. |
| DATA-05 | Preprocess: filter invalid structures, remove outliers, handle missing values | pymatgen Structure validation (proximity check, positive volume, valid species). Outlier removal via IQR on target properties. |
| DATA-06 | Document every preprocessing filter with rationale | Logging module with structured filter log. Write cleaning_log.json with filter name, count before/after, rationale. |
| REPR-01 | Fixed random seeds across all experiments | Set seeds in Python random, numpy, and document in config. No torch seeds needed yet (Phase 1 has no training). |
| REPR-02 | Pinned dependency file | requirements.txt with exact versions (==) for all packages. Also provide environment.yml for conda. |
| REPR-03 | YAML configuration files for all settings | Single config.yaml with sections: api_keys, data_sources, filters, cache, random_seeds. Loaded via PyYAML safe_load(). |
</phase_requirements>

## Standard Stack

### Core (Phase 1 Only)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pymatgen | >=2025.10.7 | Crystal structure processing, validation, neighbor finding | Industry standard for materials informatics. Required by mp-api. The lingua franca of crystal structure data |
| mp-api | >=0.45.0 | Materials Project API client | Official Python client. Built-in rate limiting, retry logic. Returns pymatgen Structure objects natively |
| qmpy-rester | 0.2.0 | OQMD REST API wrapper | Only Python client for OQMD. No API key needed. Last updated 2019 but API is stable |
| requests | >=2.31 | HTTP client for direct API fallback and Battery Data Genome | Standard HTTP library. Needed for OQMD fallback and BDG downloads |
| PyYAML | >=6.0 | YAML configuration file parsing | Standard YAML parser for Python. Use safe_load() only (security) |
| pandas | >=2.1 | Tabular data handling for cleaning and filtering | Standard data manipulation. Used for merging, deduplication, filtering |
| numpy | >=1.26,<2.0 | Numerical operations | Pin below 2.0 for downstream DGL/matgl compatibility in later phases |
| tqdm | >=4.66 | Progress bars for long API downloads | Standard progress indication |
| python-dotenv | >=1.0 | Load .env file for API keys | Keep secrets out of config files and code |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ASE | >=3.27 | Structure format conversion | Converting OQMD structures to pymatgen format if needed |
| scipy | >=1.12 | Statistical outlier detection | IQR-based outlier removal on target properties |
| pytest | >=8.0 | Unit testing framework | Testing data pipeline functions |
| black | >=24.0 | Code formatter | Consistent code style from project start |
| ruff | >=0.4 | Fast linter | Catch issues early |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyYAML | OmegaConf/Hydra | Hydra is more powerful for ML experiment configs but overkill for Phase 1. Can adopt later if needed |
| qmpy_rester | Direct HTTP to oqmd.org/api | More flexible but requires manual response parsing. Use as fallback if qmpy_rester fails |
| python-dotenv | Direct env vars | dotenv provides .env.example template for onboarding new users |
| JSON cache | SQLite cache | SQLite is better for queries but JSON is simpler for debugging and Phase 1 data volumes are small |

**Installation:**
```bash
conda create -n cathode-ml python=3.11
conda activate cathode-ml

pip install pymatgen mp-api qmpy-rester
pip install requests pyyaml pandas "numpy>=1.26,<2.0" tqdm python-dotenv
pip install ase scipy
pip install pytest black ruff
```

## Architecture Patterns

### Recommended Project Structure (Phase 1 Scope)

```
lithium-cathode-ml/
+-- cathode_ml/                 # Main package (not src/ -- enables python -m cathode_ml.data.fetch)
|   +-- __init__.py
|   +-- config.py               # YAML config loader, seed setter, path resolver
|   +-- data/
|   |   +-- __init__.py
|   |   +-- fetch.py            # __main__ entry: orchestrates all fetchers
|   |   +-- mp_fetcher.py       # Materials Project API client with caching
|   |   +-- oqmd_fetcher.py     # OQMD API client with caching + HTTP fallback
|   |   +-- bdg_fetcher.py      # Battery Data Genome downloader/parser
|   |   +-- clean.py            # Validation, deduplication, filtering, logging
|   |   +-- cache.py            # Shared cache read/write utilities
|   |   +-- schemas.py          # Data validation schemas / dataclasses
|
+-- configs/
|   +-- data.yaml               # All pipeline parameters
|
+-- data/                       # Data directory (gitignored)
|   +-- raw/                    # Cached API responses
|   +-- processed/              # Cleaned structures + targets
|   +-- logs/                   # Cleaning logs
|   +-- .gitkeep
|
+-- tests/
|   +-- __init__.py
|   +-- conftest.py             # Shared fixtures (mock structures, mock API responses)
|   +-- test_config.py          # Config loading tests
|   +-- test_mp_fetcher.py      # MP API client tests (mocked)
|   +-- test_oqmd_fetcher.py    # OQMD client tests (mocked)
|   +-- test_clean.py           # Cleaning/validation tests
|   +-- test_cache.py           # Cache read/write tests
|
+-- requirements.txt            # Pinned dependencies
+-- environment.yml             # Conda environment spec
+-- .env.example                # API key template
+-- .gitignore                  # Ignore data/, .env, __pycache__
+-- pyproject.toml              # Project metadata
```

### Pattern 1: Config-Driven Pipeline

**What:** All pipeline behavior controlled by a single YAML config file. No hardcoded values in code.
**When to use:** Always -- this is the foundation for reproducibility (REPR-03).

```python
# configs/data.yaml
random_seeds:
  python: 42
  numpy: 42

data_sources:
  materials_project:
    enabled: true
    elements_must_contain: ["Li"]
    fields: ["material_id", "formula_pretty", "structure", "formation_energy_per_atom",
             "energy_above_hull", "is_stable", "symmetry"]
    energy_above_hull_max: 0.1  # eV/atom -- only near-hull-stable materials
  oqmd:
    enabled: true
    element_set: "Li"
    stability_max: 0.1
  battery_data_genome:
    enabled: true
    source_url: "https://..."

filters:
  min_sites: 2
  max_sites: 200
  formation_energy_range: [-5.0, 0.5]  # eV/atom
  required_properties: ["formation_energy_per_atom"]
  remove_noble_gases: true

cache:
  directory: "data/raw"
  use_cache: true
```

```python
# cathode_ml/config.py
import yaml
import random
import numpy as np
from pathlib import Path

def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def set_seeds(config: dict) -> None:
    seeds = config["random_seeds"]
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
```

### Pattern 2: Fetcher Interface with Caching

**What:** Each data source has a dedicated fetcher class that checks cache before making API calls.
**When to use:** All three data sources (DATA-01, DATA-02, DATA-03, DATA-04).

```python
# cathode_ml/data/cache.py
import json
import hashlib
from pathlib import Path
from datetime import datetime

class DataCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, source: str, params: dict) -> str:
        param_str = json.dumps(params, sort_keys=True)
        return f"{source}_{hashlib.md5(param_str.encode()).hexdigest()}"

    def has(self, key: str) -> bool:
        return (self.cache_dir / f"{key}.json").exists()

    def load(self, key: str) -> dict:
        with open(self.cache_dir / f"{key}.json") as f:
            return json.load(f)

    def save(self, key: str, data: dict, metadata: dict = None) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "data": data,
        }
        with open(self.cache_dir / f"{key}.json", "w") as f:
            json.dump(payload, f)
```

### Pattern 3: Structured Cleaning Log (DATA-06)

**What:** Every filter operation logs what was removed, how many, and why.
**When to use:** All preprocessing steps.

```python
# cathode_ml/data/clean.py
import logging
from dataclasses import dataclass, field, asdict

@dataclass
class FilterRecord:
    filter_name: str
    description: str
    rationale: str
    count_before: int
    count_after: int
    count_removed: int

class CleaningPipeline:
    def __init__(self):
        self.log: list[FilterRecord] = []
        self.logger = logging.getLogger("cathode_ml.cleaning")

    def apply_filter(self, data, filter_fn, name, rationale):
        count_before = len(data)
        data = [d for d in data if filter_fn(d)]
        count_after = len(data)
        record = FilterRecord(
            filter_name=name,
            description=f"Applied {name}",
            rationale=rationale,
            count_before=count_before,
            count_after=count_after,
            count_removed=count_before - count_after,
        )
        self.log.append(record)
        self.logger.info(f"{name}: {count_before} -> {count_after} (removed {record.count_removed})")
        return data

    def save_log(self, path):
        import json
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.log], f, indent=2)
```

### Anti-Patterns to Avoid

- **Hardcoded API keys:** Never put MP_API_KEY in code. Use .env file loaded by python-dotenv, referenced in config.yaml as `${MP_API_KEY}` or loaded from environment.
- **Monolithic fetch script:** Don't put all three data source fetchers in one function. Each source has different APIs, error modes, and cache strategies.
- **Pickle-only serialization:** Don't cache pymatgen Structures as pickle only. Also save as CIF or JSON (pymatgen's as_dict()) for portability across pymatgen versions.
- **Skipping structure validation:** Every structure from every source must pass validation before entering the processed dataset. OQMD structures in particular may have issues.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Crystal structure parsing/validation | Custom CIF parser or coordinate validator | pymatgen.core.Structure + Structure.is_valid() | Edge cases in CIF format, spacegroup handling, fractional coordinate normalization |
| Materials Project data access | Direct HTTP to MP API | mp-api MPRester | Built-in rate limiting, retry, pagination, returns pymatgen objects |
| OQMD data access | Raw HTTP parsing | qmpy_rester (with HTTP fallback) | Handles pagination and response format |
| YAML config loading | Custom config parser | PyYAML yaml.safe_load() | Type conversion, nested structures, anchors/aliases |
| Progress bars for downloads | Print statements | tqdm | Handles terminal width, ETA, rate display |
| Environment variable management | os.environ boilerplate | python-dotenv load_dotenv() | .env file support, .env.example template pattern |

**Key insight:** The materials science ecosystem has mature, well-tested libraries for every data access and structure manipulation task. Hand-rolling any of these will introduce bugs that take weeks to find -- especially around crystal symmetry operations and coordinate transformations.

## Common Pitfalls

### Pitfall 1: MP API Rate Limits and Incomplete Downloads
**What goes wrong:** Scripts hit the 25 req/s rate limit, get HTTP 429 errors, and produce incomplete datasets. Worse: partial downloads get cached, so re-running skips the missing data.
**Why it happens:** Fetching thousands of materials without proper pagination or retry logic.
**How to avoid:** Use mp-api's MPRester which has built-in rate limiting. Download in bulk by specifying all needed fields in a single search() call rather than per-material queries. Cache the complete response only after verifying the expected count. Record the MP database version in cache metadata.
**Warning signs:** HTTP 429 in logs, dataset size varies between runs, missing material_ids.

### Pitfall 2: Battery Data Genome Is Not a Single API
**What goes wrong:** Developers expect a REST API like MP or OQMD and waste time looking for one.
**Why it happens:** "Battery Data Genome" sounds like a database with an API, but it is a NREL initiative encompassing multiple datasets, tools, and data format standards. There is no single `bdg.search()` endpoint.
**How to avoid:** For this project (crystal structure -> property prediction), the relevant cathode data comes from: (1) Materials Project insertion electrodes (voltage, capacity), (2) the auto-generated battery materials database (doi:10.1038/s41597-020-00602-2) which provides text-mined cathode properties, and (3) NREL's downloadable electrode datasets. Implement bdg_fetcher as a file downloader + parser, not an API client.
**Warning signs:** Searching for "Battery Data Genome API" returns nothing useful.

### Pitfall 3: Deduplication Across Data Sources
**What goes wrong:** The same material appears in both MP and OQMD with slightly different structures (different DFT settings produce different relaxed geometries). Without deduplication, the dataset contains near-duplicates that inflate metrics.
**Why it happens:** MP and OQMD use different DFT codes (VASP vs VASP with different settings) and different structure relaxation protocols.
**How to avoid:** Deduplicate by reduced formula + space group number. For materials that match on these, prefer the MP entry (higher data quality, more properties). Log all deduplication decisions. Use pymatgen's StructureMatcher for fuzzy matching if needed.
**Warning signs:** Dataset has unexpected duplicates after merging, same composition appears multiple times with slightly different formation energies.

### Pitfall 4: Pymatgen Structure Serialization Across Versions
**What goes wrong:** Structures cached as Python pickle with pymatgen 2025.x cannot be loaded with a future pymatgen version. The project breaks when dependencies are updated.
**How to avoid:** Always serialize structures using pymatgen's `structure.as_dict()` (JSON-serializable dict) or CIF format alongside any pickle caching. The as_dict() format is version-stable.
**Warning signs:** Import errors or attribute errors when loading cached structures after pip upgrade.

### Pitfall 5: Missing Seeds in Preprocessing
**What goes wrong:** Random operations in data shuffling or train/test splitting during cleaning use unseeded random, producing different processed datasets on each run.
**Why it happens:** Seeds are set for model training but forgotten in data preprocessing steps.
**How to avoid:** Set seeds at the very start of the pipeline (config.py set_seeds()). Use numpy RandomState or Python random with explicit seed for any shuffling. Document all random operations in the cleaning log.
**Warning signs:** Running the pipeline twice produces different processed dataset sizes or orderings.

## Code Examples

### Materials Project: Fetching Cathode Materials

```python
# Source: mp-api docs + community examples
from mp_api.client import MPRester
import os

def fetch_mp_cathodes(config: dict) -> list[dict]:
    """Fetch lithium cathode materials from Materials Project."""
    mp_config = config["data_sources"]["materials_project"]

    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        # 1. Fetch summary data for Li-containing materials
        docs = mpr.materials.summary.search(
            elements=mp_config["elements_must_contain"],
            energy_above_hull=(0, mp_config["energy_above_hull_max"]),
            fields=mp_config["fields"],
        )

        # 2. Fetch insertion electrode (battery) data
        electrode_docs = mpr.insertion_electrodes.search(
            fields=["battery_id", "material_ids", "average_voltage",
                    "capacity_grav", "stability_charge", "stability_discharge",
                    "working_ion", "framework_formula"]
        )
        # Filter for Li-ion electrodes
        li_electrodes = [d for d in electrode_docs if d.working_ion == "Li"]

    return docs, li_electrodes
```

### OQMD: Fetching with Fallback

```python
# Source: qmpy_rester docs + OQMD REST API docs
import qmpy_rester as qr
import requests

def fetch_oqmd_cathodes(config: dict) -> list[dict]:
    """Fetch lithium-containing materials from OQMD."""
    oqmd_config = config["data_sources"]["oqmd"]
    results = []

    try:
        with qr.QMPYRester() as q:
            data = q.get_oqmd_phases(
                element_set="Li",
                stability=f"<{oqmd_config['stability_max']}",
                verbose=False,
            )
            results = data.get("data", [])
    except Exception as e:
        # Fallback to direct HTTP
        logging.warning(f"qmpy_rester failed ({e}), using direct HTTP")
        url = "http://oqmd.org/oqmdapi/formationenergy"
        params = {
            "element_set": "Li",
            "stability": f"<{oqmd_config['stability_max']}",
            "limit": 100,
            "offset": 0,
        }
        while True:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            page = resp.json()
            results.extend(page.get("data", []))
            if not page.get("next"):
                break
            params["offset"] += params["limit"]

    return results
```

### Structure Validation

```python
# Source: pymatgen docs
from pymatgen.core import Structure

def validate_structure(structure: Structure) -> tuple[bool, str]:
    """Validate a crystal structure for use in ML pipeline."""
    # Check basic validity
    if structure.volume <= 0:
        return False, "non-positive volume"

    if len(structure) < 2:
        return False, "fewer than 2 sites"

    if len(structure) > 200:
        return False, "more than 200 sites (too large for GNN)"

    # Check for overlapping atoms (distance < 0.5 Angstrom)
    for i, site_i in enumerate(structure):
        for j, site_j in enumerate(structure):
            if i >= j:
                continue
            dist = site_i.distance(site_j)
            if dist < 0.5:
                return False, f"overlapping atoms at sites {i},{j} (dist={dist:.3f})"

    # Check for valid species (no dummy species)
    for site in structure:
        if not hasattr(site.specie, 'Z'):
            return False, f"invalid species: {site.specie}"

    return True, "valid"
```

### YAML Config Loading with Seed Setting

```python
# cathode_ml/config.py
import yaml
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load YAML configuration file."""
    load_dotenv()  # Load .env file for API keys
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def set_seeds(config: dict) -> None:
    """Set random seeds for reproducibility."""
    seeds = config.get("random_seeds", {})
    py_seed = seeds.get("python", 42)
    np_seed = seeds.get("numpy", 42)
    random.seed(py_seed)
    np.random.seed(np_seed)
```

### CLI Entry Point

```python
# cathode_ml/data/fetch.py
"""
Entry point: python -m cathode_ml.data.fetch
Fetches, cleans, and caches cathode material data from all sources.
"""
import argparse
import logging
from cathode_ml.config import load_config, set_seeds

def main():
    parser = argparse.ArgumentParser(description="Fetch cathode material data")
    parser.add_argument("--config", default="configs/data.yaml", help="Config file path")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache, re-download")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seeds(config)
    logging.basicConfig(level=logging.INFO)

    # ... fetch from each source, clean, validate, save ...

if __name__ == "__main__":
    main()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Legacy MP API (MPRester from pymatgen.ext.matproj) | mp-api package (MPRester from mp_api.client) | 2022-2023 | Old API deprecated. New API has different endpoints, rate limiting, field names |
| qmpy full install (requires PostgreSQL) | qmpy_rester (REST API only) | 2019 | Dramatically simpler -- just pip install, no DB setup |
| CSV/pickle for data caching | JSON with metadata (timestamp, version, query) | Best practice | Enables cache invalidation and reproducibility tracking |
| Hardcoded parameters in scripts | YAML config files | ML best practice | Enables reproducible experiments, config versioning |
| requirements.txt with ranges (>=) | requirements.txt with exact pins (==) | pip-tools / pip freeze | Exact reproducibility across machines |

**Deprecated/outdated:**
- `pymatgen.ext.matproj.MPRester` -- use `mp_api.client.MPRester` instead
- OQMD v1 API format -- qmpy_rester wraps the current REST API
- `yaml.load()` without Loader -- security risk, always use `yaml.safe_load()`

## Open Questions

1. **Battery Data Genome: exact dataset to use**
   - What we know: BDG is an NREL initiative, not a single API. Multiple datasets exist.
   - What's unclear: Which specific BDG dataset contains crystal structure + property data suitable for cathode ML. The auto-generated battery materials database (doi:10.1038/s41597-020-00602-2) is one candidate.
   - Recommendation: Implement bdg_fetcher as a download + CSV/JSON parser. If no suitable crystal structure dataset is found, document this and rely on MP + OQMD. The requirement says "Battery Data Genome datasets" (plural) -- may need to aggregate from multiple BDG sources.

2. **MP Insertion Electrode data join strategy**
   - What we know: Electrode docs contain battery_id, material_ids, voltage, capacity. Summary docs contain structure, formation energy, stability.
   - What's unclear: Exact join key between electrode docs and summary docs. material_ids in electrode docs may be a list.
   - Recommendation: Join on material_id. One electrode doc may reference multiple material_ids (working/charged states). Handle this during cleaning.

3. **OQMD structure format conversion**
   - What we know: OQMD returns structures in its own format, not pymatgen Structure directly.
   - What's unclear: Whether qmpy_rester returns enough data to reconstruct a pymatgen Structure (lattice + species + coords).
   - Recommendation: Test qmpy_rester response format. May need to use POSCAR strings or construct Structure from lattice vectors + fractional coordinates. ASE can help as intermediary.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | None -- Wave 0 must create pytest.ini or pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v --tb=short` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | MP fetcher returns structures with properties | unit (mocked API) | `pytest tests/test_mp_fetcher.py -x` | No -- Wave 0 |
| DATA-02 | OQMD fetcher returns structures with properties | unit (mocked API) | `pytest tests/test_oqmd_fetcher.py -x` | No -- Wave 0 |
| DATA-03 | BDG fetcher downloads and parses datasets | unit (mocked HTTP) | `pytest tests/test_bdg_fetcher.py -x` | No -- Wave 0 |
| DATA-04 | Cache prevents repeated API calls | unit | `pytest tests/test_cache.py -x` | No -- Wave 0 |
| DATA-05 | Invalid structures filtered, outliers removed | unit | `pytest tests/test_clean.py -x` | No -- Wave 0 |
| DATA-06 | Cleaning log documents every filter | unit | `pytest tests/test_clean.py::test_filter_logging -x` | No -- Wave 0 |
| REPR-01 | Fixed seeds produce identical outputs | unit | `pytest tests/test_config.py::test_seed_reproducibility -x` | No -- Wave 0 |
| REPR-02 | requirements.txt installs all deps | smoke | `pip install -r requirements.txt --dry-run` | No -- Wave 0 |
| REPR-03 | YAML config controls pipeline behavior | unit | `pytest tests/test_config.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green before verify-work

### Wave 0 Gaps
- [ ] `pyproject.toml` with `[tool.pytest.ini_options]` section
- [ ] `tests/conftest.py` -- shared fixtures (mock pymatgen structures, mock API responses, temp cache dirs)
- [ ] `tests/test_config.py` -- config loading and seed setting tests
- [ ] `tests/test_mp_fetcher.py` -- MP API client tests with mocked responses
- [ ] `tests/test_oqmd_fetcher.py` -- OQMD client tests with mocked responses
- [ ] `tests/test_bdg_fetcher.py` -- BDG download/parse tests with mocked HTTP
- [ ] `tests/test_cache.py` -- cache read/write/invalidation tests
- [ ] `tests/test_clean.py` -- structure validation, filtering, dedup, log tests
- [ ] pytest install: `pip install pytest` (already in requirements.txt)

## Sources

### Primary (HIGH confidence)
- [Materials Project API Documentation](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started) - API access, rate limits, MPRester usage
- [Materials Project API Examples](https://docs.materialsproject.org/downloading-data/using-the-api/examples) - Query patterns
- [MP API Querying Data](https://docs.materialsproject.org/downloading-data/using-the-api/querying-data) - Available fields, search parameters
- [mp-api GitHub (mprester.py)](https://github.com/materialsproject/api/blob/main/mp_api/client/mprester.py) - MPRester implementation, available methods
- [OQMD REST API Documentation](http://oqmd.org/static/docs/restful.html) - OQMD query parameters
- [qmpy_rester GitHub](https://github.com/mohanliu/qmpy_rester) - Python OQMD wrapper usage
- [pymatgen Documentation](https://pymatgen.org/) - Structure validation, serialization

### Secondary (MEDIUM confidence)
- [MatSci Forum: Battery Explorer MPRester](https://matsci.org/t/extracting-battery-explorer-data-using-mprester/46630) - insertion_electrodes.search() usage
- [Auto-generated battery materials database](https://www.nature.com/articles/s41597-020-00602-2) - Text-mined battery properties dataset
- [NREL Battery Data Tools](https://github.com/NREL/battery_data_tools) - Battery Data Genome tooling

### Tertiary (LOW confidence)
- Battery Data Genome as "single API" - confirmed this is NOT the case; it is a collection of datasets and standards. Needs validation during implementation of bdg_fetcher.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - well-established materials informatics libraries, versions verified
- Architecture: HIGH - modular package structure is standard practice, patterns from ARCHITECTURE.md research
- Data source APIs: HIGH (MP), MEDIUM (OQMD -- unmaintained client), LOW (BDG -- unclear access pattern)
- Pitfalls: HIGH - well-documented in materials ML literature

**Research date:** 2026-03-05
**Valid until:** 2026-04-05 (stable domain, libraries change slowly)
