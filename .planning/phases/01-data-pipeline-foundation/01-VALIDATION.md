---
phase: 1
slug: data-pipeline-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-05
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.0 |
| **Config file** | pyproject.toml [tool.pytest.ini_options] — Wave 0 creates |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v --tb=short` |
| **Estimated runtime** | ~5 seconds (mocked APIs) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Req ID | Requirement | Test Type | Automated Command | File Exists | Status |
|--------|-------------|-----------|-------------------|-------------|--------|
| DATA-01 | MP fetcher returns structures with properties | unit (mocked API) | `pytest tests/test_mp_fetcher.py -x` | ❌ W0 | ⬜ pending |
| DATA-02 | OQMD fetcher returns structures with properties | unit (mocked API) | `pytest tests/test_oqmd_fetcher.py -x` | ❌ W0 | ⬜ pending |
| DATA-03 | BDG fetcher downloads and parses datasets | unit (mocked HTTP) | `pytest tests/test_bdg_fetcher.py -x` | ❌ W0 | ⬜ pending |
| DATA-04 | Cache prevents repeated API calls | unit | `pytest tests/test_cache.py -x` | ❌ W0 | ⬜ pending |
| DATA-05 | Invalid structures filtered, outliers removed | unit | `pytest tests/test_clean.py -x` | ❌ W0 | ⬜ pending |
| DATA-06 | Cleaning log documents every filter | unit | `pytest tests/test_clean.py::test_filter_logging -x` | ❌ W0 | ⬜ pending |
| REPR-01 | Fixed seeds produce identical outputs | unit | `pytest tests/test_config.py::test_seed_reproducibility -x` | ❌ W0 | ⬜ pending |
| REPR-02 | requirements.txt installs all deps | smoke | `pip install -r requirements.txt --dry-run` | ❌ W0 | ⬜ pending |
| REPR-03 | YAML config controls pipeline behavior | unit | `pytest tests/test_config.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` with `[tool.pytest.ini_options]` section
- [ ] `tests/conftest.py` — shared fixtures (mock pymatgen structures, mock API responses, temp cache dirs)
- [ ] `tests/test_config.py` — config loading and seed setting tests
- [ ] `tests/test_mp_fetcher.py` — MP API client tests with mocked responses
- [ ] `tests/test_oqmd_fetcher.py` — OQMD client tests with mocked responses
- [ ] `tests/test_bdg_fetcher.py` — BDG download/parse tests with mocked HTTP
- [ ] `tests/test_cache.py` — cache read/write/invalidation tests
- [ ] `tests/test_clean.py` — structure validation, filtering, dedup, log tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| deps install on fresh machine | REPR-02 | Requires clean venv | `python -m venv test_env && source test_env/bin/activate && pip install -r requirements.txt` |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
