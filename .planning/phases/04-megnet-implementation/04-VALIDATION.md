---
phase: 4
slug: megnet-implementation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-06
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 |
| **Config file** | tests/ directory (uses defaults) |
| **Quick run command** | `pytest tests/test_megnet.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~30 seconds (unit), ~120 seconds (integration with training) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_megnet.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | MODL-02 | unit | `pytest tests/test_megnet.py::test_load_pretrained -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | MODL-02 | integration | `pytest tests/test_megnet.py::test_megnet_training -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | MODL-02 | integration | `pytest tests/test_megnet.py::test_artifact_format -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | MODL-02 | unit | `pytest tests/test_megnet.py::test_same_splits -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_megnet.py` — stubs for MODL-02 (MEGNet model loading, training, artifact format, split consistency)
- [ ] Tests requiring matgl/DGL marked with `@pytest.mark.skipif` when matgl is not installed (consistent with lazy import pattern)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| DGL+PyG coexistence | MODL-02 | Environment-specific | `pip install matgl==1.3.0` in existing env, verify `import matgl` and `import torch_geometric` both work |
| Docker fallback | MODL-02 | Infrastructure | If DGL conflicts, build Dockerfile and verify `docker run` trains MEGNet |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
