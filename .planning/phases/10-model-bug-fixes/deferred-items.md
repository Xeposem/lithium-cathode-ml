# Deferred Items - Phase 10

## Pre-existing Test Failure

**File:** `tests/test_tensornet.py::TestConvertLightningLogsTensorNet::test_convert_lightning_logs`
**Issue:** Test asserts `float(rows[0]["train_loss"]) == pytest.approx(2.0)` but `rows[0]["train_loss"]` is an empty string. The test expectation doesn't match the actual behavior of `convert_lightning_logs` which shifts train metrics by one epoch (epoch 0 train row becomes empty). The M3GNet version of this test (`test_m3gnet.py`) correctly expects `rows[0]["train_loss"] == ""`.
**Recommendation:** Fix the test assertion to match the M3GNet test pattern.
