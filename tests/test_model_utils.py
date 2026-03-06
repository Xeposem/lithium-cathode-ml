"""Tests for shared model evaluation utilities.

Tests cover compute_metrics (MAE, RMSE, R2 computation) and save_results
(JSON serialization with directory creation).
"""

import json
import math

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cathode_ml.models.utils import compute_metrics, save_results


class TestComputeMetricsValues:
    """compute_metrics produces correct MAE, RMSE, R2 matching sklearn."""

    def test_compute_metrics_values(self):
        """Known inputs produce correct MAE/RMSE/R2."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.7])
        n_train = 10

        result = compute_metrics(y_true, y_pred, n_train)

        expected_mae = float(mean_absolute_error(y_true, y_pred))
        expected_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        expected_r2 = float(r2_score(y_true, y_pred))

        assert math.isclose(result["mae"], expected_mae, rel_tol=1e-9)
        assert math.isclose(result["rmse"], expected_rmse, rel_tol=1e-9)
        assert math.isclose(result["r2"], expected_r2, rel_tol=1e-9)

    def test_compute_metrics_keys(self):
        """Returned dict has exactly {mae, rmse, r2, n_train, n_test} keys."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        result = compute_metrics(y_true, y_pred, n_train=10)
        assert set(result.keys()) == {"mae", "rmse", "r2", "n_train", "n_test"}
        assert result["n_train"] == 10
        assert result["n_test"] == 3


class TestSaveResults:
    """save_results writes valid JSON and creates directories."""

    def test_save_results_creates_json(self, tmp_path):
        """Writes valid JSON to specified path."""
        data = {"mae": 0.15, "rmse": 0.22, "r2": 0.95}
        out_path = str(tmp_path / "results.json")
        save_results(data, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_results_creates_dirs(self, tmp_path):
        """Creates missing parent directories."""
        data = {"mae": 0.1}
        out_path = str(tmp_path / "deep" / "nested" / "dir" / "results.json")
        save_results(data, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == data
