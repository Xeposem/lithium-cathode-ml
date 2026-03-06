"""Tests for baseline model training and evaluation pipeline.

Tests cover Random Forest and XGBoost training, metric computation,
JSON result saving, and the full run_baselines orchestrator.
"""

import json
import math
from unittest.mock import patch

import numpy as np
import pytest

from cathode_ml.data.schemas import MaterialRecord
from cathode_ml.models.baselines import (
    evaluate_model,
    run_baselines,
    save_results,
    train_baseline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Generate synthetic training and test data."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(80, 10)
    y_train = rng.randn(80)
    X_test = rng.randn(20, 10)
    y_test = rng.randn(20)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def rf_config():
    """Random Forest hyperparameters."""
    return {"n_estimators": 10, "max_depth": 5, "min_samples_leaf": 2, "n_jobs": 1}


@pytest.fixture
def xgb_config():
    """XGBoost hyperparameters."""
    return {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    }


@pytest.fixture
def fake_records():
    """Create fake MaterialRecord objects for integration testing."""
    formulas = [
        "LiCoO2", "LiCoO2", "LiMnO2", "LiMnO2",
        "LiFePO4", "LiFePO4", "LiNiO2", "LiNiO2",
        "LiTiO2", "LiTiO2", "LiVO2", "LiVO2",
        "LiCrO2", "LiCrO2", "LiMoO2", "LiMoO2",
        "LiWO2", "LiWO2", "LiZnO2", "LiZnO2",
    ]
    records = []
    rng = np.random.RandomState(99)
    for i, f in enumerate(formulas):
        records.append(MaterialRecord(
            material_id=f"test-{i}",
            formula=f,
            structure_dict={},
            source="test",
            formation_energy_per_atom=rng.uniform(-3, 0),
            energy_above_hull=rng.uniform(0, 0.1),
            voltage=rng.uniform(2.5, 4.5),
            capacity=rng.uniform(100, 300),
        ))
    return records


@pytest.fixture
def baselines_config(tmp_path):
    """Baselines configuration dict."""
    return {
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
        },
        "results_dir": str(tmp_path / "results"),
    }


@pytest.fixture
def features_cfg():
    """Features config with target properties and splitting params."""
    return {
        "target_properties": [
            "formation_energy_per_atom",
            "voltage",
        ],
        "splitting": {
            "test_size": 0.2,
            "val_size": 0.1,
        },
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_train_random_forest(synthetic_data, rf_config):
    """train_baseline with model_type='rf' returns a fitted RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor

    X_train, y_train, _, _ = synthetic_data
    model = train_baseline(X_train, y_train, model_type="rf", config=rf_config)
    assert isinstance(model, RandomForestRegressor)
    # Model should be fitted (has estimators_)
    assert hasattr(model, "estimators_")


def test_train_xgboost(synthetic_data, xgb_config):
    """train_baseline with model_type='xgb' returns a fitted XGBRegressor."""
    from xgboost import XGBRegressor

    X_train, y_train, _, _ = synthetic_data
    model = train_baseline(X_train, y_train, model_type="xgb", config=xgb_config)
    assert isinstance(model, XGBRegressor)


def test_evaluate_metrics(synthetic_data, rf_config):
    """evaluate_model returns dict with mae, rmse, r2, n_train, n_test."""
    X_train, y_train, X_test, y_test = synthetic_data
    model = train_baseline(X_train, y_train, model_type="rf", config=rf_config)
    metrics = evaluate_model(model, X_test, y_test, n_train=len(X_train))
    assert set(metrics.keys()) >= {"mae", "rmse", "r2", "n_train", "n_test"}
    assert metrics["n_train"] == 80
    assert metrics["n_test"] == 20


def test_metrics_are_finite(synthetic_data, rf_config):
    """All metric values are finite floats (not NaN or inf)."""
    X_train, y_train, X_test, y_test = synthetic_data
    model = train_baseline(X_train, y_train, model_type="rf", config=rf_config)
    metrics = evaluate_model(model, X_test, y_test, n_train=len(X_train))
    for key in ("mae", "rmse", "r2"):
        assert isinstance(metrics[key], float), f"{key} is not float"
        assert math.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"


def test_save_results_json(tmp_path):
    """save_results creates a valid JSON file loadable with json.load."""
    results = {"formation_energy_per_atom": {"rf": {"mae": 0.1}}}
    out_path = str(tmp_path / "subdir" / "results.json")
    save_results(results, out_path)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == results


def test_run_baselines_per_property(fake_records, baselines_config, features_cfg):
    """run_baselines trains separate models per target property."""
    # Mock featurize_compositions to avoid slow matminer
    n = len(fake_records)
    fake_X = np.random.RandomState(42).randn(n, 10)
    fake_labels = [f"feat_{i}" for i in range(10)]

    with patch(
        "cathode_ml.models.baselines.featurize_compositions",
        return_value=(fake_X, fake_labels),
    ):
        results = run_baselines(
            fake_records, features_cfg, baselines_config, seed=42
        )

    # Should have results for each target property
    for prop in features_cfg["target_properties"]:
        assert prop in results, f"Missing property {prop} in results"


def test_results_contain_both_models(fake_records, baselines_config, features_cfg):
    """Each property result dict contains both 'rf' and 'xgb' sub-dicts."""
    n = len(fake_records)
    fake_X = np.random.RandomState(42).randn(n, 10)
    fake_labels = [f"feat_{i}" for i in range(10)]

    with patch(
        "cathode_ml.models.baselines.featurize_compositions",
        return_value=(fake_X, fake_labels),
    ):
        results = run_baselines(
            fake_records, features_cfg, baselines_config, seed=42
        )

    for prop in features_cfg["target_properties"]:
        assert "rf" in results[prop], f"Missing 'rf' in {prop}"
        assert "xgb" in results[prop], f"Missing 'xgb' in {prop}"
        # Each model result should have metrics
        for model_type in ("rf", "xgb"):
            assert "mae" in results[prop][model_type]
            assert "rmse" in results[prop][model_type]
            assert "r2" in results[prop][model_type]
