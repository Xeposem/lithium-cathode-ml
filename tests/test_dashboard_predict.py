"""Tests for baseline model persistence and model loader utilities.

Covers joblib model saving in run_baselines, model loading functions,
and composition-based prediction pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test: run_baselines saves joblib files
# ---------------------------------------------------------------------------


class TestBaselinesPersistence:
    """Verify that run_baselines saves joblib model files alongside JSON."""

    def test_baselines_save_joblib(self, tmp_path):
        """After run_baselines, joblib files exist at baselines/{model}_{prop}.joblib."""
        import joblib

        from cathode_ml.data.schemas import MaterialRecord

        # Create minimal records with formation_energy_per_atom
        # Need diverse formulas for compositional group splitting
        formulas = [
            "LiCoO2", "LiFePO4", "LiMnO2", "LiNiO2", "LiCrO2",
            "NaCoO2", "NaFePO4", "NaMnO2", "NaNiO2", "NaCrO2",
            "KCoO2", "KFePO4", "KMnO2", "KNiO2", "KCrO2",
            "CsCoO2", "CsFePO4", "CsMnO2", "CsNiO2", "CsCrO2",
        ]
        records = []
        for i, formula in enumerate(formulas):
            records.append(
                MaterialRecord(
                    material_id=f"mp-{i}",
                    formula=formula,
                    structure_dict={},
                    source="mp",
                    formation_energy_per_atom=float(i) * 0.1,
                    energy_above_hull=None,
                    voltage=None,
                    capacity=None,
                    is_stable=None,
                    space_group=None,
                )
            )

        features_config = {
            "target_properties": ["formation_energy_per_atom"],
            "splitting": {"test_size": 0.2, "val_size": 0.1},
        }
        baselines_config = {
            "random_forest": {"n_estimators": 10},
            "xgboost": {"n_estimators": 10},
            "results_dir": str(tmp_path / "results"),
        }

        from cathode_ml.models.baselines import run_baselines

        run_baselines(records, features_config, baselines_config, seed=42)

        baselines_dir = tmp_path / "results" / "baselines"
        rf_path = baselines_dir / "rf_formation_energy_per_atom.joblib"
        xgb_path = baselines_dir / "xgb_formation_energy_per_atom.joblib"

        assert rf_path.exists(), f"RF joblib not found at {rf_path}"
        assert xgb_path.exists(), f"XGB joblib not found at {xgb_path}"

        # Verify they are loadable and have predict method
        rf_model = joblib.load(rf_path)
        assert hasattr(rf_model, "predict")

        xgb_model = joblib.load(xgb_path)
        assert hasattr(xgb_model, "predict")


# ---------------------------------------------------------------------------
# Test: model loader utilities
# ---------------------------------------------------------------------------


class TestModelLoader:
    """Tests for dashboard/utils/model_loader.py functions."""

    def test_load_baseline_model(self, tmp_path):
        """load_baseline_model returns object with .predict() method."""
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        # Create a mock model file
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(np.random.rand(10, 5), np.random.rand(10))

        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        joblib.dump(model, baselines_dir / "rf_voltage.joblib")

        from dashboard.utils.model_loader import load_baseline_model

        loaded = load_baseline_model("rf", "voltage", results_base=str(tmp_path))
        assert loaded is not None
        assert hasattr(loaded, "predict")

    def test_load_baseline_model_missing(self, tmp_path, caplog):
        """Returns None and logs warning when file missing."""
        from dashboard.utils.model_loader import load_baseline_model

        with caplog.at_level(logging.WARNING):
            result = load_baseline_model(
                "rf", "voltage", results_base=str(tmp_path / "nonexistent")
            )

        assert result is None
        assert any("not found" in msg.lower() or "missing" in msg.lower() for msg in caplog.messages)

    def test_predict_from_composition(self, tmp_path):
        """predict_from_composition returns dict with property keys and float values."""
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        # Create mock models for one property
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        # Train on dummy Magpie-like features (132-dim)
        X_dummy = np.random.rand(20, 132)
        y_dummy = np.random.rand(20)
        model.fit(X_dummy, y_dummy)
        joblib.dump(model, baselines_dir / "rf_voltage.joblib")

        from dashboard.utils.model_loader import predict_from_composition

        result = predict_from_composition("LiFePO4", results_base=str(tmp_path))

        assert isinstance(result, dict)
        assert "voltage" in result
        assert "rf" in result["voltage"]
        assert isinstance(result["voltage"]["rf"], float)

    def test_predict_from_composition_no_models(self, tmp_path):
        """Returns empty dict when no models persisted."""
        from dashboard.utils.model_loader import predict_from_composition

        result = predict_from_composition(
            "LiFePO4", results_base=str(tmp_path / "nonexistent")
        )
        assert result == {}
