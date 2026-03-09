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


# ---------------------------------------------------------------------------
# Test: Structure prediction wiring (cross-phase)
# ---------------------------------------------------------------------------


class TestStructurePrediction:
    """Tests verifying correct wiring between dashboard and core modules."""

    def test_structure_to_graph_import(self):
        """Verify structure_to_graph is importable and accepts (structure, config)."""
        import inspect

        from cathode_ml.features.graph import structure_to_graph

        sig = inspect.signature(structure_to_graph)
        params = list(sig.parameters.keys())
        assert params == ["structure", "config"], (
            f"Expected (structure, config) but got {params}"
        )

    def test_model_loader_uses_structure_to_graph(self):
        """model_loader.py imports structure_to_graph, not structure_to_pyg_data."""
        import ast

        source_path = Path("dashboard/utils/model_loader.py")
        source = source_path.read_text()
        tree = ast.parse(source)

        imports_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "cathode_ml.features.graph" in node.module:
                    for alias in node.names:
                        imports_found.append(alias.name)

        assert "structure_to_graph" in imports_found, (
            f"Expected 'structure_to_graph' import, found: {imports_found}"
        )
        assert "structure_to_pyg_data" not in imports_found, (
            "model_loader.py should NOT import structure_to_pyg_data"
        )

    def test_m3gnet_raw_state_dict_loading(self):
        """M3GNet loader handles raw state_dict (no 'model_state_dict' key)."""
        import ast

        source_path = Path("dashboard/utils/model_loader.py")
        source = source_path.read_text()

        # Verify the code handles both formats (raw and wrapped)
        assert "isinstance(checkpoint, dict)" in source or (
            "model_state_dict" in source
            and "else" in source
        ), "M3GNet loader should handle both raw state_dict and wrapped format"

        # More specifically: should not unconditionally access checkpoint["model_state_dict"]
        tree = ast.parse(source)
        assert 'checkpoint["model_state_dict"]' not in source or (
            "isinstance(checkpoint" in source
            or "if" in source.split('checkpoint["model_state_dict"]')[0].split("\n")[-1]
        ), "M3GNet loader should not unconditionally access checkpoint['model_state_dict']"

    def test_tensornet_state_dict_loading(self):
        """TensorNet loader section exists and handles state_dict loading."""
        source_path = Path("dashboard/utils/model_loader.py")
        source = source_path.read_text()

        # Verify TensorNet branch exists in load_gnn_model
        assert "tensornet" in source, "model_loader.py should handle tensornet"
        assert "build_tensornet_from_config" in source, (
            "TensorNet loader should use build_tensornet_from_config"
        )


class TestPageModuleLevelMain:
    """Tests verifying pages call main() at module level (not behind __name__ guard)."""

    def test_predict_page_module_level_main(self):
        """predict.py calls main() at module level, not behind __name__ guard."""
        import ast

        source_path = Path("dashboard/pages/predict.py")
        source = source_path.read_text()
        tree = ast.parse(source)

        # Check for top-level main() call
        has_top_level_main_call = False
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "main":
                    has_top_level_main_call = True

        assert has_top_level_main_call, (
            "predict.py must call main() at module level (not behind __name__ guard)"
        )

    def test_crystal_viewer_module_level_main(self):
        """crystal_viewer.py calls main() at module level, not behind __name__ guard."""
        import ast

        source_path = Path("dashboard/pages/crystal_viewer.py")
        source = source_path.read_text()
        tree = ast.parse(source)

        # Check for top-level main() call
        has_top_level_main_call = False
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "main":
                    has_top_level_main_call = True

        assert has_top_level_main_call, (
            "crystal_viewer.py must call main() at module level (not behind __name__ guard)"
        )
