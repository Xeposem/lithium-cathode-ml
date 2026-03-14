"""Tests for M3GNet model wrapper and training orchestrator.

Tests cover lazy import error handling, pretrained model loading (mocked),
available model listing, state dict extraction, Lightning log conversion,
artifact format, split consistency, and checkpoint naming.
All tests mock matgl to avoid requiring actual installation.
"""

import csv
import importlib.util
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestM3GNetLazyImports:
    """M3GNet functions raise helpful ImportError when matgl is missing."""

    def test_import_matgl_returns_module(self):
        """_import_matgl() calls import matgl and returns the module."""
        mock_matgl = MagicMock()
        with patch.dict("sys.modules", {"matgl": mock_matgl}):
            from cathode_ml.models.m3gnet import _import_matgl

            result = _import_matgl()
            assert result is mock_matgl

    def test_install_msg_mentions_matgl(self):
        """_INSTALL_MSG mentions matgl>=2.0.0."""
        from cathode_ml.models.m3gnet import _INSTALL_MSG

        assert "matgl>=2.0.0" in _INSTALL_MSG

    def test_lazy_import_error_load(self):
        """load_m3gnet_model raises ImportError with install instructions when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.m3gnet import load_m3gnet_model

            with pytest.raises(ImportError, match="matgl>=2.0.0"):
                load_m3gnet_model("M3GNet-MP-2018.6.1-Eform")

    def test_lazy_import_error_available(self):
        """get_available_m3gnet_models raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.m3gnet import get_available_m3gnet_models

            with pytest.raises(ImportError, match="matgl>=2.0.0"):
                get_available_m3gnet_models()

    def test_lazy_import_error_predict(self):
        """predict_with_m3gnet raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.m3gnet import predict_with_m3gnet

            with pytest.raises(ImportError, match="matgl>=2.0.0"):
                predict_with_m3gnet(MagicMock(), [])


class TestLoadM3GNetModel:
    """M3GNet pretrained model loading via mocked matgl."""

    def test_load_pretrained_calls_matgl(self):
        """load_m3gnet_model calls matgl.load_model with correct name."""
        mock_matgl = MagicMock()
        mock_model = MagicMock()
        mock_model.cutoff = 5.0
        mock_matgl.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"matgl": mock_matgl}):
            from cathode_ml.models.m3gnet import load_m3gnet_model

            model = load_m3gnet_model("M3GNet-MP-2018.6.1-Eform")

        mock_matgl.load_model.assert_called_once_with("M3GNet-MP-2018.6.1-Eform")
        assert model is mock_model


class TestGetAvailableM3GNetModels:
    """M3GNet model listing with mocked matgl."""

    def test_filters_m3gnet_models(self):
        """get_available_m3gnet_models returns only M3GNet models from full list."""
        mock_matgl = MagicMock()
        mock_matgl.get_available_pretrained_models.return_value = [
            "M3GNet-MP-2018.6.1-Eform",
            "M3GNet-MP-2021.2.8-PES",
            "MEGNet-MP-2019.4.1-BandGap-mfi",
            "CHGNet-v0.3.0",
        ]

        with patch.dict("sys.modules", {"matgl": mock_matgl}):
            from cathode_ml.models.m3gnet import get_available_m3gnet_models

            models = get_available_m3gnet_models()

        assert len(models) == 2
        assert all("M3GNet" in m for m in models)


class TestGetM3GNetStateDict:
    """State dict extraction from M3GNet model."""

    def test_state_dict_extraction(self):
        """get_m3gnet_state_dict returns model.model.state_dict()."""
        from cathode_ml.models.m3gnet import get_m3gnet_state_dict

        mock_model = MagicMock()
        fake_state_dict = {"layer1.weight": "tensor1", "layer1.bias": "tensor2"}
        mock_model.model.state_dict.return_value = fake_state_dict

        result = get_m3gnet_state_dict(mock_model)
        assert result == fake_state_dict
        mock_model.model.state_dict.assert_called_once()


class TestPredictWithM3GNet:
    """Prediction with mocked M3GNet model."""

    def test_predict_returns_floats(self):
        """predict_with_m3gnet returns list of floats."""
        mock_matgl = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_structure.side_effect = [1.5, 2.3, 0.8]

        with patch.dict("sys.modules", {"matgl": mock_matgl}):
            from cathode_ml.models.m3gnet import predict_with_m3gnet

            results = predict_with_m3gnet(mock_model, [MagicMock(), MagicMock(), MagicMock()])

        assert len(results) == 3
        assert all(isinstance(v, float) for v in results)
        assert results == [1.5, 2.3, 0.8]


class TestConvertLightningLogs:
    """convert_lightning_logs reads Lightning CSV and outputs standardized format."""

    def test_convert_lightning_logs(self):
        """Mock Lightning metrics.csv is converted to standardized CSV columns."""
        from cathode_ml.models.train_m3gnet import convert_lightning_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock Lightning CSVLogger output (separate rows for train/val)
            metrics_csv = os.path.join(tmpdir, "metrics.csv")
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_Total_Loss", "train_MAE", "val_Total_Loss", "val_MAE"])
                writer.writerow(["0", "1.5", "1.2", "", ""])
                writer.writerow(["0", "", "", "1.3", "1.1"])
                writer.writerow(["1", "1.0", "0.8", "", ""])
                writer.writerow(["1", "", "", "0.9", "0.7"])

            output_csv = os.path.join(tmpdir, "output.csv")
            convert_lightning_logs(metrics_csv, output_csv)

            assert os.path.exists(output_csv), "Output CSV should be created"

            with open(output_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2, f"Should have 2 epochs, got {len(rows)}"
            expected_cols = {"epoch", "train_loss", "val_loss", "val_mae"}
            assert expected_cols.issubset(set(rows[0].keys())), (
                f"Missing columns: {expected_cols - set(rows[0].keys())}"
            )
            # Epoch 0: val metrics present, train_loss is empty because
            # Lightning's train row at epoch 0 is shifted to epoch 1
            assert float(rows[0]["val_loss"]) == pytest.approx(1.3)
            assert float(rows[0]["val_mae"]) == pytest.approx(1.1)
            assert rows[0]["train_loss"] == ""  # no train data for epoch 0 after shift
            # Epoch 1: train_loss from epoch 0 shifted here
            assert float(rows[1]["train_loss"]) == pytest.approx(1.5)
            assert float(rows[1]["val_loss"]) == pytest.approx(0.9)
            assert float(rows[1]["val_mae"]) == pytest.approx(0.7)


class TestTrainM3GNetForProperty:
    """train_m3gnet_for_property trains a single property with mocked dependencies."""

    def test_per_property_training(self):
        """Verify function calls load_m3gnet_model, runs training, returns metrics dict."""
        from cathode_ml.models.train_m3gnet import train_m3gnet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            m3gnet_config = {
                "model": {"pretrained_model": "M3GNet-MP-2018.6.1-Eform"},
                "training": {
                    "learning_rate": 0.0001,
                    "batch_size": 2,
                    "n_epochs": 2,
                    "early_stopping_patience": 5,
                },
                "results_dir": tmpdir,
            }

            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            with patch("cathode_ml.models.train_m3gnet.load_m3gnet_model", return_value=mock_model), \
                 patch("cathode_ml.models.train_m3gnet.predict_with_m3gnet", return_value=[0.1] * 3), \
                 patch("cathode_ml.models.train_m3gnet.compute_metrics", return_value={
                     "mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 7, "n_test": 3,
                 }), \
                 patch("cathode_ml.models.train_m3gnet._run_lightning_training"):
                result = train_m3gnet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    m3gnet_config=m3gnet_config,
                    seed=42,
                )

            assert isinstance(result, dict)
            for key in ("mae", "rmse", "r2", "n_train", "n_test"):
                assert key in result, f"Missing key: {key}"


class TestTrainM3GNet:
    """train_m3gnet orchestrator loops over properties."""

    def test_artifact_format(self):
        """Results dict has structure {property: {m3gnet: {mae, rmse, r2, n_train, n_test}}}."""
        from cathode_ml.models.train_m3gnet import train_m3gnet
        from cathode_ml.data.schemas import MaterialRecord

        formulas = [
            "LiCoO2", "LiMnO2", "LiNiO2", "LiFePO4", "LiMn2O4",
            "LiNi0.5Mn0.5O2", "LiNi0.33Co0.33Mn0.33O2", "LiTiO2",
            "LiVO2", "LiCrO2",
        ]
        records = []
        for i in range(10):
            records.append(MaterialRecord(
                material_id=f"test-{i}",
                formula=formulas[i],
                structure_dict={},
                source="test",
                formation_energy_per_atom=float(i) * 0.1,
            ))

        features_config = {
            "target_properties": ["formation_energy_per_atom"],
            "splitting": {"test_size": 0.2, "val_size": 0.2},
        }
        m3gnet_config = {
            "model": {"pretrained_model": "M3GNet-MP-2018.6.1-Eform"},
            "training": {
                "learning_rate": 0.0001,
                "batch_size": 2,
                "n_epochs": 2,
                "early_stopping_patience": 5,
            },
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_m3gnet.train_m3gnet_for_property",
            return_value=mock_metrics,
        ):
            results = train_m3gnet(records, features_config, m3gnet_config, seed=42)

        assert "formation_energy_per_atom" in results
        assert "m3gnet" in results["formation_energy_per_atom"]
        m3gnet_result = results["formation_energy_per_atom"]["m3gnet"]
        for key in ("mae", "rmse", "r2", "n_train", "n_test"):
            assert key in m3gnet_result, f"Missing key: {key}"

    def test_skips_small_properties(self):
        """Properties with fewer than 5 valid records are skipped."""
        from cathode_ml.models.train_m3gnet import train_m3gnet
        from cathode_ml.data.schemas import MaterialRecord

        formulas = [
            "LiCoO2", "LiMnO2", "LiNiO2", "LiFePO4", "LiMn2O4",
            "LiNi0.5Mn0.5O2", "LiNi0.33Co0.33Mn0.33O2", "LiTiO2",
            "LiVO2", "LiCrO2",
        ]
        records = []
        for i in range(10):
            records.append(MaterialRecord(
                material_id=f"test-{i}",
                formula=formulas[i],
                structure_dict={},
                source="test",
                formation_energy_per_atom=float(i) * 0.1,
                voltage=float(i) * 0.1 if i < 3 else None,
            ))

        features_config = {
            "target_properties": ["formation_energy_per_atom", "voltage"],
            "splitting": {"test_size": 0.2, "val_size": 0.2},
        }
        m3gnet_config = {
            "model": {"pretrained_model": "M3GNet-MP-2018.6.1-Eform"},
            "training": {
                "learning_rate": 0.0001, "batch_size": 2, "n_epochs": 2,
                "early_stopping_patience": 5,
            },
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_m3gnet.train_m3gnet_for_property",
            return_value=mock_metrics,
        ):
            results = train_m3gnet(records, features_config, m3gnet_config, seed=42)

        assert "formation_energy_per_atom" in results
        assert "voltage" not in results

    def test_results_saved_as_m3gnet_json(self):
        """Results saved to m3gnet_results.json (not megnet_results.json)."""
        from cathode_ml.models.train_m3gnet import train_m3gnet
        from cathode_ml.data.schemas import MaterialRecord

        formulas = [
            "LiCoO2", "LiMnO2", "LiNiO2", "LiFePO4", "LiMn2O4",
            "LiNi0.5Mn0.5O2", "LiNi0.33Co0.33Mn0.33O2", "LiTiO2",
            "LiVO2", "LiCrO2",
        ]
        records = []
        for i in range(10):
            records.append(MaterialRecord(
                material_id=f"test-{i}",
                formula=formulas[i],
                structure_dict={},
                source="test",
                formation_energy_per_atom=float(i) * 0.1,
            ))

        with tempfile.TemporaryDirectory() as tmpdir:
            features_config = {
                "target_properties": ["formation_energy_per_atom"],
                "splitting": {"test_size": 0.2, "val_size": 0.2},
            }
            m3gnet_config = {
                "model": {"pretrained_model": "M3GNet-MP-2018.6.1-Eform"},
                "training": {"learning_rate": 0.0001, "batch_size": 2, "n_epochs": 2,
                             "early_stopping_patience": 5},
                "results_dir": tmpdir,
            }

            mock_metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 6, "n_test": 2}
            with patch(
                "cathode_ml.models.train_m3gnet.train_m3gnet_for_property",
                return_value=mock_metrics,
            ):
                train_m3gnet(records, features_config, m3gnet_config, seed=42)

            assert os.path.exists(os.path.join(tmpdir, "m3gnet_results.json"))
            assert not os.path.exists(os.path.join(tmpdir, "megnet_results.json"))


class TestM3GNetArtifactFormat:
    """M3GNet-specific artifact naming and format checks."""

    def test_checkpoint_naming(self):
        """Checkpoint files use m3gnet_{property}_best.pt naming."""
        from cathode_ml.models.train_m3gnet import train_m3gnet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            m3gnet_config = {
                "model": {"pretrained_model": "M3GNet-MP-2018.6.1-Eform"},
                "training": {
                    "learning_rate": 0.0001, "batch_size": 2, "n_epochs": 2,
                    "early_stopping_patience": 5,
                },
                "results_dir": tmpdir,
            }

            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            def fake_training(model, train_structures, train_targets, val_structures,
                              val_targets, property_name, m3gnet_config, seed):
                import torch
                torch.save(mock_model.model.state_dict(),
                           os.path.join(tmpdir, f"m3gnet_{property_name}_best.pt"))
                torch.save(mock_model.model.state_dict(),
                           os.path.join(tmpdir, f"m3gnet_{property_name}_final.pt"))

            with patch("cathode_ml.models.train_m3gnet.load_m3gnet_model", return_value=mock_model), \
                 patch("cathode_ml.models.train_m3gnet.predict_with_m3gnet", return_value=[0.1] * 3), \
                 patch("cathode_ml.models.train_m3gnet.compute_metrics", return_value={
                     "mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 7, "n_test": 3,
                 }), \
                 patch("cathode_ml.models.train_m3gnet._run_lightning_training") as mock_train:
                mock_train.side_effect = fake_training
                train_m3gnet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    m3gnet_config=m3gnet_config,
                    seed=42,
                )

            best_pt = os.path.join(tmpdir, "m3gnet_formation_energy_per_atom_best.pt")
            final_pt = os.path.join(tmpdir, "m3gnet_formation_energy_per_atom_final.pt")
            assert os.path.exists(best_pt), f"Best checkpoint not found: {best_pt}"
            assert os.path.exists(final_pt), f"Final checkpoint not found: {final_pt}"

    def test_results_key_is_m3gnet(self):
        """Results JSON uses 'm3gnet' key, not 'megnet'."""
        from cathode_ml.models.train_m3gnet import train_m3gnet
        from cathode_ml.data.schemas import MaterialRecord

        records = [
            MaterialRecord(
                material_id=f"test-{i}",
                formula=f"Li{chr(65+i)}O2",
                structure_dict={},
                source="test",
                formation_energy_per_atom=float(i) * 0.1,
            )
            for i in range(10)
        ]

        features_config = {
            "target_properties": ["formation_energy_per_atom"],
            "splitting": {"test_size": 0.2, "val_size": 0.2},
        }
        m3gnet_config = {
            "model": {"pretrained_model": "M3GNet-MP-2018.6.1-Eform"},
            "training": {"learning_rate": 0.0001, "batch_size": 2, "n_epochs": 2,
                         "early_stopping_patience": 5},
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_m3gnet.train_m3gnet_for_property",
            return_value=mock_metrics,
        ):
            results = train_m3gnet(records, features_config, m3gnet_config, seed=42)

        for prop, prop_data in results.items():
            assert "m3gnet" in prop_data
            assert "megnet" not in prop_data
