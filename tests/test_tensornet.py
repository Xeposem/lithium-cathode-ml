"""Tests for TensorNet model wrapper and training orchestrator.

Tests cover lazy import error handling, config-driven construction (mocked),
state dict extraction, Lightning log conversion, artifact format, from-scratch
training loop, and element_types handling.
All tests mock matgl to avoid requiring actual installation.
"""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTensorNetLazyImports:
    """TensorNet functions raise helpful ImportError when matgl is missing."""

    def test_import_matgl_returns_module(self):
        """_import_matgl() calls import matgl and returns the module."""
        mock_matgl = MagicMock()
        with patch.dict("sys.modules", {"matgl": mock_matgl}):
            from cathode_ml.models.tensornet import _import_matgl

            result = _import_matgl()
            assert result is mock_matgl

    def test_install_msg_mentions_matgl(self):
        """_INSTALL_MSG mentions matgl>=2.0.0."""
        from cathode_ml.models.tensornet import _INSTALL_MSG

        assert "matgl>=2.0.0" in _INSTALL_MSG

    def test_lazy_import_error_predict(self):
        """predict_with_tensornet raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.tensornet import predict_with_tensornet

            with pytest.raises(ImportError, match="matgl>=2.0.0"):
                predict_with_tensornet(MagicMock(), [])


class TestBuildTensorNetFromConfig:
    """TensorNet construction from config with mocked matgl."""

    def test_build_calls_tensornet_constructor(self):
        """build_tensornet_from_config calls TensorNet with correct params."""
        mock_matgl = MagicMock()
        mock_tensornet_cls = MagicMock()
        mock_tensornet_instance = MagicMock()
        mock_tensornet_cls.return_value = mock_tensornet_instance

        # Mock the matgl.models module
        mock_models = MagicMock()
        mock_models.TensorNet = mock_tensornet_cls

        with patch.dict("sys.modules", {
            "matgl": mock_matgl,
            "matgl.models": mock_models,
        }):
            from cathode_ml.models.tensornet import build_tensornet_from_config

            model_config = {
                "units": 128,
                "nblocks": 3,
                "cutoff": 6.0,
            }
            element_types = ["Li", "Co", "O"]

            result = build_tensornet_from_config(model_config, element_types)

        mock_tensornet_cls.assert_called_once()
        call_kwargs = mock_tensornet_cls.call_args[1]
        assert call_kwargs["element_types"] == ["Li", "Co", "O"]
        assert call_kwargs["units"] == 128
        assert call_kwargs["nblocks"] == 3
        assert call_kwargs["cutoff"] == 6.0
        assert result is mock_tensornet_instance

    def test_build_uses_defaults(self):
        """build_tensornet_from_config uses defaults for missing config keys."""
        mock_matgl = MagicMock()
        mock_tensornet_cls = MagicMock()
        mock_models = MagicMock()
        mock_models.TensorNet = mock_tensornet_cls

        with patch.dict("sys.modules", {
            "matgl": mock_matgl,
            "matgl.models": mock_models,
        }):
            from cathode_ml.models.tensornet import build_tensornet_from_config

            build_tensornet_from_config({}, ["Li", "O"])

        call_kwargs = mock_tensornet_cls.call_args[1]
        assert call_kwargs["units"] == 64
        assert call_kwargs["nblocks"] == 2
        assert call_kwargs["cutoff"] == 5.0

    def test_element_types_passed_through(self):
        """element_types is passed directly to TensorNet constructor."""
        mock_matgl = MagicMock()
        mock_tensornet_cls = MagicMock()
        mock_models = MagicMock()
        mock_models.TensorNet = mock_tensornet_cls

        with patch.dict("sys.modules", {
            "matgl": mock_matgl,
            "matgl.models": mock_models,
        }):
            from cathode_ml.models.tensornet import build_tensornet_from_config

            elements = ["Li", "Fe", "P", "O"]
            build_tensornet_from_config({}, elements)

        call_kwargs = mock_tensornet_cls.call_args[1]
        assert call_kwargs["element_types"] == ["Li", "Fe", "P", "O"]


class TestGetTensorNetStateDict:
    """State dict extraction from TensorNet model."""

    def test_state_dict_extraction(self):
        """get_tensornet_state_dict returns model.model.state_dict()."""
        from cathode_ml.models.tensornet import get_tensornet_state_dict

        mock_model = MagicMock()
        fake_state_dict = {"layer1.weight": "tensor1", "layer1.bias": "tensor2"}
        mock_model.model.state_dict.return_value = fake_state_dict

        result = get_tensornet_state_dict(mock_model)
        assert result == fake_state_dict
        mock_model.model.state_dict.assert_called_once()


class TestPredictWithTensorNet:
    """Prediction with mocked TensorNet model."""

    def test_predict_returns_floats(self):
        """predict_with_tensornet returns list of floats."""
        mock_matgl = MagicMock()
        mock_model = MagicMock()
        mock_model.predict_structure.side_effect = [1.5, 2.3, 0.8]

        with patch.dict("sys.modules", {"matgl": mock_matgl}):
            from cathode_ml.models.tensornet import predict_with_tensornet

            results = predict_with_tensornet(mock_model, [MagicMock(), MagicMock(), MagicMock()])

        assert len(results) == 3
        assert all(isinstance(v, float) for v in results)
        assert results == [1.5, 2.3, 0.8]


class TestConvertLightningLogsTensorNet:
    """convert_lightning_logs from TensorNet training module."""

    def test_convert_lightning_logs(self):
        """Mock Lightning metrics.csv is converted to standardized CSV columns."""
        from cathode_ml.models.train_tensornet import convert_lightning_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_csv = os.path.join(tmpdir, "metrics.csv")
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_Total_Loss", "train_MAE", "val_Total_Loss", "val_MAE"])
                writer.writerow(["0", "2.0", "1.5", "", ""])
                writer.writerow(["0", "", "", "1.8", "1.3"])
                writer.writerow(["1", "1.5", "1.0", "", ""])
                writer.writerow(["1", "", "", "1.2", "0.9"])

            output_csv = os.path.join(tmpdir, "output.csv")
            convert_lightning_logs(metrics_csv, output_csv)

            assert os.path.exists(output_csv)

            with open(output_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert float(rows[0]["train_loss"]) == pytest.approx(2.0)
            assert float(rows[1]["val_mae"]) == pytest.approx(0.9)


class TestTrainTensorNetForProperty:
    """train_tensornet_for_property trains via config-driven model construction."""

    def test_per_property_training_from_scratch(self):
        """Verify model is built from config (not pretrained), returns metrics dict."""
        from cathode_ml.models.train_tensornet import train_tensornet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            tensornet_config = {
                "model": {
                    "units": 64,
                    "nblocks": 2,
                    "cutoff": 5.0,
                },
                "training": {
                    "learning_rate": 0.001,
                    "batch_size": 2,
                    "n_epochs": 2,
                    "early_stopping_patience": 5,
                },
                "results_dir": tmpdir,
            }

            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            # Pre-populate sys.modules with mocked matgl modules so lazy imports succeed
            mock_matgl = MagicMock()
            mock_matgl_ext = MagicMock()
            mock_matgl_ext_pymatgen = MagicMock()
            mock_matgl_ext_pymatgen.get_element_list = MagicMock(return_value=["Li", "Co", "O"])

            with patch.dict("sys.modules", {
                "matgl": mock_matgl,
                "matgl.ext": mock_matgl_ext,
                "matgl.ext.pymatgen": mock_matgl_ext_pymatgen,
            }), \
                 patch("cathode_ml.models.train_tensornet.build_tensornet_from_config",
                        return_value=mock_model) as mock_build, \
                 patch("cathode_ml.models.train_tensornet.predict_with_tensornet",
                        return_value=[0.1] * 3), \
                 patch("cathode_ml.models.train_tensornet.compute_metrics", return_value={
                     "mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 7, "n_test": 3,
                 }), \
                 patch("cathode_ml.models.train_tensornet._run_lightning_training"):
                result = train_tensornet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    tensornet_config=tensornet_config,
                    seed=42,
                )

            # Verify build_tensornet_from_config was called (not load_model)
            mock_build.assert_called_once()
            assert isinstance(result, dict)
            for key in ("mae", "rmse", "r2", "n_train", "n_test"):
                assert key in result


class TestTrainTensorNet:
    """train_tensornet orchestrator loops over properties."""

    def test_artifact_format(self):
        """Results dict has structure {property: {tensornet: {mae, rmse, r2, n_train, n_test}}}."""
        from cathode_ml.models.train_tensornet import train_tensornet
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
        tensornet_config = {
            "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
            "training": {
                "learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                "early_stopping_patience": 5,
            },
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_tensornet.train_tensornet_for_property",
            return_value=mock_metrics,
        ):
            results = train_tensornet(records, features_config, tensornet_config, seed=42)

        assert "formation_energy_per_atom" in results
        assert "tensornet" in results["formation_energy_per_atom"]
        tensornet_result = results["formation_energy_per_atom"]["tensornet"]
        for key in ("mae", "rmse", "r2", "n_train", "n_test"):
            assert key in tensornet_result

    def test_skips_small_properties(self):
        """Properties with fewer than 5 valid records are skipped."""
        from cathode_ml.models.train_tensornet import train_tensornet
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
        tensornet_config = {
            "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
            "training": {
                "learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                "early_stopping_patience": 5,
            },
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_tensornet.train_tensornet_for_property",
            return_value=mock_metrics,
        ):
            results = train_tensornet(records, features_config, tensornet_config, seed=42)

        assert "formation_energy_per_atom" in results
        assert "voltage" not in results

    def test_results_saved_as_tensornet_json(self):
        """Results saved to tensornet_results.json."""
        from cathode_ml.models.train_tensornet import train_tensornet
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
            tensornet_config = {
                "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
                "training": {"learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                             "early_stopping_patience": 5},
                "results_dir": tmpdir,
            }

            mock_metrics = {"mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 6, "n_test": 2}
            with patch(
                "cathode_ml.models.train_tensornet.train_tensornet_for_property",
                return_value=mock_metrics,
            ):
                train_tensornet(records, features_config, tensornet_config, seed=42)

            assert os.path.exists(os.path.join(tmpdir, "tensornet_results.json"))


class TestTensorNetArtifactFormat:
    """TensorNet-specific artifact naming and format checks."""

    def test_checkpoint_naming(self):
        """Checkpoint files use tensornet_{property}_best.pt naming."""
        from cathode_ml.models.train_tensornet import train_tensornet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            tensornet_config = {
                "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
                "training": {
                    "learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                    "early_stopping_patience": 5,
                },
                "results_dir": tmpdir,
            }

            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            def fake_training(model, train_structures, train_targets, val_structures,
                              val_targets, property_name, tensornet_config, seed):
                import torch
                torch.save(mock_model.model.state_dict(),
                           os.path.join(tmpdir, f"tensornet_{property_name}_best.pt"))
                torch.save(mock_model.model.state_dict(),
                           os.path.join(tmpdir, f"tensornet_{property_name}_final.pt"))

            mock_matgl = MagicMock()
            mock_matgl_ext = MagicMock()
            mock_matgl_ext_pymatgen = MagicMock()
            mock_matgl_ext_pymatgen.get_element_list = MagicMock(return_value=["Li", "Co", "O"])

            with patch.dict("sys.modules", {
                "matgl": mock_matgl,
                "matgl.ext": mock_matgl_ext,
                "matgl.ext.pymatgen": mock_matgl_ext_pymatgen,
            }), \
                 patch("cathode_ml.models.train_tensornet.build_tensornet_from_config",
                        return_value=mock_model), \
                 patch("cathode_ml.models.train_tensornet.predict_with_tensornet",
                        return_value=[0.1] * 3), \
                 patch("cathode_ml.models.train_tensornet.compute_metrics", return_value={
                     "mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 7, "n_test": 3,
                 }), \
                 patch("cathode_ml.models.train_tensornet._run_lightning_training") as mock_train:
                mock_train.side_effect = fake_training
                train_tensornet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    tensornet_config=tensornet_config,
                    seed=42,
                )

            best_pt = os.path.join(tmpdir, "tensornet_formation_energy_per_atom_best.pt")
            final_pt = os.path.join(tmpdir, "tensornet_formation_energy_per_atom_final.pt")
            assert os.path.exists(best_pt), f"Best checkpoint not found: {best_pt}"
            assert os.path.exists(final_pt), f"Final checkpoint not found: {final_pt}"

    def test_results_key_is_tensornet(self):
        """Results JSON uses 'tensornet' key, not 'megnet'."""
        from cathode_ml.models.train_tensornet import train_tensornet
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
        tensornet_config = {
            "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
            "training": {"learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                         "early_stopping_patience": 5},
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_tensornet.train_tensornet_for_property",
            return_value=mock_metrics,
        ):
            results = train_tensornet(records, features_config, tensornet_config, seed=42)

        for prop, prop_data in results.items():
            assert "tensornet" in prop_data
            assert "megnet" not in prop_data

    def test_no_pretrained_model_loading(self):
        """TensorNet does not call load_model (trains from scratch)."""
        from cathode_ml.models.train_tensornet import train_tensornet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            tensornet_config = {
                "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
                "training": {
                    "learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                    "early_stopping_patience": 5,
                },
                "results_dir": tmpdir,
            }

            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            mock_matgl = MagicMock()
            mock_matgl_ext = MagicMock()
            mock_matgl_ext_pymatgen = MagicMock()
            mock_matgl_ext_pymatgen.get_element_list = MagicMock(return_value=["Li", "Co", "O"])

            with patch.dict("sys.modules", {
                "matgl": mock_matgl,
                "matgl.ext": mock_matgl_ext,
                "matgl.ext.pymatgen": mock_matgl_ext_pymatgen,
            }), \
                 patch("cathode_ml.models.train_tensornet.build_tensornet_from_config",
                        return_value=mock_model) as mock_build, \
                 patch("cathode_ml.models.train_tensornet.predict_with_tensornet",
                        return_value=[0.1] * 3), \
                 patch("cathode_ml.models.train_tensornet.compute_metrics", return_value={
                     "mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": 7, "n_test": 3,
                 }), \
                 patch("cathode_ml.models.train_tensornet._run_lightning_training"):
                train_tensornet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    tensornet_config=tensornet_config,
                    seed=42,
                )

            # build_tensornet_from_config was called (not load_model)
            mock_build.assert_called_once()


class TestNoDenormalization:
    """Verify predict_with_tensornet output is used directly without rescaling."""

    def test_predictions_not_double_denormalized(self):
        """compute_metrics receives raw predict_with_tensornet values, not rescaled ones.

        If double-denormalization bug is present, compute_metrics would receive
        [1.5*3+2=6.5, 2.3*3+2=8.9, 0.8*3+2=4.4] instead of [1.5, 2.3, 0.8].
        """
        from cathode_ml.models.train_tensornet import train_tensornet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            tensornet_config = {
                "model": {"units": 64, "nblocks": 2, "cutoff": 5.0},
                "training": {
                    "learning_rate": 0.001, "batch_size": 2, "n_epochs": 2,
                    "early_stopping_patience": 5,
                },
                "results_dir": tmpdir,
            }

            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            # predict_with_tensornet returns already-denormalized values
            raw_preds = [1.5, 2.3, 0.8]

            # _run_lightning_training returns data_mean=2.0, data_std=3.0
            captured_args = {}

            def capture_compute_metrics(y_true, y_pred, n_train):
                captured_args["y_pred"] = y_pred.tolist()
                return {"mae": 0.2, "rmse": 0.25, "r2": 0.8, "n_train": n_train, "n_test": len(y_pred)}

            mock_matgl = MagicMock()
            mock_matgl_ext = MagicMock()
            mock_matgl_ext_pymatgen = MagicMock()
            mock_matgl_ext_pymatgen.get_element_list = MagicMock(return_value=["Li", "Co", "O"])

            with patch.dict("sys.modules", {
                "matgl": mock_matgl,
                "matgl.ext": mock_matgl_ext,
                "matgl.ext.pymatgen": mock_matgl_ext_pymatgen,
            }), \
                 patch("cathode_ml.models.train_tensornet.build_tensornet_from_config",
                        return_value=mock_model), \
                 patch("cathode_ml.models.train_tensornet.predict_with_tensornet", return_value=raw_preds), \
                 patch("cathode_ml.models.train_tensornet.compute_metrics", side_effect=capture_compute_metrics), \
                 patch("cathode_ml.models.train_tensornet._run_lightning_training", return_value=(2.0, 3.0)):
                train_tensornet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    tensornet_config=tensornet_config,
                    seed=42,
                )

            # y_pred should be the raw predictions, NOT rescaled
            assert captured_args["y_pred"] == pytest.approx(raw_preds), (
                f"Expected raw predictions {raw_preds}, got {captured_args['y_pred']}. "
                "Double-denormalization bug is present."
            )
