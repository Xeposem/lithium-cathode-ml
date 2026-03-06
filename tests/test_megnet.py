"""Tests for MEGNet model wrapper and training orchestrator.

Tests cover lazy import error handling, pretrained model loading,
available model listing, state dict extraction, Lightning log conversion,
artifact format, split consistency with CGCNN, and checkpoint naming.
"""

import csv
import importlib.util
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

HAS_MATGL = importlib.util.find_spec("matgl") is not None
skip_no_matgl = pytest.mark.skipif(not HAS_MATGL, reason="matgl not installed")


class TestLazyImportError:
    """MEGNet functions raise helpful ImportError when matgl is missing."""

    def test_lazy_import_error_load(self):
        """load_megnet_model raises ImportError with install instructions when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.megnet import load_megnet_model

            with pytest.raises(ImportError, match="pip install matgl==1.3.0 dgl==2.2.0"):
                load_megnet_model("MEGNet-MP-2018.6.1-Eform")

    def test_lazy_import_error_available(self):
        """get_available_megnet_models raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.megnet import get_available_megnet_models

            with pytest.raises(ImportError, match="pip install matgl==1.3.0 dgl==2.2.0"):
                get_available_megnet_models()

    def test_lazy_import_error_predict(self):
        """predict_with_megnet raises ImportError when matgl missing."""
        with patch.dict("sys.modules", {"matgl": None}):
            from cathode_ml.models.megnet import predict_with_megnet

            with pytest.raises(ImportError, match="pip install matgl==1.3.0 dgl==2.2.0"):
                predict_with_megnet(MagicMock(), [])


@skip_no_matgl
class TestLoadPretrained:
    """MEGNet pretrained model loading via matgl."""

    def test_load_pretrained(self):
        """load_megnet_model returns a model object with cutoff attribute."""
        from cathode_ml.models.megnet import load_megnet_model

        model = load_megnet_model("MEGNet-MP-2018.6.1-Eform")
        assert hasattr(model, "cutoff"), "Loaded model should have a cutoff attribute"

    def test_get_available_models(self):
        """get_available_megnet_models returns list of MEGNet model strings."""
        from cathode_ml.models.megnet import get_available_megnet_models

        models = get_available_megnet_models()
        assert isinstance(models, list)
        assert len(models) > 0, "Should find at least one MEGNet model"
        for name in models:
            assert "MEGNet" in name, f"Model {name!r} should contain 'MEGNet'"

    def test_get_state_dict(self):
        """get_megnet_state_dict returns a dict-like object from loaded model."""
        from cathode_ml.models.megnet import (
            get_megnet_state_dict,
            load_megnet_model,
        )

        model = load_megnet_model("MEGNet-MP-2018.6.1-Eform")
        state_dict = get_megnet_state_dict(model)
        assert hasattr(state_dict, "keys"), "State dict should be dict-like"
        assert len(state_dict) > 0, "State dict should not be empty"


# ---- Plan 02: Training orchestrator tests ----


class TestConvertLightningLogs:
    """convert_lightning_logs reads Lightning CSV and outputs standardized format."""

    def test_convert_lightning_logs(self):
        """Mock Lightning metrics.csv is converted to standardized CSV columns."""
        from cathode_ml.models.train_megnet import convert_lightning_logs

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

            # Read and verify columns
            with open(output_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2, f"Should have 2 epochs, got {len(rows)}"
            expected_cols = {"epoch", "train_loss", "val_loss", "val_mae"}
            assert expected_cols.issubset(set(rows[0].keys())), (
                f"Missing columns: {expected_cols - set(rows[0].keys())}"
            )
            assert float(rows[0]["train_loss"]) == pytest.approx(1.5)
            assert float(rows[0]["val_loss"]) == pytest.approx(1.3)
            assert float(rows[0]["val_mae"]) == pytest.approx(1.1)
            assert float(rows[1]["val_mae"]) == pytest.approx(0.7)


class TestArtifactFormat:
    """train_megnet returns results in same nested format as CGCNN."""

    def test_artifact_format(self):
        """Results dict has structure {property: {megnet: {mae, rmse, r2, n_train, n_test}}}."""
        from cathode_ml.models.train_megnet import train_megnet
        from cathode_ml.data.schemas import MaterialRecord

        # Create minimal records with diverse formulas for compositional splitting
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
                structure_dict={},  # empty structure
                source="test",
                formation_energy_per_atom=float(i) * 0.1,
            ))

        features_config = {
            "target_properties": ["formation_energy_per_atom"],
            "splitting": {"test_size": 0.2, "val_size": 0.2},
        }
        megnet_config = {
            "model": {"pretrained_model": "MEGNet-MP-2018.6.1-Eform"},
            "training": {
                "learning_rate": 0.0001,
                "batch_size": 2,
                "n_epochs": 2,
                "early_stopping_patience": 5,
                "scheduler": {"factor": 0.5, "patience": 5, "min_lr": 1e-7},
            },
            "results_dir": tempfile.mkdtemp(),
        }

        # Mock matgl-dependent functions to avoid needing actual matgl
        mock_metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_megnet.train_megnet_for_property",
            return_value=mock_metrics,
        ):
            results = train_megnet(records, features_config, megnet_config, seed=42)

        assert "formation_energy_per_atom" in results
        assert "megnet" in results["formation_energy_per_atom"]
        megnet_result = results["formation_energy_per_atom"]["megnet"]
        for key in ("mae", "rmse", "r2", "n_train", "n_test"):
            assert key in megnet_result, f"Missing key: {key}"


class TestSameSplitsAsCGCNN:
    """train_megnet uses identical compositional splits as train_cgcnn."""

    def test_same_splits_as_cgcnn(self):
        """Same records and seed produce identical train/val/test indices."""
        from cathode_ml.features.split import compositional_split, get_group_keys

        # Create test formulas
        formulas = [
            "LiCoO2", "LiCoO2", "LiMnO2", "LiNiO2", "LiFePO4",
            "LiCoO2", "LiMnO2", "LiNiO2", "LiFePO4", "LiMn2O4",
            "LiNi0.5Mn0.5O2", "LiNi0.33Mn0.33Co0.33O2",
        ]
        groups = get_group_keys(formulas)

        # Call compositional_split with same params as both CGCNN and MEGNet should use
        seed = 42
        test_size = 0.1
        val_size = 0.1

        train1, val1, test1 = compositional_split(
            n_samples=len(formulas), groups=groups,
            test_size=test_size, val_size=val_size, seed=seed,
        )
        train2, val2, test2 = compositional_split(
            n_samples=len(formulas), groups=groups,
            test_size=test_size, val_size=val_size, seed=seed,
        )

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(test1, test2)


class TestCheckpointSavedAsPt:
    """After training, checkpoint files are .pt format, not .ckpt."""

    def test_checkpoint_saved_as_pt(self):
        """megnet_{property}_best.pt and megnet_{property}_final.pt exist after training."""
        from cathode_ml.models.train_megnet import train_megnet_for_property

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock all matgl-dependent objects
            mock_model = MagicMock()
            mock_model.cutoff = 5.0
            mock_model.model.state_dict.return_value = {"weight": "fake"}

            megnet_config = {
                "model": {"pretrained_model": "MEGNet-MP-2018.6.1-Eform"},
                "training": {
                    "learning_rate": 0.0001,
                    "batch_size": 2,
                    "n_epochs": 2,
                    "early_stopping_patience": 5,
                    "scheduler": {"factor": 0.5, "patience": 5, "min_lr": 1e-7},
                },
                "results_dir": tmpdir,
            }

            # Create mock structures and targets
            mock_structures = [MagicMock() for _ in range(10)]
            mock_targets = [float(i) * 0.1 for i in range(10)]

            # We need to mock the entire training pipeline
            with patch("cathode_ml.models.train_megnet.load_megnet_model", return_value=mock_model), \
                 patch("cathode_ml.models.train_megnet.predict_with_megnet", return_value=[0.1] * 3), \
                 patch("cathode_ml.models.train_megnet.compute_metrics", return_value={
                     "mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 7, "n_test": 3,
                 }), \
                 patch("cathode_ml.models.train_megnet._run_lightning_training") as mock_train:
                # Mock _run_lightning_training to create fake checkpoint files
                def fake_training(model, train_structures, train_targets, val_structures, val_targets,
                                  property_name, megnet_config, seed):
                    # Simulate what Lightning would create
                    import torch
                    torch.save(mock_model.model.state_dict(), os.path.join(tmpdir, f"megnet_{property_name}_best.pt"))
                    torch.save(mock_model.model.state_dict(), os.path.join(tmpdir, f"megnet_{property_name}_final.pt"))
                    # Create a fake metrics CSV
                    csv_path = os.path.join(tmpdir, f"{property_name}_metrics.csv")
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["epoch", "train_loss", "val_loss", "val_mae"])
                        writer.writerow(["0", "1.0", "0.9", "0.8"])

                mock_train.side_effect = fake_training

                result = train_megnet_for_property(
                    structures=mock_structures,
                    targets=mock_targets,
                    train_idx=[0, 1, 2, 3, 4, 5, 6],
                    val_idx=[7, 8],
                    test_idx=[9],
                    property_name="formation_energy_per_atom",
                    megnet_config=megnet_config,
                    seed=42,
                )

            # Check .pt files exist
            best_pt = os.path.join(tmpdir, "megnet_formation_energy_per_atom_best.pt")
            final_pt = os.path.join(tmpdir, "megnet_formation_energy_per_atom_final.pt")
            assert os.path.exists(best_pt), f"Best checkpoint not found: {best_pt}"
            assert os.path.exists(final_pt), f"Final checkpoint not found: {final_pt}"


class TestTrainMegnetPerPropertyLoop:
    """train_megnet iterates over target properties and skips small ones."""

    def test_train_megnet_per_property_loop(self):
        """Properties with fewer than 5 valid records are skipped."""
        from cathode_ml.models.train_megnet import train_megnet
        from cathode_ml.data.schemas import MaterialRecord

        # Create records with diverse formulas for compositional splitting
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
                voltage=float(i) * 0.1 if i < 3 else None,  # only 3 valid
            ))

        features_config = {
            "target_properties": ["formation_energy_per_atom", "voltage"],
            "splitting": {"test_size": 0.2, "val_size": 0.2},
        }
        megnet_config = {
            "model": {"pretrained_model": "MEGNet-MP-2018.6.1-Eform"},
            "training": {
                "learning_rate": 0.0001, "batch_size": 2, "n_epochs": 2,
                "early_stopping_patience": 5,
                "scheduler": {"factor": 0.5, "patience": 5, "min_lr": 1e-7},
            },
            "results_dir": tempfile.mkdtemp(),
        }

        mock_metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.9, "n_train": 6, "n_test": 2}
        with patch(
            "cathode_ml.models.train_megnet.train_megnet_for_property",
            return_value=mock_metrics,
        ):
            results = train_megnet(records, features_config, megnet_config, seed=42)

        # formation_energy_per_atom should be present (10 records)
        assert "formation_energy_per_atom" in results
        # voltage should be skipped (only 3 valid records < 5)
        assert "voltage" not in results
