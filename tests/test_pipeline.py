"""Tests for the CLI pipeline orchestrator.

Tests argument parsing (build_parser) and stage orchestration (run_pipeline)
with fully mocked dependencies -- no actual data fetching, training, or
evaluation occurs during testing.
"""

from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from cathode_ml.pipeline import build_parser, run_pipeline


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for build_parser defaults and flag handling."""

    def test_build_parser_defaults(self):
        """Default args: skip_fetch=False, skip_train=False, all models, seed=42."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.skip_fetch is False
        assert args.skip_train is False
        assert args.models == ["rf", "xgb", "cgcnn", "m3gnet", "tensornet"]
        assert args.seed == 42

    def test_build_parser_skip_flags(self):
        """--skip-fetch and --skip-train set to True when provided."""
        parser = build_parser()
        args = parser.parse_args(["--skip-fetch", "--skip-train"])
        assert args.skip_fetch is True
        assert args.skip_train is True

    def test_build_parser_models_flag(self):
        """--models rf cgcnn sets models list correctly."""
        parser = build_parser()
        args = parser.parse_args(["--models", "rf", "cgcnn"])
        assert args.models == ["rf", "cgcnn"]

    def test_build_parser_seed(self):
        """--seed 123 sets seed to 123."""
        parser = build_parser()
        args = parser.parse_args(["--seed", "123"])
        assert args.seed == 123


# ---------------------------------------------------------------------------
# Pipeline execution tests
# ---------------------------------------------------------------------------

class TestRunPipeline:
    """Tests for run_pipeline stage orchestration with mocked stages."""

    def _make_args(self, **overrides):
        """Create a namespace with default pipeline args, applying overrides."""
        parser = build_parser()
        args = parser.parse_args([])
        for key, val in overrides.items():
            setattr(args, key, val)
        return args

    @patch("cathode_ml.pipeline.run_evaluate_stage")
    @patch("cathode_ml.pipeline.run_train_stage")
    @patch("cathode_ml.pipeline.run_featurize_stage")
    @patch("cathode_ml.pipeline.run_fetch_stage")
    def test_run_pipeline_skips_fetch(
        self, mock_fetch, mock_feat, mock_train, mock_eval
    ):
        """With skip_fetch=True, fetch stage function is not called."""
        args = self._make_args(skip_fetch=True)
        run_pipeline(args)
        mock_fetch.assert_not_called()
        mock_train.assert_called_once()
        mock_eval.assert_called_once()

    @patch("cathode_ml.pipeline.run_evaluate_stage")
    @patch("cathode_ml.pipeline.run_train_stage")
    @patch("cathode_ml.pipeline.run_featurize_stage")
    @patch("cathode_ml.pipeline.run_fetch_stage")
    def test_run_pipeline_skips_train(
        self, mock_fetch, mock_feat, mock_train, mock_eval
    ):
        """With skip_train=True, train stage function is not called."""
        args = self._make_args(skip_train=True)
        run_pipeline(args)
        mock_train.assert_not_called()
        mock_fetch.assert_called_once()
        mock_eval.assert_called_once()

    @patch("cathode_ml.pipeline.run_evaluate_stage")
    @patch("cathode_ml.pipeline.run_train_stage")
    @patch("cathode_ml.pipeline.run_featurize_stage")
    @patch("cathode_ml.pipeline.run_fetch_stage")
    def test_run_pipeline_calls_evaluate(
        self, mock_fetch, mock_feat, mock_train, mock_eval
    ):
        """Evaluate stage is always called (never skipped)."""
        args = self._make_args()
        run_pipeline(args)
        mock_eval.assert_called_once()

    @patch("cathode_ml.pipeline.run_evaluate_stage")
    @patch("cathode_ml.pipeline.run_train_stage")
    @patch("cathode_ml.pipeline.run_featurize_stage")
    @patch("cathode_ml.pipeline.run_fetch_stage")
    def test_run_pipeline_stage_banners(
        self, mock_fetch, mock_feat, mock_train, mock_eval, caplog
    ):
        """Each stage logs a banner with stage number and name."""
        args = self._make_args()
        with caplog.at_level(logging.INFO, logger="cathode_ml.pipeline"):
            run_pipeline(args)

        banner_lines = [r.message for r in caplog.records if "=== Stage" in r.message]
        assert len(banner_lines) == 4
        assert "Stage 1/4" in banner_lines[0]
        assert "Fetching Data" in banner_lines[0]
        assert "Stage 4/4" in banner_lines[3]
        assert "Evaluating" in banner_lines[3]

    @patch("cathode_ml.pipeline.run_evaluate_stage")
    @patch("cathode_ml.pipeline.run_train_stage")
    @patch("cathode_ml.pipeline.run_featurize_stage")
    @patch("cathode_ml.pipeline.run_fetch_stage")
    def test_run_pipeline_skip_banner(
        self, mock_fetch, mock_feat, mock_train, mock_eval, caplog
    ):
        """Skipped stages show SKIPPED in their banner."""
        args = self._make_args(skip_fetch=True)
        with caplog.at_level(logging.INFO, logger="cathode_ml.pipeline"):
            run_pipeline(args)

        banner_lines = [r.message for r in caplog.records if "=== Stage" in r.message]
        assert "SKIPPED" in banner_lines[0]


# ---------------------------------------------------------------------------
# Module entry point test
# ---------------------------------------------------------------------------

class TestRunTrainStage:
    """Tests verifying run_train_stage config and data loading wiring."""

    def _make_args(self, **overrides):
        """Create a namespace with default pipeline args, applying overrides."""
        parser = build_parser()
        args = parser.parse_args([])
        for key, val in overrides.items():
            setattr(args, key, val)
        return args

    @patch("cathode_ml.models.train_tensornet.train_tensornet")
    @patch("cathode_ml.models.train_m3gnet.train_m3gnet")
    @patch("cathode_ml.models.train_cgcnn.train_cgcnn")
    @patch("cathode_ml.models.baselines.run_baselines")
    @patch("cathode_ml.config.load_config")
    def test_loads_separate_config_files(
        self, mock_load_config, mock_baselines, mock_cgcnn, mock_m3gnet, mock_tensornet
    ):
        """run_train_stage calls load_config with features.yaml, baselines.yaml, cgcnn.yaml, m3gnet.yaml, tensornet.yaml -- not data.yaml."""
        # load_config returns a minimal dict for each call
        mock_load_config.return_value = {
            "target_properties": ["formation_energy_per_atom"],
            "splitting": {"test_size": 0.1, "val_size": 0.1},
        }

        # Patch json.load and open for data reading
        fake_records = [
            {
                "material_id": "test-1",
                "formula": "LiCoO2",
                "structure_dict": {},
                "source": "test",
            }
        ]
        with patch("builtins.open", MagicMock()), \
             patch("json.load", return_value=fake_records):
            args = self._make_args(
                models=["rf", "xgb", "cgcnn", "m3gnet", "tensornet"], config_dir="configs"
            )
            from cathode_ml.pipeline import run_train_stage

            run_train_stage(args)

        # Collect all paths load_config was called with
        called_paths = [str(call.args[0]) for call in mock_load_config.call_args_list]

        # Should have been called with separate YAML files
        assert any("features.yaml" in p for p in called_paths), (
            f"Expected features.yaml in calls, got: {called_paths}"
        )
        assert any("baselines.yaml" in p for p in called_paths), (
            f"Expected baselines.yaml in calls, got: {called_paths}"
        )
        assert any("cgcnn.yaml" in p for p in called_paths), (
            f"Expected cgcnn.yaml in calls, got: {called_paths}"
        )
        assert any("m3gnet.yaml" in p for p in called_paths), (
            f"Expected m3gnet.yaml in calls, got: {called_paths}"
        )
        assert any("tensornet.yaml" in p for p in called_paths), (
            f"Expected tensornet.yaml in calls, got: {called_paths}"
        )
        # Should NOT use data.yaml
        assert not any("data.yaml" in p for p in called_paths), (
            f"Should not call load_config with data.yaml, got: {called_paths}"
        )

    @patch("cathode_ml.models.train_tensornet.train_tensornet")
    @patch("cathode_ml.models.train_m3gnet.train_m3gnet")
    @patch("cathode_ml.models.train_cgcnn.train_cgcnn")
    @patch("cathode_ml.models.baselines.run_baselines")
    @patch("cathode_ml.config.load_config")
    def test_loads_processed_records(
        self, mock_load_config, mock_baselines, mock_cgcnn, mock_m3gnet, mock_tensornet
    ):
        """run_train_stage reads records from data/processed/materials.json (not DataCache)."""
        mock_load_config.return_value = {
            "target_properties": ["formation_energy_per_atom"],
            "splitting": {"test_size": 0.1, "val_size": 0.1},
        }

        fake_records = [
            {
                "material_id": "test-1",
                "formula": "LiCoO2",
                "structure_dict": {},
                "source": "test",
            }
        ]

        mock_file = MagicMock()
        with patch("builtins.open", mock_file), \
             patch("json.load", return_value=fake_records):
            args = self._make_args(
                models=["rf", "xgb", "cgcnn", "m3gnet", "tensornet"], config_dir="configs"
            )
            from cathode_ml.pipeline import run_train_stage

            run_train_stage(args)

        # Verify open was called with a path containing materials.json
        open_calls = mock_file.call_args_list
        open_paths = [str(call.args[0]) if call.args else "" for call in open_calls]
        assert any("materials.json" in p for p in open_paths), (
            f"Expected open() called with materials.json path, got: {open_paths}"
        )


class TestModuleEntryPoint:
    """Test that cathode_ml/__main__.py can be imported."""

    def test_module_entry_point(self):
        """cathode_ml/__main__.py can be imported without error."""
        mod = importlib.import_module("cathode_ml.__main__")
        assert hasattr(mod, "main")
