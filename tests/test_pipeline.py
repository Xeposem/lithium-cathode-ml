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
        assert args.models == ["rf", "xgb", "cgcnn", "megnet"]
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

class TestModuleEntryPoint:
    """Test that cathode_ml/__main__.py can be imported."""

    def test_module_entry_point(self):
        """cathode_ml/__main__.py can be imported without error."""
        mod = importlib.import_module("cathode_ml.__main__")
        assert hasattr(mod, "main")
