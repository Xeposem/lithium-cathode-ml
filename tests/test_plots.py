"""Tests for cathode_ml.evaluation.plots module.

Verifies Nature-style parity plots, bar chart comparison,
and learning curves generation with headless matplotlib backend.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_all_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture()
def mock_predictions() -> dict:
    """Create mock model predictions for parity plot testing."""
    rng = np.random.default_rng(42)
    preds = {}
    for model_key in ["rf", "xgb", "cgcnn", "megnet"]:
        y_true = rng.standard_normal(50)
        y_pred = y_true + rng.standard_normal(50) * 0.1
        preds[model_key] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "r2": 0.95,
        }
    return preds


@pytest.fixture()
def mock_csv_dir(tmp_path: Path) -> Path:
    """Create mock CSV files for learning curve testing."""
    for model in ["cgcnn", "megnet"]:
        model_dir = tmp_path / model
        model_dir.mkdir()
        for prop in ["formation_energy_per_atom", "voltage"]:
            csv_path = model_dir / f"{prop}_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if model == "cgcnn":
                    writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "lr"])
                    for epoch in range(1, 11):
                        writer.writerow([epoch, 1.0 / epoch, 1.1 / epoch, 0.5 / epoch, 0.001])
                else:
                    # MEGNet: no lr column
                    writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "train_mae"])
                    for epoch in range(1, 11):
                        writer.writerow([epoch, 1.0 / epoch, 1.1 / epoch, 0.5 / epoch, 0.6 / epoch])
    return tmp_path


class TestApplyNatureStyle:
    """Test Nature journal style application."""

    def test_apply_nature_style(self):
        """After calling apply_nature_style(), rcParams match NATURE_STYLE."""
        from cathode_ml.evaluation.plots import NATURE_STYLE, apply_nature_style

        apply_nature_style()

        assert plt.rcParams["axes.grid"] is False
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["savefig.dpi"] == 300


class TestPlotParity:
    """Test parity plot generation."""

    def test_plot_parity_creates_file(self, tmp_path: Path, mock_predictions: dict):
        """plot_parity creates a PNG file at the specified path."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_parity

        apply_nature_style()
        out = tmp_path / "parity.png"
        plot_parity("formation_energy_per_atom", mock_predictions, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_parity_2x2_layout(self, tmp_path: Path, mock_predictions: dict):
        """Generated figure has 2x2 subplot grid."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_parity

        apply_nature_style()
        out = tmp_path / "parity.png"
        # Capture the figure by monkeypatching savefig
        fig_holder = {}

        original_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            fig_holder["fig"] = self
            return original_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_parity("formation_energy_per_atom", mock_predictions, str(out))
        finally:
            plt.Figure.savefig = original_savefig

        fig = fig_holder["fig"]
        axes = fig.get_axes()
        assert len(axes) == 4

    def test_plot_parity_annotations(self, tmp_path: Path, mock_predictions: dict):
        """Each panel has R-squared and MAE text annotations."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_parity

        apply_nature_style()
        out = tmp_path / "parity.png"

        fig_holder = {}
        original_savefig = plt.Figure.savefig

        def capture_savefig(self, *args, **kwargs):
            fig_holder["fig"] = self
            return original_savefig(self, *args, **kwargs)

        plt.Figure.savefig = capture_savefig
        try:
            plot_parity("formation_energy_per_atom", mock_predictions, str(out))
        finally:
            plt.Figure.savefig = original_savefig

        fig = fig_holder["fig"]
        for ax in fig.get_axes():
            texts = [t.get_text() for t in ax.texts]
            text_combined = " ".join(texts)
            # Each visible panel should have R^2 and MAE annotations
            if ax.get_visible():
                assert "R" in text_combined or "MAE" in text_combined

    def test_plot_parity_handles_missing_model(self, tmp_path: Path):
        """Missing model results in invisible panel, not crash."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_parity

        apply_nature_style()
        out = tmp_path / "parity.png"
        partial = {
            "rf": {
                "y_true": np.array([1.0, 2.0]),
                "y_pred": np.array([1.1, 2.1]),
                "mae": 0.1,
                "r2": 0.99,
            }
        }
        plot_parity("voltage", partial, str(out))
        assert out.exists()


class TestPlotBarComparison:
    """Test bar chart comparison generation."""

    def test_plot_bar_comparison_creates_file(self, tmp_path: Path):
        """plot_bar_comparison creates a PNG file."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_bar_comparison

        apply_nature_style()
        out = tmp_path / "bar.png"
        results = {
            "formation_energy_per_atom": {
                "rf": {"mae": 0.1, "rmse": 0.15, "r2": 0.95},
                "xgb": {"mae": 0.08, "rmse": 0.12, "r2": 0.97},
            },
            "voltage": {
                "rf": {"mae": 0.2, "rmse": 0.25, "r2": 0.90},
            },
        }
        plot_bar_comparison(results, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_bar_comparison_handles_missing(self, tmp_path: Path):
        """Missing model for a property does not crash."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_bar_comparison

        apply_nature_style()
        out = tmp_path / "bar.png"
        results = {
            "formation_energy_per_atom": {
                "rf": {"mae": 0.1, "rmse": 0.15, "r2": 0.95},
            },
        }
        plot_bar_comparison(results, str(out))
        assert out.exists()


class TestPlotLearningCurves:
    """Test learning curves plot generation."""

    def test_plot_learning_curves_creates_file(self, tmp_path: Path, mock_csv_dir: Path):
        """plot_learning_curves creates a PNG from mock CSV data."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_learning_curves

        apply_nature_style()
        out = tmp_path / "curves.png"
        plot_learning_curves(str(mock_csv_dir), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_learning_curves_handles_missing_csv(self, tmp_path: Path):
        """Missing CSV files result in 'No data' panels, not crashes."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_learning_curves

        apply_nature_style()
        empty_dir = tmp_path / "empty_results"
        empty_dir.mkdir()
        out = tmp_path / "curves.png"
        plot_learning_curves(str(empty_dir), str(out))
        assert out.exists()

    def test_plot_learning_curves_handles_missing_lr(self, tmp_path: Path, mock_csv_dir: Path):
        """MEGNet CSV without lr column plots successfully."""
        from cathode_ml.evaluation.plots import apply_nature_style, plot_learning_curves

        apply_nature_style()
        # mock_csv_dir already has MEGNet CSVs without lr column
        out = tmp_path / "curves.png"
        plot_learning_curves(str(mock_csv_dir), str(out))
        assert out.exists()
