"""Unit tests for dashboard utility functions.

Tests data loading and chart factory functions without requiring
a running Streamlit server. Mocks st.cache_data as identity decorator.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mock streamlit before importing dashboard modules
# ---------------------------------------------------------------------------
_st_mock = MagicMock()


def _identity_decorator(func=None, **kwargs):
    """Mock st.cache_data as a no-op decorator."""
    if func is not None:
        return func
    return lambda f: f


_st_mock.cache_data = _identity_decorator
_st_mock.cache_resource = _identity_decorator
sys.modules.setdefault("streamlit", _st_mock)


from dashboard.utils.data_loader import (  # noqa: E402
    get_all_results,
    get_cached_records,
    get_training_csv,
)
from dashboard.utils.charts import (  # noqa: E402
    make_bar_comparison,
    make_training_curves,
    make_parity_plot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_results():
    """Unified results dict matching load_all_results output."""
    return {
        "formation_energy_per_atom": {
            "rf": {"mae": 0.15, "rmse": 0.20, "r2": 0.90, "n_train": 80, "n_test": 20},
            "xgb": {"mae": 0.12, "rmse": 0.18, "r2": 0.92, "n_train": 80, "n_test": 20},
            "cgcnn": {"mae": 0.10, "rmse": 0.15, "r2": 0.95, "n_train": 80, "n_test": 20},
        },
        "voltage": {
            "rf": {"mae": 0.30, "rmse": 0.40, "r2": 0.80, "n_train": 60, "n_test": 15},
        },
    }


@pytest.fixture
def sample_training_df():
    """Training CSV DataFrame."""
    return pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5],
        "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
        "val_loss": [1.2, 0.9, 0.7, 0.6, 0.55],
        "val_mae": [0.5, 0.4, 0.3, 0.25, 0.22],
        "lr": [0.01, 0.01, 0.005, 0.005, 0.001],
    })


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

class TestGetAllResults:
    def test_get_all_results(self, tmp_path):
        """get_all_results returns dict with expected structure when results exist."""
        # Create mock baseline results
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        results = {
            "formation_energy_per_atom": {
                "rf": {"mae": 0.15, "rmse": 0.20, "r2": 0.90, "n_train": 80, "n_test": 20},
            }
        }
        (baselines_dir / "baseline_results.json").write_text(json.dumps(results))

        result = get_all_results(str(tmp_path))
        assert "formation_energy_per_atom" in result
        assert "rf" in result["formation_energy_per_atom"]
        assert "mae" in result["formation_energy_per_atom"]["rf"]

    def test_get_all_results_empty(self, tmp_path):
        """Returns empty dict when no results directory content."""
        result = get_all_results(str(tmp_path / "nonexistent"))
        assert result == {}


class TestGetCachedRecords:
    def test_get_cached_records(self, tmp_path):
        """get_cached_records returns list of dicts."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create a fake cache file matching DataCache format
        payload = {
            "timestamp": "2026-01-01T00:00:00",
            "metadata": {},
            "data": [
                {"material_id": "mp-1", "formula": "LiCoO2"},
                {"material_id": "mp-2", "formula": "LiFePO4"},
            ],
        }
        (cache_dir / "cleaned_records.json").write_text(json.dumps(payload))

        records = get_cached_records(str(cache_dir))
        assert isinstance(records, list)
        assert len(records) == 2
        assert records[0]["material_id"] == "mp-1"


class TestGetTrainingCsv:
    def test_get_training_csv(self, tmp_path):
        """Loads training CSV when it exists."""
        model_dir = tmp_path / "cgcnn"
        model_dir.mkdir()
        df = pd.DataFrame({"epoch": [1, 2], "train_loss": [1.0, 0.8]})
        df.to_csv(model_dir / "formation_energy_per_atom_metrics.csv", index=False)

        result = get_training_csv(str(tmp_path), "cgcnn", "formation_energy_per_atom")
        assert result is not None
        assert len(result) == 2

    def test_get_training_csv_missing(self, tmp_path):
        """Returns None when CSV not found."""
        result = get_training_csv(str(tmp_path), "cgcnn", "voltage")
        assert result is None


# ---------------------------------------------------------------------------
# Chart tests
# ---------------------------------------------------------------------------

class TestMakeBarComparison:
    def test_make_bar_comparison(self, sample_results):
        """Returns Plotly Figure with correct number of traces matching available models."""
        import plotly.graph_objects as go

        fig = make_bar_comparison(
            sample_results,
            ["formation_energy_per_atom", "voltage"],
        )
        assert isinstance(fig, go.Figure)
        # Should have traces for models that appear in the results
        # rf, xgb, cgcnn all appear across the two properties
        model_names_in_traces = {t.name for t in fig.data}
        assert len(model_names_in_traces) >= 2  # at least rf and one other

    def test_make_bar_comparison_colors(self, sample_results):
        """Bar colors match MODEL_COLORS values."""
        from cathode_ml.evaluation.metrics import MODEL_COLORS, MODEL_LABELS

        fig = make_bar_comparison(
            sample_results,
            ["formation_energy_per_atom"],
        )
        color_values = set(MODEL_COLORS.values())
        for trace in fig.data:
            assert trace.marker.color in color_values


class TestMakeTrainingCurves:
    def test_make_training_curves(self, sample_training_df):
        """Returns Plotly Figure with train/val traces from CSV data."""
        import plotly.graph_objects as go

        fig = make_training_curves(sample_training_df, "cgcnn", "formation_energy_per_atom")
        assert isinstance(fig, go.Figure)
        # Should have at least 2 traces (train loss, val loss)
        assert len(fig.data) >= 2
        trace_names = [t.name.lower() for t in fig.data]
        assert any("train" in n for n in trace_names)
        assert any("val" in n for n in trace_names)

    def test_make_training_curves_missing(self):
        """Returns None when DataFrame is None."""
        result = make_training_curves(None, "cgcnn", "voltage")
        assert result is None


class TestMakeParityPlot:
    def test_make_parity_plot(self):
        """Returns Plotly Figure with scatter trace and diagonal line."""
        import plotly.graph_objects as go

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])

        fig = make_parity_plot(y_true, y_pred, "cgcnn", "formation_energy_per_atom")
        assert isinstance(fig, go.Figure)
        # Should have scatter trace + diagonal line shape or trace
        assert len(fig.data) >= 1
        # Check scatter trace exists
        scatter_traces = [t for t in fig.data if hasattr(t, "mode") and "markers" in str(t.mode)]
        assert len(scatter_traces) >= 1


# ---------------------------------------------------------------------------
# Data Explorer tests (Plan 03)
# ---------------------------------------------------------------------------

from dashboard.pages.data_explorer import (  # noqa: E402
    _make_histogram,
    _make_scatter_matrix,
)


@pytest.fixture
def sample_materials_df():
    """Sample materials DataFrame for explorer tests."""
    return pd.DataFrame({
        "material_id": ["mp-1", "mp-2", "mp-3", "mp-4", "mp-5"],
        "formula": ["LiCoO2", "LiFePO4", "LiMnO2", "LiNiO2", "LiFeO2"],
        "source": ["materials_project", "materials_project", "oqmd", "oqmd", "battery_data_genome"],
        "formation_energy_per_atom": [-1.5, -1.2, -0.8, -1.0, -0.5],
        "voltage": [3.9, 3.4, 4.0, 3.7, 3.2],
        "capacity": [140.0, 170.0, 148.0, 200.0, 120.0],
        "energy_above_hull": [0.0, 0.0, 0.05, 0.1, 0.2],
        "is_stable": [True, True, False, False, False],
        "space_group": [166, 62, 166, 166, 62],
    })


def test_data_explorer_load(tmp_path):
    """get_cached_records returns non-empty list when cache exists."""
    import json

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    payload = {
        "timestamp": "2026-01-01T00:00:00",
        "metadata": {},
        "data": [
            {"material_id": "mp-1", "formula": "LiCoO2", "voltage": 3.9},
            {"material_id": "mp-2", "formula": "LiFePO4", "voltage": 3.4},
        ],
    }
    (cache_dir / "cleaned_records.json").write_text(json.dumps(payload))

    records = get_cached_records(str(cache_dir))
    assert isinstance(records, list)
    assert len(records) == 2


def test_histogram_creation(sample_materials_df):
    """Histogram function returns Plotly Figure with expected axis labels."""
    import plotly.graph_objects as go

    fig = _make_histogram(sample_materials_df, "voltage", "#0072B2")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Check axis label contains property name
    assert "voltage" in fig.layout.xaxis.title.text.lower() or "Voltage" in fig.layout.xaxis.title.text


def test_scatter_matrix_creation(sample_materials_df):
    """Scatter matrix function returns Plotly Figure."""
    import plotly.graph_objects as go

    fig = _make_scatter_matrix(
        sample_materials_df,
        ["formation_energy_per_atom", "voltage", "capacity"],
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
