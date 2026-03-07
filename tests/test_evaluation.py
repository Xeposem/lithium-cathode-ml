"""Tests for cathode_ml.evaluation.metrics module.

Validates unified result loading, comparison table generation,
and model label conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cathode_ml.evaluation.metrics import (
    MODEL_LABELS,
    MODELS_ORDER,
    PROPERTIES,
    generate_all_tables,
    generate_comparison_table,
    load_all_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BASELINE_RESULTS = {
    "formation_energy_per_atom": {
        "rf": {"mae": 0.15, "rmse": 0.20, "r2": 0.85, "n_train": 100, "n_test": 25},
        "xgb": {"mae": 0.12, "rmse": 0.18, "r2": 0.88, "n_train": 100, "n_test": 25},
    },
    "voltage": {
        "rf": {"mae": 0.30, "rmse": 0.40, "r2": 0.70, "n_train": 100, "n_test": 25},
        "xgb": {"mae": 0.25, "rmse": 0.35, "r2": 0.75, "n_train": 100, "n_test": 25},
    },
}

SAMPLE_CGCNN_RESULTS = {
    "formation_energy_per_atom": {
        "cgcnn": {"mae": 0.10, "rmse": 0.14, "r2": 0.92, "n_train": 100, "n_test": 25},
    },
    "voltage": {
        "cgcnn": {"mae": 0.20, "rmse": 0.28, "r2": 0.80, "n_train": 100, "n_test": 25},
    },
}

SAMPLE_MEGNET_RESULTS = {
    "formation_energy_per_atom": {
        "megnet": {"mae": 0.08, "rmse": 0.11, "r2": 0.95, "n_train": 100, "n_test": 25},
    },
    "voltage": {
        "megnet": {"mae": 0.18, "rmse": 0.25, "r2": 0.82, "n_train": 100, "n_test": 25},
    },
}


def _write_result_files(
    base: Path,
    baselines: dict | None = None,
    cgcnn: dict | None = None,
    megnet: dict | None = None,
) -> None:
    """Helper to write mock result JSON files."""
    if baselines is not None:
        p = base / "baselines" / "baseline_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(baselines))
    if cgcnn is not None:
        p = base / "cgcnn" / "cgcnn_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cgcnn))
    if megnet is not None:
        p = base / "megnet" / "megnet_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(megnet))


# ---------------------------------------------------------------------------
# Tests: load_all_results
# ---------------------------------------------------------------------------


class TestLoadAllResults:
    """Tests for load_all_results function."""

    def test_load_all_results_empty(self, tmp_path: Path) -> None:
        """Returns empty dict when no result files exist."""
        result = load_all_results(str(tmp_path))
        assert result == {}

    def test_load_all_results_baselines_only(self, tmp_path: Path) -> None:
        """Correctly loads baseline JSON with rf and xgb under properties."""
        _write_result_files(tmp_path, baselines=SAMPLE_BASELINE_RESULTS)
        result = load_all_results(str(tmp_path))

        assert "formation_energy_per_atom" in result
        assert "rf" in result["formation_energy_per_atom"]
        assert "xgb" in result["formation_energy_per_atom"]
        assert result["formation_energy_per_atom"]["rf"]["mae"] == 0.15
        assert result["formation_energy_per_atom"]["xgb"]["r2"] == 0.88

    def test_load_all_results_all_models(self, tmp_path: Path) -> None:
        """Loads baselines + cgcnn + megnet into unified dict."""
        _write_result_files(
            tmp_path,
            baselines=SAMPLE_BASELINE_RESULTS,
            cgcnn=SAMPLE_CGCNN_RESULTS,
            megnet=SAMPLE_MEGNET_RESULTS,
        )
        result = load_all_results(str(tmp_path))

        # All four models present for formation_energy_per_atom
        fe = result["formation_energy_per_atom"]
        assert set(fe.keys()) == {"rf", "xgb", "cgcnn", "megnet"}
        assert fe["megnet"]["mae"] == 0.08
        assert fe["cgcnn"]["r2"] == 0.92

    def test_load_all_results_missing_model(self, tmp_path: Path) -> None:
        """Gracefully handles missing model directories (no crash)."""
        _write_result_files(tmp_path, baselines=SAMPLE_BASELINE_RESULTS)
        # cgcnn and megnet directories don't exist -- should not crash
        result = load_all_results(str(tmp_path))
        assert "formation_energy_per_atom" in result
        assert "cgcnn" not in result.get("formation_energy_per_atom", {})


# ---------------------------------------------------------------------------
# Tests: generate_comparison_table
# ---------------------------------------------------------------------------


class TestGenerateComparisonTable:
    """Tests for generate_comparison_table function."""

    def _build_all_results(self) -> dict:
        """Merge sample results into unified format."""
        unified: dict = {}
        for src in [SAMPLE_BASELINE_RESULTS, SAMPLE_CGCNN_RESULTS, SAMPLE_MEGNET_RESULTS]:
            for prop, models in src.items():
                unified.setdefault(prop, {}).update(models)
        return unified

    def test_generate_comparison_table_format(self) -> None:
        """Output is valid markdown table with header, separator, data rows."""
        results = self._build_all_results()
        table = generate_comparison_table(results, "formation_energy_per_atom")

        lines = [l for l in table.strip().splitlines() if l.strip()]
        # Find the table header line (may be preceded by a heading)
        table_lines = [l for l in lines if l.startswith("|")]
        assert len(table_lines) >= 3, "Expected header, separator, and data rows"
        assert "Model" in table_lines[0]
        assert "MAE" in table_lines[0]
        assert "---" in table_lines[1]

    def test_generate_comparison_table_bold_best(self) -> None:
        """Best MAE/RMSE (lowest) and R2 (highest) are bolded."""
        results = self._build_all_results()
        table = generate_comparison_table(results, "formation_energy_per_atom")

        # MEGNet has best MAE (0.08), RMSE (0.11), R2 (0.95)
        assert "**0.0800**" in table  # best MAE
        assert "**0.1100**" in table  # best RMSE
        assert "**0.9500**" in table  # best R2

    def test_generate_comparison_table_megnet_dagger(self) -> None:
        """MEGNet label includes dagger symbol and footnote."""
        results = self._build_all_results()
        table = generate_comparison_table(results, "formation_energy_per_atom")

        assert "\u2020" in table  # dagger symbol
        assert "Fine-tuned" in table or "pretrained" in table.lower()

    def test_generate_comparison_table_missing_model(self) -> None:
        """Missing model produces no row (not crash)."""
        # Only baselines, no cgcnn/megnet
        results = {"formation_energy_per_atom": SAMPLE_BASELINE_RESULTS["formation_energy_per_atom"].copy()}
        table = generate_comparison_table(results, "formation_energy_per_atom")

        assert "RF" in table
        assert "XGBoost" in table
        assert "CGCNN" not in table


# ---------------------------------------------------------------------------
# Tests: generate_all_tables
# ---------------------------------------------------------------------------


class TestGenerateAllTables:
    """Tests for generate_all_tables function."""

    def test_generate_all_tables(self, tmp_path: Path) -> None:
        """Produces one table per property, writes markdown and JSON."""
        _write_result_files(
            tmp_path,
            baselines=SAMPLE_BASELINE_RESULTS,
            cgcnn=SAMPLE_CGCNN_RESULTS,
            megnet=SAMPLE_MEGNET_RESULTS,
        )
        generate_all_tables(str(tmp_path))

        md_path = tmp_path / "comparison" / "comparison.md"
        json_path = tmp_path / "comparison" / "comparison.json"

        assert md_path.exists(), "comparison.md not created"
        assert json_path.exists(), "comparison.json not created"

        md_content = md_path.read_text()
        # Should have tables for both properties present in sample data
        assert "formation_energy_per_atom" in md_content
        assert "voltage" in md_content

        json_data = json.loads(json_path.read_text())
        assert "formation_energy_per_atom" in json_data
        assert "voltage" in json_data


# ---------------------------------------------------------------------------
# Tests: model labels
# ---------------------------------------------------------------------------


class TestModelLabels:
    """Tests for MODEL_LABELS constant."""

    def test_model_labels(self) -> None:
        """RF='RF', XGB='XGBoost', CGCNN='CGCNN', MEGNet has dagger."""
        assert MODEL_LABELS["rf"] == "RF"
        assert MODEL_LABELS["xgb"] == "XGBoost"
        assert MODEL_LABELS["cgcnn"] == "CGCNN"
        assert "\u2020" in MODEL_LABELS["megnet"]
        assert "MEGNet" in MODEL_LABELS["megnet"]
