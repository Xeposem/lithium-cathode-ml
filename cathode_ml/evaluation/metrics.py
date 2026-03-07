"""Unified result loading and comparison table generation.

Loads JSON result artifacts from all four model types (RF, XGBoost,
CGCNN, MEGNet), normalizes them into a unified comparison structure,
and generates publication-quality markdown + JSON comparison tables.
"""

from __future__ import annotations

MODEL_COLORS: dict[str, str] = {}
MODEL_LABELS: dict[str, str] = {}
MODELS_ORDER: list[str] = []
PROPERTIES: list[str] = []


def load_all_results(results_base: str = "data/results") -> dict:
    """Stub."""
    raise NotImplementedError


def generate_comparison_table(all_results: dict, property_name: str) -> str:
    """Stub."""
    raise NotImplementedError


def generate_all_tables(results_base: str = "data/results") -> None:
    """Stub."""
    raise NotImplementedError
