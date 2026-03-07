"""Cached data loading functions for the Streamlit dashboard.

Wraps the evaluation and cache modules with Streamlit caching
to avoid redundant I/O on page reruns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from cathode_ml.evaluation.metrics import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
    PROPERTIES,
    load_all_results,
)

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "get_all_results",
    "get_cached_records",
    "get_training_csv",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODELS_ORDER",
    "PROPERTIES",
]


@st.cache_data
def get_all_results(results_base: str = "data/results") -> dict:
    """Load all model results with Streamlit caching.

    Wraps :func:`cathode_ml.evaluation.metrics.load_all_results`
    with ``@st.cache_data`` to avoid redundant file reads.

    Args:
        results_base: Root directory containing model result subdirectories.

    Returns:
        Unified results dict: ``{property: {model: {mae, rmse, r2, ...}}}``.
    """
    return load_all_results(results_base)


@st.cache_data
def get_cached_records(data_dir: str = "data/processed") -> list[dict]:
    """Load processed material records from JSON.

    Reads ``materials.json`` from the processed data directory.
    Handles both plain JSON list format and legacy DataCache wrapper.

    Args:
        data_dir: Path to the processed data directory.

    Returns:
        List of material record dicts. Empty list if not found.
    """
    data_path = Path(data_dir) / "materials.json"
    if not data_path.exists():
        logger.warning("Processed data not found: %s", data_path)
        return []

    try:
        with open(data_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Handle legacy DataCache wrapper format
        return data.get("data", [data])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load processed data: %s", exc)
        return []


@st.cache_data
def get_training_csv(
    results_base: str,
    model: str,
    prop: str,
) -> Optional[pd.DataFrame]:
    """Load training metrics CSV for a model/property combination.

    Reads from ``{results_base}/{model}/{prop}_metrics.csv``.

    Args:
        results_base: Root results directory.
        model: Model key (e.g., ``"cgcnn"``).
        prop: Property name (e.g., ``"formation_energy_per_atom"``).

    Returns:
        DataFrame with training metrics, or None if CSV not found.
    """
    csv_path = Path(results_base) / model / f"{prop}_metrics.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Failed to read training CSV %s: %s", csv_path, exc)
        return None
