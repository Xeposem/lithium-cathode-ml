"""Evaluation metrics and comparison utilities for cathode ML models.

Provides unified result loading across all model types (RF, XGBoost,
CGCNN, MEGNet) and publication-quality comparison table generation.
"""

from cathode_ml.evaluation.metrics import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS_ORDER,
    PROPERTIES,
    generate_all_tables,
    generate_comparison_table,
    load_all_results,
)

__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODELS_ORDER",
    "PROPERTIES",
    "generate_all_tables",
    "generate_comparison_table",
    "load_all_results",
]
