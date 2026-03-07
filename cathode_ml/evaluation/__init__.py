"""Evaluation metrics, comparison tables, and figure generation for cathode ML models.

Provides unified result loading across all model types (RF, XGBoost,
CGCNN, MEGNet), publication-quality comparison table generation,
and Nature-style figure plotting (parity, bar charts, learning curves).
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
from cathode_ml.evaluation.plots import (
    apply_nature_style,
    plot_bar_comparison,
    plot_learning_curves,
    plot_parity,
)

__all__ = [
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODELS_ORDER",
    "PROPERTIES",
    "apply_nature_style",
    "generate_all_tables",
    "generate_comparison_table",
    "load_all_results",
    "plot_bar_comparison",
    "plot_learning_curves",
    "plot_parity",
]
