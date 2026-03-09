"""Shared evaluation utilities for cathode ML models.

Provides metric computation and result saving used by both baseline
(RF, XGBoost) and deep learning (CGCNN, M3GNet, TensorNet) model pipelines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
) -> dict:
    """Compute regression metrics from true and predicted values.

    Args:
        y_true: Ground truth target values, shape (n_test,).
        y_pred: Predicted values, shape (n_test,).
        n_train: Number of training samples (recorded in metrics).

    Returns:
        Dict with keys: mae, rmse, r2, n_train, n_test.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
    }


def save_results(results: dict, path: str) -> None:
    """Save results dictionary to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        results: Results dictionary to serialize.
        path: Output file path for the JSON file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)
