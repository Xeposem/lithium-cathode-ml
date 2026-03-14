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


def convert_lightning_logs(log_path: str, output_csv: str) -> None:
    """Convert Lightning CSVLogger output to project-standard CSV format.

    Lightning logs train and val metrics on **separate rows** for the same
    epoch.  The train row for epoch *N* actually carries the metrics from
    epoch *N-1*'s training step (a known CSVLogger lag).  To get correct
    per-epoch figures we:

    1. Split into val-only and train-only frames.
    2. Shift train metrics forward by one epoch so they align with the
       epoch in which training actually occurred.
    3. Join on epoch.

    The result uses the project standard columns:
    epoch, train_loss, val_loss, val_mae, train_mae.

    Args:
        log_path: Path to the Lightning metrics.csv file.
        output_csv: Path to write the standardized CSV.
    """
    import pandas as pd

    df = pd.read_csv(log_path)

    # Identify column name mappings (case-insensitive matching)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "epoch":
            col_map[col] = "epoch"
        elif "train" in col_lower and "loss" in col_lower:
            col_map[col] = "train_loss"
        elif "val" in col_lower and "loss" in col_lower:
            col_map[col] = "val_loss"
        elif "train" in col_lower and "mae" in col_lower:
            col_map[col] = "train_mae"
        elif "val" in col_lower and "mae" in col_lower:
            col_map[col] = "val_mae"

    df = df.rename(columns=col_map)

    # Select only standardized columns that exist
    standard_cols = ["epoch", "train_loss", "val_loss", "val_mae", "train_mae"]
    out_cols = [c for c in standard_cols if c in df.columns]
    df = df[out_cols]

    # Split val rows (have val_loss) and train rows (have train_loss)
    val_cols = [c for c in ["val_loss", "val_mae"] if c in df.columns]
    train_cols = [c for c in ["train_loss", "train_mae"] if c in df.columns]

    val_df = df.dropna(subset=val_cols[:1])[["epoch"] + val_cols].copy()
    train_df = df.dropna(subset=train_cols[:1])[["epoch"] + train_cols].copy()

    # Lightning's train row at epoch N holds metrics from epoch N-1.
    # Shift: assign the train metrics to epoch N+1 so they align correctly.
    if not train_df.empty:
        train_df["epoch"] = train_df["epoch"] + 1

    # Join on epoch
    merged = val_df.merge(train_df, on="epoch", how="left")
    merged = merged.sort_values("epoch").reset_index(drop=True)

    merged.to_csv(output_csv, index=False)
    logger.info("Converted Lightning logs to %s (%d epochs)", output_csv, len(merged))


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
