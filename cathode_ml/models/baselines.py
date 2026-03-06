"""Baseline models (Random Forest, XGBoost) for cathode property prediction.

Trains separate RF and XGBoost regressors per target property using Magpie
composition features and compositional group splitting to prevent leakage.
Results (MAE, RMSE, R2) are saved as JSON artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cathode_ml.data.schemas import MaterialRecord
from cathode_ml.features.composition import featurize_compositions
from cathode_ml.features.split import compositional_split, get_group_keys

logger = logging.getLogger(__name__)


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    config: dict,
    seed: int = 42,
) -> Any:
    """Train a baseline regression model.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training target values of shape (n_samples,).
        model_type: Either "rf" (Random Forest) or "xgb" (XGBoost).
        config: Hyperparameters dict for the chosen model type.
        seed: Random seed for reproducibility.

    Returns:
        Fitted sklearn/xgboost model instance.

    Raises:
        ValueError: If model_type is not "rf" or "xgb".
    """
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", None),
            min_samples_leaf=config.get("min_samples_leaf", 2),
            n_jobs=config.get("n_jobs", -1),
            random_state=seed,
        )
    elif model_type == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=config.get("n_estimators", 500),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            n_jobs=config.get("n_jobs", -1),
            random_state=seed,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'rf' or 'xgb'.")

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_train: int,
) -> dict:
    """Evaluate a fitted model on test data.

    Args:
        model: Fitted regression model with a predict method.
        X_test: Test feature matrix of shape (n_test, n_features).
        y_test: Test target values of shape (n_test,).
        n_train: Number of training samples (recorded in metrics).

    Returns:
        Dict with keys: mae, rmse, r2, n_train, n_test.
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
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


def run_baselines(
    records: list[MaterialRecord],
    features_config: dict,
    baselines_config: dict,
    seed: int = 42,
) -> dict:
    """Train RF and XGBoost baselines for each target property.

    Orchestrates the full pipeline: featurization, splitting, training,
    evaluation, and result saving. Each target property gets separate
    models (not multi-output).

    Args:
        records: List of MaterialRecord objects with formulas and properties.
        features_config: Config with target_properties list and splitting params.
        baselines_config: Config with random_forest, xgboost hyperparams and results_dir.
        seed: Random seed for reproducibility.

    Returns:
        Nested dict: {property_name: {model_type: {metric: value}}}.
    """
    target_properties = features_config["target_properties"]
    split_cfg = features_config.get("splitting", {})
    test_size = split_cfg.get("test_size", 0.1)
    val_size = split_cfg.get("val_size", 0.1)

    # Extract formulas and featurize once
    formulas = [r.formula for r in records]
    X_all, feature_labels = featurize_compositions(formulas)
    logger.info("Featurized %d records -> %d features", len(records), X_all.shape[1])

    # Compute group keys for compositional splitting
    groups = get_group_keys(formulas)

    results: dict = {}

    for prop in target_properties:
        logger.info("Training baselines for: %s", prop)

        # Filter to records that have this property (not None)
        valid_mask = np.array([getattr(r, prop) is not None for r in records])
        if valid_mask.sum() < 5:
            logger.warning(
                "Skipping %s: only %d valid records (need >= 5)", prop, valid_mask.sum()
            )
            continue

        X_prop = X_all[valid_mask]
        y_prop = np.array([getattr(r, prop) for r in records if getattr(r, prop) is not None])
        groups_prop = [g for g, v in zip(groups, valid_mask) if v]

        # Split with compositional grouping
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=len(y_prop),
            groups=groups_prop,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        X_train, y_train = X_prop[train_idx], y_prop[train_idx]
        X_test, y_test = X_prop[test_idx], y_prop[test_idx]

        prop_results: dict = {}

        # Train and evaluate RF
        rf_model = train_baseline(
            X_train, y_train, model_type="rf",
            config=baselines_config["random_forest"], seed=seed,
        )
        prop_results["rf"] = evaluate_model(rf_model, X_test, y_test, n_train=len(X_train))
        logger.info("  RF  MAE=%.4f  RMSE=%.4f  R2=%.4f",
                     prop_results["rf"]["mae"], prop_results["rf"]["rmse"], prop_results["rf"]["r2"])

        # Train and evaluate XGBoost
        xgb_model = train_baseline(
            X_train, y_train, model_type="xgb",
            config=baselines_config["xgboost"], seed=seed,
        )
        prop_results["xgb"] = evaluate_model(xgb_model, X_test, y_test, n_train=len(X_train))
        logger.info("  XGB MAE=%.4f  RMSE=%.4f  R2=%.4f",
                     prop_results["xgb"]["mae"], prop_results["xgb"]["rmse"], prop_results["xgb"]["r2"])

        results[prop] = prop_results

    # Save combined results
    results_dir = baselines_config.get("results_dir", "data/results")
    results_path = str(Path(results_dir) / "baseline_results.json")
    save_results(results, results_path)

    return results
