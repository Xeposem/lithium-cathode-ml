"""MEGNet training orchestrator with matgl Lightning integration.

Trains separate MEGNet models for each target property using compositional
group splitting, fine-tuning from pretrained weights via matgl's Lightning
trainer. Produces checkpoints, per-epoch CSV metrics, and JSON evaluation
results compatible with CGCNN/baselines format.

Usage:
    python -m cathode_ml.models.train_megnet
    python -m cathode_ml.models.train_megnet --seed 123
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Structure

from cathode_ml.data.schemas import MaterialRecord
from cathode_ml.features.split import compositional_split, get_group_keys
from cathode_ml.models.megnet import (
    get_megnet_state_dict,
    load_megnet_model,
    predict_with_megnet,
)
from cathode_ml.models.utils import compute_metrics, save_results

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_lightning_logs(log_path: str, output_csv: str) -> None:
    """Convert Lightning CSVLogger output to project-standard CSV format.

    Lightning may log train and val metrics on separate rows for the same
    epoch. This function merges them by epoch and renames columns to the
    project standard: epoch, train_loss, val_loss, val_mae, train_mae.

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

    # Group by epoch and aggregate (take first non-NaN for each column)
    if "epoch" in df.columns:
        df = df.groupby("epoch", as_index=False).first()

    # Select only standardized columns that exist
    standard_cols = ["epoch", "train_loss", "val_loss", "val_mae", "train_mae"]
    out_cols = [c for c in standard_cols if c in df.columns]
    df = df[out_cols]

    df.to_csv(output_csv, index=False)
    logger.info("Converted Lightning logs to %s (%d epochs)", output_csv, len(df))


def _run_lightning_training(
    model,
    train_structures: list,
    train_targets: list,
    val_structures: list,
    val_targets: list,
    property_name: str,
    megnet_config: dict,
    seed: int,
) -> None:
    """Run Lightning training loop for a single property.

    This function contains all matgl/DGL/Lightning lazy imports.
    After training it saves .pt checkpoints and converts logs.

    Args:
        model: A matgl MEGNet model from load_megnet_model.
        train_structures: Pymatgen Structures for training.
        train_targets: Target values for training.
        val_structures: Pymatgen Structures for validation.
        val_targets: Target values for validation.
        property_name: Name of the property being trained.
        megnet_config: MEGNet configuration dict.
        seed: Random seed.
    """
    # Lazy imports for matgl/DGL/Lightning
    import matgl
    from matgl.ext.pymatgen import Structure2Graph, get_element_list
    from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph
    from matgl.utils.training import ModelLightningModule

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    training_cfg = megnet_config["training"]
    results_dir = Path(megnet_config.get("results_dir", "data/results/megnet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get element list from all structures
    all_structures = train_structures + val_structures
    element_types = get_element_list(all_structures)

    # Build Structure2Graph converter using pretrained model's cutoff
    converter = Structure2Graph(element_types=element_types, cutoff=model.cutoff)

    # Create datasets
    train_dataset = MGLDataset(
        structures=train_structures,
        labels={"Eform": train_targets},
        converter=converter,
    )
    val_dataset = MGLDataset(
        structures=val_structures,
        labels={"Eform": val_targets},
        converter=converter,
    )

    # Create dataloaders
    batch_size = training_cfg.get("batch_size", 128)
    train_loader = MGLDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_graph,
    )
    val_loader = MGLDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_graph,
    )

    # Create Lightning module
    lr = training_cfg.get("learning_rate", 0.0001)
    lit_module = ModelLightningModule(model=model, lr=lr)

    # Configure callbacks
    sched_cfg = training_cfg.get("scheduler", {})
    patience = training_cfg.get("early_stopping_patience", 100)
    max_epochs = training_cfg.get("n_epochs", 1000)

    csv_logger = CSVLogger(save_dir=str(results_dir), name=property_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(results_dir),
        filename=f"megnet_{property_name}_best",
        monitor="val_MAE",
        mode="min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_MAE",
        patience=patience,
        mode="min",
    )

    # Configure trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
    )

    # Train
    _set_seeds(seed)
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Convert Lightning .ckpt to .pt format
    best_state_dict = get_megnet_state_dict(model)
    torch.save(best_state_dict, str(results_dir / f"megnet_{property_name}_best.pt"))
    torch.save(best_state_dict, str(results_dir / f"megnet_{property_name}_final.pt"))

    # Convert Lightning logs to standard CSV format
    log_dir = csv_logger.log_dir
    lightning_csv = Path(log_dir) / "metrics.csv"
    if lightning_csv.exists():
        output_csv = str(results_dir / f"{property_name}_metrics.csv")
        convert_lightning_logs(str(lightning_csv), output_csv)

    logger.info("Training complete for %s", property_name)


def train_megnet_for_property(
    structures: list,
    targets: list,
    train_idx: list,
    val_idx: list,
    test_idx: list,
    property_name: str,
    megnet_config: dict,
    seed: int,
) -> dict:
    """Train MEGNet for a single property and evaluate on test set.

    Args:
        structures: List of pymatgen Structure objects.
        targets: List of target float values.
        train_idx: Indices for training set.
        val_idx: Indices for validation set.
        test_idx: Indices for test set.
        property_name: Name of the target property.
        megnet_config: MEGNet configuration dict.
        seed: Random seed.

    Returns:
        Dict with keys: mae, rmse, r2, n_train, n_test.
    """
    results_dir = Path(megnet_config.get("results_dir", "data/results/megnet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    train_structures = [structures[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_structures = [structures[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]
    test_structures = [structures[i] for i in test_idx]
    test_targets = [targets[i] for i in test_idx]

    # Load pretrained model
    model_name = megnet_config["model"]["pretrained_model"]
    model = load_megnet_model(model_name)

    # Run Lightning training
    _run_lightning_training(
        model=model,
        train_structures=train_structures,
        train_targets=train_targets,
        val_structures=val_structures,
        val_targets=val_targets,
        property_name=property_name,
        megnet_config=megnet_config,
        seed=seed,
    )

    # Evaluate on test set
    predictions = predict_with_megnet(model, test_structures)
    y_true = np.array(test_targets)
    y_pred = np.array(predictions)

    metrics = compute_metrics(y_true, y_pred, n_train=len(train_targets))

    logger.info(
        "  %s test: MAE=%.4f  RMSE=%.4f  R2=%.4f",
        property_name,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )

    return metrics


def train_megnet(
    records: List[MaterialRecord],
    features_config: dict,
    megnet_config: dict,
    seed: int = 42,
) -> dict:
    """Train MEGNet models for each target property.

    Orchestrates the full training pipeline: data filtering,
    compositional splitting, per-property model training, evaluation,
    and result saving. Uses identical splits as CGCNN.

    Args:
        records: List of MaterialRecord objects with structures and properties.
        features_config: Parsed configs/features.yaml with target info.
        megnet_config: Parsed configs/megnet.yaml with model/training/results_dir.
        seed: Random seed for reproducibility.

    Returns:
        Nested dict: {property_name: {megnet: {mae, rmse, r2, n_train, n_test}}}.
    """
    _set_seeds(seed)

    target_properties = features_config["target_properties"]
    results_dir = Path(megnet_config.get("results_dir", "data/results/megnet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = features_config.get("splitting", {})
    test_size = split_cfg.get("test_size", 0.1)
    val_size = split_cfg.get("val_size", 0.1)

    results = {}

    for prop in target_properties:
        logger.info("=== Training MEGNet for: %s ===", prop)

        # Filter to records with valid property values and valid structures
        valid_records = []
        for record in records:
            value = getattr(record, prop, None)
            if value is not None:
                valid_records.append(record)

        if len(valid_records) < 5:
            logger.warning(
                "Skipping %s: only %d valid records (need >= 5)",
                prop,
                len(valid_records),
            )
            continue

        # Compositional split (same as CGCNN)
        formulas = [r.formula for r in valid_records]
        groups = get_group_keys(formulas)
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=len(valid_records),
            groups=groups,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        # Convert records to pymatgen Structures and extract targets
        structures = []
        targets = []
        for record in valid_records:
            if record.structure_dict:
                structures.append(Structure.from_dict(record.structure_dict))
            else:
                # Create a dummy structure placeholder for records without structure
                structures.append(None)
            targets.append(getattr(record, prop))

        logger.info(
            "  Split: train=%d, val=%d, test=%d",
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )

        # Reset seeds before each property for reproducible initialization
        _set_seeds(seed)

        # Train and evaluate
        metrics = train_megnet_for_property(
            structures=structures,
            targets=targets,
            train_idx=train_idx.tolist(),
            val_idx=val_idx.tolist(),
            test_idx=test_idx.tolist(),
            property_name=prop,
            megnet_config=megnet_config,
            seed=seed,
        )

        results[prop] = {"megnet": metrics}

    # Save combined results
    results_path = str(results_dir / "megnet_results.json")
    save_results(results, results_path)

    # Log summary
    logger.info("=== MEGNet Training Summary ===")
    for prop, prop_results in results.items():
        m = prop_results["megnet"]
        logger.info(
            "  %s: MAE=%.4f  RMSE=%.4f  R2=%.4f  (n_train=%d, n_test=%d)",
            prop,
            m["mae"],
            m["rmse"],
            m["r2"],
            m["n_train"],
            m["n_test"],
        )

    return results


if __name__ == "__main__":
    import argparse

    from cathode_ml.config import load_config

    parser = argparse.ArgumentParser(description="Train MEGNet models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    # Load configs
    features_config = load_config("configs/features.yaml")
    megnet_config = load_config("configs/megnet.yaml")

    # Load cached processed records
    processed_path = Path("data/processed/materials.json")
    if not processed_path.exists():
        logger.error(
            "No processed data found at %s. Run data pipeline first.",
            processed_path,
        )
        raise SystemExit(1)

    with open(processed_path) as f:
        raw_records = json.load(f)

    records = [MaterialRecord(**r) for r in raw_records]
    logger.info("Loaded %d records from %s", len(records), processed_path)

    results = train_megnet(records, features_config, megnet_config, seed=args.seed)

    print(f"\nTraining complete. Results saved to {megnet_config['results_dir']}/")
    for prop, prop_results in results.items():
        m = prop_results["megnet"]
        print(
            f"  {prop}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}"
        )
