"""TensorNet training orchestrator with matgl 2.x Lightning integration.

Trains TensorNet models from scratch (no pretrained weights) using the
O(3)-equivariant tensor network architecture. Uses compositional group
splitting and per-property sequential training for fair comparison with
other models.

Usage:
    python -m cathode_ml.models.train_tensornet
    python -m cathode_ml.models.train_tensornet --seed 123
"""

from __future__ import annotations

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
from cathode_ml.models.tensornet import (
    build_tensornet_from_config,
    get_tensornet_state_dict,
    predict_with_tensornet,
)
from cathode_ml.models.utils import compute_metrics, convert_lightning_logs, save_results

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_lightning_training(
    model,
    train_structures: list,
    train_targets: list,
    val_structures: list,
    val_targets: list,
    property_name: str,
    tensornet_config: dict,
    seed: int,
) -> None:
    """Run Lightning training loop for a single property.

    This function contains all matgl/DGL/Lightning lazy imports.
    After training it saves .pt checkpoints and converts logs.

    Args:
        model: A matgl TensorNet model from build_tensornet_from_config.
        train_structures: Pymatgen Structures for training.
        train_targets: Target values for training.
        val_structures: Pymatgen Structures for validation.
        val_targets: Target values for validation.
        property_name: Name of the property being trained.
        tensornet_config: TensorNet configuration dict.
        seed: Random seed.
    """
    # Lazy imports for matgl/DGL/Lightning
    from functools import partial

    import matgl
    from matgl.ext.pymatgen import Structure2Graph, get_element_list
    from matgl.graph.data import MGLDataset, collate_fn_graph
    from matgl.utils.training import ModelLightningModule
    from torch.utils.data import DataLoader

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    training_cfg = tensornet_config["training"]
    results_dir = Path(tensornet_config.get("results_dir", "data/results/tensornet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get element list from all structures
    all_structures = train_structures + val_structures
    element_types = get_element_list(all_structures)

    # Build Structure2Graph converter using model's cutoff
    converter = Structure2Graph(element_types=element_types, cutoff=model.cutoff)

    # Create datasets -- TensorNet does NOT use threebody_cutoff or line graph
    train_dataset = MGLDataset(
        structures=train_structures,
        converter=converter,
        labels={"Eform": train_targets},
        include_line_graph=False,
    )
    val_dataset = MGLDataset(
        structures=val_structures,
        converter=converter,
        labels={"Eform": val_targets},
        include_line_graph=False,
    )

    # Create dataloaders using torch DataLoader with matgl collate_fn
    batch_size = training_cfg.get("batch_size", 128)
    collate_fn = partial(collate_fn_graph, include_line_graph=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Compute target normalization from training data so the model's
    # output head works across different property scales (eV/atom, V, mAh/g).
    import numpy as np

    _targets_arr = np.array(train_targets, dtype=np.float64)
    data_mean = float(_targets_arr.mean())
    data_std = float(_targets_arr.std()) or 1.0

    # Create Lightning module with include_line_graph=False for TensorNet
    lr = training_cfg.get("learning_rate", 0.001)
    lit_module = ModelLightningModule(
        model=model,
        include_line_graph=False,
        lr=lr,
        data_mean=data_mean,
        data_std=data_std,
    )

    # Configure callbacks
    patience = training_cfg.get("early_stopping_patience", 100)
    max_epochs = training_cfg.get("n_epochs", 1000)

    csv_logger = CSVLogger(save_dir=str(results_dir), name=property_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(results_dir),
        filename=f"tensornet_{property_name}_best",
        monitor="val_MAE",
        mode="min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_MAE",
        patience=patience,
        mode="min",
    )

    # Configure trainer — detect DGL CUDA support to avoid runtime error
    # when DGL is CPU-only but PyTorch sees a GPU
    _accel = "auto"
    try:
        import dgl

        dgl.DGLGraph().to("cuda")
    except Exception:
        _accel = "cpu"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=_accel,
        logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # Prevent Lightning from adding duplicate console handlers
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    # Train
    _set_seeds(seed)
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Convert Lightning .ckpt to .pt format
    best_state_dict = get_tensornet_state_dict(model)
    torch.save(best_state_dict, str(results_dir / f"tensornet_{property_name}_best.pt"))
    torch.save(best_state_dict, str(results_dir / f"tensornet_{property_name}_final.pt"))

    # Convert Lightning logs to standard CSV format
    log_dir = csv_logger.log_dir
    lightning_csv = Path(log_dir) / "metrics.csv"
    if lightning_csv.exists():
        output_csv = str(results_dir / f"{property_name}_metrics.csv")
        convert_lightning_logs(str(lightning_csv), output_csv)

    logger.info("Training complete for %s", property_name)
    return data_mean, data_std


def train_tensornet_for_property(
    structures: list,
    targets: list,
    train_idx: list,
    val_idx: list,
    test_idx: list,
    property_name: str,
    tensornet_config: dict,
    seed: int,
) -> dict:
    """Train TensorNet for a single property and evaluate on test set.

    Unlike M3GNet, TensorNet is trained from scratch -- no pretrained
    weights are loaded. The model is constructed from config parameters
    with element types derived from the training data.

    Args:
        structures: List of pymatgen Structure objects.
        targets: List of target float values.
        train_idx: Indices for training set.
        val_idx: Indices for validation set.
        test_idx: Indices for test set.
        property_name: Name of the target property.
        tensornet_config: TensorNet configuration dict.
        seed: Random seed.

    Returns:
        Dict with keys: mae, rmse, r2, n_train, n_test.
    """
    from matgl.ext.pymatgen import get_element_list

    results_dir = Path(tensornet_config.get("results_dir", "data/results/tensornet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    train_structures = [structures[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_structures = [structures[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]
    test_structures = [structures[i] for i in test_idx]
    test_targets = [targets[i] for i in test_idx]

    # Build model from config (trained from scratch, no pretrained weights)
    all_structures = train_structures + val_structures + test_structures
    element_types = get_element_list(all_structures)
    model = build_tensornet_from_config(tensornet_config["model"], element_types)

    # Run Lightning training
    _run_lightning_training(
        model=model,
        train_structures=train_structures,
        train_targets=train_targets,
        val_structures=val_structures,
        val_targets=val_targets,
        property_name=property_name,
        tensornet_config=tensornet_config,
        seed=seed,
    )

    # Evaluate on test set -- predict_structure() already returns
    # denormalized values; ModelLightningModule handles normalization internally
    predictions = predict_with_tensornet(model, test_structures)
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


def train_tensornet(
    records: List[MaterialRecord],
    features_config: dict,
    tensornet_config: dict,
    seed: int = 42,
) -> dict:
    """Train TensorNet models for each target property.

    Orchestrates the full training pipeline: data filtering,
    compositional splitting, per-property model training, evaluation,
    and result saving. Uses identical splits as CGCNN/baselines.

    Args:
        records: List of MaterialRecord objects with structures and properties.
        features_config: Parsed configs/features.yaml with target info.
        tensornet_config: Parsed configs/tensornet.yaml with model/training/results_dir.
        seed: Random seed for reproducibility.

    Returns:
        Nested dict: {property_name: {tensornet: {mae, rmse, r2, n_train, n_test}}}.
    """
    _set_seeds(seed)

    target_properties = features_config["target_properties"]
    results_dir = Path(tensornet_config.get("results_dir", "data/results/tensornet"))
    results_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = features_config.get("splitting", {})
    test_size = split_cfg.get("test_size", 0.1)
    val_size = split_cfg.get("val_size", 0.1)

    results = {}

    for prop in target_properties:
        logger.info("=== Training TensorNet for: %s ===", prop)

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

        # Compositional split (same as CGCNN/baselines)
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
        metrics = train_tensornet_for_property(
            structures=structures,
            targets=targets,
            train_idx=train_idx.tolist(),
            val_idx=val_idx.tolist(),
            test_idx=test_idx.tolist(),
            property_name=prop,
            tensornet_config=tensornet_config,
            seed=seed,
        )

        results[prop] = {"tensornet": metrics}

    # Save combined results
    results_path = str(results_dir / "tensornet_results.json")
    save_results(results, results_path)

    # Log summary
    logger.info("=== TensorNet Training Summary ===")
    for prop, prop_results in results.items():
        m = prop_results["tensornet"]
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

    parser = argparse.ArgumentParser(description="Train TensorNet models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    # Load configs
    features_config = load_config("configs/features.yaml")
    tensornet_config = load_config("configs/tensornet.yaml")

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

    results = train_tensornet(records, features_config, tensornet_config, seed=args.seed)

    print(f"\nTraining complete. Results saved to {tensornet_config['results_dir']}/")
    for prop, prop_results in results.items():
        m = prop_results["tensornet"]
        print(
            f"  {prop}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}"
        )
