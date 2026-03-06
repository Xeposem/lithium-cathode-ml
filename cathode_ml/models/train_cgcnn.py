"""CGCNN training orchestrator wiring model, data, and trainer.

Trains separate CGCNN models for each target property using compositional
group splitting. Produces checkpoints, per-epoch CSV metrics, and JSON
evaluation results compatible with baselines format.

Usage:
    python -m cathode_ml.models.train_cgcnn
    python -m cathode_ml.models.train_cgcnn --seed 123
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cathode_ml.data.schemas import MaterialRecord
from cathode_ml.features.graph import structure_to_graph, validate_graph
from cathode_ml.features.split import compositional_split, get_group_keys
from cathode_ml.models.cgcnn import build_cgcnn_from_config
from cathode_ml.models.trainer import GNNTrainer
from cathode_ml.models.utils import save_results

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _precompute_graphs(
    records: List[MaterialRecord],
    features_config: dict,
) -> List[Tuple[Data, MaterialRecord]]:
    """Convert all records with valid structures to PyG graphs.

    Args:
        records: List of MaterialRecord objects.
        features_config: Config dict with graph section.

    Returns:
        List of (graph_data, record) tuples for valid entries.
    """
    valid_pairs = []
    skipped = 0

    for record in records:
        if not record.structure_dict:
            skipped += 1
            continue

        try:
            structure = Structure.from_dict(record.structure_dict)
            graph = structure_to_graph(structure, features_config)
            is_valid, reason = validate_graph(graph)
            if not is_valid:
                logger.warning(
                    "Invalid graph for %s: %s", record.material_id, reason
                )
                skipped += 1
                continue
            valid_pairs.append((graph, record))
        except Exception as e:
            logger.warning(
                "Failed to convert %s to graph: %s", record.material_id, e
            )
            skipped += 1

    logger.info(
        "Pre-computed %d graphs (%d skipped)", len(valid_pairs), skipped
    )
    return valid_pairs


def train_cgcnn(
    records: List[MaterialRecord],
    features_config: dict,
    cgcnn_config: dict,
    seed: int = 42,
) -> dict:
    """Train CGCNN models for each target property.

    Orchestrates the full training pipeline: graph construction,
    compositional splitting, per-property model training, evaluation,
    and result saving.

    Args:
        records: List of MaterialRecord objects with structures and properties.
        features_config: Parsed configs/features.yaml with graph and target info.
        cgcnn_config: Parsed configs/cgcnn.yaml with model/training/results_dir.
        seed: Random seed for reproducibility.

    Returns:
        Nested dict: {property_name: {cgcnn: {mae, rmse, r2, n_train, n_test}}}.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    _set_seeds(seed)

    target_properties = features_config["target_properties"]
    training_cfg = cgcnn_config["training"]
    results_dir = Path(cgcnn_config.get("results_dir", "data/results/cgcnn"))
    results_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = features_config.get("splitting", {})
    test_size = split_cfg.get("test_size", 0.1)
    val_size = split_cfg.get("val_size", 0.1)

    # Pre-compute graphs once
    graph_pairs = _precompute_graphs(records, features_config)

    if not graph_pairs:
        logger.error("No valid graphs found. Aborting training.")
        return {}

    results = {}

    # Per-property training loop
    for prop in target_properties:
        logger.info("=== Training CGCNN for: %s ===", prop)

        # Filter to records with valid property values
        valid_data = []
        for graph, record in graph_pairs:
            value = getattr(record, prop, None)
            if value is not None:
                # Clone graph to avoid mutating original
                g = graph.clone()
                g.y = torch.tensor([value], dtype=torch.float32)
                valid_data.append((g, record))

        if len(valid_data) < 5:
            logger.warning(
                "Skipping %s: only %d valid records (need >= 5)",
                prop, len(valid_data),
            )
            continue

        graphs = [g for g, _ in valid_data]
        valid_records = [r for _, r in valid_data]

        # Compositional split
        formulas = [r.formula for r in valid_records]
        groups = get_group_keys(formulas)
        train_idx, val_idx, test_idx = compositional_split(
            n_samples=len(graphs),
            groups=groups,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

        train_data = [graphs[i] for i in train_idx]
        val_data = [graphs[i] for i in val_idx]
        test_data = [graphs[i] for i in test_idx]

        logger.info(
            "  Split: train=%d, val=%d, test=%d",
            len(train_data), len(val_data), len(test_data),
        )

        # Create DataLoaders
        batch_size = training_cfg.get("batch_size", 64)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Build model
        _set_seeds(seed)  # Reset seeds for consistent initialization
        model = build_cgcnn_from_config(cgcnn_config, features_config)

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_cfg.get("learning_rate", 0.001),
            weight_decay=training_cfg.get("weight_decay", 0.0),
        )

        sched_cfg = training_cfg.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 30),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )

        # Create trainer
        trainer = GNNTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            patience=training_cfg.get("early_stopping_patience", 50),
            checkpoint_prefix=f"cgcnn_{prop}",
            results_dir=str(results_dir),
        )

        # Train
        n_epochs = training_cfg.get("n_epochs", 400)
        csv_path = str(results_dir / f"{prop}_metrics.csv")
        history = trainer.fit(train_loader, val_loader, n_epochs, csv_path=csv_path)

        logger.info(
            "  Training done: %d epochs, early_stopped=%s, best_val_loss=%.6f",
            history["epochs_trained"],
            history["early_stopped"],
            trainer.best_val_loss,
        )

        # Evaluate on test set
        metrics = trainer.evaluate(test_loader, n_train=len(train_data))

        logger.info(
            "  Test metrics: MAE=%.4f  RMSE=%.4f  R2=%.4f",
            metrics["mae"], metrics["rmse"], metrics["r2"],
        )

        results[prop] = {"cgcnn": metrics}

    # Save combined results
    results_path = str(results_dir / "cgcnn_results.json")
    save_results(results, results_path)

    # Log summary
    logger.info("=== CGCNN Training Summary ===")
    for prop, prop_results in results.items():
        m = prop_results["cgcnn"]
        logger.info(
            "  %s: MAE=%.4f  RMSE=%.4f  R2=%.4f  (n_train=%d, n_test=%d)",
            prop, m["mae"], m["rmse"], m["r2"], m["n_train"], m["n_test"],
        )

    return results


if __name__ == "__main__":
    import argparse

    from cathode_ml.config import load_config

    parser = argparse.ArgumentParser(description="Train CGCNN models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    # Load configs
    features_config = load_config("configs/features.yaml")
    cgcnn_config = load_config("configs/cgcnn.yaml")

    # Load cached processed records
    processed_path = Path("data/processed/materials.json")
    if not processed_path.exists():
        logger.error(
            "No processed data found at %s. Run data pipeline first.", processed_path
        )
        raise SystemExit(1)

    with open(processed_path) as f:
        raw_records = json.load(f)

    records = [MaterialRecord(**r) for r in raw_records]
    logger.info("Loaded %d records from %s", len(records), processed_path)

    results = train_cgcnn(records, features_config, cgcnn_config, seed=args.seed)

    print(f"\nTraining complete. Results saved to {cgcnn_config['results_dir']}/")
    for prop, prop_results in results.items():
        m = prop_results["cgcnn"]
        print(f"  {prop}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}")
