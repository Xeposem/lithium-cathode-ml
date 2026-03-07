"""Model loading and prediction utilities for the Streamlit dashboard.

Provides cached model loaders for baseline (joblib) and GNN (torch) models,
plus prediction functions for composition strings and crystal structures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Target properties and their units
PROPERTIES = ["formation_energy_per_atom", "energy_above_hull", "voltage", "capacity"]
PROPERTY_UNITS = {
    "voltage": "V",
    "capacity": "mAh/g",
    "formation_energy_per_atom": "eV/atom",
    "energy_above_hull": "eV/atom",
}
BASELINE_TYPES = ["rf", "xgb"]


def _cache_resource(func):
    """Apply st.cache_resource if streamlit is available, otherwise identity."""
    try:
        import streamlit as st
        return st.cache_resource(func)
    except ImportError:
        return func


@_cache_resource
def load_baseline_model(
    model_type: str,
    property_name: str,
    results_base: str = "data/results",
) -> Optional[Any]:
    """Load a persisted baseline model from joblib file.

    Args:
        model_type: Model type identifier ("rf" or "xgb").
        property_name: Target property name (e.g. "voltage").
        results_base: Base results directory containing baselines/ subfolder.

    Returns:
        Fitted sklearn/xgboost model with .predict(), or None if not found.
    """
    path = Path(results_base) / "baselines" / f"{model_type}_{property_name}.joblib"
    if not path.exists():
        logger.warning("Baseline model not found: %s", path)
        return None

    import joblib
    model = joblib.load(path)
    logger.info("Loaded baseline model: %s", path)
    return model


@_cache_resource
def load_gnn_model(
    model_name: str,
    property_name: str,
    results_base: str = "data/results",
    configs_dir: str = "configs",
) -> Optional[Any]:
    """Load a GNN model checkpoint and reconstruct the model.

    Args:
        model_name: Model name ("cgcnn" or "megnet").
        property_name: Target property name (e.g. "voltage").
        results_base: Base results directory containing model checkpoints.
        configs_dir: Directory with YAML config files.

    Returns:
        Model in eval mode, or None if checkpoint not found.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available -- cannot load GNN models")
        return None

    if model_name == "cgcnn":
        checkpoint_path = (
            Path(results_base) / "cgcnn" / f"cgcnn_{property_name}_best.pt"
        )
        if not checkpoint_path.exists():
            logger.warning("CGCNN checkpoint not found: %s", checkpoint_path)
            return None

        try:
            import yaml
            from cathode_ml.models.cgcnn import build_cgcnn_from_config

            with open(Path(configs_dir) / "cgcnn.yaml") as f:
                cgcnn_config = yaml.safe_load(f)
            with open(Path(configs_dir) / "features.yaml") as f:
                features_config = yaml.safe_load(f)

            model = build_cgcnn_from_config(cgcnn_config, features_config)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            logger.info("Loaded CGCNN model for %s", property_name)
            return model
        except Exception as exc:
            logger.error("Failed to load CGCNN for %s: %s", property_name, exc)
            return None

    elif model_name == "megnet":
        checkpoint_path = (
            Path(results_base) / "megnet" / f"megnet_{property_name}_best.pt"
        )
        if not checkpoint_path.exists():
            logger.warning("MEGNet checkpoint not found: %s", checkpoint_path)
            return None

        try:
            from cathode_ml.models.megnet import _import_matgl

            matgl = _import_matgl()
            if matgl is None:
                logger.warning("matgl not available -- cannot load MEGNet")
                return None

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # MEGNet state dict is extracted from matgl model
            # Reconstruct from pretrained then load fine-tuned weights
            import yaml

            with open(Path(configs_dir) / "megnet.yaml") as f:
                megnet_config = yaml.safe_load(f)

            pretrained_name = megnet_config["model"]["pretrained_model"]
            model = matgl.load_model(pretrained_name)
            model.model.load_state_dict(checkpoint["model_state_dict"])
            model.model.eval()
            logger.info("Loaded MEGNet model for %s", property_name)
            return model
        except Exception as exc:
            logger.error("Failed to load MEGNet for %s: %s", property_name, exc)
            return None

    else:
        logger.warning("Unknown GNN model: %s", model_name)
        return None


def predict_from_composition(
    formula: str,
    results_base: str = "data/results",
) -> dict:
    """Predict properties from a composition formula using baseline models.

    Featurizes the formula using Magpie descriptors, loads all available
    baseline models, and returns predictions organized by property and model.

    Args:
        formula: Chemical formula string (e.g. "LiFePO4").
        results_base: Base results directory containing baselines/ subfolder.

    Returns:
        Nested dict: {property: {model_type: predicted_value}}.
        Empty dict if no models are available.
    """
    baselines_dir = Path(results_base) / "baselines"
    if not baselines_dir.exists():
        logger.warning("Baselines directory not found: %s", baselines_dir)
        return {}

    # Discover available model files
    available = {}
    for joblib_path in baselines_dir.glob("*.joblib"):
        stem = joblib_path.stem  # e.g. "rf_voltage"
        parts = stem.split("_", 1)
        if len(parts) == 2:
            model_type, prop = parts
            available.setdefault(prop, []).append(model_type)

    if not available:
        logger.warning("No baseline model files found in %s", baselines_dir)
        return {}

    # Featurize the composition
    from cathode_ml.features.composition import featurize_compositions

    X, _ = featurize_compositions([formula])

    results: dict = {}
    for prop, model_types in available.items():
        prop_results = {}
        for mt in model_types:
            model = load_baseline_model(mt, prop, results_base=results_base)
            if model is not None:
                pred = model.predict(X)
                prop_results[mt] = float(pred[0])
        if prop_results:
            results[prop] = prop_results

    return results


def predict_from_structure(
    structure,
    results_base: str = "data/results",
    configs_dir: str = "configs",
) -> dict:
    """Predict properties from a pymatgen Structure using GNN models.

    Converts the structure to a PyG graph, loads available GNN checkpoints,
    and returns predictions organized by property and model.

    Args:
        structure: pymatgen Structure object.
        results_base: Base results directory containing GNN checkpoints.
        configs_dir: Directory with YAML config files.

    Returns:
        Nested dict: {property: {model_name: predicted_value}}.
        Empty dict if no GNN models are available.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available -- cannot run GNN predictions")
        return {}

    results: dict = {}

    for gnn_name in ["cgcnn", "megnet"]:
        for prop in PROPERTIES:
            model = load_gnn_model(
                gnn_name, prop,
                results_base=results_base,
                configs_dir=configs_dir,
            )
            if model is None:
                continue

            try:
                if gnn_name == "cgcnn":
                    from cathode_ml.features.graph import structure_to_pyg_data

                    import yaml
                    with open(Path(configs_dir) / "features.yaml") as f:
                        feat_cfg = yaml.safe_load(f)

                    graph_cfg = feat_cfg["graph"]
                    data = structure_to_pyg_data(
                        structure,
                        cutoff=graph_cfg["cutoff_radius"],
                        max_neighbors=graph_cfg["max_neighbors"],
                        gaussian_config=graph_cfg["gaussian"],
                    )
                    # Add batch dimension
                    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
                    with torch.no_grad():
                        pred = model(data).item()
                elif gnn_name == "megnet":
                    # MEGNet uses matgl's predict_structure
                    with torch.no_grad():
                        pred = float(model.predict_structure(structure))

                results.setdefault(prop, {})[gnn_name] = pred
            except Exception as exc:
                logger.error(
                    "GNN prediction failed for %s/%s: %s", gnn_name, prop, exc
                )

    return results
