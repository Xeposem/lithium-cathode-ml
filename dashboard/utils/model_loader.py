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
GNN_TYPES = ["cgcnn", "m3gnet", "tensornet"]


def get_best_models(results_base: str = "data/results") -> dict[str, str]:
    """Determine the best model for each property based on lowest test MAE.

    Scans all *_results.json files and returns a mapping of property name
    to the model key with the lowest MAE.

    Returns:
        Dict mapping property name to best model key (e.g. {"voltage": "xgb"}).
    """
    import json

    # Collect (property, model, mae) from all result files
    all_metrics: dict[str, dict[str, float]] = {}
    results_dir = Path(results_base)

    for json_path in results_dir.rglob("*_results.json"):
        try:
            data = json.load(open(json_path))
        except (json.JSONDecodeError, OSError):
            continue

        for prop, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            # Direct metrics (GNN results): {prop: {model: {mae: ...}}}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    mae = value.get("mae") or value.get("test_mae")
                    if mae is not None:
                        all_metrics.setdefault(prop, {})[key] = mae
                elif key in ("mae", "test_mae") and isinstance(value, (int, float)):
                    # Flat format: {prop: {mae: ...}} — infer model from filename
                    model = json_path.stem.replace("_results", "")
                    all_metrics.setdefault(prop, {})[model] = value
                    break

    best: dict[str, str] = {}
    for prop, model_maes in all_metrics.items():
        best[prop] = min(model_maes, key=model_maes.get)  # type: ignore[arg-type]
    return best


def _cache_resource(func):
    """Apply st.cache_resource if streamlit is available, otherwise identity."""
    try:
        import streamlit as st
        return st.cache_resource(func)
    except (ImportError, TypeError, AttributeError):
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
        model_name: Model name ("cgcnn", "m3gnet", or "tensornet").
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

    elif model_name == "m3gnet":
        checkpoint_path = (
            Path(results_base) / "m3gnet" / f"m3gnet_{property_name}_best.pt"
        )
        if not checkpoint_path.exists():
            logger.warning("M3GNet checkpoint not found: %s", checkpoint_path)
            return None

        try:
            from cathode_ml.models.m3gnet import _import_matgl

            matgl = _import_matgl()
            if matgl is None:
                logger.warning("matgl not available -- cannot load M3GNet")
                return None

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            import yaml

            with open(Path(configs_dir) / "m3gnet.yaml") as f:
                m3gnet_config = yaml.safe_load(f)

            pretrained_name = m3gnet_config["model"]["pretrained_model"]
            model = matgl.load_model(pretrained_name)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.model.load_state_dict(state_dict)
            model.model.eval()
            logger.info("Loaded M3GNet model for %s", property_name)
            return model
        except Exception as exc:
            logger.error("Failed to load M3GNet for %s: %s", property_name, exc)
            return None

    elif model_name == "tensornet":
        checkpoint_path = (
            Path(results_base) / "tensornet" / f"tensornet_{property_name}_best.pt"
        )
        if not checkpoint_path.exists():
            logger.warning("TensorNet checkpoint not found: %s", checkpoint_path)
            return None

        try:
            from cathode_ml.models.tensornet import build_tensornet_from_config

            import yaml

            with open(Path(configs_dir) / "tensornet.yaml") as f:
                tensornet_config = yaml.safe_load(f)

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "element_types" in checkpoint:
                element_types = checkpoint["element_types"]
                state_dict = checkpoint.get("model_state_dict", checkpoint)
            else:
                # Raw state_dict saved by get_tensornet_state_dict().
                # Infer the number of element types from the embedding weight
                # shape (tensor_embedding.emb.weight: [n_elements, units]).
                # Reconstruct element list as the first N elements by atomic
                # number, which matches the ordering used during training.
                state_dict = checkpoint
                emb_key = "tensor_embedding.emb.weight"
                if emb_key in state_dict:
                    n_elements = state_dict[emb_key].shape[0]
                else:
                    n_elements = 89  # safe upper bound for cathode materials
                from pymatgen.core import Element as _Element
                element_types = [e.symbol for e in _Element][:n_elements]
            model = build_tensornet_from_config(
                tensornet_config["model"], element_types
            )
            # TensorNet is a plain nn.Module in matgl 2.x (no .model wrapper).
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Loaded TensorNet model for %s", property_name)
            return model
        except Exception as exc:
            logger.error("Failed to load TensorNet for %s: %s", property_name, exc)
            return None

    else:
        logger.warning("Unknown GNN model: %s", model_name)
        return None


def _load_formula_index(data_dir: str = "data/processed") -> dict[str, list[dict]]:
    """Build an index mapping reduced formula to material records.

    Loads materials.json once and indexes by reduced formula for fast lookup.
    """
    import json

    from pymatgen.core import Composition

    data_path = Path(data_dir) / "materials.json"
    if not data_path.exists():
        return {}

    try:
        with open(data_path) as f:
            records = json.load(f)
        if not isinstance(records, list):
            records = records.get("data", [])
    except (json.JSONDecodeError, OSError):
        return {}

    index: dict[str, list[dict]] = {}
    for r in records:
        try:
            key = Composition(r["formula"]).reduced_formula
            index.setdefault(key, []).append(r)
        except Exception:
            continue
    return index


# Apply Streamlit caching if available
_load_formula_index = _cache_resource(_load_formula_index)


def lookup_structure(
    formula: str,
    data_dir: str = "data/processed",
) -> Optional[tuple[Any, dict]]:
    """Look up the most stable known crystal structure for a composition.

    Searches the local materials database for entries matching the given
    reduced formula and returns the one with the lowest energy above hull.

    Args:
        formula: Chemical formula string (e.g. "LiFePO4").
        data_dir: Path to processed data directory containing materials.json.

    Returns:
        Tuple of (pymatgen Structure, record dict) for the best match,
        or None if no matching structure is found.
    """
    from pymatgen.core import Composition, Structure

    target = Composition(formula).reduced_formula
    index = _load_formula_index(data_dir)
    matches = index.get(target)

    if not matches:
        return None

    # Pick the most stable structure (lowest energy_above_hull)
    best = min(
        matches,
        key=lambda r: r.get("energy_above_hull") if r.get("energy_above_hull") is not None else float("inf"),
    )

    try:
        structure = Structure.from_dict(best["structure_dict"])
        return structure, best
    except Exception as exc:
        logger.error("Failed to reconstruct structure for %s: %s", formula, exc)
        return None


def predict_from_composition(
    formula: str,
    results_base: str = "data/results",
    data_dir: str = "data/processed",
    configs_dir: str = "configs",
) -> tuple[dict, Optional[dict]]:
    """Predict properties from a composition formula using all available models.

    Runs baseline models on Magpie descriptors. Also looks up the most stable
    known crystal structure for the composition and runs GNN predictions if
    a structure is found.

    Args:
        formula: Chemical formula string (e.g. "LiFePO4").
        results_base: Base results directory.
        data_dir: Path to processed data directory for structure lookup.
        configs_dir: Directory with YAML config files for GNN models.

    Returns:
        Tuple of (results_dict, structure_info) where:
            results_dict: {property: {model_type: predicted_value}}.
            structure_info: dict with keys 'material_id', 'source',
                'energy_above_hull', 'space_group' for the matched structure,
                or None if no structure was found.
    """
    results: dict = {}

    # --- Baseline predictions ---
    baselines_dir = Path(results_base) / "baselines"
    if baselines_dir.exists():
        available = {}
        for joblib_path in baselines_dir.glob("*.joblib"):
            stem = joblib_path.stem
            parts = stem.split("_", 1)
            if len(parts) == 2:
                model_type, prop = parts
                available.setdefault(prop, []).append(model_type)

        if available:
            from cathode_ml.features.composition import featurize_compositions

            X, _ = featurize_compositions([formula])

            for prop, model_types in available.items():
                for mt in model_types:
                    model = load_baseline_model(mt, prop, results_base=results_base)
                    if model is not None:
                        pred = model.predict(X)
                        results.setdefault(prop, {})[mt] = float(pred[0])

    # --- GNN predictions via structure lookup ---
    structure_info = None
    match = lookup_structure(formula, data_dir=data_dir)
    if match is not None:
        structure, record = match
        structure_info = {
            "material_id": record.get("material_id"),
            "source": record.get("source"),
            "energy_above_hull": record.get("energy_above_hull"),
            "space_group": record.get("space_group"),
        }
        gnn_results = predict_from_structure(
            structure, results_base=results_base, configs_dir=configs_dir
        )
        for prop, preds in gnn_results.items():
            results.setdefault(prop, {}).update(preds)

    return results, structure_info


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

    for gnn_name in ["cgcnn", "m3gnet", "tensornet"]:
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
                    from cathode_ml.features.graph import structure_to_graph

                    import yaml
                    with open(Path(configs_dir) / "features.yaml") as f:
                        feat_cfg = yaml.safe_load(f)

                    data = structure_to_graph(structure, feat_cfg)
                    # Add batch dimension
                    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
                    with torch.no_grad():
                        pred = model(data).item()
                elif gnn_name in ("m3gnet", "tensornet"):
                    # M3GNet and TensorNet use matgl's predict_structure
                    with torch.no_grad():
                        pred = float(model.predict_structure(structure))

                results.setdefault(prop, {})[gnn_name] = pred
            except Exception as exc:
                logger.error(
                    "GNN prediction failed for %s/%s: %s", gnn_name, prop, exc
                )

    return results
