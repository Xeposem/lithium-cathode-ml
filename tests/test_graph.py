"""Tests for crystal structure to graph conversion."""

import pytest
import torch
from torch_geometric.data import Data

from cathode_ml.features.graph import (
    gaussian_expansion,
    structure_to_graph,
    validate_graph,
)


# --- Gaussian expansion tests ---


def test_gaussian_expansion_shape():
    """gaussian_expansion returns correct output shape."""
    distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gaussian_expansion(distances, dmin=0.0, dmax=5.0, num_gaussians=80)
    assert result.shape == (5, 80)


def test_gaussian_expansion_values():
    """Peak values occur at correct centers."""
    distances = torch.tensor([2.5])
    result = gaussian_expansion(distances, dmin=0.0, dmax=5.0, num_gaussians=80)
    # Center index 40 corresponds to distance 2.5 in linspace(0, 5, 80)
    # The peak should be at or very near index 40
    peak_idx = result[0].argmax().item()
    # Allow +/- 1 index tolerance due to discrete centers
    assert abs(peak_idx - 40) <= 1, f"Peak at index {peak_idx}, expected near 40"


# --- structure_to_graph tests ---


def test_structure_to_graph_returns_data(sample_pymatgen_structure, features_config):
    """structure_to_graph returns a PyG Data object with required attributes."""
    data = structure_to_graph(sample_pymatgen_structure, features_config)
    assert isinstance(data, Data)
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")


def test_node_features_one_hot(sample_pymatgen_structure, features_config):
    """Node features are one-hot encoded atomic numbers with shape (num_atoms, 100)."""
    data = structure_to_graph(sample_pymatgen_structure, features_config)
    num_atoms = len(sample_pymatgen_structure)
    assert data.x.shape == (num_atoms, 100)
    # Each row should have exactly one 1.0
    for i in range(num_atoms):
        assert data.x[i].sum().item() == pytest.approx(1.0)
        assert data.x[i].max().item() == pytest.approx(1.0)


def test_edge_attr_gaussian(sample_pymatgen_structure, features_config):
    """Edge attributes have shape (num_edges, num_gaussians)."""
    data = structure_to_graph(sample_pymatgen_structure, features_config)
    num_gaussians = features_config["graph"]["gaussian"]["num_gaussians"]
    num_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape == (num_edges, num_gaussians)


def test_no_disconnected_graphs(sample_pymatgen_structure, features_config):
    """Every node has at least one edge."""
    data = structure_to_graph(sample_pymatgen_structure, features_config)
    is_valid, reason = validate_graph(data)
    assert is_valid, f"Graph validation failed: {reason}"


def test_configurable_cutoff(sample_pymatgen_structure, features_config):
    """Changing cutoff_radius changes the number of edges."""
    import copy

    config_small = copy.deepcopy(features_config)
    config_large = copy.deepcopy(features_config)
    config_small["graph"]["cutoff_radius"] = 3.0
    config_large["graph"]["cutoff_radius"] = 10.0

    data_small = structure_to_graph(sample_pymatgen_structure, config_small)
    data_large = structure_to_graph(sample_pymatgen_structure, config_large)

    assert data_large.edge_index.shape[1] >= data_small.edge_index.shape[1], (
        f"Larger cutoff should produce >= edges: "
        f"small={data_small.edge_index.shape[1]}, large={data_large.edge_index.shape[1]}"
    )


# --- validate_graph tests ---


def test_validate_graph_catches_empty():
    """validate_graph returns (False, reason) for a Data object with no edges."""
    data = Data(
        x=torch.ones(3, 100),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 80),
    )
    is_valid, reason = validate_graph(data)
    assert not is_valid
    assert "edge" in reason.lower() or "empty" in reason.lower()
