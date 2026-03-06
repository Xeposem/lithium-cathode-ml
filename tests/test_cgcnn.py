"""Tests for CGCNN model architecture and forward pass.

Tests cover CGCNNModel construction, forward pass shapes, architecture
verification, and config-driven model building.
"""

import torch
import pytest
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv

from cathode_ml.models.cgcnn import CGCNNModel, build_cgcnn_from_config


class TestCGCNNForward:
    """CGCNNModel forward pass produces correct output shapes."""

    def test_cgcnn_forward(self, sample_graph_data):
        """Single graph through DataLoader produces output shape (1,) with finite values."""
        model = CGCNNModel(node_feature_dim=100, edge_feature_dim=80, hidden_dim=128, n_conv=3)
        model.eval()

        loader = DataLoader([sample_graph_data], batch_size=1)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(batch)

        assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"
        assert torch.isfinite(out).all(), f"Output contains non-finite values: {out}"

    def test_cgcnn_batch_forward(self, sample_graph_data):
        """Batch of 3 graphs produces output shape (3,)."""
        model = CGCNNModel(node_feature_dim=100, edge_feature_dim=80, hidden_dim=128, n_conv=3)
        model.eval()

        # Create 3 copies with different targets
        graphs = []
        for i in range(3):
            g = Data(
                x=sample_graph_data.x.clone(),
                edge_index=sample_graph_data.edge_index.clone(),
                edge_attr=sample_graph_data.edge_attr.clone(),
                y=torch.tensor([float(i)]),
            )
            graphs.append(g)

        loader = DataLoader(graphs, batch_size=3)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(batch)

        assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"
        assert torch.isfinite(out).all()


class TestCGCNNArchitecture:
    """CGCNNModel has correct architecture components."""

    def test_cgcnn_architecture(self):
        """Model has correct number of CGConv layers and embedding dimensions."""
        model = CGCNNModel(
            node_feature_dim=100, edge_feature_dim=80,
            hidden_dim=128, n_conv=3, batch_norm=True,
        )
        assert len(model.convs) == 3
        assert isinstance(model.convs[0], CGConv)
        assert model.embedding.in_features == 100
        assert model.embedding.out_features == 128

    def test_cgcnn_custom_params(self):
        """Custom parameters produce matching architecture."""
        model = CGCNNModel(
            node_feature_dim=100, edge_feature_dim=80,
            hidden_dim=64, n_conv=5,
        )
        assert len(model.convs) == 5
        assert model.embedding.out_features == 64


class TestCGCNNConfigDriven:
    """build_cgcnn_from_config constructs model from YAML configs."""

    def test_cgcnn_config_driven(self, cgcnn_config, features_config):
        """Config-driven construction produces model with correct params."""
        model = build_cgcnn_from_config(cgcnn_config, features_config)

        assert isinstance(model, CGCNNModel)
        assert model.embedding.out_features == 128  # hidden_dim from cgcnn.yaml
        assert len(model.convs) == 3  # n_conv from cgcnn.yaml
        assert model.embedding.in_features == 100  # node_feature_dim from features.yaml
