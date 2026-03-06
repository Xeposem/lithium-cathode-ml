"""CGCNN model for cathode property prediction.

Implements Crystal Graph Convolutional Neural Network (Xie & Grossman, 2018)
using PyG CGConv layers with configurable architecture. Takes PyG Data batches
with one-hot atomic number node features and Gaussian-expanded edge features,
and predicts a scalar property per graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool


class CGCNNModel(nn.Module):
    """Crystal Graph Convolutional Neural Network.

    Architecture:
        1. Linear embedding: node_feature_dim -> hidden_dim
        2. N CGConv layers (channels=hidden_dim, dim=edge_feature_dim)
        3. Global mean pooling (graph-level readout)
        4. FC head with Softplus activation -> scalar output

    Args:
        node_feature_dim: Dimensionality of input node features (e.g. 100 for one-hot atomic number).
        edge_feature_dim: Dimensionality of edge features (e.g. 80 for Gaussian expansion).
        hidden_dim: Hidden channel size for conv layers and FC head.
        n_conv: Number of CGConv message-passing layers.
        n_fc: Number of fully-connected hidden layers before output.
        batch_norm: Whether to use batch normalization in CGConv layers.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        n_conv: int = 3,
        n_fc: int = 1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        # Node feature embedding
        self.embedding = nn.Linear(node_feature_dim, hidden_dim)

        # CGConv message-passing layers
        self.convs = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_feature_dim, batch_norm=batch_norm)
            for _ in range(n_conv)
        ])

        # FC head
        fc_layers = []
        for _ in range(n_fc):
            fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            fc_layers.append(nn.Softplus())
        self.fc = nn.Sequential(*fc_layers) if fc_layers else nn.Identity()

        # Output projection
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data: "torch_geometric.data.Data") -> torch.Tensor:
        """Forward pass on a PyG Data batch.

        Args:
            data: PyG Data or Batch object with x, edge_index, edge_attr, batch.

        Returns:
            Predictions tensor of shape (batch_size,).
        """
        x = self.embedding(data.x)

        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)

        # Graph-level readout
        x = global_mean_pool(x, data.batch)

        # FC head + output
        x = self.fc(x)
        x = self.out(x)

        return x.squeeze(-1)


def build_cgcnn_from_config(
    cgcnn_config: dict,
    features_config: dict,
) -> CGCNNModel:
    """Construct a CGCNNModel from YAML configuration dicts.

    Args:
        cgcnn_config: Parsed configs/cgcnn.yaml with 'model' section.
        features_config: Parsed configs/features.yaml with 'graph' section.

    Returns:
        Configured CGCNNModel instance (not yet trained).
    """
    model_cfg = cgcnn_config["model"]
    graph_cfg = features_config["graph"]

    return CGCNNModel(
        node_feature_dim=graph_cfg["node_feature_dim"],
        edge_feature_dim=graph_cfg["gaussian"]["num_gaussians"],
        hidden_dim=model_cfg["hidden_dim"],
        n_conv=model_cfg["n_conv"],
        n_fc=model_cfg.get("n_fc", 1),
        batch_norm=model_cfg.get("batch_norm", True),
    )
