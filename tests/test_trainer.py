"""Tests for GNNTrainer class with early stopping, checkpointing, and CSV logging."""

import csv
import json

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cathode_ml.models.cgcnn import CGCNNModel
from cathode_ml.models.trainer import GNNTrainer


def _make_synthetic_graphs(n_graphs=6, node_feat_dim=10, edge_feat_dim=5):
    """Create synthetic PyG Data objects for testing."""
    graphs = []
    for i in range(n_graphs):
        x = torch.randn(4, node_feat_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, edge_feat_dim)
        y = torch.tensor([float(i) * 0.5], dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs


@pytest.fixture
def tiny_model():
    """Create a small CGCNNModel for testing."""
    return CGCNNModel(
        node_feature_dim=10, edge_feature_dim=5, hidden_dim=8, n_conv=1, n_fc=1
    )


@pytest.fixture
def trainer_setup(tiny_model, tmp_path):
    """Set up a GNNTrainer with model, optimizer, scheduler, and data loaders."""
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    graphs = _make_synthetic_graphs(n_graphs=6)
    train_loader = DataLoader(graphs[:4], batch_size=2, shuffle=False)
    val_loader = DataLoader(graphs[4:], batch_size=2, shuffle=False)
    test_loader = DataLoader(graphs[4:], batch_size=2, shuffle=False)

    trainer = GNNTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=50,
        checkpoint_prefix="test_model",
        results_dir=str(tmp_path),
    )

    return trainer, train_loader, val_loader, test_loader, tmp_path


class TestTrainerFit:
    def test_trainer_fit_returns_history(self, trainer_setup):
        """GNNTrainer.fit runs for a few epochs and returns history dict."""
        trainer, train_loader, val_loader, _, tmp_path = trainer_setup
        csv_path = str(tmp_path / "metrics.csv")

        history = trainer.fit(train_loader, val_loader, n_epochs=5, csv_path=csv_path)

        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        assert "epochs_trained" in history
        assert history["epochs_trained"] == 5

    def test_checkpoint_saving(self, trainer_setup):
        """After fit, best and final .pt files exist with expected keys."""
        trainer, train_loader, val_loader, _, tmp_path = trainer_setup

        trainer.fit(train_loader, val_loader, n_epochs=3)

        best_path = tmp_path / "test_model_best.pt"
        final_path = tmp_path / "test_model_final.pt"

        assert best_path.exists(), "Best checkpoint not found"
        assert final_path.exists(), "Final checkpoint not found"

        ckpt = torch.load(best_path, weights_only=False)
        expected_keys = {"model_state_dict", "optimizer_state_dict", "epoch", "val_loss", "config"}
        assert expected_keys.issubset(set(ckpt.keys()))

    def test_csv_logging(self, trainer_setup):
        """After fit, CSV file has correct columns and row count."""
        trainer, train_loader, val_loader, _, tmp_path = trainer_setup
        csv_path = str(tmp_path / "metrics.csv")

        trainer.fit(train_loader, val_loader, n_epochs=4, csv_path=csv_path)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 4
        expected_cols = {"epoch", "train_loss", "val_loss", "val_mae", "lr"}
        assert expected_cols == set(reader.fieldnames)

    def test_early_stopping(self, trainer_setup):
        """Training stops before n_epochs when patience is exceeded."""
        trainer, train_loader, val_loader, _, tmp_path = trainer_setup
        # Override patience to 2 for quick early stopping
        trainer.patience = 2
        # Set best_val_loss very low so validation never improves
        trainer.best_val_loss = -1e10

        history = trainer.fit(train_loader, val_loader, n_epochs=100)

        assert history["early_stopped"] is True
        assert history["epochs_trained"] < 100
        # With patience=2, should stop at epoch 3 at most
        assert history["epochs_trained"] <= 3


class TestTrainerEvaluate:
    def test_evaluate_returns_metrics(self, trainer_setup):
        """trainer.evaluate returns dict with mae, rmse, r2 keys."""
        trainer, _, _, test_loader, _ = trainer_setup

        metrics = trainer.evaluate(test_loader, n_train=10)

        assert isinstance(metrics, dict)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["n_train"] == 10
        assert metrics["n_test"] > 0

    def test_results_json_format(self, trainer_setup):
        """Trainer evaluation results are JSON-serializable and match baselines format."""
        trainer, train_loader, val_loader, test_loader, tmp_path = trainer_setup

        trainer.fit(train_loader, val_loader, n_epochs=2)
        metrics = trainer.evaluate(test_loader, n_train=4)

        # Save and reload to verify JSON compatibility
        results = {"formation_energy_per_atom": {"cgcnn": metrics}}
        results_path = str(tmp_path / "test_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f)

        with open(results_path, "r") as f:
            loaded = json.load(f)

        assert "formation_energy_per_atom" in loaded
        assert "cgcnn" in loaded["formation_energy_per_atom"]
        assert "mae" in loaded["formation_energy_per_atom"]["cgcnn"]


class TestPerPropertyTraining:
    def test_per_property_training(self, tmp_path):
        """Training two properties sequentially produces separate artifacts."""
        device = torch.device("cpu")
        graphs = _make_synthetic_graphs(n_graphs=6)

        for prop_name in ["prop_a", "prop_b"]:
            model = CGCNNModel(
                node_feature_dim=10, edge_feature_dim=5, hidden_dim=8, n_conv=1
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )

            trainer = GNNTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                patience=50,
                checkpoint_prefix=f"cgcnn_{prop_name}",
                results_dir=str(tmp_path),
            )

            train_loader = DataLoader(graphs[:4], batch_size=2, shuffle=False)
            val_loader = DataLoader(graphs[4:], batch_size=2, shuffle=False)

            csv_path = str(tmp_path / f"{prop_name}_metrics.csv")
            trainer.fit(train_loader, val_loader, n_epochs=2, csv_path=csv_path)

        # Both properties should have separate checkpoints
        assert (tmp_path / "cgcnn_prop_a_best.pt").exists()
        assert (tmp_path / "cgcnn_prop_a_final.pt").exists()
        assert (tmp_path / "cgcnn_prop_b_best.pt").exists()
        assert (tmp_path / "cgcnn_prop_b_final.pt").exists()
        assert (tmp_path / "prop_a_metrics.csv").exists()
        assert (tmp_path / "prop_b_metrics.csv").exists()
