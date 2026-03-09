"""Generic GNN trainer with early stopping, checkpointing, and CSV logging.

Provides a reusable training loop for any PyG-compatible nn.Module.
Used by CGCNN (Phase 3) and MEGNet (Phase 4) training pipelines.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cathode_ml.models.utils import compute_metrics

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Generic trainer for graph neural network models.

    Handles training loop, validation, early stopping, checkpoint saving,
    CSV metric logging, and test evaluation. Model-agnostic: works with
    any nn.Module that accepts PyG Data batches.

    Args:
        model: PyTorch model accepting PyG Data batches.
        optimizer: Configured optimizer instance.
        scheduler: Learning rate scheduler (ReduceLROnPlateau expected).
        device: Torch device for training.
        patience: Early stopping patience (epochs without improvement).
        checkpoint_prefix: Prefix for checkpoint filenames.
        results_dir: Directory to save checkpoints and metrics.
        loss_fn: Loss function (default: nn.MSELoss()).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        patience: int = 50,
        checkpoint_prefix: str = "model",
        results_dir: str = "data/results",
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.checkpoint_prefix = checkpoint_prefix
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = loss_fn or nn.MSELoss()

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader) -> float:
        """Run one training epoch.

        Args:
            train_loader: PyG DataLoader for training data.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader) -> dict:
        """Run validation and compute loss and MAE.

        Args:
            val_loader: PyG DataLoader for validation data.

        Returns:
            Dict with val_loss (MSE) and val_mae (L1).
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y)
                total_loss += loss.item() * len(batch.y)
                total_mae += torch.nn.functional.l1_loss(
                    pred, batch.y, reduction="sum"
                ).item()
                n_samples += len(batch.y)

        return {
            "val_loss": total_loss / max(n_samples, 1),
            "val_mae": total_mae / max(n_samples, 1),
        }

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs: int,
        csv_path: str | None = None,
    ) -> dict:
        """Train model with early stopping and checkpointing.

        Args:
            train_loader: PyG DataLoader for training data.
            val_loader: PyG DataLoader for validation data.
            n_epochs: Maximum number of training epochs.
            csv_path: Path for per-epoch CSV metrics (optional).

        Returns:
            History dict with train_loss, val_loss, val_mae, lr lists,
            epochs_trained count, and early_stopped flag.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "lr": [],
            "epochs_trained": 0,
            "early_stopped": False,
        }

        log_interval = max(1, n_epochs // 20)  # Log ~20 times during training

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]
            val_mae = val_metrics["val_mae"]

            # Step scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["lr"].append(current_lr)
            history["epochs_trained"] = epoch

            # CSV logging
            if csv_path:
                self._log_csv(csv_path, epoch, train_loss, val_loss, val_mae, current_lr)

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = self.results_dir / f"{self.checkpoint_prefix}_best.pt"
                self.save_checkpoint(str(best_path), epoch, val_loss)
                if epoch % log_interval == 0 or epoch == 1:
                    logger.info(
                        "  Epoch %d/%d: train=%.6f val=%.6f mae=%.4f lr=%.1e (new best)",
                        epoch, n_epochs, train_loss, val_loss, val_mae, current_lr,
                    )
            else:
                self.patience_counter += 1
                if epoch % log_interval == 0:
                    logger.info(
                        "  Epoch %d/%d: train=%.6f val=%.6f mae=%.4f lr=%.1e (patience %d/%d)",
                        epoch, n_epochs, train_loss, val_loss, val_mae, current_lr,
                        self.patience_counter, self.patience,
                    )

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                history["early_stopped"] = True
                break

        # Save final checkpoint
        final_path = self.results_dir / f"{self.checkpoint_prefix}_final.pt"
        final_val = history["val_loss"][-1] if history["val_loss"] else float("inf")
        self.save_checkpoint(str(final_path), history["epochs_trained"], final_val)

        logger.info(
            "Training complete: %d epochs, best_val_loss=%.6f, early_stopped=%s",
            history["epochs_trained"],
            self.best_val_loss,
            history["early_stopped"],
        )

        return history

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        config: dict | None = None,
    ) -> None:
        """Save model checkpoint to disk.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            val_loss: Validation loss at this checkpoint.
            config: Optional configuration dict to include.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config or {},
        }
        torch.save(checkpoint, path)

    def evaluate(self, test_loader, n_train: int) -> dict:
        """Evaluate model on test data and return metrics.

        Args:
            test_loader: PyG DataLoader for test data.
            n_train: Number of training samples (recorded in metrics).

        Returns:
            Dict with mae, rmse, r2, n_train, n_test.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        return compute_metrics(y_true, y_pred, n_train)

    def _log_csv(
        self,
        csv_path: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_mae: float,
        lr: float,
    ) -> None:
        """Append one row to the CSV metrics log.

        Creates the file with header on first call.

        Args:
            csv_path: Path to CSV file.
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            val_mae: Validation MAE for this epoch.
            lr: Current learning rate.
        """
        path = Path(csv_path)
        write_header = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "lr"])
            writer.writerow([epoch, train_loss, val_loss, val_mae, lr])
