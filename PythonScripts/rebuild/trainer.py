"""Training utilities for depth reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class DepthTrainer:
    """Encapsulates training, validation, and evaluation routines for depth reconstruction."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model: Neural network model
            device: Device to train on
            lr: Learning rate
            weight_decay: L2 regularization factor
            patience: Early stopping patience
            save_path: Path to save best model
        """
        self.model = model
        self.device = device
        self.criterion = nn.SmoothL1Loss()  # More robust than MSE for outliers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.history: List[Dict[str, float]] = []

        self.save_path = Path(save_path or "./rebuild_best.pt")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> List[Dict[str, float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        
        Returns:
            Training history
        """
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(
                train_loader,
                training=True,
                epoch=epoch,
                total_epochs=epochs
            )
            val_loss = self._run_epoch(
                val_loader,
                training=False,
                epoch=epoch,
                total_epochs=epochs
            )
            
            self.history.append({
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  → Best model saved (val_loss: {val_loss:.6f})")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered after epoch {epoch}.")
                    break

        # Load best model
        if self.save_path.exists():
            state_dict = torch.load(self.save_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Loaded best model from {self.save_path}")
        
        return self.history

    def _run_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Run a single training or validation epoch."""
        mode = "Train" if training else "Val"
        self.model.train(training)
        running_loss = 0.0
        num_batches = 0

        loop = tqdm(
            dataloader,
            desc=f"{mode} {epoch}/{total_epochs}",
            leave=False
        )

        for batch in loop:
            # Input shape: (B, H, W, wave_len) → (B, wave_len, H, W)
            wave_input = batch["input"].to(self.device).permute(0, 3, 1, 2)
            # Target shape: (B, H, W) → (B, 1, H, W)
            depth_target = batch["target"].to(self.device).unsqueeze(1)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                predictions = self.model(wave_input)
                loss = self.criterion(predictions, depth_target)

            if training:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=loss.item())

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model and compute metrics.
        
        Args:
            dataloader: Test data loader
        
        Returns:
            Dictionary of metrics (MAE, RMSE, etc.)
        """
        self.model.eval()
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            wave_input = batch["input"].to(self.device).permute(0, 3, 1, 2)
            depth_target = batch["target"].to(self.device)
            
            output = self.model(wave_input)
            preds.append(output.squeeze(1).cpu())
            targets.append(depth_target.cpu())

        pred_tensor = torch.cat(preds, dim=0).numpy()
        target_tensor = torch.cat(targets, dim=0).numpy()

        return self._compute_metrics(pred_tensor, target_tensor)

    @torch.no_grad()
    def predict(self, wave_data: torch.Tensor) -> torch.Tensor:
        """
        Predict depth map from wave data.
        
        Args:
            wave_data: Wave tensor of shape (H, W, wave_len) or (B, H, W, wave_len)
        
        Returns:
            Depth map of shape (H, W) or (B, H, W)
        """
        self.model.eval()
        
        # Handle different input shapes
        original_shape = wave_data.shape
        if wave_data.dim() == 3:
            # (H, W, wave_len) → (1, wave_len, H, W)
            wave_data = wave_data.unsqueeze(0).permute(0, 3, 1, 2)
        elif wave_data.dim() == 4:
            # (B, H, W, wave_len) → (B, wave_len, H, W)
            wave_data = wave_data.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {wave_data.dim()}D")
        
        wave_data = wave_data.to(self.device)
        output = self.model(wave_data)
        result = output.squeeze(1).cpu()
        
        # Return original batch structure
        if len(original_shape) == 3:
            result = result.squeeze(0)
        
        return result

    def _compute_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        errors = preds - targets
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(np.abs(errors))
        
        # Relative error (avoid division by zero)
        mask = targets > 1e-6
        if np.any(mask):
            relative_errors = np.abs(errors[mask] / targets[mask])
            mape = np.mean(relative_errors) * 100
        else:
            mape = 0.0
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Max_Error": float(max_error),
            "MAPE(%)": float(mape),
        }
