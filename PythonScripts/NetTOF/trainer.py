"""Training utilities shared by NetTOF strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ParameterMetrics = Dict[str, Dict[str, float]]


class Trainer:
    """Encapsulates training, validation, and evaluation routines."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        n_iter_outer: int,
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-5,
        patience: int = 10,
        save_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.n_iter_outer = n_iter_outer
        self.denormalize_fn = denormalize_fn
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.history: List[Dict[str, float]] = []

        self.save_path = Path(save_path or "./nettof_best.pt")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> List[Dict[str, float]]:
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True, epoch=epoch, total_epochs=epochs)
            val_loss = self._run_epoch(val_loader, training=False, epoch=epoch, total_epochs=epochs)
            self.history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered after epoch {epoch}.")
                    break

        if self.save_path.exists():
            state_dict = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        return self.history

    def _run_epoch(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
        epoch: int,
        total_epochs: int,
    ) -> float:
        mode = "train" if training else "eval"
        self.model.train(training)
        running_loss = 0.0
        num_batches = 0

        loop = tqdm(dataloader, desc=f"{mode.capitalize()} {epoch}/{total_epochs}", leave=False)

        for batch in loop:
            waveforms = batch["waveform"].to(self.device)
            targets = batch["target"].to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                predictions = self.model(waveforms)
                loss = self.criterion(predictions, targets)

            if training:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            loop.set_postfix(loss=loss.item())

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> ParameterMetrics:
        self.model.eval()
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for batch in dataloader:
            waveforms = batch["waveform"].to(self.device)
            target = batch["target"].to(self.device)
            output = self.model(waveforms)
            preds.append(output.cpu())
            targets.append(target.cpu())

        pred_tensor = torch.cat(preds, dim=0)
        target_tensor = torch.cat(targets, dim=0)

        denorm = denormalize_fn or self.denormalize_fn
        pred_physical = denorm(pred_tensor).cpu().numpy()
        target_physical = denorm(target_tensor).cpu().numpy()

        return self._compute_metrics(pred_physical, target_physical)

    @torch.no_grad()
    def predict(
        self,
        waveform: torch.Tensor,
        *,
        denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        self.model.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        output = self.model(waveform)
        denorm = denormalize_fn or self.denormalize_fn
        result = denorm(output.cpu())
        return result.squeeze(0)

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> ParameterMetrics:
        errors = preds - targets
        mae = np.mean(np.abs(errors), axis=(0, 1))
        rmse = np.sqrt(np.mean(errors ** 2, axis=(0, 1)))

        param_names = ["fc", "beta", "alpha", "r", "tau", "psi", "phi"]
        metrics: ParameterMetrics = {"mae": {}, "rmse": {}}
        for idx, name in enumerate(param_names):
            metrics["mae"][name] = float(mae[idx])
            metrics["rmse"][name] = float(rmse[idx])
        return metrics


__all__ = ["Trainer"]
