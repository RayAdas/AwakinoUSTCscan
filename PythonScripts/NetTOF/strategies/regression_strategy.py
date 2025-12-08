"""Regression-based neural strategy for NetTOF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_strategy import BaseNetStrategy


class RegressionHead(nn.Module):
    """Combined 1D CNN + MLP head for parameter regression."""

    def __init__(self, n_iter_outer: int) -> None:
        super().__init__()
        self.n_iter_outer = n_iter_outer

        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, n_iter_outer * 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv_stack(x)
        raw = self.mlp(features)
        raw = raw.view(-1, self.n_iter_outer, 7)

        fc, beta, alpha, r, tau, psi, phi = torch.chunk(raw, chunks=7, dim=-1)

        fc = torch.sigmoid(F.softplus(fc))
        alpha = torch.sigmoid(F.softplus(alpha))
        psi = torch.sigmoid(F.softplus(psi))

        beta = torch.tanh(beta)
        r = torch.tanh(r)
        tau = torch.tanh(tau)
        phi = torch.tanh(phi)

        return torch.cat([fc, beta, alpha, r, tau, psi, phi], dim=-1)


class RegressionNetStrategy(BaseNetStrategy):
    """Concrete strategy implementing the regression network."""

    def __init__(
        self,
        n_iter_outer: int,
        waveform_length: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self._trainer = None
        self._denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        super().__init__(n_iter_outer=n_iter_outer, waveform_length=waveform_length, device=device)

    def build_model(self) -> nn.Module:  # type: ignore[override]
        return RegressionHead(self.n_iter_outer)

    def train(
        self,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        *,
        epochs: int,
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
        save_path: Optional[Path] = None,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-5,
        patience: int = 10,
    ) -> List[Dict[str, float]]:
        from ..trainer import Trainer

        save_path = save_path or Path("./nettof_regression.pt")

        self._denormalize_fn = denormalize_fn

        trainer = Trainer(
            model=self.model,
            device=self.device,
            n_iter_outer=self.n_iter_outer,
            denormalize_fn=denormalize_fn,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            save_path=save_path,
        )
        history = trainer.fit(dataloader_train, dataloader_val, epochs=epochs)
        self._trainer = trainer
        return history

    def evaluate(
        self,
        dataloader_test: DataLoader,
        *,
        denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, Dict[str, float]]:
        if self._trainer is None:
            raise RuntimeError("Strategy must be trained before evaluation.")
        denorm = denormalize_fn or self._denormalize_fn
        if denorm is None:
            raise RuntimeError("A denormalization function is required for evaluation.")
        return self._trainer.evaluate(dataloader_test, denormalize_fn=denorm)

    def predict(
        self,
        waveform: torch.Tensor,
        *,
        denormalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self._trainer is None:
            raise RuntimeError("Strategy must be trained before prediction.")
        denorm = denormalize_fn or self._denormalize_fn
        if denorm is None:
            raise RuntimeError("A denormalization function is required for prediction.")
        return self._trainer.predict(waveform, denormalize_fn=denorm)
