"""Abstract base definitions for NetTOF strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader


class BaseNetStrategy(ABC):
    """Common interface for model strategies."""

    def __init__(self, n_iter_outer: int, waveform_length: int, device: Optional[torch.device] = None) -> None:
        self.n_iter_outer = n_iter_outer
        self.waveform_length = waveform_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Create and return the underlying neural network."""

    @abstractmethod
    def train(
        self,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        *,
        epochs: int,
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
        save_path: Optional[Path] = None,
        lr: float,
        weight_decay: float,
        patience: int,
    ) -> list:
        """Run model training and return the logged history."""

    @abstractmethod
    def evaluate(
        self,
        dataloader_test: DataLoader,
        *,
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict:
        """Evaluate the model and return aggregated metrics."""

    @abstractmethod
    def predict(
        self,
        waveform: torch.Tensor,
        *,
        denormalize_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Predict echo parameters for a single waveform."""
