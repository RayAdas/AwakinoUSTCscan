"""Abstract base definitions for depth reconstruction strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


class BaseDepthStrategy(ABC):
    """Common interface for depth reconstruction model strategies."""

    def __init__(
        self,
        input_channels: int,
        spatial_size: int,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Args:
            input_channels: Number of input channels (wave_len)
            spatial_size: Spatial dimension size (receptive_field_size)
            device: Device to run model on
        """
        self.input_channels = input_channels
        self.spatial_size = spatial_size
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
        save_path: Optional[Path] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
    ) -> list:
        """Run model training and return the logged history."""

    @abstractmethod
    def evaluate(
        self,
        dataloader_test: DataLoader,
    ) -> dict:
        """Evaluate the model and return aggregated metrics."""

    @abstractmethod
    def predict(
        self,
        wave_data: torch.Tensor,
    ) -> torch.Tensor:
        """Predict depth map for input wave data."""
