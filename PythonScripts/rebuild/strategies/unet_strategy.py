"""U-Net strategy implementation for depth reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_strategy import BaseDepthStrategy
from ..models import UNet


class UNetStrategy(BaseDepthStrategy):
    """Concrete strategy implementing U-Net for depth reconstruction."""

    def __init__(
        self,
        input_channels: int = 128,
        spatial_size: int = 41,
        base_channels: int = 64,
        bilinear: bool = True,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            input_channels: Number of wave channels (wave_len)
            spatial_size: Spatial dimension (receptive_field_size)
            base_channels: Base number of channels in U-Net
            bilinear: Use bilinear upsampling or transposed convolution
            dropout: Dropout rate for regularization
            device: Device to run on
        """
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.dropout = dropout
        self._trainer = None
        super().__init__(
            input_channels=input_channels,
            spatial_size=spatial_size,
            device=device
        )

    def build_model(self) -> nn.Module:
        """Build the U-Net model."""
        return UNet(
            in_channels=self.input_channels,
            base_channels=self.base_channels,
            bilinear=self.bilinear,
            dropout=self.dropout
        )

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
    ) -> List[Dict[str, float]]:
        """Train the U-Net model."""
        from ..trainer import DepthTrainer

        save_path = save_path or Path("./rebuild_unet.pt")

        trainer = DepthTrainer(
            model=self.model,
            device=self.device,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            save_path=save_path,
        )

        self._trainer = trainer
        return trainer.fit(dataloader_train, dataloader_val, epochs=epochs)

    def evaluate(self, dataloader_test: DataLoader) -> dict:
        """Evaluate the model on test data."""
        if self._trainer is None:
            raise RuntimeError("Model must be trained before evaluation.")
        return self._trainer.evaluate(dataloader_test)

    def predict(self, wave_data: torch.Tensor) -> torch.Tensor:
        """
        Predict depth map from wave data.
        
        Args:
            wave_data: Input wave tensor of shape (H, W, wave_len) or (B, H, W, wave_len)
        
        Returns:
            Depth map of shape (H, W) or (B, H, W)
        """
        if self._trainer is None:
            raise RuntimeError("Model must be trained before prediction.")
        return self._trainer.predict(wave_data)
