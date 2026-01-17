"""Transformer strategy implementation for depth reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_strategy import BaseDepthStrategy
from ..models.transformer import TransformerDepthReconstructor


class TransformerStrategy(BaseDepthStrategy):
    """Concrete strategy implementing Transformer Encoder for depth reconstruction."""

    def __init__(
        self,
        input_channels: int = 128,
        spatial_size: int = 41,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            input_channels: Number of wave channels (wave_len)
            spatial_size: Spatial dimension (receptive_field_size)
            embed_dim: Dimension of the embedding space
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate for regularization
            device: Device to run on
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self._trainer = None
        super().__init__(
            input_channels=input_channels,
            spatial_size=spatial_size,
            device=device
        )

    def build_model(self) -> nn.Module:
        """Build the Transformer model."""
        return TransformerDepthReconstructor(
            wave_len=self.input_channels,
            spatial_size=self.spatial_size,
            d_model=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout
        )

    def train(
        self,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        *,
        epochs: int,
        save_path: Optional[Path] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10,
    ) -> List[Dict[str, float]]:
        """Train the Transformer model."""
        from ..trainer import DepthTrainer

        save_path = save_path or Path("./rebuild_transformer.pt")

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
