"""Strategy pattern implementations for depth reconstruction."""

from .base_strategy import BaseDepthStrategy
from .unet_strategy import UNetStrategy
from .transformer_strategy import TransformerStrategy

STRATEGY_REGISTRY = {
    "unet": UNetStrategy,
    "transformer": TransformerStrategy,
}

__all__ = ["STRATEGY_REGISTRY", "BaseDepthStrategy", "UNetStrategy", "TransformerStrategy"]
