"""Strategy pattern implementations for depth reconstruction."""

from .base_strategy import BaseDepthStrategy
from .unet_strategy import UNetStrategy

STRATEGY_REGISTRY = {
    "unet": UNetStrategy,
}

__all__ = ["STRATEGY_REGISTRY", "BaseDepthStrategy", "UNetStrategy"]
