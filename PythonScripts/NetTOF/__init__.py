"""High-level package exports for the NetTOF synthetic echo prediction framework."""

from . import dataset, echo_model, trainer
from .strategies import STRATEGY_REGISTRY

__all__ = [
    "dataset",
    "echo_model",
    "trainer",
    "STRATEGY_REGISTRY",
]
