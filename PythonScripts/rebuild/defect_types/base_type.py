from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from ._clamped_value import ClampedValue

class BaseDefectType(ABC):
    """Abstract base class for defect type definitions."""
    
    def __init__(self) -> None:
        super().__init__()
        self.rotation_angle = ClampedValue(0.0, 360.0)
        self.offset_x = ClampedValue(0, 0)
        self.offset_y = ClampedValue(0, 0)

    @abstractmethod
    def get_depth(self, x:Tensor, y:Tensor) -> Tensor:
        """Get depth at (x, y)."""

    @abstractmethod
    def get_envelop(self) -> tuple[float, float, float, float]:
        """Get the area as (minx, miny, maxx, maxy)."""
