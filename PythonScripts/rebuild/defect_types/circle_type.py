from torch import Tensor
import torch
from .base_type import BaseDefectType
from ._clamped_value import ClampedValue

class CircleDefectType(BaseDefectType):
    """Concrete defect type representing a circular defect."""

    def __init__(self) -> None:
        super().__init__()
        self.radius = ClampedValue(0.001, 0.01)
        self.depth = ClampedValue(0.001, 0.01)

    def get_depth(self, x:Tensor, y:Tensor) -> Tensor:
        """Calculate the depth of the defect based on input x and y."""
        device = x.device
        ox = torch.as_tensor(self.offset_x.value, device=device, dtype=x.dtype)
        oy = torch.as_tensor(self.offset_y.value, device=device, dtype=x.dtype)
        r = torch.as_tensor(self.radius.value, device=device, dtype=x.dtype)
        dp = torch.as_tensor(self.depth.value, device=device, dtype=x.dtype)
        squared_distance_ = (x - ox)**2 + (y - oy)**2
        inside_circle = squared_distance_ <= r**2
        depth_tensor = dp * inside_circle.float()
        return depth_tensor
        
    def get_envelop(self) -> tuple:
        """Get the area as tuple (xmin, ymin, xmax, ymax)."""
        r = self.radius.value
        x = self.offset_x.value
        y = self.offset_y.value
        return (x - r, y - r, x + r, y + r)