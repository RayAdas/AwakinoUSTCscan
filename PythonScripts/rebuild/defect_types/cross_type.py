import torch
from torch import Tensor

from .base_type import BaseDefectType
from ._clamped_value import ClampedValue

class CrossDefectType(BaseDefectType):
    """Concrete defect type representing a cross-shaped defect."""

    def __init__(self) -> None:
        super().__init__()
        self.line_width = ClampedValue(0.001, 0.01)
        self.radius = ClampedValue(0.003, 0.02)
        self.depth = ClampedValue(0.001, 0.01)

    def get_depth(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate the depth of the cross-shaped defect."""
        device = x.device
        dtype = x.dtype

        ox = torch.as_tensor(self.offset_x.value, device=device, dtype=dtype)
        oy = torch.as_tensor(self.offset_y.value, device=device, dtype=dtype)
        r = torch.as_tensor(self.radius.value, device=device, dtype=dtype)
        w = torch.as_tensor(self.line_width.value, device=device, dtype=dtype) / 2
        dp = torch.as_tensor(self.depth.value, device=device, dtype=dtype)

        dx = x - ox
        dy = y - oy

        # Horizontal rectangle
        inside_horizontal = (dx.abs() <= r) & (dy.abs() <= w)

        # Vertical rectangle
        inside_vertical = (dy.abs() <= r) & (dx.abs() <= w)

        inside_cross = inside_horizontal | inside_vertical

        depth_tensor = dp * inside_cross.float()
        return depth_tensor

    def get_envelop(self) -> tuple:
        """Get the bounding box (xmin, ymin, xmax, ymax)."""
        r = self.radius.value
        x = self.offset_x.value
        y = self.offset_y.value
        return (x - r, y - r, x + r, y + r)
