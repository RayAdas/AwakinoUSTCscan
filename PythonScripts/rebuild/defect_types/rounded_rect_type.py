import math

import torch
from torch import Tensor

from ._clamped_value import ClampedValue
from .base_type import BaseDefectType


class RoundedRectDefectType(BaseDefectType):
    """Defect type representing a rounded rectangle pit."""

    def __init__(self) -> None:
        super().__init__()
        self.width = ClampedValue(0.006, 0.02)
        self.height = ClampedValue(0.006, 0.018)
        self.corner_radius = ClampedValue(0.001, 0.004)
        self.depth = ClampedValue(0.001, 0.01)

    def get_depth(self, x: Tensor, y: Tensor) -> Tensor:
        device = x.device
        dtype = x.dtype

        ox = torch.as_tensor(self.offset_x.value, device=device, dtype=dtype)
        oy = torch.as_tensor(self.offset_y.value, device=device, dtype=dtype)
        w = torch.as_tensor(self.width.value, device=device, dtype=dtype)
        h = torch.as_tensor(self.height.value, device=device, dtype=dtype)
        cr_raw = torch.as_tensor(self.corner_radius.value, device=device, dtype=dtype)
        dp = torch.as_tensor(self.depth.value, device=device, dtype=dtype)

        max_cr = torch.minimum(w, h) / 2
        cr = torch.minimum(cr_raw, max_cr)

        theta = math.radians(float(self.rotation_angle.value))
        c = torch.as_tensor(math.cos(theta), device=device, dtype=dtype)
        s = torch.as_tensor(math.sin(theta), device=device, dtype=dtype)

        dx = x - ox
        dy = y - oy
        xr = dx * c + dy * s
        yr = -dx * s + dy * c

        qx = torch.clamp(xr.abs() - (w / 2 - cr), min=0)
        qy = torch.clamp(yr.abs() - (h / 2 - cr), min=0)
        d = torch.sqrt(qx * qx + qy * qy)

        inside = d <= cr
        return dp * inside.float()

    def get_envelop(self) -> tuple[float, float, float, float]:
        x = self.offset_x.value
        y = self.offset_y.value
        half_w = self.width.value / 2
        half_h = self.height.value / 2
        r = max(half_w, half_h)
        return (x - r, y - r, x + r, y + r)
