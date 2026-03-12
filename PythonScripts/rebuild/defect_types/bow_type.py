import math

import torch
from torch import Tensor

from ._clamped_value import ClampedValue
from .base_type import BaseDefectType


class BowDefectType(BaseDefectType):
    """Defect type representing a bow-shaped cut extruded with finite width."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = ClampedValue(0.0015, 0.01)
        self.k = ClampedValue(2.5, 3.5)
        self.width = ClampedValue(0.002, 0.008)

    def get_depth(self, x: Tensor, y: Tensor) -> Tensor:
        device = x.device
        dtype = x.dtype

        ox = torch.as_tensor(self.offset_x.value, device=device, dtype=dtype)
        oy = torch.as_tensor(self.offset_y.value, device=device, dtype=dtype)
        a = torch.as_tensor(self.depth.value, device=device, dtype=dtype)
        k = torch.as_tensor(self.k.value, device=device, dtype=dtype)
        c = torch.as_tensor(self.width.value, device=device, dtype=dtype)

        b = k * a
        half_chord = b / 2

        theta = math.radians(float(self.rotation_angle.value))
        cos_t = torch.as_tensor(math.cos(theta), device=device, dtype=dtype)
        sin_t = torch.as_tensor(math.sin(theta), device=device, dtype=dtype)

        dx = x - ox
        dy = y - oy

        s = dx * cos_t + dy * sin_t
        t = -dx * sin_t + dy * cos_t

        a_safe = torch.clamp(a, min=torch.as_tensor(1e-9, device=device, dtype=dtype))
        radius = a_safe / 2 + (b * b) / (8 * a_safe)
        base = radius - a_safe

        arc_term = torch.sqrt(torch.clamp(radius * radius - s * s, min=0.0))
        bow_depth = torch.clamp(arc_term - base, min=0.0)

        inside = (s.abs() <= half_chord) & (t.abs() <= c / 2)
        return bow_depth * inside.float()

    def get_envelop(self) -> tuple[float, float, float, float]:
        x = self.offset_x.value
        y = self.offset_y.value
        half_chord = (self.k.value * self.depth.value) / 2
        half_width = self.width.value / 2
        r = max(half_chord, half_width)
        return (x - r, y - r, x + r, y + r)
