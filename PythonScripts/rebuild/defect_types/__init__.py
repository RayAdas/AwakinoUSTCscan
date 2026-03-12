from rebuild.defect_types.base_type import BaseDefectType
from rebuild.defect_types.cross_type import CrossDefectType
from .circle_type import CircleDefectType
from .bow_type import BowDefectType
from .rounded_rect_type import RoundedRectDefectType

TYPE_REGISTRY = {
    "circle": CircleDefectType,
    "cross": CrossDefectType,
    "bow": BowDefectType,
    "rounded_rect": RoundedRectDefectType,
}

__all__ = [
    "TYPE_REGISTRY",
    "BaseDefectType",
    "CircleDefectType",
    "CrossDefectType",
    "BowDefectType",
    "RoundedRectDefectType",
]