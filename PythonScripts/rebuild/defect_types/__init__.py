from rebuild.defect_types.base_type import BaseDefectType
from rebuild.defect_types.cross_type import CrossDefectType
from .circle_type import CircleDefectType

TYPE_REGISTRY = {
    "circle": CircleDefectType,
    "cross": CrossDefectType,
}

__all__ = ["TYPE_REGISTRY", "BaseDefectType", "CircleDefectType", "CrossDefectType"]