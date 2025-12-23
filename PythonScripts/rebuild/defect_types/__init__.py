from rebuild.defect_types.base_type import BaseDefectType
from .circle_type import CircleDefectType

TYPE_REGISTRY = {
    "circle": CircleDefectType,
}

__all__ = ["TYPE_REGISTRY", "BaseDefectType", "CircleDefectType"]