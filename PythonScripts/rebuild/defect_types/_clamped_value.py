from typing import Optional
from numpy import clip
from random import uniform
from warnings import warn

class ClampedValue:
    def __init__(self, min_value: float, max_value: float, value: Optional[float] = None):
        self.min_value = min_value
        self.max_value = max_value
        if value is not None:
            self.value = value
        else:
            self.rerand()

    @property
    def value(self) -> float:
        return self._value
    
    @value.setter
    def value(self, new_value: float):
        if new_value < self.min_value or new_value > self.max_value:
            warn(f"Requested value {new_value}, but clamped to {self._value} within [{self.min_value}, {self.max_value}].")

        self._value = clip(new_value, self.min_value, self.max_value)

    def rerand(self):
        self.value = uniform(self.min_value, self.max_value)