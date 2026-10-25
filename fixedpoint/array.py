"""整数码值载荷。复数分别保存 I、Q，避免 complex128 承载整数。"""
from dataclasses import dataclass
import numpy as np
from .format import FixedPointFormat


@dataclass
class FixedPointArray:
    real: np.ndarray
    format: FixedPointFormat
    imag: np.ndarray = None

    def __post_init__(self):
        self.real = np.asarray(self.real, dtype=np.int64)
        if self.imag is not None:
            self.imag = np.asarray(self.imag, dtype=np.int64)
            if self.imag.shape != self.real.shape:
                raise ValueError("I/Q shapes differ")
        for component in (self.real, self.imag):
            if component is not None and component.size:
                if component.min() < self.format.min_code or component.max() > self.format.max_code:
                    raise ValueError("integer code outside declared format")

    def dequantize(self):
        scale = 2.0 ** -self.format.fractional_length
        result = self.real.astype(np.float64) * scale
        return result if self.imag is None else result + 1j * (self.imag.astype(np.float64) * scale)

    @property
    def shape(self):
        return self.real.shape
