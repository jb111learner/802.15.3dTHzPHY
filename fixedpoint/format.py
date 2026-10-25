"""显式二进制定点格式；有符号 Qm.n 的 m 包含符号位。"""
from dataclasses import dataclass
from typing import Mapping


ROUNDINGS = {"nearest_even", "nearest_away", "toward_zero", "floor", "ceil"}


@dataclass(frozen=True)
class FixedPointFormat:
    word_length: int = 16
    fractional_length: int = 12
    signed: bool = True
    rounding: str = "nearest_even"
    saturation: bool = True
    overflow_policy: str = "error"

    def __post_init__(self):
        if type(self.word_length) is not int or not 1 <= self.word_length <= 62:
            raise ValueError("word_length must be an integer in [1, 62]")
        if (type(self.fractional_length) is not int
                or not 0 <= self.fractional_length <= self.word_length - int(self.signed)):
            raise ValueError("fractional_length does not fit the word length")
        if self.rounding not in ROUNDINGS:
            raise ValueError("unsupported rounding: " + str(self.rounding))
        if self.overflow_policy not in {"error", "wrap"}:
            raise ValueError("overflow_policy must be error or wrap")
        if type(self.signed) is not bool or type(self.saturation) is not bool:
            raise TypeError("signed and saturation must be bool")

    @property
    def min_code(self):
        return -(1 << (self.word_length - 1)) if self.signed else 0

    @property
    def max_code(self):
        return (1 << (self.word_length - int(self.signed))) - 1

    @property
    def minimum(self):
        return self.min_code * 2.0 ** -self.fractional_length

    @property
    def maximum(self):
        return self.max_code * 2.0 ** -self.fractional_length

    @property
    def q_format(self):
        return ("Q" if self.signed else "UQ") + str(self.word_length - self.fractional_length) + "." + str(self.fractional_length)

    @property
    def lsb(self):
        return 2.0 ** -self.fractional_length

    def to_dict(self):
        return {"word_length": self.word_length, "fractional_length": self.fractional_length,
                "signed": self.signed, "rounding": self.rounding,
                "saturation": self.saturation, "overflow_policy": self.overflow_policy}

    @classmethod
    def from_dict(cls, value=None, default=None):
        if isinstance(value, cls):
            return value
        default = default or cls()
        if value is None:
            return default
        if not isinstance(value, Mapping):
            raise TypeError("fixed-point format must be a mapping")
        fields = default.to_dict()
        unknown = set(value) - set(fields)
        if unknown:
            raise ValueError("unknown format fields: " + ", ".join(sorted(unknown)))
        fields.update(value)
        return cls(**fields)
