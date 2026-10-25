import numpy as np
from ..array import FixedPointArray
from ..arithmetic import mul, add, from_scaled


def fixed_fir(x, h, accumulator_format, product_format, mode="full", event=None):
    if x.real.ndim != 1 or h.real.ndim != 1:
        raise ValueError("fixed FIR requires one-dimensional inputs")
    if not x.real.size or not h.real.size:
        raise ValueError("fixed FIR inputs must not be empty")
    if mode not in {"full", "same", "valid"}:
        raise ValueError("unsupported convolution mode")
    length = x.real.size + h.real.size - 1
    complex_result = x.imag is not None or h.imag is not None
    acc = FixedPointArray(np.zeros(length, dtype=np.int64), accumulator_format,
                         np.zeros(length, dtype=np.int64) if complex_result else None)
    for index in range(h.real.size):
        coefficient = FixedPointArray(h.real[index], h.format,
                                     None if h.imag is None else h.imag[index])
        product = mul(x, coefficient, product_format, event)
        current = FixedPointArray(acc.real[index:index + x.real.size], accumulator_format,
                                 None if acc.imag is None else acc.imag[index:index + x.real.size])
        current = add(current, product, accumulator_format, event)
        acc.real[index:index + x.real.size] = current.real
        if acc.imag is not None:
            acc.imag[index:index + x.real.size] = 0 if current.imag is None else current.imag
    if mode == "full":
        return acc
    output_size = max(x.real.size, h.real.size) if mode == "same" else abs(x.real.size - h.real.size) + 1
    start = (length - output_size) // 2
    return FixedPointArray(acc.real[start:start + output_size], accumulator_format,
                           None if acc.imag is None else acc.imag[start:start + output_size])


def fixed_correlate(x, h, accumulator_format, product_format, mode="valid", event=None):
    reversed_h = from_scaled(h.real[::-1], None if h.imag is None else -h.imag[::-1],
                             h.format.fractional_length, h.format, event)
    return fixed_fir(x, reversed_h, accumulator_format, product_format, mode, event)
