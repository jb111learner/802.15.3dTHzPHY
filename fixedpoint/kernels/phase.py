"""整数 CORDIC 相位/旋转内核；浮点仅用于初始化常数表。"""
from __future__ import annotations

import math
import numpy as np

from ..array import FixedPointArray
from ..format import FixedPointFormat
from ..arithmetic import quantize
from ..arithmetic import requantize


def _shift(value, bits, rounding):
    if not bits:
        return value
    divisor = 1 << bits
    if rounding == "floor":
        return value // divisor
    if rounding == "ceil":
        return -((-value) // divisor)
    sign = -1 if value < 0 else 1
    q, r = divmod(abs(value), divisor)
    if rounding in {"nearest_even", "nearest_away"}:
        twice = r * 2
        if twice > divisor or (twice == divisor and (rounding == "nearest_away" or q % 2)):
            q += 1
    return sign * q


def _limit(value, fmt, event):
    overflow = not fmt.min_code <= value <= fmt.max_code
    if event:
        event({"overflow_count": int(overflow), "count": 1})
    if fmt.min_code <= value <= fmt.max_code:
        return value
    if fmt.saturation:
        return min(max(value, fmt.min_code), fmt.max_code)
    if fmt.overflow_policy == "error":
        raise OverflowError("CORDIC result exceeds declared format")
    value %= 1 << fmt.word_length
    if fmt.signed and value >= 1 << (fmt.word_length - 1):
        value -= 1 << fmt.word_length
    return value


def cordic_angle(value, input_format, phase_format=None, iterations=None, event=None, state_format=None):
    phase_format = phase_format or FixedPointFormat(24, 20)
    if not phase_format.signed or phase_format.maximum < math.pi:
        raise ValueError("phase format must represent the complete [-pi, pi] interval")
    data = quantize(value, input_format, event=event)
    state_format = state_format or FixedPointFormat(
        min(62, input_format.word_length + 3), input_format.fractional_length,
        rounding=input_format.rounding, saturation=input_format.saturation,
        overflow_policy=input_format.overflow_policy)
    if not state_format.signed:
        raise ValueError("CORDIC state format must be signed")
    data = requantize(data, state_format, event=event)
    real = data.real.reshape(-1)
    imag = np.zeros_like(real) if data.imag is None else data.imag.reshape(-1)
    iterations = int(iterations or min(32, max(input_format.fractional_length, phase_format.fractional_length) + 2))
    if not 1 <= iterations <= 62:
        raise ValueError("CORDIC iterations must be in [1, 62]")
    atan = quantize(np.arctan(2.0 ** -np.arange(iterations)), phase_format).real
    pi = int(quantize(math.pi, phase_format).real)
    result = np.empty(real.shape, dtype=np.int64)
    for index, (r, j) in enumerate(zip(real, imag)):
        x, y, z = int(r), int(j), 0
        if x == 0 and y == 0:
            result[index] = 0
            continue
        if x < 0:
            z = pi if y >= 0 else -pi
            x, y = _limit(-x, state_format, event), _limit(-y, state_format, event)
        for step, angle in enumerate(atan):
            if y == 0:
                break
            direction = 1 if y > 0 else -1
            x, y = (_limit(x + direction * _shift(y, step, state_format.rounding), state_format, event),
                    _limit(y - direction * _shift(x, step, state_format.rounding), state_format, event))
            z = _limit(z + direction * int(angle), phase_format, event)
        result[index] = _limit(z, phase_format, event)
    return FixedPointArray(result.reshape(data.shape), phase_format)


def cordic_phasor(theta, coefficient_format, phase_format=None, iterations=None, event=None, state_format=None):
    phase_format = phase_format or FixedPointFormat(24, 20)
    if not phase_format.signed or phase_format.maximum < math.pi:
        raise ValueError("phase format must represent [-pi, pi]")
    wide_phase = FixedPointFormat(62, phase_format.fractional_length,
                                  rounding=phase_format.rounding,
                                  saturation=False, overflow_policy="error")
    raw = quantize(theta, wide_phase, event=event)
    scale = 1 << phase_format.fractional_length
    pi, two_pi = round(math.pi * scale), round(2 * math.pi * scale)
    wrapped_codes = np.array([(int(code) + pi) % two_pi - pi
                              for code in raw.real.reshape(-1)], dtype=np.int64)
    data = FixedPointArray(wrapped_codes.reshape(raw.shape), phase_format)
    iterations = int(iterations or min(32, max(coefficient_format.fractional_length, phase_format.fractional_length) + 2))
    if not 1 <= iterations <= 62:
        raise ValueError("CORDIC iterations must be in [1, 62]")
    atan = quantize(np.arctan(2.0 ** -np.arange(iterations)), phase_format).real
    pi, half_pi = round(math.pi * scale), round(math.pi / 2 * scale)
    gain = math.prod(math.sqrt(1 + 2.0 ** (-2 * step)) for step in range(iterations))
    state_format = state_format or FixedPointFormat(
        min(62, coefficient_format.word_length + 3), coefficient_format.fractional_length,
        rounding=coefficient_format.rounding, saturation=coefficient_format.saturation,
        overflow_policy=coefficient_format.overflow_policy)
    if not state_format.signed or not coefficient_format.signed:
        raise ValueError("CORDIC state/coefficient formats must be signed")
    initial_x = int(quantize(1 / gain, state_format).real)
    real, imag = np.empty(data.real.size, dtype=np.int64), np.empty(data.real.size, dtype=np.int64)
    for index, raw in enumerate(data.real.reshape(-1)):
        x, y, z, sign = initial_x, 0, int(raw), 1
        if z > half_pi:
            z, sign = z - pi, -1
        elif z < -half_pi:
            z, sign = z + pi, -1
        for step, angle in enumerate(atan):
            direction = 1 if z >= 0 else -1
            x, y = (_limit(x - direction * _shift(y, step, state_format.rounding), state_format, event),
                    _limit(y + direction * _shift(x, step, state_format.rounding), state_format, event))
            z = _limit(z - direction * int(angle), phase_format, event)
        real[index] = _limit(sign * x, state_format, event)
        imag[index] = _limit(sign * y, state_format, event)
    return requantize(FixedPointArray(real.reshape(data.shape), state_format, imag.reshape(data.shape)),
                      coefficient_format, event=event)
