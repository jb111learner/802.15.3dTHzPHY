"""混合基数 FFT。480 点使用 2/3/5 分解，绝不改动变换长度。"""
import numpy as np
from ..array import FixedPointArray
from ..arithmetic import quantize, requantize, mul, add, shift, checked_mul, round_ratio, from_scaled
from ..format import FixedPointFormat
import math


def _radix(n):
    for radix in (2, 3, 5):
        if n % radix == 0:
            return radix
    raise NotImplementedError("operation FFT only supports lengths with factors 2, 3, 5: " + str(n))


def fixed_fft(value, state_format, coefficient_format, output_format, inverse=False,
              axis=-1, norm=None, n=None, stage_shifts=(), event=None, observe=None):
    if norm not in {None, "backward", "ortho", "forward"}:
        raise ValueError("unsupported FFT norm")
    if value.real.ndim == 0:
        raise ValueError("FFT input must have an axis")
    real = np.moveaxis(value.real, axis, -1)
    imag = np.zeros_like(real) if value.imag is None else np.moveaxis(value.imag, axis, -1)
    size = real.shape[-1] if n is None else int(n)
    if size < 1:
        raise ValueError("FFT length must be positive")
    if size != real.shape[-1]:
        if size < real.shape[-1]:
            real, imag = real[..., :size], imag[..., :size]
        else:
            width = [(0, 0)] * real.ndim
            width[-1] = (0, size - real.shape[-1])
            real, imag = np.pad(real, width), np.pad(imag, width)
    stage_shifts = tuple(int(v) for v in stage_shifts)
    if any(v < 0 for v in stage_shifts):
        raise ValueError("FFT stage shifts must be nonnegative")
    check, depth = size, 0
    while check > 1:
        check //= _radix(check)
        depth += 1
    if len(stage_shifts) > depth:
        raise ValueError("more FFT stage shifts than transform stages")
    sign = 1 if inverse else -1

    def recurse(x, level):
        length = x.real.shape[-1]
        if length == 1:
            return requantize(x, state_format, event)
        radix = _radix(length)
        subsize = length // radix
        k = np.arange(length)
        acc = FixedPointArray(np.zeros(x.real.shape, dtype=np.int64), state_format,
                              np.zeros(x.real.shape, dtype=np.int64))
        # 使用首级对应 stage_shifts[0]；子问题采用后续移位。
        stage_shift = stage_shifts[level] if level < len(stage_shifts) else 0
        for residue in range(radix):
            sub = FixedPointArray(x.real[..., residue::radix], x.format,
                                  x.imag[..., residue::radix])
            sub = recurse(sub, level + 1)
            expanded = FixedPointArray(sub.real[..., k % subsize], state_format,
                                       sub.imag[..., k % subsize])
            twiddle = quantize(np.exp(sign * 2j * np.pi * residue * k / length), coefficient_format, event)
            product = mul(expanded, twiddle, state_format, event)
            if stage_shift:
                product = shift(product, -stage_shift, state_format, event)
            acc = add(acc, product, state_format, event)
        if observe is not None:
            observe("stage_" + str(level + 1).zfill(2), acc.dequantize())
        return acc

    result = recurse(requantize(FixedPointArray(real, value.format, imag), state_format, event), 0)
    # 把归一化和内部缩放补偿合并后才转换到输出格式。
    # 非ortho使用整数除N，避免把1/480先量化到Q2.16造成确定性增益偏差。
    if norm != "ortho":
        divisor = size if ((norm == "forward") != inverse) else 1
        exponent = output_format.fractional_length - state_format.fractional_length + sum(stage_shifts)
        compensation = 1 << max(0, exponent)
        divisor *= 1 << max(0, -exponent)
        if divisor > np.iinfo(np.int64).max:
            raise OverflowError("native FFT normalization denominator does not fit int64")
        real = round_ratio(checked_mul(result.real, compensation), divisor, output_format.rounding)
        imag = round_ratio(checked_mul(result.imag, compensation), divisor, output_format.rounding)
        result = from_scaled(real, imag, output_format.fractional_length, output_format, event)
    else:
        mantissa, exponent = math.frexp(2.0 ** sum(stage_shifts) / math.sqrt(size))
        coefficient = quantize(mantissa, coefficient_format, event)
        # 完整乘积，指数补偿不经窄系数格式饱和。
        wide_format = FixedPointFormat(min(62, state_format.word_length + coefficient_format.word_length),
                                       state_format.fractional_length + coefficient_format.fractional_length,
                                       rounding=output_format.rounding, saturation=output_format.saturation,
                                       overflow_policy=output_format.overflow_policy)
        result = mul(result, coefficient, wide_format, event)
        real, imag = result.real, result.imag
        if exponent > 0:
            real, imag = checked_mul(real, 1 << exponent), checked_mul(imag, 1 << exponent)
        fractional = wide_format.fractional_length + max(0, -exponent)
        result = from_scaled(real, imag, fractional, output_format, event)
    return FixedPointArray(np.moveaxis(result.real, -1, axis), output_format,
                           np.moveaxis(result.imag, -1, axis))
