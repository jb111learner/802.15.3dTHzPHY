"""显式整数算术；每次目标格式转换执行配置的舍入与溢出规则。

使用宿主 int64 的运算先检查风险；除法采用声明位宽的完整 Python 整数。
任何路径都不允许 NumPy 静默回绕。
"""
import numpy as np
from .array import FixedPointArray

NATIVE_MAX = np.iinfo(np.int64).max


def _absmax(value):
    value = np.asarray(value)
    return max(abs(int(value.min())), abs(int(value.max()))) if value.size else 0


def checked_add(a, b):
    if _absmax(a) + _absmax(b) > NATIVE_MAX:
        raise OverflowError("native int64 addition may overflow; reduce internal word lengths")
    return np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64)


def checked_mul(a, b):
    if _absmax(a) * _absmax(b) > NATIVE_MAX:
        raise OverflowError("native int64 multiplication may overflow; reduce internal word lengths")
    return np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)


def _left_shift(value, bits):
    if bits < 0:
        raise ValueError("left shift must be nonnegative")
    if _absmax(value) * (1 << bits) > NATIVE_MAX:
        raise OverflowError("native int64 shift may overflow; reduce fractional length")
    return np.asarray(value, dtype=np.int64) * (1 << bits)


def round_ratio(numerator, denominator, rounding):
    """分母必须为正；以余数实现负数和半 LSB 舍入。"""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.int64), np.asarray(denominator, dtype=np.int64))
    if np.any(denominator <= 0):
        raise ZeroDivisionError("fixed-point denominator must be positive")
    floor = np.floor_divide(numerator, denominator)
    rem = np.remainder(numerator, denominator)
    if rounding == "floor":
        return floor
    if rounding == "ceil":
        return floor + (rem != 0)
    if rounding == "toward_zero":
        return floor + ((numerator < 0) & (rem != 0))
    greater = rem > (denominator - rem)
    equal = rem == (denominator - rem)
    if rounding == "nearest_even":
        return floor + (greater | (equal & ((floor % 2) != 0)))
    if rounding == "nearest_away":
        return floor + (greater | (equal & (numerator >= 0)))
    raise ValueError("unsupported rounding: " + str(rounding))


def _overflow(value, fmt, event=None):
    value = np.asarray(value, dtype=np.int64)
    mask = (value < fmt.min_code) | (value > fmt.max_code)
    if event is not None:
        event({"overflow_count": int(np.count_nonzero(mask)), "count": int(value.size)})
    if not np.any(mask):
        return value
    if fmt.saturation:
        return np.clip(value, fmt.min_code, fmt.max_code)
    if fmt.overflow_policy == "error":
        raise OverflowError("fixed-point overflow in " + fmt.q_format)
    modulus = 1 << fmt.word_length
    # modulus <= 2**62；余数不会溢出，避免 (value-min_code) 相加。
    result = np.remainder(value, modulus)
    if fmt.signed:
        result = np.where(result >= (1 << (fmt.word_length - 1)), result - modulus, result)
    return result.astype(np.int64)


def _round_float(value, mode):
    if mode == "nearest_even":
        return np.rint(value)
    if mode == "nearest_away":
        return np.copysign(np.floor(np.abs(value) + 0.5), value)
    return {"toward_zero": np.trunc, "floor": np.floor, "ceil": np.ceil}[mode](value)


def quantize(value, fmt, event=None):
    if isinstance(value, FixedPointArray):
        return requantize(value, fmt, event)
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise ValueError("NaN/Inf cannot be represented by fixed-point arithmetic")

    def component(v):
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = _round_float(np.asarray(v, dtype=np.float64) * (2.0 ** fmt.fractional_length), fmt.rounding)
        # float64 将 int64.max 舍入到2**63；必须比较排他的正端点。
        native_bad = (rounded < -(1 << 63)) | (rounded >= (1 << 63))
        if np.any(native_bad) and not fmt.saturation:
            raise OverflowError("native int64 conversion would overflow")
        if fmt.saturation:
            bad = (rounded < fmt.min_code) | (rounded > fmt.max_code)
            safe = np.clip(rounded, float(fmt.min_code), float(fmt.max_code)).astype(np.int64)
            # float64 可能把大字长 max_code 舍入到 max_code+1。
            bad |= (safe < fmt.min_code) | (safe > fmt.max_code)
            if event is not None:
                event({"overflow_count": int(np.count_nonzero(bad)), "count": int(v.size)})
            return _overflow(safe, fmt)
        return _overflow(rounded.astype(np.int64), fmt, event)

    real = component(value.real)
    imag = component(value.imag) if np.iscomplexobj(value) else None
    return FixedPointArray(real, fmt, imag)


def from_scaled(real, imag, fractional_length, fmt, event=None):
    def convert(v):
        if v is None:
            return None
        shift = fmt.fractional_length - fractional_length
        value = (_left_shift(v, shift) if shift >= 0 else
                 round_ratio(v, 1 << -shift, fmt.rounding))
        return _overflow(value, fmt, event)
    return FixedPointArray(convert(real), fmt, convert(imag))


def requantize(value, fmt, event=None):
    return from_scaled(value.real, value.imag, value.format.fractional_length, fmt, event)


def to_integer(value):
    """数值转整数向零截断；与直接提取 .real 整数码值不同。"""
    if not isinstance(value, FixedPointArray) or value.imag is not None:
        raise TypeError("to_integer requires a real FixedPointArray")
    return round_ratio(value.real, 1 << value.format.fractional_length, "toward_zero")


def negate(value, target=None, event=None):
    """一元取负也执行目标格式溢出规则。"""
    target = target or value.format
    return from_scaled(-value.real, None if value.imag is None else -value.imag,
                       value.format.fractional_length, target, event)


def abs_real(value, event=None):
    """Clause 4 的实数绝对值在不可表示时饱和，包括最小负值。"""
    from dataclasses import replace
    if value.imag is not None:
        raise TypeError("abs_real does not compute complex magnitude")
    saturated = replace(value.format, saturation=True)
    codes = _overflow(np.abs(value.real), saturated, event)
    return FixedPointArray(codes, value.format)


def add(a, b, target, event=None):
    fractional = max(a.format.fractional_length, b.format.fractional_length)
    ar = _left_shift(a.real, fractional - a.format.fractional_length)
    br = _left_shift(b.real, fractional - b.format.fractional_length)
    real = checked_add(ar, br)
    imag = None
    if a.imag is not None or b.imag is not None:
        ai = 0 if a.imag is None else _left_shift(a.imag, fractional - a.format.fractional_length)
        bi = 0 if b.imag is None else _left_shift(b.imag, fractional - b.format.fractional_length)
        imag = checked_add(ai, bi)
    return from_scaled(real, imag, fractional, target, event)


def mul(a, b, target, event=None):
    fractional = a.format.fractional_length + b.format.fractional_length
    real = checked_mul(a.real, b.real)
    imag = None
    if a.imag is not None and b.imag is not None:
        real = checked_add(real, -checked_mul(a.imag, b.imag))
    if a.imag is not None or b.imag is not None:
        imag = checked_add(0 if b.imag is None else checked_mul(a.real, b.imag),
                           0 if a.imag is None else checked_mul(a.imag, b.real))
    return from_scaled(real, imag, fractional, target, event)


def div(a, b, target, event=None):
    """完整整数商；宽中间量不经 int64，目标码值最终才转换。

    例如两个 Q20.28 累加结果相除时，商可为1，但被除数左移28位
    可超过64位。共轭乘积、平方和同样需要明确的完整中间精度。
    这里从输入格式推导最大位宽，并检查实际 Python 整数不超过该界。
    """
    complex_result = a.imag is not None or b.imag is not None
    arrays = np.broadcast_arrays(a.real, b.real,
                                 0 if a.imag is None else a.imag,
                                 0 if b.imag is None else b.imag)
    shape = arrays[0].shape
    real = np.empty(shape, dtype=np.int64)
    imag = np.empty(shape, dtype=np.int64) if complex_result else None
    fractional_shift = b.format.fractional_length + target.fractional_length - a.format.fractional_length
    a_width = a.format.word_length + int(not a.format.signed)
    b_width = b.format.word_length + int(not b.format.signed)
    if b.imag is None:
        numerator_width = a_width + 1
        denominator_width = b_width + 1
    else:
        numerator_width = a_width + b_width + 1
        denominator_width = 2 * b_width + 1
    bound = max(numerator_width + max(0, fractional_shift),
                denominator_width + max(0, -fractional_shift))
    maximum_bits, overflows = 1, 0

    def track(*values):
        nonlocal maximum_bits
        # 含符号的完整二进制位宽；Python整数负责存储而不负责隐式截断。
        bits = max((int(value).bit_length() + 1 if value >= 0 else
                    (~int(value)).bit_length() + 1) for value in values)
        if bits > bound:
            raise OverflowError("division intermediate exceeds declared complete width")
        maximum_bits = max(maximum_bits, bits)

    def rounded(numerator, denominator):
        nonlocal overflows
        if denominator < 0:
            numerator, denominator = -numerator, -denominator
        if fractional_shift >= 0:
            numerator <<= fractional_shift
        else:
            denominator <<= -fractional_shift
        track(numerator, denominator)
        floor, remainder = divmod(numerator, denominator)
        mode = target.rounding
        if mode == "ceil":
            code = floor + bool(remainder)
        elif mode == "toward_zero":
            code = floor + (numerator < 0 and bool(remainder))
        elif mode in {"nearest_even", "nearest_away"}:
            complement = denominator - remainder
            increment = (remainder > complement or
                         (remainder == complement and
                          ((floor % 2 != 0) if mode == "nearest_even" else numerator >= 0)))
            code = floor + increment
        else:
            code = floor
        if code < target.min_code or code > target.max_code:
            overflows += 1
            if target.saturation:
                code = min(max(code, target.min_code), target.max_code)
            elif target.overflow_policy == "error":
                raise OverflowError("fixed-point division overflow in " + target.q_format)
            else:
                code %= 1 << target.word_length
                if target.signed and code >= 1 << (target.word_length - 1):
                    code -= 1 << target.word_length
        return code

    for index in np.ndindex(shape):
        ar, br, ai, bi = (int(component[index]) for component in arrays)
        if b.imag is None:
            denominator, numerator_r, numerator_i = br, ar, ai
            track(denominator, numerator_r, numerator_i)
        else:
            # 所有乘积和平方和使用声明的宽整数，不调用宿主 int64 乘法。
            br2, bi2, arbr, aibi, aibr, arbi = br * br, bi * bi, ar * br, ai * bi, ai * br, ar * bi
            track(br2, bi2, arbr, aibi, aibr, arbi)
            denominator = br2 + bi2
            numerator_r = arbr + aibi
            numerator_i = aibr - arbi
            track(denominator, numerator_r, numerator_i)
        if denominator == 0:
            raise ZeroDivisionError("fixed-point division by zero (possibly quantized to zero)")
        real[index] = rounded(numerator_r, denominator)
        if imag is not None:
            imag[index] = rounded(numerator_i, denominator)
    if event is not None:
        event({"overflow_count": overflows, "count": int(real.size) * (2 if complex_result else 1),
               "intermediate_bits": maximum_bits, "intermediate_bound_bits": bound,
               "integer_backend": "bounded_complete_python_integer", "operation_count": int(real.size)})
    return FixedPointArray(real, target, imag)


def shift(value, bits, target=None, event=None):
    target = target or value.format
    # 移位改变数值而不是元数据；右移在同一格式内舍入。
    def component(v):
        if v is None:
            return None
        shifted = _left_shift(v, bits) if bits >= 0 else round_ratio(v, 1 << -bits, target.rounding)
        return shifted
    return from_scaled(component(value.real), component(value.imag), value.format.fractional_length, target, event)


def sum_values(value, target, axis=None, event=None):
    if axis is None:
        real = value.real.reshape(-1)
        imag = None if value.imag is None else value.imag.reshape(-1)
    else:
        real = np.moveaxis(value.real, axis, -1)
        imag = None if value.imag is None else np.moveaxis(value.imag, axis, -1)
    shape = real.shape[:-1]
    acc = FixedPointArray(np.zeros(shape, dtype=np.int64), target,
                         None if imag is None else np.zeros(shape, dtype=np.int64))
    for index in range(real.shape[-1]):
        term = FixedPointArray(real[..., index], value.format,
                              None if imag is None else imag[..., index])
        acc = add(acc, term, target, event)
    return acc
