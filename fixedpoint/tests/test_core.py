"""运算边界和算法意义测试；不以复制实现作为预期值。"""
import numpy as np
import pytest
from fixedpoint import (FixedPointFormat, FixedPointConfig, FixedPointContext,
                        FixedPointMixin, RangeStatistics, recommend_qformats)
from fixedpoint.arithmetic import quantize, requantize, add, mul, div, shift, checked_mul


@pytest.mark.parametrize("mode,expected", [
    ("nearest_even", [-2, -2, 0, 0, 2, 2]),
    ("nearest_away", [-3, -2, -1, 1, 2, 3]),
    ("toward_zero", [-2, -1, 0, 0, 1, 2]),
    ("floor", [-3, -2, -1, 0, 1, 2]),
    ("ceil", [-2, -1, 0, 1, 2, 3]),
])
def test_rounding_positive_negative_half_lsb(mode, expected):
    values = np.array([-2.5, -1.5, -.5, .5, 1.5, 2.5])
    fmt = FixedPointFormat(8, 0, rounding=mode)
    assert np.array_equal(quantize(values, fmt).real, expected)
    wide = quantize(values, FixedPointFormat(12, 2))
    assert np.array_equal(requantize(wide, fmt).real, expected)


def test_saturation_wrap_error_and_native_overflow():
    assert np.array_equal(quantize([-100, 100], FixedPointFormat(4, 0)).real, [-8, 7])
    wrap = FixedPointFormat(4, 0, saturation=False, overflow_policy="wrap")
    assert np.array_equal(quantize([8, -9], wrap).real, [-8, 7])
    with pytest.raises(OverflowError):
        quantize([8], FixedPointFormat(4, 0, saturation=False))
    with pytest.raises(OverflowError, match="native"):
        checked_mul(np.array([1 << 50]), np.array([1 << 20]))
    with pytest.raises(ValueError):
        quantize([np.nan], FixedPointFormat())
    with pytest.raises(OverflowError, match="native"):
        quantize([float(1 << 63)], FixedPointFormat(62, 0, saturation=False, overflow_policy="wrap"))
    np.testing.assert_equal(quantize([-1e300, 1e300], FixedPointFormat(4, 0)).real, [-8, 7])


def test_complex_arithmetic_and_division_quantized_zero():
    fmt = FixedPointFormat(24, 16)
    a = quantize([1 + 2j, -1 + .5j], fmt)
    b = quantize([2 - 1j, 1 + 1j], fmt)
    np.testing.assert_allclose(mul(a, b, fmt).dequantize(), np.array([4 + 3j, -1.5 - .5j]))
    np.testing.assert_allclose(div(a, b, fmt).dequantize(), np.array([1j, -.25 + .75j]), atol=fmt.lsb)
    with pytest.raises(ZeroDivisionError):
        div(a, quantize([0, 1], fmt), fmt)


def test_full_precision_division_large_intermediate_small_output():
    fmt = FixedPointFormat(48, 28)
    a = quantize([512, -512, 10000 + 3000j], fmt)
    b = quantize([512, 512, 10000 - 3000j], fmt)
    events = []
    result = div(a, b, fmt, events.append).dequantize()
    expected = np.array([1, -1, (10000 + 3000j) / (10000 - 3000j)])
    np.testing.assert_allclose(result, expected, atol=fmt.lsb)
    assert events[0]["intermediate_bits"] > 64
    assert events[0]["intermediate_bits"] <= events[0]["intermediate_bound_bits"]
    ctx = context()
    numerator = ctx.operand([512, -512], "test", "accumulator")
    denominator = ctx.operand([512, 512], "test", "accumulator")
    np.testing.assert_equal(ctx.div(numerator, denominator, "test", role="accumulator"), [1, -1])
    stats = ctx.report()["diagnostics"]["test.div"]
    assert stats["max_intermediate_bits"] > 64 and stats["wide_division_count"] == 2


@pytest.mark.parametrize("mode,expected", [
    ("nearest_even", [-2, -2, 0, 0, 2, 2]),
    ("nearest_away", [-3, -2, -1, 1, 2, 3]),
    ("toward_zero", [-2, -1, 0, 0, 1, 2]),
    ("floor", [-3, -2, -1, 0, 1, 2]),
    ("ceil", [-2, -1, 0, 1, 2, 3]),
])
def test_wide_division_signed_half_lsb_rounding(mode, expected):
    fmt = FixedPointFormat(48, 28)
    numerator = quantize([-5, -3, -1, 1, 3, 5], fmt)
    denominator = quantize(2, fmt)
    target = FixedPointFormat(12, 0, rounding=mode)
    np.testing.assert_equal(div(numerator, denominator, target).real, expected)


def test_wide_division_target_overflow_and_unsigned_complex_width():
    source = FixedPointFormat(48, 28)
    large, small = quantize(1000, source), quantize(1, source)
    assert int(div(large, small, FixedPointFormat(4, 0)).real) == 7
    wrapped = FixedPointFormat(4, 0, saturation=False, overflow_policy="wrap")
    assert int(div(large, small, wrapped).real) == -8
    with pytest.raises(OverflowError):
        div(large, small, FixedPointFormat(4, 0, saturation=False))
    from fixedpoint import FixedPointArray
    unsigned = FixedPointFormat(62, 0, signed=False)
    huge = FixedPointArray(np.array(unsigned.max_code), unsigned, np.array(unsigned.max_code))
    result = div(huge, huge, FixedPointFormat(16, 12))
    np.testing.assert_equal(result.dequantize(), 1 + 0j)


def test_add_alignment_and_negative_shift():
    a = quantize([-1.5, 2.5], FixedPointFormat(12, 1))
    b = quantize([.25, -.25], FixedPointFormat(12, 2))
    np.testing.assert_equal(add(a, b, FixedPointFormat(16, 4)).dequantize(), [-1.25, 2.25])
    result = shift(quantize([-3, -1, 1, 3], FixedPointFormat(8, 0)), -1)
    np.testing.assert_equal(result.real, [-2, 0, 0, 2])


def context(level="operation", **options):
    return FixedPointContext({"enable_fixed": True, "modules": {"test": {
        "arithmetic_level": level,
        "input_format": {"word_length": 28, "fractional_length": 18},
        "state_format": {"word_length": 30, "fractional_length": 18},
        "output_format": {"word_length": 28, "fractional_length": 18},
        **options}}}, {"enabled": True, "reservoir_size": 64})


@pytest.mark.parametrize("length", [1, 15, 32, 480, 512])
@pytest.mark.parametrize("norm", [None, "ortho", "forward"])
@pytest.mark.parametrize("inverse", [False, True])
def test_mixed_radix_fft_matches_length_and_normalization(length, norm, inverse):
    rng = np.random.default_rng(317)
    x = .01 * (rng.normal(size=(2, length)) + 1j * rng.normal(size=(2, length)))
    ctx = context()
    result = ctx.fft(x, "test", inverse=inverse, axis=1, norm=norm)
    golden = (np.fft.ifft if inverse else np.fft.fft)(x, axis=1, norm=norm)
    assert result.shape == x.shape
    np.testing.assert_allclose(result, golden, atol=0.002, rtol=0)


def test_fft_stage_scaling_and_unsupported_length():
    ctx = context(stage_shifts=[1, 1, 1, 1, 1])
    impulse = np.zeros(32)
    impulse[0] = 1
    np.testing.assert_allclose(ctx.fft(impulse, "test"), np.ones(32), atol=0.002)
    with pytest.raises(NotImplementedError):
        ctx.fft(np.ones(7), "test")


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_integer_fir_and_correlation_match(mode):
    x = np.array([.25 + .5j, -.75 + .25j, .5, .125 - .25j])
    h = np.array([.25, -.5j, .125 + .25j])
    ctx = context()
    np.testing.assert_allclose(ctx.fir(x, h, "test", mode), np.convolve(x, h, mode), atol=1e-5)
    np.testing.assert_allclose(ctx.correlate(x, h, "test", mode), np.correlate(x, h, mode), atol=1e-5)


def test_correlation_minimum_negative_conjugation_obeys_overflow_rule():
    from fixedpoint.kernels.fir import fixed_correlate
    x = quantize([1 + 0j], FixedPointFormat(8, 0))
    wide = FixedPointFormat(16, 0)
    events = []
    h = quantize([-8j], FixedPointFormat(4, 0))
    saturated = fixed_correlate(x, h, wide, wide, event=events.append)
    np.testing.assert_equal(saturated.dequantize(), [7j])
    assert sum(event["overflow_count"] for event in events) == 1
    h_error = quantize([-8j], FixedPointFormat(4, 0, saturation=False))
    with pytest.raises(OverflowError):
        fixed_correlate(x, h_error, wide, wide)


def test_range_exact_moments_deterministic_reservoir_and_merge():
    values = np.linspace(-3, 8, 1000)
    first, second = RangeStatistics(32), RangeStatistics(32)
    first.observe(values[:300])
    first.observe(values[300:])
    second.observe(values)
    assert first.minimum == values.min() and first.maximum == values.max()
    assert abs(first.mean - values.mean()) < 1e-12
    assert abs(first.report()["std"] - values.std()) < 1e-12
    np.testing.assert_equal(np.sort(first.samples), np.sort(second.samples))
    left, right = RangeStatistics(32), RangeStatistics(32)
    left.observe(values[:400]); right.observe(values[400:]); left.merge(right)
    assert abs(left.report()["std"] - values.std()) < 1e-12
    assert left.report()["percentile_approximate"]


def test_merge_percentiles_respect_very_different_population_counts():
    large, small = RangeStatistics(256), RangeStatistics(256)
    large.observe(np.zeros(100000))
    small.observe(np.ones(1000) * 100)
    large.merge(small)
    # 小总体约占1%，不能因两个reservoir等长而占约50%。
    assert np.count_nonzero(large.samples) < 20
    assert large.report()["percentile"]["50"] == 0


def test_float_default_does_not_change_values_or_rng():
    ctx = FixedPointContext()
    assert ctx.stimulus is None
    x = np.array([.123456789, 1.234567890])
    assert ctx.quantize(x, "test") is x
    np.random.seed(88)
    before = np.random.get_state()
    ctx.observe("x", x)
    after = np.random.get_state()
    np.testing.assert_equal(before[1], after[1])
    assert before[2:] == after[2:]
    obj = FixedPointMixin.__new__(FixedPointMixin)
    assert obj.enable_fixed is False


def test_next_run_configuration_change_refreshes_context():
    class Params:
        def __init__(self):
            self.config = {"fixed_point": {}}
        def get(self, name, default=None):
            return self.config.get(name, default)
    params = Params()
    first = FixedPointContext.from_params(params)
    assert FixedPointContext.from_params(params) is first
    params.config["fixed_point"] = {"enable_fixed": True}
    second = FixedPointContext.from_params(params)
    assert second is not first and second.module_enabled("test")
    assert FixedPointContext.from_params(params) is second


def test_serialization_preserves_child_special_defaults_under_parent_switch():
    cfg = FixedPointConfig.from_dict({"enable_fixed": True, "modules": {"sc": {"arithmetic_level": "operation"}}})
    assert "formats" not in cfg.to_dict()["modules"]["sc"]
    ctx = FixedPointContext(cfg)
    assert ctx.module_config("sc.rx.equalizer").format("output").word_length == 32
    assert ctx.module_config("sc.rx.coarse_sync").format("output").word_length == 32
    assert ctx.module_config("ldpc.llr").format("output").word_length == 8


def test_module_reconfiguration_preserves_range_switch_and_rejects_conflict():
    module = FixedPointMixin()
    initial = FixedPointContext({}, {"enabled": True, "reservoir_size": 32})
    module.configure_fixed(False, fixed_context=initial)
    module.configure_fixed(True)
    assert module.enable_fixed and module.fixed_context.range_enabled
    assert module.fixed_context.range_analyzer.reservoir_size == 32
    current = module.fixed_context
    with pytest.raises(ValueError):
        module.configure_fixed(True, fixed_context=initial)
    assert module.fixed_context is current


def test_config_validation_copy_inheritance_and_recommendation():
    raw = {"enable_fixed": True, "modules": {"sc": {
        "output_format": {"word_length": 20, "fractional_length": 14}, "options": {"tag": [1]}},
        "sc.fir": {"enabled": False}}}
    cfg = FixedPointConfig.from_dict(raw)
    raw["modules"]["sc"]["options"]["tag"].append(2)
    assert cfg.module_config("sc.fir").get("tag") == [1]
    assert cfg.module_config("sc.fir").format("output").word_length == 20
    assert not cfg.module_config("sc.fir").enabled
    snapshot = FixedPointContext(cfg)
    cfg.modules["sc"]["options"]["tag"].append(3)
    assert snapshot.module_config("sc.fir").get("tag") == [1]
    for bad in ([1], 0, {"enable_fixed": "false"}, {"modules": {"x": {"enabled": "false"}}}):
        with pytest.raises((ValueError, TypeError)):
            FixedPointConfig.from_dict(bad)
    with pytest.raises((ValueError, TypeError)):
        FixedPointFormat(word_length=True)
    ctx = context()
    ctx.observe("x", np.array([-2, 3]))
    recommendation = recommend_qformats(ctx.report())
    assert recommendation["validated"] is False
    assert recommendation["nodes"]["x"]["candidates"]
    assert all(c["word_length"] >= 4 for c in recommendation["nodes"]["x"]["candidates"])


def test_operand_preserves_accumulator_and_sum_saturation():
    ctx = context()
    accumulator = ctx.operand([5000, 5000], "test", "accumulator")
    # 不能重新进入狭窄 input 格式。
    result = ctx.div(accumulator, ctx.operand(2, "test", "input"), "test", role="accumulator")
    np.testing.assert_equal(result, [2500, 2500])
    cfg = {"enable_fixed": True, "modules": {"x": {"arithmetic_level": "operation",
           "accumulator_format": {"word_length": 4, "fractional_length": 0}}}}
    assert FixedPointContext(cfg).sum([3, 3, 3], "x") == 7


def test_integer_sqrt_and_cordic():
    ctx = context()
    np.testing.assert_allclose(ctx.sqrt([0, 2, 4], "test"), np.sqrt([0, 2, 4]), atol=1e-5)
    angles = np.array([-np.pi, -.75, 0, .4, np.pi])
    np.testing.assert_allclose(ctx.phasor(angles, "test"), np.exp(1j * angles), atol=1e-4)
    error = np.angle(np.exp(1j * (ctx.angle(np.exp(1j * angles), "test") - angles)))
    np.testing.assert_allclose(error, 0, atol=1e-4)


def test_large_integer_magnitude_avoids_narrow_squared_product():
    ctx = context()
    # Q48.28乘积仅能容纳约5e5；4k相关峰平方约1.6e7，不可先夹断。
    values = ctx.operand([4000 + 3000j, -6000j], "test", "accumulator")
    result = ctx.magnitude(values, "test", role="accumulator")
    np.testing.assert_equal(result, [5000, 6000])
    assert ctx.report()["coverage"]["test"]["derived_formats"]["magnitude_full_squared_sum"]["word_length"] == 97


def test_matrix_operation_and_singular_rejection():
    ctx = context()
    a = np.array([[2., 1.], [1., 3.]])
    b = np.array([1., 4.])
    np.testing.assert_allclose(ctx.solve(a, b, "test"), np.linalg.solve(a, b), atol=1e-3)
    np.testing.assert_allclose(ctx.pinv(a, "test"), np.linalg.pinv(a), atol=1e-3)
    with pytest.raises(np.linalg.LinAlgError):
        ctx.solve(np.array([[1., 1.], [1., 1.]]), b, "test")
