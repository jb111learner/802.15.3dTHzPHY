"""发射与信道定点接入、黄金路径和噪声重放验证。"""
import numpy as np
import pytest

from channel.AWGN import AWGN
from channel.CFO import CFO
from channel.MIMOChannel import MIMOChannel
from channel.MultipathChannel import MultipathChannel
from channel.THzChannel import THzChannel
from fixedpoint.context import FixedPointContext
from params.PHYParams import PHYParams
from transmitter.Pulseshaper import TxPulseShaper
from transmitter.TxOFDMProcesser import TxOFDMProcesser


def signal_dict(samples, sample_rate=30e9):
    samples = np.asarray(samples, dtype=np.complex128)
    return {
        "signal_stream": samples, "sample_rate_Hz": sample_rate,
        "duration_seconds": samples.shape[-1] / sample_rate,
        "signal_length": samples.shape[-1], "padding_bit_num": 0,
    }


def test_adc_child_override_respects_top_level_veto():
    params = PHYParams()
    values = dict(enable_awgn=False, enable_multipath=False, enable_cfo=False,
                  enable_iq_imbalance=False, enable_pa=False)
    config = {"enable_fixed": True, "modules": {
        "channel": {"enabled": False}, "channel.rx_samples": {"enabled": True},
    }}
    params.update(**values, fixed_point=config)
    original = signal_dict([20.125 + .12345j])
    received = THzChannel(params).run(original)
    assert received["signal_stream"][0].real < 8
    config["enable_fixed"] = False
    params.update(fixed_point=config)
    np.testing.assert_array_equal(THzChannel(params).run(original)["signal_stream"], original["signal_stream"])


def test_mimo_boundary_output_after_noise_is_on_declared_grid():
    params = PHYParams()
    params.update(enable_mimo=True, num_tx=2, num_rx=2, num_spatial_streams=2,
                  random_seed=44, enable_iq_imbalance=False,
                  fixed_point={"enable_fixed": True, "modules": {
                      "channel.mimo": {"quantize_channel_arithmetic": True},
                  }})
    channel = MIMOChannel(params)
    transmitted = signal_dict(np.full((2, 32), .12345 + .45678j))
    actual = channel.apply(transmitted)["signal_stream"]
    for component in (actual.real, actual.imag):
        np.testing.assert_array_equal(component * 4096, np.rint(component * 4096))


def operation_config(module):
    return {
        "enable_fixed": True,
        "default_format": {"word_length": 32, "fractional_length": 24},
        "modules": {
            module: {
                "arithmetic_level": "operation", "quantize_channel_arithmetic": True,
                "state_format": {"word_length": 40, "fractional_length": 24},
                "coefficient_format": {"word_length": 32, "fractional_length": 28},
                "product_format": {"word_length": 56, "fractional_length": 32},
                "accumulator_format": {"word_length": 56, "fractional_length": 32},
            }
        },
    }


def test_ofdm_float_matches_original_ortho_gain_and_layout():
    params = PHYParams()
    processor = TxOFDMProcesser(params)
    count = processor.num_data_ofdm_per_subframe * processor.subwave_num
    rng = np.random.default_rng(81)
    symbols = (rng.normal(size=count) + 1j * rng.normal(size=count)) / 3
    data = signal_dict(symbols)
    data.update(frame_num=1, frame_symbol_num=count)
    actual = processor.ofdm_process(data)["signal_stream"]
    grid = np.zeros((processor.subframe_ofdm_num, processor.subwave_num), complex)
    grid[processor.pilot_block_indexes] = processor.pilot_sequences
    data_rows = np.setdiff1d(np.arange(processor.subframe_ofdm_num), processor.pilot_block_indexes)
    grid[data_rows] = symbols.reshape(len(data_rows), processor.subwave_num)
    length = processor.subwave_num * processor.oversampling
    padded = np.zeros((length, processor.subframe_ofdm_num), complex)
    half = processor.subwave_num // 2
    padded[:half] = grid.T[:half]
    padded[-half:] = grid.T[half:]
    np.testing.assert_array_equal(actual, np.fft.ifft(padded, axis=0, norm="ortho").ravel(order="F"))


def test_ofdm_operation_keeps_ortho_normalization_and_frequency_grid():
    params = PHYParams()
    params.update(oversampling=1, subframe_ofdm_num=4, pilot_block_indexes=[0, 2],
                  ofdm_iq_pilot_phase_codes_deg=[0.0, 90.0])
    rng = np.random.default_rng(92)
    data = signal_dict((rng.normal(size=1024) + 1j * rng.normal(size=1024)) / 4)
    data.update(frame_num=1, frame_symbol_num=1024)
    golden = TxOFDMProcesser(params).ofdm_process(data)["signal_stream"]
    fixed = TxOFDMProcesser(params)
    fixed.configure_fixed(True, operation_config("ofdm.tx.ifft"))
    actual = fixed.ofdm_process(data)["signal_stream"]
    np.testing.assert_allclose(actual, golden, atol=2e-6, rtol=0)
    assert actual.shape == golden.shape == (2048,)
    assert "operation" in fixed.fixed_context.coverage["ofdm.tx.ifft"]["levels"]


def test_sc_operation_filter_preserves_delay_and_output_length():
    base = {
        "oversampling": 2, "rolloff": 0.22, "filter_type": "rrc",
        "filter_length": 8, "link_mode": "sc-fde",
    }
    rng = np.random.default_rng(22)
    data = signal_dict((rng.normal(size=80) + 1j * rng.normal(size=80)) / 4)
    golden = TxPulseShaper(base).shape_pulse(data)["signal_stream"]
    fixed = TxPulseShaper({**base, "fixed_point": operation_config("sc.tx.pulse_shaping")})
    actual = fixed.shape_pulse(data)["signal_stream"]
    assert actual.shape == golden.shape == (160,)
    np.testing.assert_allclose(actual, golden, atol=2e-6, rtol=0)
    assert "operation" in fixed.fixed_context.coverage["sc.tx.pulse_shaping"]["levels"]


def test_static_integer_channel_operation_preserves_cross_frame_history():
    params = {
        "static_channel": True, "enable_multipath_memory": True,
        "multipath_paths": [
            {"delay_samples": 0, "gain": 1.0},
            {"delay_samples": 1, "gain": 0.5},
        ],
        "fixed_point": operation_config("channel.multipath"),
    }
    channel = MultipathChannel(params)
    first = channel.apply(signal_dict([1, 2, 3]))["signal_stream"]
    second = channel.apply(signal_dict([4, 5]))["signal_stream"]
    np.testing.assert_array_equal(first, [1, 2.5, 4])
    np.testing.assert_array_equal(second, [5.5, 7])
    channel.reset_state()
    np.testing.assert_array_equal(channel.apply(signal_dict([4, 5]))["signal_stream"], [4, 7])


def test_fractional_channel_operation_is_explicitly_rejected():
    channel = MultipathChannel({
        "use_frac_delay": True,
        "fixed_point": operation_config("channel.multipath"),
    })
    with pytest.raises(NotImplementedError, match="static integer-delay"):
        channel.apply(signal_dict([1, 0, 0]))


def test_noise_replay_keeps_absolute_variance_and_rng_consumption():
    params = {"SNRdB": 12.0, "bandwidth": 30e9,
              "noise_temperature": None, "noise_figure_db": None}
    floating = AWGN(params)
    fixed = AWGN({**params, "fixed_point": {"enable_fixed": True}})
    stimulus = {}
    floating.fixed_context.stimulus = stimulus
    fixed.fixed_context.stimulus = stimulus
    np.random.seed(110)
    first = floating.add_awgn(signal_dict(np.ones(96)))
    following_float = np.random.randn(10)
    np.random.seed(110)
    second = fixed.add_awgn(signal_dict(np.ones(96) / 2))
    following_fixed = np.random.randn(10)
    np.testing.assert_array_equal(floating.last_noise, fixed.last_noise)
    np.testing.assert_array_equal(following_float, following_fixed)
    assert first["noise_power"] == second["noise_power"]
    assert floating.fixed_context.stimulus["noise_power"] == first["noise_power"]


def test_normal_awgn_does_not_replay_first_frame_noise():
    awgn = AWGN({"SNRdB": 10, "bandwidth": 30e9,
                 "noise_temperature": None, "noise_figure_db": None})
    np.random.seed(15)
    awgn.add_awgn(signal_dict(np.ones(32)))
    first = awgn.last_noise.copy()
    awgn.add_awgn(signal_dict(np.ones(32)))
    assert awgn.fixed_context.stimulus is None
    assert not np.array_equal(first, awgn.last_noise)


def test_cfo_operation_quantizes_rotation_without_changing_physical_phase():
    params = {"ppm": 10, "fc": 1e9, "enable_phase_noise": False,
              "fixed_point": operation_config("channel.cfo")}
    channel = CFO(params)
    samples = np.ones(96, dtype=complex) * (0.25 + 0.5j)
    actual = channel.apply_cfo(signal_dict(samples, sample_rate=1e6))["signal_stream"]
    expected = samples * np.exp(1j * 2 * np.pi * 1e4 * np.arange(96) / 1e6)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert "operation" in channel.fixed_context.coverage["channel.cfo"]["levels"]


def test_adc_only_quantizes_receive_boundary_and_can_be_disabled():
    params = PHYParams()
    params.update(enable_awgn=False, enable_multipath=False, enable_cfo=False,
                  enable_iq_imbalance=False, enable_pa=False,
                  fixed_point={"enable_fixed": True, "default_format": {
                      "word_length": 8, "fractional_length": 4}})
    channel = THzChannel(params)
    data = signal_dict([0.1, 0.2, 10.0])
    actual = channel.run(data)["signal_stream"]
    np.testing.assert_array_equal(actual, [0.125, 0.1875, 7.9375])
    np.testing.assert_array_equal(data["signal_stream"], [0.1, 0.2, 10.0])
    disabled = FixedPointContext({"enable_fixed": True,
                                 "modules": {"channel.rx_samples": {"enabled": False}}})
    channel.configure_fixed(True, fixed_context=disabled)
    np.testing.assert_array_equal(channel.run(data)["signal_stream"], data["signal_stream"])


def test_mimo_noise_replay_keeps_later_phase_noise_realization():
    params = {
        "num_tx": 2, "num_rx": 2, "num_spatial_streams": 2,
        "enable_mimo": True, "link_mode": "ofdm", "subwave_num": 16,
        "gi_length": 4, "enable_multipath": False, "mimo_channel_model": "identity",
        "enable_awgn": True, "SNRdB": 14, "random_seed": 922,
        "enable_phase_noise": True, "phase_noise_std": 0.01,
    }
    floating = MIMOChannel(params)
    fixed = MIMOChannel({**params, "fixed_point": {"enable_fixed": True}})
    stimulus = {}
    floating.fixed_context.stimulus = stimulus
    fixed.fixed_context.stimulus = stimulus
    samples = np.ones((2, 48), dtype=complex)
    golden = floating.apply(signal_dict(samples))["signal_stream"]
    replay = fixed.apply(signal_dict(samples))["signal_stream"]
    np.testing.assert_array_equal(golden, replay)
    np.testing.assert_array_equal(floating.rng.standard_normal(10), fixed.rng.standard_normal(10))


def test_mimo_operation_uses_integer_convolution_and_wide_path_accumulation():
    params = {
        "num_tx": 2, "num_rx": 2, "num_spatial_streams": 2,
        "enable_mimo": True, "link_mode": "ofdm", "subwave_num": 16,
        "gi_length": 4, "enable_multipath": True,
        "fixed_point": operation_config("channel.mimo"),
    }
    taps = np.array([[[1.0, 0.25], [0.5, 0.0]], [[0.0, 0.5], [1.0, -0.25]]], complex)
    channel = MIMOChannel(params, taps)
    samples = np.array([[0.5, 1, -1, 0], [1, -0.5, 0.5, 1]], complex)
    actual = channel.apply(signal_dict(samples))["signal_stream"]
    expected = np.stack([
        sum(np.convolve(samples[t], taps[r, t])[:4] for t in range(2))
        for r in range(2)
    ])
    np.testing.assert_array_equal(actual, expected)
    assert "operation" in channel.fixed_context.coverage["channel.mimo"]["levels"]
