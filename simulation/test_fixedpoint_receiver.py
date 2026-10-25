"""接收端格式传播、golden 保持和 LLR 类型的回归验证。"""

from types import SimpleNamespace
import numpy as np
import pytest

from receiver.Decoder import Decoder
from receiver.Equalizer import FreqDomainEqualizer
from receiver.MIMODetector import MIMODetector
from receiver.RxOFDMProcesser import RxOFDMProcesser
from utils.Coder import LDPCCoder


def _equalizer(fixed=False, range_enabled=False):
    params = {
        "subframe_length": 480, "subframe_num": 1, "gi_length": 32,
        "gi_type": "cp", "equalizer_method": "zf",
        "fixed_point": {"enable_fixed": fixed, "modules": {
            "sc.rx.equalizer": {"arithmetic_level": "operation"},
        }},
        "range_analysis": {"enabled": range_enabled},
    }
    return FreqDomainEqualizer(SimpleNamespace(params=params, preamble=np.zeros(8)))


def test_sc_480_operation_preserves_wide_frequency_values():
    data = np.full(480, 0.5 + 0.25j)
    frame = np.concatenate((data[-32:], data))
    fixed = _equalizer(True)
    y, noise, _ = fixed._equalize_frame(frame, np.ones(480, complex), 0.01)
    # A DC FFT bin is 240+120j, which would be destroyed by Q4.12 re-input.
    np.testing.assert_allclose(y, data, atol=0.001)
    assert noise == pytest.approx(0.01, abs=1e-6)
    assert "operation" in fixed.fixed_context.coverage["sc.rx.equalizer"]["levels"]


def test_range_observation_does_not_change_float_equalizer():
    rng = np.random.default_rng(123)
    data = rng.normal(size=480) + 1j * rng.normal(size=480)
    frame = np.concatenate((data[-32:], data))
    a, _, _ = _equalizer()._equalize_frame(frame, np.ones(480, complex), 0.01)
    observed = _equalizer(range_enabled=True)
    b, _, _ = observed._equalize_frame(frame, np.ones(480, complex), 0.01)
    assert np.array_equal(a, b)
    assert observed.fixed_context.range_analyzer.report()


def test_multiframe_noise_mean_is_on_noise_grid():
    fixed = _equalizer(True)
    block = np.concatenate((np.full(32, .5), np.full(480, .5)))
    frame = np.concatenate((np.zeros(8), block))
    samples = np.concatenate((frame, frame))
    result = fixed.equalize({
        "signal_stream": samples, "sample_rate_Hz": 1., "duration_seconds": float(len(samples)),
        "signal_length": len(samples), "padding_bit_num": 0,
        "channel_freq_response": [np.ones(480, complex), np.ones(480, complex)],
        "noise_var": [.01, .0100003],
    })
    scalar = result["noise_var"] * 2**24
    assert scalar == round(scalar)
    assert result["post_equalization_noise_var"] == result["noise_var"]


def _ldpc_params(fixed=False):
    return {
        "code_type": "LDPC", "ldpc_matrix_type": "engineering",
        "ldpc_n": 12, "ldpc_k": 6, "ldpc_col_weight": 2, "ldpc_max_iter": 2,
        "ldpc_seed": 2024, "ldpc_decode_algorithm": "min_sum",
        "ldpc_llr_clip": 20.0, "ldpc_hard_llr_value": 12.0, "ldpc_min_sum_alpha": 0.8,
        "fixed_point": {"enable_fixed": fixed},
    }


def test_explicit_soft_zero_one_llr_does_not_use_hard_fallback():
    coder = LDPCCoder(_ldpc_params())
    _, soft = coder.decode(np.ones(12), input_is_soft=True)
    _, legacy = coder.decode(np.ones(12))
    assert soft["used_hard_fallback"] is False
    assert legacy["used_hard_fallback"] is True


def test_decoder_clips_then_quantizes_and_exposes_block_mapping():
    decoder = Decoder(_ldpc_params(True))
    raw = np.array([50, -50] + [1, 0] * 5, dtype=float)
    result = decoder.decode({
        "signal_stream": raw, "sample_rate_Hz": 12.0, "duration_seconds": 1.0,
        "signal_length": 12, "padding_bit_num": 0, "llr_input_kind": "soft",
    })
    debug = result["ldpc_diagnostics"]
    assert debug["used_hard_fallback"] is False
    assert debug["llr_clipping_count"] == 2
    assert debug["information_block_valid_bits"] == [6]
    assert debug["ldpc_internal_arithmetic"] == "float"
    assert np.array_equal(raw, np.array([50, -50] + [1, 0] * 5))


def test_mimo_integer_zf_matches_diagonal_channel():
    params = {"fixed_point": {"enable_fixed": True, "modules": {
        "mimo.rx.detector": {"arithmetic_level": "operation"},
    }}}
    detector = MIMODetector("zf", params=params)
    channel = np.array([[1, 0], [0, 0.5]], complex)[..., None]
    sent = np.array([[0.5 + 0.25j], [-0.25 + 0.5j]])
    received = (channel[..., 0] @ sent)[..., None]
    detected, diagnostics = detector.detect(received, channel, 0.01)
    np.testing.assert_allclose(detected[..., 0], sent, atol=0.001)
    assert diagnostics["singular_matrix_policy"] == "error"


def test_mimo_integer_rank_deficiency_has_no_float_fallback():
    params = {"fixed_point": {"enable_fixed": True, "modules": {
        "mimo.rx.detector": {"arithmetic_level": "operation"},
    }}}
    detector = MIMODetector("zf", params=params)
    with pytest.raises(np.linalg.LinAlgError):
        detector.detect(np.ones((2, 1, 1), complex), np.ones((2, 2, 1), complex))


def test_ofdm_phase_ramp_uses_integer_symbol_index():
    # Symbol index 47 must not be clipped to a Q4.12 signal value near 8.
    params = {
        "enable_channel_equalization": True, "subwave_num": 8,
        "subframe_ofdm_num": 48, "pilot_block_indexes": [0, 16, 32],
        "gi_length": 2, "oversampling": 1, "enable_iq_compensation": False,
        "fixed_point": {"enable_fixed": True, "modules": {
            "ofdm.rx.equalizer": {"arithmetic_level": "operation"},
            "ofdm.rx.channel_estimation": {"enabled": False},
        }},
    }
    tx = SimpleNamespace(params=params, preamble=np.zeros(8),
                         ofdm_processer=SimpleNamespace(pilot_seq=np.ones(8)))
    rx = RxOFDMProcesser(tx)
    rx.frame_num = 1
    phase = 0.01 * np.arange(48)
    grid = np.ones((8, 48), complex) * np.exp(1j * phase)[None, :]
    output, _, _ = rx._ls_track_and_equalize(grid, [np.ones(8)])
    np.testing.assert_allclose(output[:, 47], np.ones(8), atol=0.003)


def test_paired_sc_cfo_enabled_operation_smoke():
    from params.PHYParams import PHYParams
    from simulation.fixedpoint_comparison import run_float_fixed_comparison

    params = PHYParams()
    params.update(
        link_mode="sc-fde", duration=None, sample_length=240,
        NCBPS=2, subframe_length=480, subframe_num=1, gi_length=32,
        oversampling=2, filter_type="rrc", filter_length=8, code_type="LDPC",
        ldpc_matrix_type="engineering", ldpc_n=96, ldpc_k=48,
        ldpc_col_weight=3, ldpc_max_iter=10,
        enable_awgn=True, SNRdB=40, enable_multipath=False,
        enable_iq_compensation=False, enable_iq_imbalance=False,
        enable_cfo=True, enable_cfo_compensation=True, ppm=1.0,
        enable_phase_noise=False,
    )
    fixed = {"enable_fixed": True, "modules": {
        "tx": {"enabled": False}, "channel": {"enabled": False},
        "sc": {"enabled": False},
        "sc.rx": {"enabled": True, "arithmetic_level": "operation"},
        "rx": {"enabled": False},
        "rx.noise_estimator": {"enabled": True, "arithmetic_level": "operation"},
        "ldpc.llr": {"enabled": True},
    }}
    result = run_float_fixed_comparison(params, fixed, {"seed": 31013})
    assert result["phy_params"]["ppm"] * 1e-6 * result["phy_params"]["fc"] > 0
    assert result["stimulus"]["noise_sha256"]
    assert result["float"]["ber"] == 0
    assert result["fixed"]["ber"] == 0
    assert not result["fixed"]["length_mismatch"]
    coverage = result["fixed_report"]["coverage"]
    for module in ("sc.rx.cfo", "sc.rx.coarse_sync", "sc.rx.fine_sync",
                   "sc.rx.channel_estimation", "sc.rx.equalizer", "rx.noise_estimator"):
        assert "operation" in coverage[module]["levels"]
    assert result["nmse"]["rx.equalizer"]["value"] < 1e-3
