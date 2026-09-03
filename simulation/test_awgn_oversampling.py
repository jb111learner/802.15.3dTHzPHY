"""AWGN 带内 SNR 与过采样率一致性的回归测试。"""

import numpy as np

from channel.AWGN import AWGN
from transmitter.Pulseshaper import TxPulseShaper


def _signal_dict(signal, sample_rate, base_sample_rate):
    return {
        "signal_stream": signal,
        "sample_rate_Hz": sample_rate,
        "base_sample_rate_Hz": base_sample_rate,
        "duration_seconds": len(signal) / sample_rate,
        "signal_length": len(signal),
        "padding_bit_num": 0,
    }


def test_snr_mode_scales_discrete_noise_by_sample_rate_over_bandwidth():
    signal = np.ones(4096, dtype=np.complex128)
    params = {
        "SNRdB": 20.0,
        "bandwidth": 30e9,
        "noise_temperature": None,
        "noise_figure_db": None,
    }

    result = AWGN(params).add_awgn(_signal_dict(signal, 120e9, 30e9))

    # 4 倍过采样：全采样带宽内的噪声方差是带内噪声功率的 4 倍。
    np.testing.assert_allclose(result["noise_power"], 4.0 / 100.0)
    np.testing.assert_allclose(result["SNRdB"], 20.0)
    np.testing.assert_allclose(
        result["sample_SNRdB"], 20.0 - 10.0 * np.log10(4.0)
    )
    assert result["base_sample_rate_Hz"] == 30e9


def test_matched_filter_snr_is_invariant_to_sc_oversampling():
    rng = np.random.default_rng(2026)
    symbol_count = 40000
    symbols = (
        rng.choice((-1.0, 1.0), symbol_count)
        + 1j * rng.choice((-1.0, 1.0), symbol_count)
    ) / np.sqrt(2.0)
    target_snr_db = 12.0
    measured = []

    for sps in (2, 4, 8):
        params = {
            "oversampling": sps,
            "rolloff": 0.22,
            "filter_type": "rrc",
            "filter_length": 32,
            "link_mode": "sc-fde",
            "SNRdB": target_snr_db,
            "bandwidth": 30e9,
            "noise_temperature": None,
            "noise_figure_db": None,
        }
        shaper = TxPulseShaper(params)
        shaped_dict = shaper.shape_pulse(
            _signal_dict(symbols, 30e9, 30e9)
        )

        np.random.seed(100 + sps)
        noisy = AWGN(params).add_awgn(shaped_dict)["signal_stream"]
        clean = shaped_dict["signal_stream"]
        matched = np.conj(shaper.filter_coeffs[::-1])
        delay = (len(matched) - 1) // 2
        clean_symbols = np.convolve(clean, matched, mode="full")[delay::sps]
        noisy_symbols = np.convolve(noisy, matched, mode="full")[delay::sps]

        # 跳过有限长成形滤波器的边缘瞬态。
        guard = 2 * params["filter_length"]
        clean_symbols = clean_symbols[guard:-guard]
        error_symbols = noisy_symbols[guard:-guard] - clean_symbols
        measured.append(
            10.0
            * np.log10(
                np.mean(np.abs(clean_symbols) ** 2)
                / np.mean(np.abs(error_symbols) ** 2)
            )
        )

    np.testing.assert_allclose(measured, target_snr_db, atol=0.15)
    assert max(measured) - min(measured) < 0.15
