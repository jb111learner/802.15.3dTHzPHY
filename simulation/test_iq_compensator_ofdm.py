import numpy as np

from channel.IQImbalance import IQImbalance
from receiver.IQCompensatorOFDM import (
    BlindRXIQWhiteningCompensator,
    ConfiguredRXIQCompensator,
    OFDMWidelyLinearIQCompensator,
    resolve_iq_compensation_method,
)


class _Params:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def _signal_dict(signal):
    return {
        "signal_stream": signal,
        "sample_rate_Hz": 1.0,
        "duration_seconds": float(len(signal)),
        "signal_length": len(signal),
        "padding_bit_num": 0,
    }


def test_configured_rx_fid_inverse_recovers_normalized_impairment():
    params = _Params(
        enable_iq_imbalance=True,
        iq_imbalance_position="rx",
        iq_imbalance_model="fid",
        iq_power_normalize=True,
        rx_iq_gain_imbalance_db=2.0,
        rx_iq_phase_imbalance_deg=5.0,
    )
    rng = np.random.default_rng(12)
    reference = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    _, _, mu, nu = IQImbalance.compute_fid_coefficients(2.0, 5.0)
    scale = 1.0 / np.sqrt(abs(mu) ** 2 + abs(nu) ** 2)
    impaired = scale * (mu * reference + nu * np.conj(reference))

    result = ConfiguredRXIQCompensator(params).compensate(
        _signal_dict(impaired)
    )

    np.testing.assert_allclose(result["signal_stream"], reference, atol=1e-12)
    assert result["iq_compensation_method"] == "configured_inverse"


def test_blind_rx_whitening_removes_second_order_iq_correlation():
    params = _Params(
        enable_iq_imbalance=True,
        iq_imbalance_position="rx",
    )
    rng = np.random.default_rng(21)
    reference = rng.normal(size=50000) + 1j * rng.normal(size=50000)
    impaired = 1.18 * reference.real + 1j * (
        0.82 * reference.imag + 0.12 * reference.real
    )

    result = BlindRXIQWhiteningCompensator(params).compensate(
        _signal_dict(impaired)
    )
    corrected = result["signal_stream"]
    covariance_start = int(0.2 * len(corrected))
    covariance = np.cov(
        np.vstack((
            corrected.real[covariance_start:],
            corrected.imag[covariance_start:],
        )),
        bias=True,
    )

    assert abs(covariance[0, 1]) < 1e-10
    assert abs(covariance[0, 0] - covariance[1, 1]) < 1e-10


def test_ofdm_widely_linear_estimates_mirror_pairs_and_recovers_grid():
    rng = np.random.default_rng(4)
    n_subcarriers = 64
    n_symbols = 8
    frame_num = 2
    pilot_indexes = [0, 3, 6]
    base_pilot = rng.choice([-1.0, 1.0], n_subcarriers).astype(complex)
    phase_codes = np.array([1.0, 1j, 1.0])
    pilot_sequences = phase_codes[:, None] * base_pilot[None, :]
    params = _Params(
        iq_ofdm_response_taps=0,
        iq_compensation_mode="per_frame",
        iq_ofdm_estimation_ridge=0.0,
        iq_ofdm_equalizer_ridge=0.0,
        iq_ofdm_condition_limit=1e8,
    )
    compensator = OFDMWidelyLinearIQCompensator(
        params, pilot_indexes, pilot_sequences, n_subcarriers
    )

    transmitted = (
        rng.normal(size=(n_subcarriers, n_symbols * frame_num))
        + 1j * rng.normal(size=(n_subcarriers, n_symbols * frame_num))
    ) / np.sqrt(2)
    for pilot_order, symbol_index in enumerate(pilot_indexes):
        for frame_index in range(frame_num):
            transmitted[:, frame_index * n_symbols + symbol_index] = (
                pilot_sequences[pilot_order]
            )

    direct = np.fft.fft(np.array([0.9 + 0.1j, 0.15 - 0.05j]), n_subcarriers)
    image = np.fft.fft(np.array([0.1 - 0.04j, 0.02 + 0.01j]), n_subcarriers)
    mirror = (-np.arange(n_subcarriers)) % n_subcarriers
    received = (
        direct[:, None] * transmitted
        + image[:, None] * np.conj(transmitted[mirror])
    )

    recovered, direct_est, image_est, diagnostics = compensator.compensate(
        received, frame_num
    )

    np.testing.assert_allclose(recovered, transmitted, atol=1e-11)
    np.testing.assert_allclose(direct_est[0], direct, atol=1e-11)
    np.testing.assert_allclose(image_est[0], image, atol=1e-11)
    assert diagnostics["frames"][0]["fallback_pair_count"] == 0


def test_method_resolution_routes_legacy_ofdm_dd_without_affecting_sc_fde():
    enabled_dd = _Params(
        enable_iq_compensation=True,
        iq_compensation_method="decision_directed",
    )
    disabled = _Params(
        enable_iq_compensation=False,
        iq_compensation_method="auto",
    )

    assert resolve_iq_compensation_method(enabled_dd, "ofdm") == (
        "ofdm_widely_linear"
    )
    assert resolve_iq_compensation_method(enabled_dd, "sc-fde") == (
        "decision_directed"
    )
    assert resolve_iq_compensation_method(disabled, "ofdm") == "none"
