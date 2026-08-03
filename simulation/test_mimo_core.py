import numpy as np
import pytest

from channel.MIMOChannel import MIMOChannel
from channel.MeasuredChannel import MeasuredChannel
from params.PHYParams import PHYParams
from receiver.MIMOChannelEstimator import MIMOChannelEstimator
from receiver.MIMODetector import MIMODetector
from receiver.MIMOLayerDemapper import MIMOLayerDemapper
from transmitter.MIMOLayerMapper import MIMOLayerMapper


def _mimo_params(**overrides):
    params = PHYParams()
    values = dict(
        enable_mimo=True,
        link_mode="ofdm",
        num_tx=2,
        num_rx=2,
        num_spatial_streams=2,
        subwave_num=32,
        gi_length=8,
        mimo_num_taps=1,
        enable_multipath=False,
        enable_awgn=False,
        enable_cfo=False,
        enable_iq_imbalance=False,
    )
    values.update(overrides)
    params.update(**values)
    return params


def test_layer_mapping_round_trip():
    symbols = np.arange(17) + 1j * np.arange(17)[::-1]
    layers = MIMOLayerMapper(2).map(symbols)
    recovered = MIMOLayerDemapper.demap(layers, len(symbols))
    np.testing.assert_array_equal(recovered, symbols)


@pytest.mark.parametrize("method", ["zf", "mmse"])
def test_detector_recovers_full_rank_channel_without_noise(method):
    rng = np.random.default_rng(3)
    transmitted = rng.normal(size=(2, 5, 8)) + 1j * rng.normal(size=(2, 5, 8))
    base_h = np.array([[1.0, 0.35 + 0.1j], [0.2 - 0.1j, 0.9]])
    channel = np.repeat(base_h[:, :, None], 8, axis=2)
    received = np.einsum("rsk,smk->rmk", channel, transmitted)
    detected, diagnostics = MIMODetector(method).detect(received, channel, 0.0)
    np.testing.assert_allclose(detected, transmitted, atol=1e-10)
    assert np.all(np.isfinite(diagnostics["condition_numbers"]))


def test_ls_channel_estimator_and_nmse():
    rng = np.random.default_rng(9)
    channel = rng.normal(size=(2, 2, 16)) + 1j * rng.normal(size=(2, 2, 16))
    pilot = (2 * rng.integers(0, 2, 16) - 1).astype(complex)
    training = channel * pilot[None, None, :]
    estimated = MIMOChannelEstimator.estimate(training, pilot)
    np.testing.assert_allclose(estimated, channel)
    assert MIMOChannelEstimator.nmse(estimated, channel) < 1e-20


def test_mimo_channel_identity_shape_and_values():
    params = _mimo_params(mimo_channel_model="identity")
    identity = np.eye(2, dtype=complex)[:, :, None]
    channel = MIMOChannel(params, identity)
    signal = np.vstack((np.arange(20), np.arange(20)[::-1])).astype(complex)
    result = channel.apply(
        {
            "signal_stream": signal,
            "sample_rate_Hz": 1e6,
            "duration_seconds": 20e-6,
            "signal_length": 20,
            "padding_bit_num": 0,
        }
    )
    assert result["signal_stream"].shape == (2, 20)
    np.testing.assert_allclose(result["signal_stream"], signal)


def test_invalid_stream_count_is_rejected():
    params = PHYParams()
    with pytest.raises(ValueError, match="num_spatial_streams"):
        params.update(
            enable_mimo=True,
            link_mode="ofdm",
            num_tx=2,
            num_rx=2,
            num_spatial_streams=3,
        )


@pytest.mark.parametrize(
    "scenario, expected_length, expected_paths",
    [("8cm", 49, 6), ("12cm", 37, 4), ("50cm", 16, 5)],
)
def test_measured_channel_builds_expected_window_and_dominant_paths(
    scenario, expected_length, expected_paths
):
    channel = MeasuredChannel(
        {
            "measured_channel_scenario": scenario,
            "measured_channel_mode": "deterministic",
            "measured_channel_retained_power": 0.95,
            "random_seed": 7,
        }
    )
    taps = channel.load_taps(full_mimo=True)
    assert taps.shape == (2, 2, expected_length)
    assert len(channel.selected_path_indexes) == expected_paths
    assert np.isclose(np.sum(np.abs(taps) ** 2), 2.0)
    assert channel.diagnostics["retained_pdp_power_ratio"] >= 0.95
    assert channel.diagnostics["raw_in_window_power_ratio"] > 0.80


def test_measured_rayleigh_ensemble_converges_to_selected_pdp():
    channel = MeasuredChannel(
        {
            "measured_channel_scenario": "8cm",
            "measured_channel_mode": "pdp_rayleigh",
            "measured_channel_retained_power": 0.95,
            "random_seed": 19,
        }
    )
    ensemble_pdp = np.zeros(channel.num_effective_taps)
    realizations = 2000
    for _ in range(realizations):
        taps = channel.random_rayleigh_taps(full_mimo=True)
        ensemble_pdp += np.sum(np.abs(taps) ** 2, axis=(0, 1)) / 2.0
    ensemble_pdp /= realizations
    np.testing.assert_allclose(
        ensemble_pdp, channel.selected_pdp, atol=8e-3, rtol=5e-2
    )


def test_measured_siso_selection_is_unit_power_and_reproducible():
    config = {
        "measured_channel_scenario": "12cm",
        "measured_channel_mode": "deterministic",
        "measured_channel_tx_index": 1,
        "measured_channel_rx_index": 0,
        "random_seed": 5,
    }
    first = MeasuredChannel(config).load_taps(full_mimo=False)
    second = MeasuredChannel(config).load_taps(full_mimo=False)
    assert first.shape == (1, 1, 37)
    assert np.isclose(np.sum(np.abs(first) ** 2), 1.0)
    np.testing.assert_array_equal(first, second)
