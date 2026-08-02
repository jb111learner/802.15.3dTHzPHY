import numpy as np
import pytest

from channel.MIMOChannel import MIMOChannel
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

