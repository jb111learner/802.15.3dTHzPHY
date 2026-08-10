import numpy as np

from params.PHYParams import PHYParams
from receiver.ChannelEstimator import ChannelEstimator
from receiver.Equalizer import FreqDomainEqualizer
from receiver.THzReceiver import THzReceiver
from transmitter.PreambleInsertor import PreambleInsertor


class _FakeTransmitter:
    def __init__(self, params):
        self.params = params
        self.preamble = np.zeros(8, dtype=np.complex128)


class _ChannelEstimatorTransmitter:
    def __init__(self, params):
        self.params = params
        self.preamble_gen = PreambleInsertor(params)
        self.preamble = self.preamble_gen.preamble
        self.sync = self.preamble_gen.sync
        self.sfd = self.preamble_gen.sfd
        self.ces = self.preamble_gen.ces


def test_measured_cir_search_preserves_negative_delay_in_frequency_response():
    params = PHYParams()
    params.update(
        link_mode="sc-fde",
        gi_length=32,
        subframe_length=480,
        subframe_num=51,
        Preamble_type="short",
        multipath_source="measured",
        measured_channel_scenario="50cm",
    )
    transmitter = _ChannelEstimatorTransmitter(params)
    estimator = ChannelEstimator(transmitter)
    channel = np.array([1.0, 0.4j, 0.2], dtype=np.complex128)
    received = np.convolve(transmitter.ces, channel)
    # 接收帧起点晚了两个符号，因此主抽头应表现为 lag=-2。
    received = received[2:2 + len(transmitter.ces)]

    response, window_lag, _ = estimator.ces_corr_estimate(
        received, transmitter.ces
    )
    circular_cir = np.fft.ifft(np.fft.ifftshift(response))

    assert window_lag < 0
    assert int(np.argmax(np.abs(circular_cir))) == params.get("subframe_length") - 2


def test_zf_reports_post_equalization_noise_variance():
    params = PHYParams()
    params.update(
        link_mode="sc-fde",
        subframe_length=8,
        subframe_num=1,
        gi_length=2,
        gi_type="cp",
        equalizer_method="zf",
    )
    equalizer = FreqDomainEqualizer(_FakeTransmitter(params))
    data = np.arange(8, dtype=float).astype(np.complex128)
    frame = np.concatenate((data[-2:], data))

    _, post_noise_var, diagnostics = equalizer._equalize_frame(
        frame,
        H=np.full(8, 2.0 + 0.0j),
        N0=0.04,
    )

    np.testing.assert_allclose(post_noise_var, 0.01, rtol=1e-12)
    assert diagnostics["residual_isi_var"] < 1e-15


class _RecordingDemodulator:
    def __init__(self):
        self.sigma = None

    def demodulate(self, signal_dict, sigma):
        self.sigma = sigma
        return signal_dict


def test_receiver_demodulator_prefers_post_equalization_noise_variance():
    receiver = THzReceiver.__new__(THzReceiver)
    receiver.noise_var = 0.8
    receiver.demodulator = _RecordingDemodulator()
    signal = {
        "signal_stream": np.ones(4, dtype=np.complex128),
        "sample_rate_Hz": 1.0,
        "signal_length": 4,
        "duration_seconds": 4.0,
        "padding_bit_num": 0,
        "noise_var": 0.4,
        "post_equalization_noise_var": 0.02,
    }

    receiver.demodulate(signal)

    np.testing.assert_allclose(receiver.demodulator.sigma, 0.1, rtol=1e-12)
