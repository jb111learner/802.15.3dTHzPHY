import sys
import types

import numpy as np
import pytest

from receiver.IQCompensator import IQCompensator


class _Params:
    def __init__(self, **values):
        self.values = values

    def get(self, key):
        return self.values[key]


class _Transmitter:
    def __init__(self, frame_symbol_num=24, frame_num=2):
        self.params = _Params(subframe_length=999, gi_length=99, subframe_num=99)
        self.sync = np.array([1, -1, 1, -1], dtype=complex)
        self.sfd = np.array([1j, -1j], dtype=complex)
        self.ces = np.array(
            [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j, 1 + 0j, 0 + 1j],
            dtype=complex,
        )
        self.preamble = np.concatenate([self.sync, self.sfd, self.ces])
        self.data_with_preamble_dict = {
            "signal_stream": np.zeros(frame_symbol_num * frame_num, dtype=complex),
            "sample_rate_Hz": 1.0,
            "duration_seconds": float(frame_symbol_num * frame_num),
            "signal_length": frame_symbol_num * frame_num,
            "padding_bit_num": 0,
            "frame_symbol_num": frame_symbol_num,
            "frame_num": frame_num,
        }


def _identity_filters(_rx_ces):
    return {
        "f1": np.array([1.0]),
        "f2": np.array([0.0]),
        "f3": np.array([0.0]),
        "f4": np.array([1.0]),
        "cI": 0.0,
        "cQ": 0.0,
        "residual_norm_I": 0.0,
        "residual_norm_Q": 0.0,
        "design_condition_number": 1.0,
    }


def test_compensator_uses_transmitter_frame_metadata_and_preserves_contract():
    transmitter = _Transmitter(frame_symbol_num=24, frame_num=2)
    compensator = IQCompensator(transmitter, filter_len=1)
    compensator._estimate_and_design_filters = _identity_filters

    signal = np.arange(48, dtype=float) + 1j * np.arange(48, dtype=float)[::-1]
    data = {
        "signal_stream": signal,
        "sample_rate_Hz": 2.0,
        "duration_seconds": 24.0,
        "signal_length": 48,
        "padding_bit_num": 7,
        "upstream_marker": "keep",
    }

    result = compensator.compensate(data)

    assert compensator.frame_symbol_num == 24
    assert result["frame_symbol_num"] == 24
    assert result["frame_num"] == 2
    assert result["signal_length"] == len(result["signal_stream"]) == 48
    assert result["duration_seconds"] == 24.0
    assert result["padding_bit_num"] == 7
    assert result["upstream_marker"] == "keep"
    assert len(result["iq_compensation_filters"]) == 2
    np.testing.assert_allclose(result["signal_stream"], signal)


def test_compensator_rejects_non_integral_frames_in_chain_mode():
    transmitter = _Transmitter(frame_symbol_num=24, frame_num=2)
    compensator = IQCompensator(transmitter, filter_len=1)
    data = {
        "signal_stream": np.zeros(47, dtype=complex),
        "sample_rate_Hz": 1.0,
        "duration_seconds": 47.0,
        "signal_length": 47,
        "padding_bit_num": 0,
    }

    with pytest.raises(ValueError, match="不是帧长"):
        compensator.compensate(data, tail_mode="error")


def test_compensator_discard_mode_reports_no_complete_frame_cleanly():
    transmitter = _Transmitter(frame_symbol_num=24, frame_num=2)
    compensator = IQCompensator(transmitter, filter_len=1)
    data = {
        "signal_stream": np.zeros(10, dtype=complex),
        "sample_rate_Hz": 1.0,
        "duration_seconds": 10.0,
        "signal_length": 10,
        "padding_bit_num": 0,
    }

    with pytest.raises(ValueError, match="不包含完整帧"):
        compensator.compensate(data, tail_mode="discard")


def _import_receiver_without_optional_galois():
    if "galois" not in sys.modules:
        module = types.ModuleType("galois")
        module.FieldArray = np.ndarray
        module.ReedSolomonError = Exception
        sys.modules["galois"] = module
    from receiver.THzReceiver import THzReceiver

    return THzReceiver


def test_receiver_iq_stage_bypasses_or_caches_compensated_output():
    receiver_class = _import_receiver_without_optional_galois()
    receiver = receiver_class.__new__(receiver_class)
    signal = {"signal_stream": np.array([1 + 1j])}

    receiver.iq_compensator = None
    assert receiver.compensate_iq_imbalance(signal) is signal

    class _Compensator:
        def compensate(self, data, tail_mode):
            assert data is signal
            assert tail_mode == "error"
            return {**data, "iq_compensation_filters": ["estimated"]}

    receiver.iq_compensator = _Compensator()
    result = receiver.compensate_iq_imbalance(signal)
    assert result["iq_compensation_filters"] == ["estimated"]
    assert receiver.rx_iq_compensated is result


@pytest.mark.parametrize(
    ("link_mode", "branch"),
    [("ofdm", ["ofdm_demodulate"]), ("sc-fde", ["estimate_channel", "equalize"])],
)
def test_receiver_places_iq_compensation_before_noise_and_mode_branch(link_mode, branch):
    receiver_class = _import_receiver_without_optional_galois()
    receiver = receiver_class.__new__(receiver_class)
    receiver.link_mode = link_mode
    receiver.enable_channel_est = True
    calls = []

    stages = [
        "matched_filter",
        "coarse_sync_detect",
        "compensate_cfo_coarse",
        "fine_sync_frame",
        "downsample",
        "compensate_cfo_fine",
        "compensate_iq_imbalance",
        "compensate_iq_decision_directed",
        "ofdm_demodulate",
        "estimate_channel",
        "equalize",
        "demodulate",
        "decode",
    ]
    for stage in stages:
        setattr(
            receiver,
            stage,
            lambda signal, stage=stage: calls.append(stage) or signal,
        )
    receiver.estimate_noise = lambda signal: calls.append("estimate_noise")

    receiver.run({"signal_stream": np.array([0j])})

    expected = [
        "matched_filter",
        "coarse_sync_detect",
        "compensate_cfo_coarse",
        "fine_sync_frame",
        "downsample",
        "compensate_cfo_fine",
        "compensate_iq_imbalance",
        "estimate_noise",
        *branch,
        "compensate_iq_decision_directed",
        "demodulate",
        "decode",
    ]
    assert calls == expected


def test_receiver_skips_decision_directed_iq_without_channel_equalization():
    receiver_class = _import_receiver_without_optional_galois()
    receiver = receiver_class.__new__(receiver_class)
    receiver.link_mode = "sc-fde"
    receiver.enable_channel_est = False
    calls = []
    passthrough_stages = [
        "matched_filter", "coarse_sync_detect", "compensate_cfo_coarse",
        "fine_sync_frame", "downsample", "compensate_cfo_fine",
        "compensate_iq_imbalance", "demodulate", "decode",
    ]
    for stage in passthrough_stages:
        setattr(
            receiver, stage,
            lambda signal, stage=stage: calls.append(stage) or signal,
        )
    receiver.compensate_iq_decision_directed = (
        lambda signal: calls.append("compensate_iq_decision_directed") or signal
    )
    receiver.estimate_noise = lambda signal: calls.append("estimate_noise")

    receiver.run({"signal_stream": np.array([0j])})

    assert "compensate_iq_decision_directed" not in calls
