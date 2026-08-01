import numpy as np

from channel.THzChannel import THzChannel
from params.PHYParams import PHYParams


def _signal_dict():
    signal = np.array([0.1 + 0.2j, -0.4 + 0.3j, 0.8 - 0.1j], dtype=complex)
    return {
        "signal_stream": signal,
        "sample_rate_Hz": 3.0,
        "duration_seconds": 1.0,
        "signal_length": len(signal),
        "padding_bit_num": 0,
        "upstream_marker": "keep",
    }


def _pa_params(enabled):
    params = PHYParams()
    params.update(
        enable_awgn=False,
        enable_multipath=False,
        enable_cfo=False,
        enable_iq_imbalance=False,
        enable_pa=enabled,
        pa_load_params_from_dataset=False,
        pa_auto_input_scaling=False,
    )
    return params


def test_pa_switch_off_bypasses_signal():
    data = _signal_dict()
    result = THzChannel(_pa_params(False)).run(data)

    assert result is data
    np.testing.assert_array_equal(result["signal_stream"], data["signal_stream"])


def test_pa_switch_on_processes_signal_and_preserves_metadata():
    data = _signal_dict()
    channel = THzChannel(_pa_params(True))
    result = channel.run(data)

    assert result is not data
    assert result["upstream_marker"] == "keep"
    assert result["signal_length"] == len(result["signal_stream"])
    assert not np.allclose(result["signal_stream"], data["signal_stream"])
    assert result["pa_diagnostics"]["pa_model"] == "modified_rapp"
    assert channel.pa_diagnostics == result["pa_diagnostics"]


def test_pa_can_load_default_300_ghz_dataset_from_module_directory():
    params = _pa_params(True)
    params.update(pa_load_params_from_dataset=True, pa_fc_Hz=300e9)

    channel = THzChannel(params)

    assert np.isclose(channel.power_amplifier.G, 11.2616039517771)
    assert np.isclose(channel.power_amplifier.q1, 1.73436662122553)
