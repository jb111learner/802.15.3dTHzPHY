import numpy as np
import pytest

from channel.MultipathChannel import MultipathChannel
from params.PHYParams import PHYParams


class _Params:
    def __init__(self, **values):
        self.values = {
            "tdl_model": None,
            "tdl_delay_spread_ns": 100.0,
            "tdl_velocity_mps": 0.0,
            "fc": 1e12,
            "random_seed": 1234,
            "enable_multipath_memory": False,
        }
        self.values.update(values)

    def get(self, key):
        if key not in self.values:
            raise KeyError(key)
        return self.values[key]


def _signal_dict(length=4096, sample_rate_hz=1e9):
    signal = np.ones(length, dtype=complex)
    return {
        "signal_stream": signal,
        "sample_rate_Hz": sample_rate_hz,
        "duration_seconds": length / sample_rate_hz,
        "signal_length": length,
        "padding_bit_num": 0,
    }


@pytest.mark.parametrize(
    "model, expected_paths",
    [("TDL-A", 23), ("TDL-B", 23), ("TDL-C", 24), ("TDL-D", 13), ("TDL-E", 14)],
)
def test_tdl_profiles_have_protocol_path_counts_and_normalized_average_power(
    model, expected_paths
):
    channel = MultipathChannel(_Params(tdl_model=model))
    result = channel.apply(_signal_dict())

    assert result["tdl_enabled"] is True
    assert result["tdl_model"] == model
    assert len(result["multipath_paths_resolved"]) == expected_paths
    assert sum(path["power"] for path in result["multipath_paths_resolved"]) == pytest.approx(1.0)
    assert all(path["gain_type"] == "static" for path in result["multipath_paths_resolved"])


def test_tdl_delay_spread_scales_normalized_delays_and_uses_integer_application():
    channel = MultipathChannel(_Params(tdl_model="TDL-A", tdl_delay_spread_ns=30.0))
    result = channel.apply(_signal_dict(sample_rate_hz=1e9))
    paths = result["multipath_paths_resolved"]
    path = next(path for path in paths if path["tdl_normalized_delay"] == pytest.approx(0.3819))

    assert path["delay_samples"] == pytest.approx(0.3819 * 30.0)
    assert channel.use_frac_delay is False
    assert np.count_nonzero(np.abs(result["chan_impulse"])) > 0


@pytest.mark.parametrize("model", ["TDL-D", "TDL-E"])
def test_tdl_los_first_tap_is_one_combined_rician_path(model):
    channel = MultipathChannel(_Params(tdl_model=model))
    result = channel.apply(_signal_dict())
    first = result["multipath_paths_resolved"][0]

    assert first["tdl_fading_type"] == "rician"
    assert first["tdl_normalized_delay"] == 0.0
    assert first["K"] > 1.0
    assert sum(path["tdl_fading_type"] == "rician" for path in result["multipath_paths_resolved"]) == 1


def test_tdl_static_realization_is_reused_across_apply_calls():
    params = _Params(tdl_model="TDL-C")
    channel = MultipathChannel(params)
    first = channel.apply(_signal_dict(length=1024))
    second = channel.apply(_signal_dict(length=1024))

    np.testing.assert_array_equal(first["signal_stream"], second["signal_stream"])
    np.testing.assert_array_equal(first["chan_impulse"], second["chan_impulse"])


def test_tdl_static_realization_restarts_identically_after_reset():
    channel = MultipathChannel(_Params(tdl_model="TDL-A"))
    first = channel.apply(_signal_dict(length=1024))
    channel.reset_state()
    second = channel.apply(_signal_dict(length=1024))

    np.testing.assert_array_equal(first["signal_stream"], second["signal_stream"])
    np.testing.assert_array_equal(first["chan_impulse"], second["chan_impulse"])


def test_tdl_jakes_is_time_correlated_and_continuous_across_frames():
    values = {
        "tdl_model": "TDL-C",
        "tdl_velocity_mps": 30.0,
        "fc": 100e9,
        "enable_multipath_memory": True,
    }
    channel = MultipathChannel(_Params(**values))
    first_input = _signal_dict(length=1024, sample_rate_hz=1e6)
    second_input = _signal_dict(length=1024, sample_rate_hz=1e6)
    first = channel.apply(first_input)
    first_trace = channel._last_tdl_gain_traces[0].copy()
    second = channel.apply(second_input)
    second_trace = channel._last_tdl_gain_traces[0].copy()

    assert first["tdl_fading_time_varying"] is True
    assert first["tdl_max_doppler_hz"] == pytest.approx(30.0 * 100e9 / 299792458.0)
    assert first["tdl_los_doppler_hz"] == pytest.approx(0.7 * first["tdl_max_doppler_hz"])
    adjacent_correlation = np.corrcoef(first_trace[:-1].real, first_trace[1:].real)[0, 1]
    assert adjacent_correlation > 0.5

    reference = MultipathChannel(_Params(**values))
    reference_result = reference.apply(
        _signal_dict(length=2048, sample_rate_hz=1e6)
    )
    reference_trace = reference._last_tdl_gain_traces[0]
    np.testing.assert_allclose(
        np.concatenate([first["signal_stream"], second["signal_stream"]]),
        reference_result["signal_stream"],
    )
    np.testing.assert_allclose(
        np.concatenate([first_trace, second_trace]), reference_trace
    )

    channel.reset_state()
    reset_result = channel.apply(_signal_dict(length=1024, sample_rate_hz=1e6))
    np.testing.assert_array_equal(reset_result["signal_stream"], first["signal_stream"])


def test_tdl_model_none_keeps_legacy_custom_path_fallback():
    channel = MultipathChannel(
        _Params(
            tdl_model=None,
            multipath_paths=[
                {"delay_samples": 0, "gain_type": "static", "gain": 1.0 + 0j},
                {"delay_samples": 2, "gain_type": "static", "gain": 0.5 + 0j},
            ],
        )
    )
    result = channel.apply(_signal_dict(length=16))

    assert result["tdl_enabled"] is False
    assert result["tdl_model"] is None
    assert len(result["multipath_paths_resolved"]) == 2
    assert result["signal_stream"][0] == pytest.approx(1.0 + 0j)
    assert result["signal_stream"][2] == pytest.approx(1.5 + 0j)


def test_phy_params_exposes_only_modern_tdl_selection_for_multipath_defaults():
    params = PHYParams()

    assert params.get("tdl_model") == "TDL-A"
    assert params.get("tdl_delay_spread_ns") == pytest.approx(100.0)
    with pytest.raises(KeyError):
        params.get("multipath_paths")


def test_phy_params_accepts_explicit_legacy_multipath_keys_without_default_clutter():
    params = PHYParams()
    params.update(
        tdl_model=None,
        multipath_paths=[
            {"delay_samples": 0, "gain_type": "static", "gain": 1.0 + 0j},
            {"delay_samples": 3, "gain_type": "static", "gain": 0.25 + 0j},
        ],
    )
    result = MultipathChannel(params).apply(_signal_dict(length=16))

    assert result["tdl_enabled"] is False
    assert len(result["multipath_paths_resolved"]) == 2
    assert params.get("multipath_paths")[1]["delay_samples"] == 3
