import numpy as np
import pytest

from thz_sim_ui.services.backend import BackendService
from channel.MeasuredChannel import MeasuredChannel
from channel.THzChannel import THzChannel
from params.PHYParams import PHYParams
from receiver.THzReceiver import THzReceiver
from simulation.SimulationManager import SimulationManager
from transmitter.THzTransmitter import THzTransmitter


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_measured_cir_resampling_preserves_delay_grid_and_power(factor):
    params = PHYParams()
    params.update(
        multipath_source="measured",
        measured_channel_mode="deterministic",
        measured_channel_scenario="50cm",
        gi_length=32,
        oversampling=factor,
    )
    channel = MeasuredChannel(params)
    source = channel.load_taps(full_mimo=False, sample_rate_hz=30e9)
    resampled = channel.load_taps(
        full_mimo=False, sample_rate_hz=30e9 * factor
    )

    assert resampled.shape[-1] == (source.shape[-1] - 1) * factor + 1
    assert np.allclose(
        np.sum(np.abs(resampled) ** 2, axis=-1),
        np.sum(np.abs(source) ** 2, axis=-1),
        rtol=1e-12,
        atol=1e-12,
    )
    source_peak = int(np.argmax(np.abs(source[0, 0])))
    target_peak = int(np.argmax(np.abs(resampled[0, 0])))
    assert abs(target_peak - source_peak * factor) <= 1
    assert channel.diagnostics["resampling_factor"] == factor


@pytest.mark.parametrize("factor", [2, 4])
def test_frontend_preserves_oversampling_for_measured_channel(factor):
    mapped = BackendService.map_ui_params_to_phy_params({
        "链路模式": "SISO-OFDM",
        "过采样率": f"{factor}x",
        "采样率": 30000.0,
        "_channel_params": {
            "enable_multipath": True,
            "multipath_source": "measured",
            "measured_channel_scenario": "50cm",
        },
    })

    assert mapped["sample_rate"] == 30e9
    assert mapped["oversampling"] == factor


def test_high_rate_cp_validation_uses_physical_delay():
    params = PHYParams()
    params.update(
        multipath_source="measured",
        measured_channel_scenario="50cm",
        gi_length=15,
        oversampling=4,
    )
    params.validate()

    with pytest.raises(ValueError, match="高采样率 CP"):
        params.update(gi_length=14)


@pytest.mark.parametrize("factor", [2, 4])
def test_tx_and_measured_channel_share_high_rate_grid(factor):
    params = PHYParams()
    params.update(
        link_mode="ofdm",
        duration=1e-7,
        sample_length=None,
        sample_rate=30e9,
        oversampling=factor,
        enable_awgn=False,
        enable_multipath=True,
        multipath_source="measured",
        measured_channel_mode="deterministic",
        measured_channel_scenario="50cm",
        gi_length=32,
        enable_cfo=False,
        enable_iq_imbalance=False,
        enable_pa=False,
    )

    tx = THzTransmitter(params)
    tx_signal = tx.run()
    rx_signal = THzChannel(params).run(tx_signal)
    diagnostics = rx_signal["measured_channel_diagnostics"]

    assert tx_signal["sample_rate_Hz"] == 30e9 * factor
    assert tx_signal["base_sample_rate_Hz"] == 30e9
    assert tx_signal["oversampling_factor"] == factor
    assert diagnostics["waveform_sample_rate_Hz"] == 30e9 * factor
    assert diagnostics["resampled_tap_count"] == 15 * factor + 1
    assert diagnostics["cp_samples_high_rate"] == 32 * factor


@pytest.mark.parametrize("factor", [2, 4])
def test_high_rate_measured_ofdm_reaches_equalizer(factor):
    params = PHYParams()
    params.update(
        link_mode="ofdm",
        duration=1e-7,
        sample_length=None,
        sample_rate=30e9,
        oversampling=factor,
        enable_awgn=False,
        enable_multipath=True,
        multipath_source="measured",
        measured_channel_mode="deterministic",
        measured_channel_scenario="50cm",
        gi_length=32,
        enable_cfo=False,
        enable_cfo_compensation=False,
        enable_iq_imbalance=False,
        enable_iq_compensation=False,
        enable_pa=False,
        random_seed=7,
    )
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    rx_signal = THzChannel(params).run(tx_signal)
    receiver = THzReceiver(params, transmitter)
    signal = receiver.matched_filter(rx_signal)
    signal = receiver.coarse_sync_detect(signal)
    signal = receiver.fine_sync_frame(signal)
    signal = receiver.downsample(signal)
    receiver.estimate_noise(signal)
    equalized = receiver.ofdm_demodulate(signal)

    assert tx_signal["sample_rate_Hz"] == 30e9 * factor
    assert signal["sample_rate_Hz"] == 30e9
    assert len(equalized["signal_stream"]) > 0
    assert all(
        len(h) == receiver.rx_ofdm.channel_estimation_taps
        for h in equalized["channel_time_response"]
    )


def test_sc_fde_negative_fine_offset_keeps_full_frame_and_decodes(tmp_path):
    params = PHYParams()
    params.update(
        link_mode="sc-fde",
        duration=1e-7,
        sample_length=None,
        sample_rate=30e9,
        oversampling=2,
        NCBPS=4,
        code_type="LDPC",
        ldpc_standard_rate="14/15",
        enable_awgn=True,
        SNRdB=24.0,
        enable_multipath=True,
        multipath_source="measured",
        measured_channel_mode="pdp_rayleigh",
        measured_channel_scenario="50cm",
        measured_channel_retained_power=0.95,
        gi_length=32,
        enable_cfo=False,
        enable_cfo_compensation=False,
        random_seed=0,
    )
    manager = SimulationManager(
        base_params=params, output_dir=str(tmp_path), save_plots=False
    )

    result = manager.run_once()

    assert result["metrics"]["ber"] == 0.0
