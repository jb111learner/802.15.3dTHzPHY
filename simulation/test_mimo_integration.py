import numpy as np

from channel.THzChannel import THzChannel
from params.PHYParams import PHYParams
from receiver.THzReceiver import THzReceiver
from transmitter.THzTransmitter import THzTransmitter
from simulation.SimulationManager import SimulationManager


def _integration_params(**overrides):
    params = PHYParams()
    values = dict(
        enable_mimo=True,
        link_mode="ofdm",
        num_tx=2,
        num_rx=2,
        num_spatial_streams=2,
        mimo_scheme="spatial_multiplexing",
        mimo_detector="mmse",
        mimo_channel_model="identity",
        mimo_csi_mode="estimated",
        mimo_num_taps=1,
        subwave_num=30,
        gi_length=8,
        enable_multipath=False,
        enable_awgn=False,
        enable_cfo=False,
        enable_cfo_compensation=False,
        enable_iq_imbalance=False,
        enable_iq_compensation=False,
        NCBPS=2,
        code_type="RS",
        rs_nsym=4,
        rs_c_exp=4,
        rs_packet_size=11,
        pilot_block_indexes=[],
        subframe_ofdm_num=1,
        sample_rate=1e6,
        duration=None,
        sample_length=22,
        random_seed=7,
        mimo_channel_seed=11,
    )
    values.update(overrides)
    params.update(**values)
    return params


def _run_link(params):
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    receiver = THzReceiver(params, transmitter)
    decoded = receiver.run(rx_signal)
    return transmitter, channel, receiver, decoded


def test_identity_noiseless_mimo_ofdm_has_zero_ber():
    transmitter, _, receiver, decoded = _run_link(_integration_params())
    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    assert len(rx_bits) == len(tx_bits)
    assert np.mean(tx_bits != rx_bits) == 0.0
    assert receiver.rx_equalized["mimo_channel_nmse"] < 1e-20


def test_full_rank_frequency_selective_ideal_csi_has_zero_ber():
    params = _integration_params(
        enable_multipath=True,
        mimo_num_taps=3,
        mimo_channel_model="iid_rayleigh",
        mimo_csi_mode="ideal",
    )
    transmitter, _, _, decoded = _run_link(params)
    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    assert np.mean(tx_bits != rx_bits) == 0.0


def test_frequency_selective_ls_estimated_csi_has_zero_ber_without_noise():
    params = _integration_params(
        enable_multipath=True,
        mimo_num_taps=3,
        mimo_channel_model="iid_rayleigh",
        mimo_csi_mode="estimated",
    )
    transmitter, _, receiver, decoded = _run_link(params)
    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    assert np.mean(tx_bits != rx_bits) == 0.0
    assert receiver.rx_equalized["mimo_channel_nmse"] < 1e-20


def test_cfo_compensation_restores_noiseless_link():
    params = _integration_params(
        enable_cfo=True,
        enable_cfo_compensation=True,
        ppm=0.05,
        fc=1e9,
    )
    transmitter, _, receiver, decoded = _run_link(params)
    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    assert np.mean(tx_bits != rx_bits) == 0.0
    assert abs(receiver.rx_equalized["estimated_cfo_Hz"] - 50.0) < 1e-6


def test_rx_iq_compensation_restores_noiseless_link():
    params = _integration_params(
        enable_iq_imbalance=True,
        iq_imbalance_position="rx",
        rx_iq_gain_imbalance_db=1.5,
        rx_iq_phase_imbalance_deg=7.0,
        enable_iq_compensation=True,
    )
    transmitter, _, _, decoded = _run_link(params)
    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    assert np.mean(tx_bits != rx_bits) == 0.0


def test_high_snr_awgn_link_is_finite_and_low_error():
    params = _integration_params(
        enable_awgn=True,
        SNRdB=35.0,
        mimo_channel_model="identity",
        mimo_csi_mode="estimated",
    )
    transmitter, _, receiver, decoded = _run_link(params)
    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    assert np.mean(tx_bits != rx_bits) <= 0.05
    assert np.isfinite(receiver.rx_equalized["mimo_channel_nmse"])
    assert np.all(np.isfinite(receiver.rx_equalized["signal_stream"]))


def test_simulation_manager_exports_mimo_metrics():
    manager = SimulationManager(base_params=_integration_params(), save_plots=False)
    result = manager.run_once()
    metrics = result["metrics"]
    assert metrics["ber"] == 0.0
    assert metrics["raw_throughput_bps"] > 0
    assert metrics["effective_throughput_bps"] == metrics["raw_throughput_bps"]
    assert metrics["mimo_channel_nmse"] < 1e-20
    assert np.isfinite(metrics["mimo_mean_condition_number"])


def test_ui_spatial_mode_maps_to_real_mimo_parameters():
    from thz_sim_ui.services.backend import BackendService

    mapped = BackendService.map_ui_params_to_phy_params(
        {"空间模式分区": "2×2 MIMO", "波形类型": "OFDM"}
    )
    assert mapped["enable_mimo"] is True
    assert mapped["num_tx"] == mapped["num_rx"] == 2
    assert mapped["num_spatial_streams"] == 2
    assert mapped["link_mode"] == "ofdm"


def test_ui_mimo_parameters_are_applied_atomically_to_default_params():
    from thz_sim_ui.services.backend import BackendService

    mapped = BackendService.map_ui_params_to_phy_params(
        {"空间模式分区": "2×2 MIMO", "波形类型": "OFDM"}
    )
    params = PHYParams()
    manager = SimulationManager(base_params=params, save_plots=False)
    unknown = manager.apply_dict_to_params(params, mapped)

    assert unknown == []
    assert params.get("enable_mimo") is True
    assert params.get("num_tx") == 2
    assert params.get("num_rx") == 2
    assert params.get("num_spatial_streams") == 2


def test_ui_mimo_controls_run_from_default_simulation_manager():
    from thz_sim_ui.services.backend import BackendService

    controls = BackendService.map_ui_params_to_phy_params(
        {"空间模式分区": "2×2 MIMO", "波形类型": "OFDM"}
    )
    controls.update(
        subwave_num=30,
        gi_length=8,
        mimo_num_taps=1,
        mimo_channel_model="identity",
        enable_multipath=False,
        enable_awgn=False,
        enable_cfo=False,
        enable_iq_imbalance=False,
        enable_iq_compensation=False,
        NCBPS=2,
        code_type="RS",
        rs_nsym=4,
        rs_c_exp=4,
        rs_packet_size=11,
        pilot_block_indexes=[],
        subframe_ofdm_num=1,
        sample_rate=1e6,
        duration=None,
        sample_length=22,
        random_seed=7,
    )
    manager = SimulationManager(base_params=PHYParams(), save_plots=False)
    manager.set_control_params(controls)
    result = manager.run_once()

    assert result["params"].get("enable_mimo") is True
    assert result["metrics"]["ber"] == 0.0
