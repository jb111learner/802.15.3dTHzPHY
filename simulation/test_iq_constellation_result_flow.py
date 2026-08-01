import numpy as np

from simulation.result_utils import collect_receiver_intermediates
from thz_sim_ui.services.result_utils import (
    build_simulation_result_path,
    select_rx_constellation_data,
)


def test_frontend_prefers_iq_compensated_constellation():
    equalized = {"signal_stream": np.array([1 + 1j])}
    compensated = {"signal_stream": np.array([2 + 2j])}

    selected = select_rx_constellation_data({
        "rx_equalized": equalized,
        "rx_iq_compensated": compensated,
    })

    assert selected is compensated


def test_frontend_falls_back_to_equalized_constellation_without_iq_compensation():
    equalized = {"signal_stream": np.array([1 + 1j])}

    selected = select_rx_constellation_data({
        "rx_equalized": equalized,
        "rx_iq_compensated": None,
    })

    assert selected is equalized


def test_simulation_result_exports_iq_compensated_signal():
    equalized = {"signal_stream": np.array([1 + 1j])}
    compensated = {"signal_stream": np.array([2 + 2j])}

    class FakeReceiver:
        rx_matched = None
        rx_downsampled = None
        rx_equalized = equalized
        rx_iq_compensated = compensated

    result = collect_receiver_intermediates(FakeReceiver())

    assert result["rx_equalized"] is equalized
    assert result["rx_iq_compensated"] is compensated


def test_each_simulation_gets_an_independent_project_result_path(tmp_path):
    first = build_simulation_result_path(tmp_path, "THz-Sim-1", 1001)
    second = build_simulation_result_path(tmp_path, "THz-Sim-2", 1002)

    assert first != second
    assert first.parent == second.parent == tmp_path / "runs"
    assert first.name == "THz-Sim-1_1001"
    assert second.name == "THz-Sim-2_1002"
