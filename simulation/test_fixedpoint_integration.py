"""冻结的原浮点摘要与定点链路的真实配对回归。"""
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.THzReceiver import THzReceiver
from simulation.fixedpoint_comparison import run_float_fixed_comparison


BASELINE = Path(__file__).resolve().parent / "fixtures/fixedpoint_float_baseline.json"


def _digest(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


@pytest.mark.parametrize("mode", ["ofdm", "sc-fde", "mimo"])
def test_float_golden_reference_exact(mode):
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))[mode]
    params = PHYParams()
    params.update(**baseline["overrides"])
    state = np.random.get_state()
    try:
        np.random.seed(baseline["global_seed"])
        with contextlib.redirect_stdout(io.StringIO()):
            tx = THzTransmitter(params)
            transmitted = tx.run()
            channel = THzChannel(params)
            received = channel.run(transmitted)
            rx = THzReceiver(params, tx)
            decoded = rx.run(received)
    finally:
        np.random.set_state(state)
    assert _digest(transmitted["signal_stream"]) == baseline["tx"]
    assert _digest(received["signal_stream"]) == baseline["channel"]
    assert _digest(rx.rx_equalized["signal_stream"]) == baseline["equalized"]
    assert _digest(decoded["signal_stream"]) == baseline["bits"]


@pytest.mark.parametrize("mode", ["ofdm", "sc-fde", "mimo"])
def test_boundary_comparison_replays_noise_and_reports_ranges(mode):
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))[mode]
    params = PHYParams()
    values = baseline["overrides"].copy()
    values["sample_length"] = (46080 if mode == "ofdm" else 48960 if mode == "sc-fde" else 512) // 4
    params.update(**values)
    state = np.random.get_state()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_float_fixed_comparison(params, {"enable_fixed": True}, {"seed": 7})
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(state, after))
    assert result["stimulus"]["noise_sha256"]
    assert result["fixed_report"]["ranges"]
    assert result["float_report"]["ranges"]
    assert "tx.regions.data" in result["float_report"]["ranges"]
    assert result["nmse"]["tx.waveform"]["status"] == "ok"
    assert not result["fixed"]["length_mismatch"]
    assert result["float"]["ber"] == 0
    assert result["fixed"]["ber"] == 0


def test_numeric_configs_are_isolated_on_clone():
    params = PHYParams()
    params.update(fixed_point={"modules": {"ofdm.tx.ifft": {"output_format": {"word_length": 18}}}})
    clone = params.clone()
    clone.get("fixed_point")["modules"]["ofdm.tx.ifft"]["output_format"]["word_length"] = 24
    assert params.get("fixed_point")["modules"]["ofdm.tx.ifft"]["output_format"]["word_length"] == 18


@pytest.mark.parametrize("mode,modules", [
    ("ofdm", ["ofdm.tx.ifft", "ofdm.rx.fft", "ofdm.rx.channel_estimation", "ofdm.rx.equalizer"]),
    ("sc-fde", ["sc.tx.pulse_shaping", "sc.rx.matched_filter", "sc.rx.coarse_sync", "sc.rx.fine_sync", "sc.rx.channel_estimation", "sc.rx.equalizer"]),
    ("mimo", ["mimo.tx.ifft", "mimo.rx.fft", "mimo.rx.channel_estimation", "mimo.rx.detector"]),
])
def test_operation_chain_uses_integer_kernels(mode, modules):
    values = json.loads(BASELINE.read_text(encoding="utf-8"))[mode]["overrides"]
    params = PHYParams()
    values["sample_length"] = (46080 if mode == "ofdm" else 48960 if mode == "sc-fde" else 512) // 4
    values["enable_iq_compensation"] = False
    params.update(**values)
    cfg = {"enable_fixed": True, "modules": {name: {"arithmetic_level": "operation"} for name in modules}}
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_float_fixed_comparison(params, cfg, {"seed": 7, "range_analysis": False})
    assert result["fixed"]["ber"] == 0
    assert result["fixed"]["evm"]["rms"] < .01
    assert result["fixed_report"]["ranges"] == {}
    for module in modules:
        assert "operation" in result["fixed_report"]["coverage"][module]["levels"]
