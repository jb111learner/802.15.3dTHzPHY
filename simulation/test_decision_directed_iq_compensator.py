import numpy as np

from receiver.IQCompensator import (
    DecisionDirectedIQCompensator,
    IQRealParallelCalibrator,
)


class _Params:
    def __init__(self, **values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


def _evm(reference, measured, skip=0):
    reference = reference[skip:]
    measured = measured[skip:]
    return np.sqrt(
        np.mean(np.abs(measured - reference) ** 2)
        / np.mean(np.abs(reference) ** 2)
    )


def test_decision_directed_compensation_reduces_64qam_evm():
    params = _Params(MCS=1, NCBPS=6)
    compensator = DecisionDirectedIQCompensator(
        params, filter_len=1, ridge_lambda=0.0, iterations=1
    )
    rng = np.random.default_rng(19)
    base = rng.choice(compensator.constellation, size=8192)
    phase = np.exp(1j * np.pi * np.arange(len(base)) / 2)
    reference = base * phase
    impaired = IQRealParallelCalibrator.apply_precompensation(
        reference, [0.94], [0.04], [-0.03], [1.06]
    )
    data = {
        "signal_stream": impaired,
        "sample_rate_Hz": 1.0,
        "duration_seconds": float(len(impaired)),
        "signal_length": len(impaired),
        "padding_bit_num": 0,
    }

    result = compensator.compensate(data)
    compensated = result["signal_stream"]

    assert result["iq_compensation_method"] == "decision_directed"
    assert len(result["iq_dd_diagnostics"]) == 1
    assert _evm(reference, compensated, skip=2) < _evm(reference, impaired, skip=2)
    assert _evm(reference, compensated, skip=2) < 0.02
