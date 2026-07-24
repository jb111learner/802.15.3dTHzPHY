import numpy as np
import pytest

from receiver.IQCompensator import IQRealParallelCalibrator


def _estimate_and_compensate(reference, impaired, filter_len):
    estimate = IQRealParallelCalibrator.estimate_rx_impairment_filters(
        reference, impaired, filter_len
    )
    filters = IQRealParallelCalibrator.design_postcompensation_filters(
        estimate["e1"], estimate["e2"], estimate["e3"], estimate["e4"],
        filter_len,
    )
    dc = IQRealParallelCalibrator.design_postcompensation_dc_offset(
        filters["f1"], filters["f2"], filters["f3"], filters["f4"],
        estimate["dI"], estimate["dQ"],
    )
    compensated = IQRealParallelCalibrator.apply_postcompensation(
        impaired,
        filters["f1"], filters["f2"], filters["f3"], filters["f4"],
        dc["cI"], dc["cQ"],
    )
    return estimate, filters, compensated


def _evm(reference, measured, skip=0):
    reference = reference[skip:]
    measured = measured[skip:]
    return np.sqrt(
        np.mean(np.abs(measured - reference) ** 2)
        / np.mean(np.abs(reference) ** 2)
    )


def test_fixed_phase_bpsk_training_is_rejected():
    rng = np.random.default_rng(7)
    bpsk = 2 * rng.integers(0, 2, 512) - 1
    training = bpsk * np.exp(1j * np.pi / 4)
    impaired = 0.9 * training + 0.15 * np.conj(training)

    with pytest.raises(np.linalg.LinAlgError, match="not identifiable"):
        IQRealParallelCalibrator.estimate_rx_impairment_filters(
            training, impaired, filter_len=5
        )


@pytest.mark.parametrize(
    ("h1", "h2", "h3", "h4"),
    [
        ([0.90], [0.08], [-0.06], [1.12]),
        ([0.90, 0.06, -0.02], [0.08, -0.03],
         [-0.06, 0.02], [1.12, -0.05, 0.01]),
    ],
)
def test_complex_training_recovers_iq_impairment(h1, h2, h3, h4):
    rng = np.random.default_rng(11)
    reference = (
        (2 * rng.integers(0, 2, 4096) - 1)
        + 1j * (2 * rng.integers(0, 2, 4096) - 1)
    ) / np.sqrt(2)
    impaired = IQRealParallelCalibrator.apply_precompensation(
        reference, h1, h2, h3, h4
    )

    estimate, filters, compensated = _estimate_and_compensate(
        reference, impaired, filter_len=max(map(len, (h1, h2, h3, h4)))
    )

    skip = 2 * max(map(len, (h1, h2, h3, h4)))
    assert estimate["design_condition_number"] < 1e10
    assert filters["condition_number"] < 1e10
    assert _evm(reference, compensated, skip) < 0.03
    assert _evm(reference, compensated, skip) < _evm(reference, impaired, skip)
