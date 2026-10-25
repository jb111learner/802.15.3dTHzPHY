import numpy as np
import pytest

from simulation.fixedpoint_comparison import bit_block_metrics, nmse, evm, snr_loss
from simulation.fixedpoint_config_bridge import map_numeric_ui_parameters


def test_missing_bits_are_errors_and_blocks_fail():
    metrics = bit_block_metrics(np.zeros(8, dtype=np.uint8), np.zeros(3, dtype=np.uint8), 4, 8)
    assert metrics["bit_errors"] == 5
    assert metrics["missing_bits"] == 5
    assert metrics["block_errors"] == 2
    assert metrics["frame_errors"] == 1


def test_padding_excluded_and_undetected_codeword_error_counted():
    metrics = bit_block_metrics(np.array([0, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 1, 1]), 4, 8, 4)
    assert metrics["bits"] == 4 and metrics["bit_errors"] == 1
    assert metrics["bler"] == 1


def test_signal_metrics_do_not_hide_shape_or_gain_error():
    assert nmse(np.ones(4), np.ones(3))["status"] == "shape_mismatch"
    assert evm(np.ones(4), np.ones(4) * 2)["rms"] == 1
    assert nmse(np.zeros(4), np.zeros(4))["status"] == "zero_reference"


def test_loss_needs_bracket_and_does_not_log_zero():
    points = [{"snr_db": snr, "float": {"ber": f}, "fixed": {"ber": q}}
              for snr, f, q in ((10, .1, .2), (12, .01, .02), (14, .001, .002))]
    loss = snr_loss(points, .01)
    assert loss["loss_db"] == pytest.approx(0.6020599913)
    assert snr_loss(points, 1e-6)["loss_db"] is None
    points[-1]["fixed"]["ber"] = 0
    assert snr_loss(points, .005)["fixed_crossing"]["estimate"] is None


def test_loss_rejects_invalid_targets_and_duplicate_snrs():
    points = [{"snr_db": 1., "float": {"ber": .1}, "fixed": {"ber": .1}}]
    for target in (0, 1, float("nan")):
        with pytest.raises(ValueError):
            snr_loss(points, target)
    with pytest.raises(ValueError, match="duplicate"):
        snr_loss(points + points)


def test_snr_loss_has_count_supported_simultaneous_confidence_brackets():
    points = []
    for snr, errors in ((0., 10000), (.1, 1000), (.2, 100)):
        values = {"ber": errors / 100000, "bit_errors": errors, "bits": 100000}
        points.append({"snr_db": snr, "float": dict(values), "fixed": dict(values)})
    result = snr_loss(points, .005)
    assert result["loss_db"] == 0
    assert result["confidence_loss_interval_db"] == pytest.approx([-.1, .1])
    assert result["statistical_status"] == "count_supported_loss_bracket"
    assert result["per_point_confidence"] > .95


def test_sampled_target_does_not_prove_zero_width_confidence_crossing():
    points = []
    for snr, errors in ((0., 10), (1., 1), (2., 0)):
        values = {"ber": errors / 100, "bit_errors": errors, "bits": 100}
        points.append({"snr_db": snr, "float": dict(values), "fixed": dict(values)})
    result = snr_loss(points, .01)
    assert result["loss_db"] == 0
    assert result["float_crossing"]["lower"] == result["float_crossing"]["upper"] == 1
    assert result["confidence_loss_interval_db"] is None
    assert "insufficient_count_evidence" in result["statistical_status"]


def test_candidate_validation_requires_confidence_loss_not_empirical_loss(monkeypatch):
    import simulation.fixedpoint_comparison as comparison
    report = {"snr_loss": {"loss_interval_db": [0, .01], "confidence_loss_interval_db": None},
              "points": [{"fixed": {"max_evm_rms": .001, "max_overflow_rate": 0.,
                                      "ber_interval": {"upper": .0001}}}]}
    monkeypatch.setattr(comparison, "run_comparison_sweep", lambda *args, **kwargs: report)
    result = comparison.evaluate_qformat_candidates(None, [{}], [1, 2], target=.001)
    assert result["validated"] is False and result["recommended"] is None
    assert "confidence SNR loss not bounded within budget" in result["candidates"][0]["rejection_reasons"]


def test_numeric_bridge_has_only_two_modes_and_preserves_modules():
    assert map_numeric_ui_parameters({"链路模式": "SC"}) == {}
    mapped = map_numeric_ui_parameters({"仿真数值模式": "浮点", "定点字长": 18, "定点小数位": 14,
                                       "_fixed_point": {"enable_fixed": True, "modules": {"ofdm.tx.ifft": {"arithmetic_level": "operation"}}}})
    assert not mapped["fixed_point"]["enable_fixed"]
    assert mapped["fixed_point"]["modules"]["ofdm.tx.ifft"]["arithmetic_level"] == "operation"
    with pytest.raises(ValueError):
        map_numeric_ui_parameters({"仿真数值模式": "compare"})


def test_range_candidate_includes_positive_endpoint_lsb():
    from fixedpoint import recommend_qformats
    ranges = {"signal": {"real": {"min": -7.9, "max": 7.9}}}
    result = recommend_qformats(ranges, {"word_lengths": [4, 8], "headroom_db": 0})
    assert [fmt["word_length"] for fmt in result["nodes"]["signal"]["candidates"]] == [8]
    assert result["validated"] is False


def test_integer_conversion_is_not_code_extraction_and_abs_saturates():
    from fixedpoint import FixedPointFormat
    from fixedpoint import arithmetic as ar
    fmt = FixedPointFormat(8, 4, saturation=False)
    value = ar.quantize(np.array([-1.75, 1.75]), fmt)
    np.testing.assert_array_equal(value.real, [-28, 28])
    np.testing.assert_array_equal(ar.to_integer(value), [-1, 1])
    minimum = ar.quantize(np.array([-8.]), fmt)
    assert ar.abs_real(minimum).real[0] == 127
    with pytest.raises(OverflowError):
        ar.negate(minimum)
