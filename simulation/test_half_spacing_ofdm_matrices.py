"""Structured inverse and detector diagnostics against independent references."""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from half_spacing_ofdm_sim import Config, Waveform, discrete_channel
from simulation.analyze_half_spacing_ofdm_matrices import analyze, scaled_vandermonde_inverse, write_outputs
from simulation.ofdm_lmmse_high_snr import high_precision_spectrum
from simulation.validate_half_spacing_ofdm_model import theory_reference


class TestMatrixParameters(unittest.TestCase):
    def test_structured_inverse_midpoint_phase_and_group_order(self):
        for n in (4, 8):
            config = Config(k=n // 2, samples=n, codes=(), channel="awgn")
            inverse, log_scale, residual = scaled_vandermonde_inverse(n, 100)
            u = Waveform(config).transmit(np.eye(n)) / (config.mu * math.sqrt(n))
            assert_allclose(u @ (inverse * 10 ** log_scale), np.eye(n), atol=2e-12)
            self.assertLess(float(residual), 1e-80)

    def test_large_integer_phase_preserves_arbitrary_precision_residual(self):
        _, _, residual = scaled_vandermonde_inverse(128, 180)
        self.assertLess(float(residual), 1e-80)

    def test_extreme_spectrum_matches_independent_high_precision_eigensolver(self):
        config = Config(k=3, samples=6, tap_delays=(0, 1, 2),
                        tap_gains=("1+0.2j", "-0.2+0.1j", "0.1j"), codes=())
        with contextlib.redirect_stdout(io.StringIO()):
            report, _ = analyze(config)
        metadata, _, _ = high_precision_spectrum(config)
        hp = report["high_precision"]
        self.assertTrue(hp["computed"])
        self.assertAlmostEqual(hp["normal_min_eigenvalue_log10"], metadata["lambda_min_log10"], places=11)
        a = next(row for row in report["matrices"] if row["matrix"] == "A")
        self.assertAlmostEqual(a["condition_2_high_precision_log10"],
                               metadata["lambda_max_log10"] - metadata["lambda_min_log10"], places=11)
        _, c, _, h = theory_reference(config, discrete_channel(config))
        transfer = next(row for row in report["matrices"] if row["matrix"] == "C_N_D")
        self.assertAlmostEqual(transfer["condition_2_high_precision_log10"],
                               math.log10(np.linalg.cond(c * h[None, :])), places=9)

    def test_regularized_condition_degrees_of_freedom_and_mse_decomposition(self):
        config = Config(k=3, samples=6, channel="awgn", codes=(), ebn0_db=(0, 20, 40))
        with contextlib.redirect_stdout(io.StringIO()):
            report, _ = analyze(config)
        _, c, _, _ = theory_reference(config, np.ones(1, complex))
        for row in report["regularized"]:
            expected = np.linalg.cond(c + row["lambda"] * np.eye(6))
            self.assertLess(abs(row["condition_2"] / expected - 1), 1e-8)
            self.assertAlmostEqual(row["bias_mse"] + row["noise_mse"], row["predicted_lmmse_mse"], places=13)
            self.assertAlmostEqual(row["effective_degrees_of_freedom"], 6 * (1 - row["predicted_lmmse_mse"]), places=12)

    def test_rectangular_sampling_is_not_mislabeled_as_square_inverse(self):
        config = Config(k=3, samples=32, codes=())
        report, _ = analyze(config)
        self.assertFalse(report["high_precision"]["computed"])
        self.assertTrue(all(row["condition_2_high_precision"] is None for row in report["matrices"]))

    def test_insufficient_precision_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ArithmeticError):
            analyze(Config(k=4, samples=8, channel="awgn", codes=()), dps=20)

    def test_parameter_artifact_export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report, spectra = analyze(Config(k=2, samples=4, channel="awgn", codes=()))
        with tempfile.TemporaryDirectory() as directory:
            write_outputs(report, spectra, directory, no_plots=True)
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             {"matrix_parameters.csv", "regularized_parameters.csv", "singular_values_float64.npz", "report.json", "summary.md"})


if __name__ == "__main__":
    unittest.main()
