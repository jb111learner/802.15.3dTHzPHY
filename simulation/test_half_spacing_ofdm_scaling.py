"""Independent checks of the ill-conditioned carrier-count scan."""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from half_spacing_ofdm_sim import Config, Waveform
from simulation.analyze_half_spacing_ofdm_matrices import scaled_vandermonde_inverse
from simulation.analyze_half_spacing_ofdm_scaling import run_scan, scaled_toeplitz_inverse, scan_case
from simulation.ofdm_lmmse_high_snr import high_precision_spectrum


class TestCarrierScaling(unittest.TestCase):
    def test_inverse_matches_independent_waveform_including_group_phase(self):
        for d, n in ((4, 4), (4, 64), (8, 512)):
            config = Config(k=d // 2, samples=n, channel="awgn", codes=())
            u = Waveform(config).transmit(np.eye(d)) / (config.mu * math.sqrt(n))
            inverse, scale, check = scaled_toeplitz_inverse(d, n, 110)
            assert_allclose((u.conj().T @ u) @ (inverse * 10 ** scale), np.eye(d), atol=2e-10)
            self.assertLess(float(check["inverse_first_column_relative_residual"]), 1e-80)

    def test_beyond_float64_matches_independent_vandermonde_inverse(self):
        inverse, scale, _ = scaled_toeplitz_inverse(128, 128, 230)
        inverse_u, scale_u, _ = scaled_vandermonde_inverse(128, 230)
        expected = inverse_u @ inverse_u.conj().T
        assert_allclose(inverse / np.linalg.norm(inverse), expected / np.linalg.norm(expected), atol=5e-15)
        self.assertAlmostEqual(scale + math.log10(np.linalg.norm(inverse, 2)),
                               2 * (scale_u + math.log10(np.linalg.norm(inverse_u, 2))), places=11)

    def test_rectangular_normal_extremes_match_independent_mp_eigensolver(self):
        reference = Config(k=256, samples=512, codes=())
        for d in (8, 16):
            row, regularized, _ = scan_case(reference, d, 512)
            from dataclasses import replace
            metadata, _, _ = high_precision_spectrum(replace(reference, k=d // 2))
            self.assertAlmostEqual(row["A_min_eigenvalue_log10"], metadata["lambda_min_log10"], places=10)
            self.assertAlmostEqual(row["A_condition_log10"], metadata["lambda_max_log10"] - metadata["lambda_min_log10"], places=10)
            for item in regularized:
                self.assertAlmostEqual(item["effective_degrees_of_freedom"], d * (1 - item["predicted_lmmse_mse"]), places=12)

    def test_precision_and_sampling_guards(self):
        with self.assertRaises(ValueError):
            scaled_toeplitz_inverse(8, 4, 100)
        with self.assertRaises(ArithmeticError):
            scaled_toeplitz_inverse(32, 32, 20)

    def test_channel_frequency_response_is_fixed_when_sampling_changes(self):
        config = Config(k=256, samples=512, codes=())
        fixed, _, _ = scan_case(config, 8, 512)
        square, _, _ = scan_case(config, 8, 8)
        self.assertEqual(fixed["channel_condition"], square["channel_condition"])
        self.assertGreater(square["C_condition_log10"], fixed["C_condition_log10"])

    def test_export_and_identical_endpoint(self):
        config = Config(k=4, samples=8, codes=())
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = run_scan(config, [4, 8], directory, no_plots=True)
            self.assertEqual(len(report["scan"]), 4)
            self.assertEqual(len(report["regularized"]), 12)
            self.assertEqual(report["scan"][-1]["transfer_condition_log10"], report["scan"][-2]["transfer_condition_log10"])
            loaded = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["scan"][-1]["sampling_policy"], "N_equals_d")


if __name__ == "__main__":
    unittest.main()
