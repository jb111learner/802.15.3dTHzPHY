"""Independent checks for the high-precision reference and no-floor bound."""
import contextlib
import io
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from half_spacing_ofdm_sim import Config, coupling_matrix, gram_matrix, complex_normal, discrete_channel, fir_frequency_response
from simulation.ofdm_lmmse_high_snr import (
    eigenchannel_lmmse, full_size_rank_bound, high_precision_spectrum, run_analysis,
)


class TestHighSNRReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(k=2, samples=32, channel="awgn", codes=(), frames=200)
        cls.metadata, cls.logs, cls.vectors = high_precision_spectrum(cls.config)

    def test_high_precision_spectrum_and_phase_transformation(self):
        expected = gram_matrix(coupling_matrix(2, 32))
        reconstructed = (self.vectors * np.exp(self.logs)) @ self.vectors.conj().T
        assert_allclose(reconstructed, expected, atol=3e-14)
        assert_allclose(self.vectors.conj().T @ self.vectors, np.eye(4), atol=3e-14)
        self.assertTrue(self.metadata["noiseless_recovery_passed"])
        self.assertTrue(self.metadata["overlap_passed"])

    def test_eigenchannel_estimate_equals_normal_equation(self):
        rng = np.random.default_rng(619)
        a, noise = complex_normal(rng, (4, 20)), complex_normal(rng, (4, 20))
        b = np.diag(np.sqrt(np.exp(self.logs))) @ self.vectors.conj().T
        nu = 0.03
        received = b @ a + np.sqrt(nu) * noise
        expected = np.linalg.solve(b.conj().T @ b + nu * np.eye(4), b.conj().T @ received)
        estimated, error, _ = eigenchannel_lmmse(a, noise, self.logs, self.vectors, np.log(nu))
        assert_allclose(estimated, expected, atol=3e-13)
        assert_allclose(estimated - a, error, atol=3e-15)

    def test_complex_multipath_channel_phase_reconstruction(self):
        config = Config(k=2, samples=32, codes=())
        _, logs, vectors = high_precision_spectrum(config)
        h = fir_frequency_response(config, discrete_channel(config))
        expected = h.conj()[:, None] * gram_matrix(coupling_matrix(2, 32)) * h[None, :]
        reconstructed = (vectors * np.exp(logs)) @ vectors.conj().T
        assert_allclose(reconstructed, expected, atol=3e-14)

    def test_positive_bound_and_zero_channel_exception(self):
        bound = full_size_rank_bound(self.config)
        self.assertTrue(bound["conditions_hold"])
        self.assertLessEqual(bound["lambda_min_lower_bound_log10"], self.logs[0] / np.log(10))
        erased = Config(k=2, samples=32, tap_delays=(0, 1), tap_gains=("1", "-1"), codes=())
        self.assertFalse(full_size_rank_bound(erased)["conditions_hold"])

    def test_mse_tends_to_zero_without_float_waveform_cancellation(self):
        a = np.ones((4, 1), complex)
        noise = np.zeros_like(a)
        predictions = [eigenchannel_lmmse(a, noise, self.logs, self.vectors, -db * np.log(10) / 10)[2]
                       for db in (0, 40, 80, 160)]
        self.assertTrue(all(left > right for left, right in zip(predictions, predictions[1:])))
        estimated, error, mse = eigenchannel_lmmse(a, noise, self.logs, self.vectors, -np.inf)
        assert_allclose(estimated, a, atol=0)
        assert_allclose(error, 0, atol=0)
        self.assertEqual(mse, 0)

    def test_reference_monte_carlo_matches_theory(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = run_analysis(self.config, [2], directory, no_plots=True)
        self.assertTrue(report["passed"])
        self.assertEqual(report["results"][-1]["bit_errors"], 0)
        self.assertLess(report["results"][-1]["theoretical_ber_upper_bound"], 1e-6)


if __name__ == "__main__":
    unittest.main()
