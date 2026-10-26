"""Signal-only physical-chain tests and noise-only covariance tests."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from half_spacing_ofdm_sim import Config, Waveform
from simulation.validate_half_spacing_ofdm_model import (
    covariance_checks, run_validation, theory_reference, validate_noise, validate_signal,
)


class TestSignalOnlyTheory(unittest.TestCase):
    def test_default_512_tone_actual_chain_and_complete_operator(self):
        config = Config(codes=(), batch_size=37)
        result, arrays = validate_signal(config, frames=23)
        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["full_transfer_columns"], 512)
        assert_allclose(arrays["signal_transfer_measured"], arrays["signal_transfer_theory"], atol=3e-12)

    def test_nonunit_duration_energy_and_16qam(self):
        base = Config(k=5, samples=128, modulation=16, symbol_duration=0.7, block_energy=2.4,
                      tap_delays=(0, 3, 19), tap_gains=("0.6+0.3j", "-0.2j", "0.1-0.2j"),
                      batch_size=3, codes=())
        for channel in ("awgn", "multipath"):
            with self.subTest(channel=channel):
                result, _ = validate_signal(replace(base, channel=channel), frames=19)
                self.assertTrue(result["passed"], result["checks"])

    def test_oversampled_512_tone_actual_chain(self):
        config = Config(k=256, samples=1024, tap_delays=(0, 82, 194), codes=(), batch_size=79)
        result, _ = validate_signal(config, frames=17)
        self.assertTrue(result["passed"], result["checks"])

    def test_signal_measurement_detects_omitted_cross_group_coupling(self):
        config = Config(k=8, samples=64, codes=())
        result, arrays = validate_signal(config, frames=13)
        self.assertTrue(result["passed"])
        measured = arrays["signal_transfer_measured"]
        wrong = np.diag(measured.diagonal())
        self.assertGreater(np.linalg.norm(measured - wrong) / np.linalg.norm(measured), 0.3)

    def test_sampled_reference_converges_to_continuous_document(self):
        errors = []
        for n in (64, 128, 256, 512):
            config = Config(k=8, samples=n, codes=())
            _, sampled, continuous, _ = theory_reference(config, np.ones(1, complex))
            errors.append(np.linalg.norm(sampled - continuous) / np.linalg.norm(continuous))
        self.assertTrue(all(right < 0.27 * left for left, right in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 1e-4)


class TestNoiseOnlyTheory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(k=8, samples=64, symbol_duration=0.7, block_energy=2.4, codes=())
        cls.trials = 32768
        cls.result, cls.arrays = validate_noise(cls.config, n0=0.03, trials=cls.trials, batch_size=1024)

    def test_pure_noise_covariance_mean_and_properness(self):
        self.assertTrue(self.result["passed"], self.result["checks"])
        c = self.arrays["noise_covariance_theory_raw"] / (0.03 * self.config.symbol_duration)
        self.assertAlmostEqual(c[0, 8].real, 0)
        self.assertGreater(c[0, 8].imag, 0.63)
        self.assertLess(self.arrays["noise_covariance_empirical_raw"][8, 0].imag, 0)

    def test_covariance_test_rejects_white_conjugated_and_wrong_power_models(self):
        actual = self.arrays["noise_covariance_empirical_raw"]
        theory = self.arrays["noise_covariance_theory_raw"]
        wrong_models = (np.diag(theory.diagonal()), theory.conj(), 2 * theory)
        for wrong in wrong_models:
            checks, _ = covariance_checks(actual, wrong, self.arrays["noise_mean_raw"],
                                          self.arrays["noise_pseudocovariance_raw"], self.trials)
            self.assertFalse(all(item["passed"] for item in checks))

    def test_receiver_noise_covariance_independent_of_propagation_channel(self):
        config = replace(self.config, channel="awgn")
        result, arrays = validate_noise(config, n0=0.03, trials=8192, batch_size=512)
        self.assertTrue(result["passed"], result["checks"])
        assert_allclose(arrays["noise_covariance_theory_raw"], self.arrays["noise_covariance_theory_raw"], atol=0)
        assert_allclose(arrays["noise_covariance_exact_operator_raw"], self.arrays["noise_covariance_exact_operator_raw"], atol=0)

    def test_transmit_energy_affects_only_normalized_noise_covariance(self):
        config = replace(self.config, block_energy=2 * self.config.block_energy)
        result, arrays = validate_noise(config, n0=0.03, trials=self.trials, batch_size=1024)
        self.assertTrue(result["passed"], result["checks"])
        assert_allclose(arrays["noise_covariance_empirical_raw"], self.arrays["noise_covariance_empirical_raw"], atol=0)
        assert_allclose(2 * arrays["noise_covariance_empirical_normalized"],
                        self.arrays["noise_covariance_empirical_normalized"], atol=2e-15)

    def test_full_size_exact_actual_matched_operator_covariance(self):
        config = Config(codes=())
        wave = Waveform(config)
        _, c, _, _ = theory_reference(config, np.ones(1, complex))
        operator = wave.matched_filter(np.eye(config.samples, dtype=complex))
        n0 = 0.03
        covariance = (config.samples * n0 / config.symbol_duration) * operator @ operator.conj().T
        assert_allclose(covariance, n0 * config.symbol_duration * c, atol=2e-14)

    def test_standalone_report_and_matrix_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(k=3, samples=32, codes=(), frames=8)
            report = run_validation(config, directory, signal_frames=8, noise_trials=4096, no_plots=True)
            saved = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertTrue(saved["signal"]["passed"])
            self.assertTrue(saved["noise"]["passed"])
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             {"signal_checks.csv", "noise_checks.csv", "report.json", "summary.md", "matrices.npz"})
            with np.load(Path(directory) / "matrices.npz") as matrices:
                self.assertEqual(matrices["signal_transfer_measured"].shape, (6, 6))
                self.assertEqual(matrices["noise_covariance_empirical_raw"].shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
