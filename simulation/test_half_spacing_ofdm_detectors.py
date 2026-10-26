"""Matched-model detectors and equal-rate OFDM; run in communication."""
import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from half_spacing_ofdm_sim import (
    Config, DETECTORS, NormalInverseLMMSE, OrdinaryOFDM, TheoryLMMSE, Waveform,
    complex_normal, coupling_matrix, discrete_channel, fir_frequency_response,
    gram_matrix, main, paired_symbols, qam_constellation, qam_indices, simulate,
)


class TestEqualRateOFDM(unittest.TestCase):
    def test_default_rate_and_energy(self):
        config = Config()
        self.assertEqual(2 * config.k, 512)
        self.assertEqual(config.ordinary_modulation, 16)
        self.assertEqual(config.bits_per_block, 1024)
        self.assertEqual(config.effective_cp_length, 97)
        self.assertAlmostEqual(config.mu ** 2 * 512, config.ordinary_mu ** 2 * 256)
        self.assertAlmostEqual(config.bits_per_block / config.block_duration,
                               256 * 4 / config.block_duration)

    def test_identical_payload_and_gray_roundtrip(self):
        rng = np.random.default_rng(719)
        for order in (4, 16, 64):
            config = Config(k=5, samples=32, modulation=order)
            a, hi, oa, oi = paired_symbols(config, rng, 11)
            _, hl = qam_constellation(order)
            _, ol = qam_constellation(order ** 2)
            assert_allclose(hl[hi].transpose(1, 0, 2).reshape(11, -1),
                            ol[oi].transpose(1, 0, 2).reshape(11, -1), atol=0)
            assert_allclose(qam_indices(a, order), hi, atol=0)
            assert_allclose(qam_indices(oa, order ** 2), oi, atol=0)

    def test_common_ebn0_gives_identical_physical_noise(self):
        config = Config(k=7, samples=64, block_energy=2.4, symbol_duration=0.7)
        for db in (-5, 0, 15, 50):
            half, ordinary, variance = config.noise_parameters(db)
            self.assertAlmostEqual(half / ordinary, 2)
            self.assertAlmostEqual(variance, config.ordinary_mu ** 2 * config.samples * ordinary)
            n0 = config.block_energy / (config.bits_per_block * 10 ** (db / 10))
            self.assertAlmostEqual(variance, config.samples * n0 / config.symbol_duration)

    def test_expected_energy_budget_including_prefix(self):
        config = Config(k=4, samples=32, block_energy=2.4, symbol_duration=0.7)
        length = config.effective_cp_length
        for wave, dimension in ((Waveform(config), 8), (OrdinaryOFDM(config), 4)):
            # Trace of the transmit operator gives the exact ensemble energy.
            basis = wave.with_cp(np.eye(dimension), length)
            energy = config.symbol_duration / config.samples * np.linalg.norm(basis) ** 2
            self.assertAlmostEqual(energy, config.block_energy * (1 + length / config.samples), places=12)

    def test_ordinary_fft_one_tap_and_noise_scaling(self):
        config = Config(k=5, samples=32, block_energy=2.4, symbol_duration=0.7)
        wave = OrdinaryOFDM(config)
        rng = np.random.default_rng(49)
        a = complex_normal(rng, (5, 6))
        h = np.array([1, 0.2j, -0.3+0.4j, 1.2-0.7j, 0.8])
        received = wave.transmit(h[:, None] * a)
        assert_allclose(wave.detect(received, h), a, atol=2e-14)
        _, nu, variance = config.noise_parameters(10)
        noise = math.sqrt(variance) * complex_normal(rng, (32, 6000))
        measured = np.mean(abs(wave.detect(noise, h)) ** 2)
        self.assertLess(abs(measured - wave.mse(nu, h)), 0.03 * wave.mse(nu, h))

    def test_ordinary_scalar_mmse_gain_compensation(self):
        config = Config(k=3, samples=16)
        wave = OrdinaryOFDM(config)
        received = complex_normal(np.random.default_rng(49), (16, 4))
        h, nu = np.array([0.1j, 0.5+0.7j, -1.2]), 0.2
        observation = wave.observe(received)
        scalar_mmse = h.conj()[:, None] * observation / (abs(h[:, None]) ** 2 + nu)
        gain = abs(h) ** 2 / (abs(h) ** 2 + nu)
        assert_allclose(wave.detect(received, h), scalar_mmse / gain[:, None], atol=1e-14)

    def test_invalid_configuration(self):
        for changes in ({"k": 0}, {"samples": 7, "k": 4}, {"modulation": 8},
                        {"cp_length": -1}, {"frames": 0}, {"tap_gains": ("1",)},
                        {"tap_gains": ("0", "0", "0")}, {"ebn0_db": (float("nan"),)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(Config(), **changes).validate()


class TestMatchedModelDetectors(unittest.TestCase):
    def test_inverse_and_svd_equal_colored_normal_solution(self):
        config = Config(k=4, samples=64, block_energy=2.4, symbol_duration=0.7)
        wave = Waveform(config)
        h = fir_frequency_response(config, discrete_channel(config))
        inverse, svd = NormalInverseLMMSE(wave, h), TheoryLMMSE(wave, h)
        received = complex_normal(np.random.default_rng(719), (64, 7))
        c = gram_matrix(coupling_matrix(4, 64))
        normal = h.conj()[:, None] * c * h[None, :]
        assert_allclose(svd.b.conj().T @ svd.b, normal, atol=3e-14)
        for nu in (0.5, 0.03, 0.001):
            inverse.prepare(nu)
            z = wave.matched_filter(received) / (config.mu * config.symbol_duration)
            reference = np.linalg.solve(normal + nu * np.eye(8), h.conj()[:, None] * z)
            assert_allclose(inverse.detect(received), reference, rtol=3e-11, atol=3e-11)
            assert_allclose(svd.detect(received, nu), reference, rtol=3e-11, atol=3e-11)
            self.assertAlmostEqual(inverse.mse(nu, svd.b), svd.mse(nu), places=12)

    def test_full_512_tone_numerical_comparison(self):
        config = Config()
        wave = Waveform(config)
        h = fir_frequency_response(config, discrete_channel(config))
        inverse, svd = NormalInverseLMMSE(wave, h), TheoryLMMSE(wave, h)
        nu, _, _ = config.noise_parameters(30)
        inverse.prepare(nu)
        received = complex_normal(np.random.default_rng(901), (512, 3))
        assert_allclose(inverse.detect(received), svd.detect(received, nu), rtol=2e-8, atol=2e-8)
        self.assertGreater(inverse.condition_number, 1000)
        self.assertLess(inverse.inverse_residual, 1e-9)

    def test_both_half_spacing_estimators_remain_linear(self):
        config = Config(k=3, samples=32)
        wave = Waveform(config)
        h = fir_frequency_response(config, discrete_channel(config))
        inverse, svd = NormalInverseLMMSE(wave, h), TheoryLMMSE(wave, h)
        inverse.prepare(0.03)
        rng = np.random.default_rng(431)
        x, y = complex_normal(rng, (32, 3)), complex_normal(rng, (32, 3))
        for detect in (inverse.detect, lambda r: svd.detect(r, 0.03)):
            assert_allclose(detect(0.3*x + 0.7j*y), 0.3*detect(x) + 0.7j*detect(y), atol=3e-12)

    def test_uncoded_monte_carlo_and_predictions(self):
        config = Config(k=5, samples=64, frames=400, batch_size=73, ebn0_db=(0, 10, 25), codes=())
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _ = simulate(config)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertTrue(row["mse_theory_passed"])
            self.assertEqual(row["bit_count"], config.frames * config.bits_per_block)
            self.assertFalse(row["finite_constellation_prior_in_lmmse"])
        for db in config.ebn0_db:
            svd, ordinary = [row for row in rows if row["ebn0_db"] == db]
            self.assertEqual(svd["label"], "LMMSE")
            self.assertEqual(ordinary["modulation"], 16)

    def test_output_contract_and_removed_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["half_spacing_ofdm_sim.py", "--k", "2", "--samples", "32", "--frames", "20",
                    "--noise-trials", "1000", "--ebn0-db", "10", "--codes", "none", "--no-plots", "--output-dir", directory]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["detectors"], list(DETECTORS))
            self.assertTrue(report["comparison"]["same_payload_bits"])
            self.assertFalse(report["detection_notes"]["CP_validation_executed"])
            self.assertNotIn("cp_validation", report)
            self.assertNotIn("mmse_parameter_analysis", report)
            self.assertNotIn("normal_inverse_lmmse", report["curves"])
            self.assertEqual({path.name for path in Path(directory).iterdir()},
                             {"metrics.csv", "report.json", "summary.md", "constellations.npz"})


if __name__ == "__main__":
    unittest.main()
