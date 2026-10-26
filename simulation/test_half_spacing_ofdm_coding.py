"""FEC bit order, soft-information calibration and project decoder equivalence."""
import contextlib
import io
import unittest
import os
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from half_spacing_ofdm_sim import (
    Config, TheoryLMMSE, Waveform, discrete_channel, fir_frequency_response,
    qam_bit_llrs, qam_constellation, simulate,
)
from simulation.ofdm_channel_coding import CodingLayout
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "numba"))
from utils.Coder import LDPCCoder, RSCoder


class TestChannelCoding(unittest.TestCase):
    def test_project_rs_corrects_two_symbol_errors(self):
        code = RSCoder({"rs_c_exp": 4, "rs_packet_size": 11, "rs_nsym": 4})
        information = np.random.default_rng(798).integers(0, 2, size=3 * 44, dtype=np.uint8)
        encoded = code.encode(information)
        corrupted = encoded.reshape(3, 15, 4).copy()
        corrupted[:, 1, 0] ^= 1
        corrupted[:, 12, 2] ^= 1
        decoded = code.decode(1.0 - 2.0 * corrupted.ravel(), len(information))
        assert_allclose(decoded, information, atol=0)

    def test_complete_codewords_and_interleaving_inverse(self):
        for n, k, frames, words in ((60, 44, 15, 256), (1440, 1056, 45, 32)):
            for enabled in (True, False):
                layout = CodingLayout(n, k, 1024, 2026, enabled)
                self.assertEqual((layout.frames, layout.words), (frames, words))
                values = np.linspace(-3, 3, 2 * layout.coded_bits)
                assert_allclose(layout.deinterleave(layout.interleave(values)), values, atol=0)
                self.assertEqual(layout.coded_bits, frames * 1024)
                self.assertEqual(layout.info_bits, words * k)

    def test_code_rate_noise_and_net_rate(self):
        config = Config()
        for rate in (11 / 15, 14 / 15):
            half, ordinary, variance = config.noise_parameters(10, rate)
            self.assertAlmostEqual(variance, config.noise_parameters(10)[2] / rate)
            self.assertAlmostEqual(variance, config.ordinary_mu ** 2 * config.samples * ordinary)
            self.assertAlmostEqual(config.bits_per_block * rate / config.block_duration,
                                   config.k * 4 * rate / config.block_duration)
        for changes in ({"codes": ("bad",)}, {"rs_packet_size": 12}, {"rs_c_exp": 9},
                        {"ldpc_max_iter": 0}, {"ldpc_rate": "1/2"}, {"ldpc_min_sum_alpha": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes).validate()

    def test_lmmse_gain_and_residual_match_full_matrix(self):
        config = Config(k=4, samples=32)
        wave = Waveform(config)
        receiver = TheoryLMMSE(wave, fir_frequency_response(config, discrete_channel(config)))
        nu = 0.05
        gain, residual = receiver.soft_parameters(nu)
        weight = (receiver.v * (receiver.singular_values / (receiver.singular_values ** 2 + nu))) @ receiver.u.conj().T
        effective = weight @ receiver.b
        assert_allclose(gain, np.diag(effective).real, atol=3e-14)
        np.fill_diagonal(effective, 0)
        expected = np.sum(abs(effective) ** 2, axis=1) + nu * np.sum(abs(weight) ** 2, axis=1)
        assert_allclose(residual, expected, atol=3e-14)

    def test_qam_llr_sign_scaling_and_gain_against_full_constellation(self):
        rng = np.random.default_rng(17)
        estimate = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
        gain, variance = np.array([0.2, 0.6, 1.0]), np.array([0.1, 0.3, 0.8])
        for order in (4, 16, 64):
            points, labels = qam_constellation(order)
            distances = abs(estimate[:, :, None] - gain[:, None, None] * points) ** 2 / variance[:, None, None]
            expected = np.stack([np.min(distances[:, :, labels[:, bit] == 1], axis=2) -
                                 np.min(distances[:, :, labels[:, bit] == 0], axis=2)
                                 for bit in range(labels.shape[1])], axis=2)
            assert_allclose(qam_bit_llrs(estimate, order, gain, variance), np.clip(expected, -20, 20), atol=3e-14)
        erased = qam_bit_llrs(estimate, 4, np.zeros(3), np.zeros(3))
        assert_allclose(erased, 0, atol=0)

    def test_coded_link_roundtrip_and_information_bit_accounting(self):
        config = Config(k=2, samples=32, frames=30, ebn0_db=(35,), noise_trials=1000)
        analysis = {}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(RSCoder, "encode", autospec=True, side_effect=RSCoder.encode) as rs_encode, \
                patch.object(RSCoder, "decode", autospec=True, side_effect=RSCoder.decode) as rs_decode, \
                patch.object(LDPCCoder, "encode", autospec=True, side_effect=LDPCCoder.encode) as ldpc_encode, \
                patch.object(LDPCCoder, "decode", autospec=True, side_effect=LDPCCoder.decode) as ldpc_decode:
            rows, _ = simulate(config, analysis)
            self.assertTrue(all(method.called for method in (rs_encode, rs_decode, ldpc_encode, ldpc_decode)))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["label"] for row in rows},
                         {"LMMSE", "OFDM", "LMMSE + RS", "OFDM + RS", "LMMSE + LDPC", "OFDM + LDPC"})
        for row in rows:
            self.assertEqual(row["bit_errors"], 0)
            if row["coding"] != "rs":
                self.assertEqual(row["decoder_failed_codewords"], 0)
            self.assertAlmostEqual(row["bit_count"], row["coded_bit_count"] * row["code_rate"])
        self.assertEqual(analysis["rs"]["decoder"], "hard")
        self.assertEqual(analysis["ldpc"]["input"], "soft LLR")
        self.assertEqual(analysis["rs"]["implementation"], "utils.Coder.RSCoder")
        self.assertEqual(analysis["ldpc"]["implementation"], "utils.Coder.LDPCCoder")
        self.assertTrue(all(data["padding_bits"] == 0 for data in analysis.values()))


if __name__ == "__main__":
    unittest.main()
