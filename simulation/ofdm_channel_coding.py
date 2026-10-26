"""OFDM codeword framing and interleaving only; codecs live in utils.Coder."""
from __future__ import annotations

import math
import numpy as np


class CodingLayout:
    """An integer number of codewords and OFDM blocks, with no padding bits."""
    def __init__(self, n, k, bits_per_block, seed, interleave=True):
        divisor = math.gcd(n, bits_per_block)
        self.frames = n // divisor
        self.words = bits_per_block // divisor
        self.coded_bits = self.words * n
        self.info_bits = self.words * k
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31, n, k]))
        self.permutation = rng.permutation(self.coded_bits) if interleave else np.arange(self.coded_bits)

    def interleave(self, bits):
        return np.asarray(bits).reshape(-1, self.coded_bits)[:, self.permutation].ravel()

    def deinterleave(self, values):
        source = np.asarray(values).reshape(-1, self.coded_bits)
        result = np.empty_like(source)
        result[:, self.permutation] = source
        return result.ravel()
