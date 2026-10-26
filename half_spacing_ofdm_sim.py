"""半子载波双组 OFDM：LMMSE、普通 OFDM 与 RS/LDPC 信道编码对比。

Run in the communication environment:
    python half_spacing_ofdm_sim.py
    python half_spacing_ofdm_sim.py --channel awgn
    python half_spacing_ofdm_sim.py --verify-only

半间隔系统使用 2K 个 M-QAM 子载波，普通 OFDM 使用 K 个 M**2-QAM 子载波。
两者共享载荷比特、有效时长、采样数、FIR、CP 长度及期望发射能量。
Eb/N0 按信息比特定义、不计 CP；编码码率 R 对应 nu=1/(R*b*EbN0)。
含 CP 的总期望能量均为 Eblock*(1+Lcp/N)。

Matched model: z=C_N D a+eta, R_eta=nu*C_N.
SVD: B=S D/sqrt(N)=U diag(sigma) V^H, y=r/(mu*sqrt(N)),
     a_hat=V diag(sigma/(sigma**2+nu)) U^H y.
This implements theoretical LMMSE without finite-constellation priors.
Ordinary OFDM uses FFT + coherent one-tap ZF, equivalent to gain-compensated
scalar MMSE for hard decisions. All receivers use final nearest-QAM slicing.
RS 仅做硬译码；LDPC 使用归一化 min-sum 和增益/残余方差校准的软信息。
仅保留矩形脉冲理论匹配模型，不执行脉冲成形、选参、EP 或 CP 验证实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


DETECTORS = ("svd_lmmse", "ordinary_ofdm_zf")
CURVE_LABELS = {"svd_lmmse": "LMMSE", "ordinary_ofdm_zf": "OFDM",
                "svd_lmmse_rs": "LMMSE + RS", "ordinary_ofdm_zf_rs": "OFDM + RS",
                "svd_lmmse_ldpc": "LMMSE + LDPC", "ordinary_ofdm_zf_ldpc": "OFDM + LDPC"}
CURVE_STYLES = {
    "svd_lmmse": dict(color="#0072B2", marker="o", linestyle="-"),
    "ordinary_ofdm_zf": dict(color="#D55E00", marker="s", linestyle="--"),
    "svd_lmmse_rs": dict(color="#009E73", marker="^", linestyle="-."),
    "ordinary_ofdm_zf_rs": dict(color="#CC79A7", marker="D", linestyle=":"),
    "svd_lmmse_ldpc": dict(color="#332288", marker="v", linestyle=(0, (6, 2))),
    "ordinary_ofdm_zf_ldpc": dict(color="#E69F00", marker="P", linestyle=(0, (5, 1, 1, 1))),
}
DEFAULT_EBN0_DB = (0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 120)


@dataclass(frozen=True)
class Config:
    k: int = 256
    samples: int = 512
    modulation: int = 4
    frames: int = 2000
    batch_size: int = 128
    ebn0_db: tuple[float, ...] = DEFAULT_EBN0_DB
    symbol_duration: float = 1.0
    block_energy: float = 1.0
    channel: str = "multipath"
    seed: int = 2026
    noise_trials: int = 12000
    cp_length: int | None = None
    tap_delays: tuple[int, ...] | None = None
    tap_gains: tuple[str, ...] = ("1", "0.45+0.2j", "0.2-0.15j")
    codes: tuple[str, ...] = ("rs", "ldpc")
    rs_c_exp: int = 4
    rs_packet_size: int = 11
    rs_nsym: int = 4
    ldpc_rate: str = "11/15"
    ldpc_max_iter: int = 30
    ldpc_min_sum_alpha: float = 0.8
    interleave: bool = True
    coding_frames: int | None = None

    @property
    def bits_per_symbol(self):
        return int(math.log2(self.modulation))

    @property
    def ordinary_modulation(self):
        return self.modulation ** 2

    @property
    def bits_per_block(self):
        return 2 * self.k * self.bits_per_symbol

    @property
    def mu(self):
        return math.sqrt(self.block_energy / (2 * self.k * self.symbol_duration))

    @property
    def ordinary_mu(self):
        return math.sqrt(self.block_energy / (self.k * self.symbol_duration))

    @property
    def effective_cp_length(self):
        return len(discrete_channel(self)) - 1 if self.cp_length is None else self.cp_length

    @property
    def block_duration(self):
        return self.symbol_duration * (1 + self.effective_cp_length / self.samples)

    def noise_parameters(self, db, code_rate=1.0):
        if not 0 < code_rate <= 1:
            raise ValueError("Require 0<code_rate<=1.")
        half_nu = 10.0 ** (-db / 10) / (code_rate * self.bits_per_symbol)
        ordinary_nu = half_nu / 2
        return half_nu, ordinary_nu, self.mu ** 2 * self.samples * half_nu

    def validate(self):
        if self.coding_frames is not None and self.coding_frames < 1:
            raise ValueError("coding-frames must be positive.")
        if len(set(self.codes)) != len(self.codes) or any(code not in ("rs", "ldpc") for code in self.codes):
            raise ValueError("codes must contain distinct rs/ldpc entries.")
        if "rs" in self.codes:
            n = self.rs_packet_size + self.rs_nsym
            if not 2 <= self.rs_c_exp <= 8 or min(self.rs_packet_size, self.rs_nsym) < 1:
                raise ValueError("RS requires 2<=c_exp<=8 and positive information/parity symbol counts.")
            if (2 ** self.rs_c_exp - 1) % n:
                raise ValueError("RS length must divide 2**c_exp-1 (default RS(15,11), GF(16)).")
        if self.ldpc_rate not in ("11/15", "14/15") or self.ldpc_max_iter < 1:
            raise ValueError("LDPC requires rate 11/15 or 14/15 and positive max iterations.")
        if not np.isfinite(self.ldpc_min_sum_alpha) or not 0 < self.ldpc_min_sum_alpha <= 1:
            raise ValueError("Require 0<ldpc-min-sum-alpha<=1.")
        if self.k < 1 or self.samples < 2 * self.k:
            raise ValueError("Require K>=1 and samples>=2K.")
        side = math.isqrt(self.modulation) if self.modulation > 0 else 0
        if side < 2 or side * side != self.modulation or side & (side - 1):
            raise ValueError("modulation must be square power-of-two QAM (4, 16, 64, ...).")
        if min(self.frames, self.batch_size) < 1 or self.noise_trials < 1000 or self.seed < 0:
            raise ValueError("Require positive frame/batch counts, noise-trials>=1000 and seed>=0.")
        if any(not np.isfinite(x) or x <= 0 for x in (self.symbol_duration, self.block_energy)):
            raise ValueError("Duration and energy must be finite and positive.")
        if not self.ebn0_db or not np.all(np.isfinite(self.ebn0_db)):
            raise ValueError("Eb/N0 must be a nonempty finite list.")
        if min(self.ebn0_db) < -100 or max(self.ebn0_db) > 120:
            raise ValueError("Double-precision waveform Eb/N0 is restricted to [-100,120] dB.")
        if self.channel not in ("awgn", "multipath"):
            raise ValueError("Unknown channel.")
        if self.cp_length is not None and not 0 <= self.cp_length <= self.samples:
            raise ValueError("Require 0<=cp-length<=N.")
        if self.tap_delays is not None and (not self.tap_delays or min(self.tap_delays) < 0 or
                                            max(self.tap_delays) >= self.samples):
            raise ValueError("tap-delays must lie in [0,N).")
        if len(self.tap_gains) != (len(self.tap_delays) if self.tap_delays is not None else 3):
            raise ValueError("tap-gains must match the number of delays.")
        discrete_channel(replace(self, channel="multipath"))
        if self.effective_cp_length < len(discrete_channel(self)) - 1:
            raise ValueError("CP must cover the largest nonzero FIR delay.")


def coupling_matrix(k, samples=None):
    delta = np.arange(k)[None, :] - np.arange(k)[:, None] + 0.5
    return 1j / (np.pi * delta) if samples is None else 1j / (samples * np.sin(np.pi * delta / samples))


def gram_matrix(p):
    return np.block([[np.eye(len(p)), p], [p.conj().T, np.eye(len(p))]])


def complex_normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), np.finfo(float).tiny))


def qam_constellation(order):
    side = math.isqrt(order)
    i, q = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    gi, gq = i.ravel() ^ (i.ravel() >> 1), q.ravel() ^ (q.ravel() >> 1)
    shifts = np.arange(int(math.log2(side)) - 1, -1, -1)
    labels = np.concatenate(((gi[:, None] >> shifts) & 1, (gq[:, None] >> shifts) & 1), axis=1).astype(np.uint8)
    points = (2 * i.ravel() - side + 1) + 1j * (2 * q.ravel() - side + 1)
    return points / math.sqrt(2 * (order - 1) / 3), labels


def qam_indices(a, order):
    side = math.isqrt(order)
    a = a * math.sqrt(2 * (order - 1) / 3)
    i = np.clip(np.rint((a.real + side - 1) / 2), 0, side - 1).astype(int)
    q = np.clip(np.rint((a.imag + side - 1) / 2), 0, side - 1).astype(int)
    return i * side + q


def paired_symbols(config, rng, frames):
    """Map exactly the same block payload to 2K M-QAM and K M**2-QAM tones."""
    hp, hl = qam_constellation(config.modulation)
    op, ol = qam_constellation(config.ordinary_modulation)
    hi = rng.integers(config.modulation, size=(2 * config.k, frames))
    payload = hl[hi].transpose(1, 0, 2).reshape(frames, config.bits_per_block)
    ordinary_bits = payload.reshape(frames, config.k, 2 * config.bits_per_symbol).transpose(1, 0, 2)
    weights = 1 << np.arange(2 * config.bits_per_symbol - 1, -1, -1)
    lookup = np.empty(config.ordinary_modulation, dtype=int)
    lookup[ol @ weights] = np.arange(config.ordinary_modulation)
    oi = lookup[ordinary_bits @ weights]
    return hp[hi], hi, op[oi], oi


def payload_symbols(config, payload):
    """Map a (frames, bits_per_block) binary payload to either waveform."""
    frames = len(payload)
    result = []
    for count, order in ((2 * config.k, config.modulation), (config.k, config.ordinary_modulation)):
        points, labels = qam_constellation(order)
        width = int(math.log2(order))
        weights = 1 << np.arange(width - 1, -1, -1)
        lookup = np.empty(order, dtype=int)
        lookup[labels @ weights] = np.arange(order)
        bits = np.asarray(payload).reshape(frames, count, width).transpose(1, 0, 2)
        indices = lookup[bits @ weights]
        result.extend((points[indices], indices))
    return tuple(result)


def qam_bit_llrs(estimate, order, gain, residual_variance):
    """Max-log LLR log P(bit=0)/P(bit=1) for estimate=gain*a+residual.

    residual_variance is COMPLEX variance, hence squared real-axis distances
    are divided by that variance (not by half that variance). Half-spacing
    residual interference is Gaussian-approximated; cross-tone correlation
    is not represented by this scalar demapper. No feedback to LMMSE is used.
    """
    side = math.isqrt(order)
    width = int(math.log2(side))
    levels = (2 * np.arange(side) - side + 1) / math.sqrt(2 * (order - 1) / 3)
    gray = np.arange(side) ^ (np.arange(side) >> 1)
    labels = (gray[:, None] >> np.arange(width - 1, -1, -1)) & 1
    gain = np.broadcast_to(np.asarray(gain, float), (len(estimate),))
    variance = np.broadcast_to(np.asarray(residual_variance, float), (len(estimate),))
    valid = (variance > 0) & np.isfinite(variance) & (gain > 0)
    output = []
    for component in (estimate.real, estimate.imag):
        distances = (component[:, :, None] - gain[:, None, None] * levels) ** 2
        for bit in range(width):
            delta = np.min(distances[:, :, labels[:, bit] == 1], axis=2) - np.min(distances[:, :, labels[:, bit] == 0], axis=2)
            output.append(np.divide(delta, variance[:, None], out=np.zeros_like(delta), where=valid[:, None]))
    return np.clip(np.stack(output, axis=2), -20.0, 20.0)


class Waveform:
    def __init__(self, config):
        self.config = config
        self.u = (np.arange(config.samples) + 0.5) / config.samples
        self.phase = np.exp(1j * np.pi * np.arange(config.k) / config.samples)
        self.shift = np.exp(1j * np.pi * self.u)

    def components(self, a):
        k, n = self.config.k, self.config.samples
        return tuple(self.config.mu * n * np.fft.ifft(group * self.phase[:, None], n=n, axis=0)
                     for group in (a[:k], a[k:]))

    def transmit(self, a):
        x0, x1 = self.components(a)
        return x0 + self.shift[:, None] * x1

    def with_cp(self, a, length):
        """Retain the established per-group CP followed by half-subcarrier shift."""
        if not 0 <= length <= self.config.samples:
            raise ValueError("CP length must lie in [0,N].")
        core = self.transmit(a)
        if length == 0:
            return core
        x0, x1 = self.components(a)
        negative_time = (np.arange(-length, 0) + 0.5) / self.config.samples
        prefix = x0[-length:] + np.exp(1j * np.pi * negative_time[:, None]) * x1[-length:]
        return np.vstack((prefix, core))

    def matched_filter(self, r):
        k = self.config.k
        y0 = np.fft.fft(r, axis=0)[:k] * self.phase.conj()[:, None]
        y1 = np.fft.fft(r * self.shift.conj()[:, None], axis=0)[:k] * self.phase.conj()[:, None]
        return self.config.symbol_duration / self.config.samples * np.vstack((y0, y1))


class OrdinaryOFDM:
    """K integer-spaced tones with ordinary cyclic prefix and M**2-QAM."""
    def __init__(self, config):
        self.config = config
        self.phase = np.exp(1j * np.pi * np.arange(config.k) / config.samples)

    def transmit(self, a):
        return self.config.ordinary_mu * self.config.samples * np.fft.ifft(
            a * self.phase[:, None], n=self.config.samples, axis=0)

    def with_cp(self, a, length):
        if not 0 <= length <= self.config.samples:
            raise ValueError("CP length must lie in [0,N].")
        core = self.transmit(a)
        return np.vstack((core[-length:], core)) if length else core

    def observe(self, received):
        return (np.fft.fft(received, axis=0)[:self.config.k] * self.phase.conj()[:, None] /
                (self.config.ordinary_mu * self.config.samples))

    def detect(self, received, h):
        observation = self.observe(received)
        return np.divide(observation, h[:, None], out=np.zeros_like(observation), where=abs(h[:, None]) > 0)

    def mse(self, nu, h):
        power = abs(h) ** 2
        error = np.divide(nu, power, out=np.ones_like(power), where=power > 0)
        return float(np.mean(error))


def discrete_channel(config):
    if config.channel == "awgn":
        return np.ones(1, complex)
    delays = np.asarray(config.tap_delays if config.tap_delays is not None else
                        np.rint(config.samples * np.array([0, 0.08, 0.19])), dtype=int)
    gains = np.asarray([complex(value) for value in config.tap_gains])
    if len(delays) != len(gains) or not np.all(np.isfinite(gains)):
        raise ValueError("FIR gains must be finite and match delays.")
    impulse = np.zeros(int(delays.max()) + 1, complex)
    np.add.at(impulse, delays, gains)
    norm = np.linalg.norm(impulse)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("The combined FIR must have positive finite energy.")
    return impulse[:np.flatnonzero(impulse)[-1] + 1] / norm


def fir_frequency_response(config, impulse):
    bins = np.r_[np.arange(config.k), np.arange(config.k) + 0.5]
    return np.exp(-2j * np.pi * bins[:, None] * np.arange(len(impulse)) / config.samples) @ impulse


class FIRLink:
    """Serialize blocks and preserve the convolution state between batches."""
    def __init__(self, wave, impulse, cp_length):
        self.wave, self.impulse, self.cp_length = wave, impulse, cp_length
        self.state = np.zeros(len(impulse) - 1, complex)

    def receive(self, a):
        blocks = self.wave.with_cp(a, self.cp_length)
        received, self.state = lfilter(self.impulse, [1.0], blocks.ravel(order="F"), zi=self.state)
        return received.reshape(blocks.shape, order="F")[self.cp_length:]


class TheoryLMMSE:
    """Current SVD receiver, equivalent to the colored matched-filter LMMSE."""
    def __init__(self, wave, h):
        self.config = wave.config
        self.b = wave.transmit(np.diag(h)) / (self.config.mu * math.sqrt(self.config.samples))
        self.u, self.singular_values, vh = np.linalg.svd(self.b, full_matrices=False)
        self.v = vh.conj().T

    def project(self, received):
        return self.u.conj().T @ received / (self.config.mu * math.sqrt(self.config.samples))

    def detect_projected(self, observation, nu):
        sigma = self.singular_values
        weights = np.divide(sigma, sigma ** 2 + nu, out=np.zeros_like(sigma), where=sigma ** 2 + nu > 0)
        return self.v @ (weights[:, None] * observation)

    def detect(self, received, nu):
        return self.detect_projected(self.project(received), nu)

    def mse(self, nu):
        return float(np.mean(nu / (self.singular_values ** 2 + nu)))

    def soft_parameters(self, nu):
        power = self.singular_values ** 2
        gain = abs(self.v) ** 2 @ (power / (power + nu))
        gain = np.clip(gain, 0.0, 1 - np.finfo(float).eps)
        return gain, gain * (1 - gain)


class NormalInverseLMMSE:
    """Explicit inverse of the theoretical normal matrix; no hidden fallback."""
    def __init__(self, wave, h):
        self.wave, self.h = wave, h
        c = gram_matrix(coupling_matrix(wave.config.k, wave.config.samples))
        self.normal = h.conj()[:, None] * c * h[None, :]

    def prepare(self, nu):
        matrix = self.normal + nu * np.eye(len(self.h))
        self.inverse = np.linalg.inv(matrix)
        self.condition_number = float(np.linalg.cond(matrix))
        self.inverse_residual = relative_error(matrix @ self.inverse, np.eye(len(self.h)))

    def detect(self, received):
        config = self.wave.config
        z = self.wave.matched_filter(received) / (config.mu * config.symbol_duration)
        return self.inverse @ (self.h.conj()[:, None] * z)

    def mse(self, nu, b):
        """MSE of the actual inverse-based map, including inverse roundoff."""
        weight = self.inverse @ b.conj().T
        bias = weight @ b - np.eye(len(self.h))
        return float((np.linalg.norm(bias) ** 2 + nu * np.linalg.norm(weight) ** 2) / len(self.h))


def verify_theory(config):
    """Verify the matched model and equal-rate bookkeeping, without CP experiments."""
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    wave = Waveform(config)
    n, k, t, mu = config.samples, config.k, config.symbol_duration, config.mu
    bins = np.r_[np.arange(k), np.arange(k) + 0.5]
    basis = np.exp(2j * np.pi * wave.u[:, None] * bins)
    c = gram_matrix(coupling_matrix(k, n))
    checks = []

    def check(name, value, limit):
        passed = bool(np.isfinite(value) and value <= limit)
        checks.append({"name": name, "value": float(value), "limit": float(limit), "passed": passed})
        print(f"  {'PASS' if passed else 'FAIL'} {name}: {value:.3e}", flush=True)

    check("sampled_Gram", relative_error(basis.conj().T @ basis / n, c), 3e-12)
    a = complex_normal(rng, (2 * k, 6))
    check("IFFT_transmitter", relative_error(wave.transmit(a), mu * basis @ a), 3e-12)
    check("FFT_matched_filter", relative_error(wave.matched_filter(wave.transmit(a)), mu * t * c @ a), 3e-12)
    covariance = np.zeros_like(c)
    nu = 0.3
    for start in range(0, config.noise_trials, config.batch_size):
        size = min(config.batch_size, config.noise_trials - start)
        white = mu * math.sqrt(nu * n) * complex_normal(rng, (n, size))
        eta = wave.matched_filter(white) / (mu * t * math.sqrt(nu))
        covariance += eta @ eta.conj().T
    check("matched_noise_covariance", relative_error(covariance / config.noise_trials, c),
          3 * math.sqrt(2 * k / config.noise_trials))
    # Verify algebra on a well-conditioned small system. Numerical disagreement
    # at the requested full size is measured in simulate(), not asserted away.
    small = replace(config, k=min(k, 3))
    sw = Waveform(small)
    h = fir_frequency_response(small, discrete_channel(small))
    inverse, svd = NormalInverseLMMSE(sw, h), TheoryLMMSE(sw, h)
    inverse.prepare(nu)
    noisy = complex_normal(rng, (n, 5))
    check("inverse_vs_SVD_small_system", relative_error(inverse.detect(noisy), svd.detect(noisy, nu)), 3e-11)
    predicted = nu * np.trace(inverse.inverse).real / (2 * small.k)
    check("LMMSE_MSE_formula", abs(svd.mse(nu) - predicted), 3e-12)
    ordinary = OrdinaryOFDM(config)
    ob = ordinary.transmit(np.eye(k)) / (config.ordinary_mu * math.sqrt(n))
    check("ordinary_orthogonality", relative_error(ob.conj().T @ ob, np.eye(k)), 3e-12)
    ha, hi, oa, oi = paired_symbols(config, rng, 3)
    hp, hl = qam_constellation(config.modulation)
    op, ol = qam_constellation(config.ordinary_modulation)
    half_payload = hl[hi].transpose(1, 0, 2).reshape(3, -1)
    ordinary_payload = ol[oi].transpose(1, 0, 2).reshape(3, -1)
    check("identical_payload_bits", np.count_nonzero(half_payload != ordinary_payload), 0)
    check("half_QAM_roundtrip", np.max(abs(qam_indices(ha, config.modulation) - hi)), 0)
    check("ordinary_QAM_roundtrip", np.max(abs(qam_indices(oa, config.ordinary_modulation) - oi)), 0)
    check("half_QAM_energy", abs(np.mean(abs(hp) ** 2) - 1), 1e-14)
    check("ordinary_QAM_energy", abs(np.mean(abs(op) ** 2) - 1), 1e-14)
    _, ordinary_nu, variance = config.noise_parameters(10)
    check("shared_noise_variance", abs(variance - config.ordinary_mu ** 2 * n * ordinary_nu), 1e-14 * variance)
    check("equal_bits_per_block", abs(config.bits_per_block - k * int(math.log2(config.ordinary_modulation))), 0)
    result = {"checks": checks, "passed": all(item["passed"] for item in checks),
              "sampling_P_relative_error": relative_error(coupling_matrix(k, n), coupling_matrix(k))}
    if not result["passed"]:
        raise RuntimeError("Matched-filter theory verification failed.")
    return result


def simulate(config, coding_analysis=None):
    wave, ordinary = Waveform(config), OrdinaryOFDM(config)
    impulse = discrete_channel(config)
    h = fir_frequency_response(config, impulse)
    ordinary_h = h[:config.k]
    receiver = TheoryLMMSE(wave, h)
    hp, hl = qam_constellation(config.modulation)
    op, ol = qam_constellation(config.ordinary_modulation)
    rows, constellation = [], {}
    if config.codes:
        # galois uses Numba internally; store its cache in the writable workspace.
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "numba"))
        from simulation.ofdm_channel_coding import CodingLayout
        from utils.Coder import LDPCCoder, RSCoder
    for code_index, code_name in enumerate(("none",) + config.codes):
        code, metadata = None, {}
        if code_name == "rs":
            code = RSCoder({"rs_c_exp": config.rs_c_exp, "rs_nsym": config.rs_nsym, "rs_packet_size": config.rs_packet_size})
            n_bits, k_bits = code.n * code.c_exp, code.k * code.c_exp
            metadata = {"decoder": "hard", "n_symbols": code.n, "k_symbols": code.k, "erasures": False}
        elif code_name == "ldpc":
            code = LDPCCoder({"ldpc_matrix_type": "ieee802153d_1440", "ldpc_standard_rate": config.ldpc_rate,
                              "ldpc_rate11_direction": "minus_literal", "ldpc_max_iter": config.ldpc_max_iter,
                              "ldpc_col_weight": 3, "ldpc_dv": 3, "ldpc_seed": config.seed,
                              "ldpc_decode_algorithm": "normalized_min_sum", "ldpc_llr_clip": 20.0,
                              "ldpc_hard_llr_value": 12.0, "ldpc_min_sum_alpha": config.ldpc_min_sum_alpha})
            n_bits, k_bits = code.n, code.k
            metadata = {"decoder": code.decode_algorithm, "input": "soft LLR", "max_iterations": code.max_iter}
        rate = 1.0 if code is None else code.efficiency
        layout = None if code is None else CodingLayout(n_bits, k_bits, config.bits_per_block,
                                                       config.seed, config.interleave)
        unit_frames = 1 if layout is None else layout.frames
        requested_frames = config.frames if code is None or config.coding_frames is None else config.coding_frames
        total_units = math.ceil(requested_frames / unit_frames)
        actual_frames = total_units * unit_frames
        units_per_batch = max(1, config.batch_size // unit_frames)
        if coding_analysis is not None and code is not None:
            coding_analysis[code_name] = {**metadata, "code": code_name, "n_bits": n_bits, "k_bits": k_bits, "rate": rate,
                                         "implementation": "utils.Coder." + type(code).__name__,
                                         "frames_per_interleaver_unit": layout.frames,
                                         "codewords_per_unit": layout.words, "interleaved": config.interleave,
                                         "requested_frames": requested_frames, "actual_frames": actual_frames,
                                         "padding_bits": 0, "information_bit_rate": rate * config.bits_per_block / config.block_duration}
        # Preserve the original uncoded RNG stream. Coded streams are independent.
        uncoded_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2]))
        for db_index, db in enumerate(config.ebn0_db):
            rng = uncoded_rng if code is None else np.random.default_rng(
                np.random.SeedSequence([config.seed, 20, code_index, db_index]))
            nu, ordinary_nu, time_variance = config.noise_parameters(db, rate)
            gain, residual = receiver.soft_parameters(nu)
            ordinary_variance = np.divide(ordinary_nu, abs(ordinary_h) ** 2,
                                          out=np.full(config.k, np.inf), where=abs(ordinary_h) > 0)
            links = (FIRLink(wave, impulse, config.effective_cp_length),
                     FIRLink(ordinary, impulse, config.effective_cp_length))
            warmup, _, ordinary_warmup, _ = paired_symbols(config, rng, 1)
            links[0].receive(warmup)
            links[1].receive(ordinary_warmup)
            stats = [{key: [] for key in ("mse", "ber", "gain")} for _ in DETECTORS]
            for item in stats:
                item.update(bit_errors=0, raw_bit_errors=0, symbol_errors=0, bit_count=0,
                            codewords=0, word_errors=0, decoder_failures=0, iteration_sum=0)
            for start in range(0, total_units, units_per_batch):
                unit_count = min(units_per_batch, total_units - start)
                size = unit_count * unit_frames
                if code is None:
                    a, indices, ordinary_a, ordinary_indices = paired_symbols(config, rng, size)
                    information = hl[indices].transpose(1, 0, 2).ravel()
                    coded = information
                else:
                    information = rng.integers(0, 2, size=unit_count * layout.info_bits, dtype=np.uint8)
                    coded = code.encode(information)
                    payload = layout.interleave(coded).reshape(size, config.bits_per_block)
                    a, indices, ordinary_a, ordinary_indices = payload_symbols(config, payload)
                noise = math.sqrt(time_variance) * complex_normal(rng, (config.samples, size))
                received = links[0].receive(a) + noise
                ordinary_received = links[1].receive(ordinary_a) + noise
                estimates = (receiver.detect(received, nu), ordinary.detect(ordinary_received, ordinary_h))
                reference_ber = None
                for position, estimate in enumerate(estimates):
                    labels, sent, sent_indices, order = ((hl, a, indices, config.modulation) if position == 0 else
                                                         (ol, ordinary_a, ordinary_indices, config.ordinary_modulation))
                    decisions = qam_indices(estimate, order)
                    hard = labels[decisions].transpose(1, 0, 2).ravel()
                    decoded, debug = hard, None
                    if code is not None:
                        if code_name == "rs":
                            # RS receives only nearest-QAM hard bits; no reliability/erasure path.
                            decoder_input = layout.deinterleave(hard)
                        else:
                            llrs = qam_bit_llrs(estimate, order, gain if position == 0 else np.ones(config.k),
                                               residual if position == 0 else ordinary_variance)
                            decoder_input = layout.deinterleave(llrs.transpose(1, 0, 2).ravel())
                        if code_name == "rs":
                            decoded = code.decode(1.0 - 2.0 * decoder_input, len(information))
                        else:
                            decoded, debug = code.decode(decoder_input, original_bit_len=len(information), ldpc_padding_bits=0)
                    if len(decoded) != len(information):
                        raise RuntimeError("Decoder returned the wrong number of information bits.")
                    errors = decoded != information
                    block_ber = np.mean(errors.reshape(unit_count, -1), axis=1)
                    if reference_ber is None:
                        reference_ber = block_ber
                    item = stats[position]
                    item["bit_errors"] += int(np.count_nonzero(errors))
                    item["bit_count"] += len(information)
                    raw = hard if code is None else layout.deinterleave(hard)
                    item["raw_bit_errors"] += int(np.count_nonzero(raw != coded))
                    item["symbol_errors"] += int(np.count_nonzero(decisions != sent_indices))
                    item["mse"].append(np.mean(abs(estimate - sent) ** 2, axis=0))
                    item["ber"].append(block_ber)
                    item["gain"].append(reference_ber - block_ber)
                    if code is not None:
                        item["codewords"] += len(information) // k_bits
                        item["word_errors"] += int(np.count_nonzero(np.any(errors.reshape(-1, k_bits), axis=1)))
                    if debug is not None:
                        item["decoder_failures"] += len(debug["failed_blocks"])
                        item["iteration_sum"] += sum(debug["iterations"])
                if start == 0 and code is None:
                    constellation = {name: estimate.ravel()[:2000] for name, estimate in zip(DETECTORS, estimates)}
                    constellation.update({"half_ideal_points": hp, "ordinary_ideal_points": op, "ebn0_db": np.asarray(db)})
            predictions = (receiver.mse(nu), ordinary.mse(ordinary_nu, ordinary_h))
            for position, item in enumerate(stats):
                metrics = {key: np.concatenate(item[key]) for key in ("mse", "ber", "gain")}

                def standard_error(key):
                    values = metrics[key]
                    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None

                name = DETECTORS[position] + ("" if code is None else "_" + code_name)
                mse, se = float(np.mean(metrics["mse"])), standard_error("mse")
                tones = 2 * config.k if position == 0 else config.k
                row = {"detector": name, "label": CURVE_LABELS[name],
                       "system": "half_spacing" if position == 0 else "ordinary_ofdm", "coding": code_name,
                       "ebn0_db": db, "subcarriers": tones,
                       "modulation": config.modulation if position == 0 else config.ordinary_modulation,
                       "code_rate": rate, "frames": actual_frames, "bits_per_block": config.bits_per_block,
                       "information_bit_rate": rate * config.bits_per_block / config.block_duration,
                       "bit_count": item["bit_count"], "bit_errors": item["bit_errors"],
                       "ber": item["bit_errors"] / item["bit_count"],
                       "ber_block_standard_error": standard_error("ber"),
                       "standard_error_unit": "OFDM block" if code is None else "coding/interleaver unit",
                       "independent_units": total_units, "coded_bit_count": actual_frames * config.bits_per_block,
                       "raw_bit_errors": item["raw_bit_errors"],
                       "raw_ber": item["raw_bit_errors"] / (actual_frames * config.bits_per_block),
                       "ser": item["symbol_errors"] / (tones * actual_frames), "mse": mse,
                       "predicted_mse": predictions[position], "mse_standard_error": se,
                       "mse_theory_passed": bool(abs(mse - predictions[position]) <= max(6 * se, 1e-12 * predictions[position]))
                       if code is None and actual_frames >= 100 else None,
                       "paired_ber_gain_vs_lmmse_same_code": float(np.mean(metrics["gain"])),
                       "paired_gain_block_standard_error": standard_error("gain"),
                       "decoder_mode": "none" if code is None else ("hard" if code_name == "rs" else "soft"),
                       "decoder_codewords": item["codewords"],
                       "decoder_failed_codewords": None if code_name == "rs" else item["decoder_failures"],
                       "information_codeword_fer": item["word_errors"] / item["codewords"] if code is not None else None,
                       "mean_decoder_iterations": item["iteration_sum"] / item["codewords"] if code_name == "ldpc" else None,
                       "interleaver_frames": unit_frames if config.interleave and code is not None else 0,
                       "normalized_noise_variance": nu if position == 0 else ordinary_nu,
                       "time_noise_variance": time_variance, "regularization_lambda": nu if position == 0 else None,
                       "finite_constellation_prior_in_lmmse": False}
                rows.append(row)
                print(f"  {CURVE_LABELS[name]} {db:g} dB: BER={row['ber']:.6g}, raw BER={row['raw_ber']:.6g}, "
                      f"R={rate:.6g}, frames={actual_frames}, information bits={item['bit_count']}", flush=True)
    return rows, constellation


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_results(config, report, constellation, output):
    if not report.get("results"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw_ber(axis, detail=False):
        for name in dict.fromkeys(row["detector"] for row in report["results"]):
            selected = [row for row in report["results"] if row["detector"] == name]
            style = CURVE_STYLES[name]
            axis.semilogy([row["ebn0_db"] for row in selected],
                          [row["ber"] if row["bit_errors"] else np.nan for row in selected],
                          **style, linewidth=1.8, markersize=6, markerfacecolor="white",
                          markeredgewidth=1.3, label=CURVE_LABELS[name])
            zeros = [row for row in selected if not row["bit_errors"]]
            if zeros:
                axis.scatter([row["ebn0_db"] for row in zeros], [1 / row["bit_count"] for row in zeros],
                             marker=style["marker"], facecolors="none", edgecolors=style["color"], s=48)
                for row in zeros:
                    axis.annotate("", xy=(row["ebn0_db"], 0.55 / row["bit_count"]),
                                  xytext=(row["ebn0_db"], 0.95 / row["bit_count"]),
                                  arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8))
        axis.set(xlabel="Information Eb/N0 (dB)", ylabel="Information-bit BER")
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.10)
        axis.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    handlelength=3.3, framealpha=0.95)
        axis.text(0.98, 0.08, "Arrows: zero errors (markers at 1/Nbits)",
                  transform=axis.transAxes, ha="right", fontsize=8)
        if detail:
            inset = axis.inset_axes([0.46, 0.49, 0.51, 0.40])
            for name in dict.fromkeys(row["detector"] for row in report["results"]):
                selected = [row for row in report["results"] if row["detector"] == name and row["system"] == "half_spacing"]
                if selected:
                    inset.plot([row["ebn0_db"] for row in selected], [row["ber"] for row in selected],
                               **CURVE_STYLES[name], linewidth=1.5, markersize=4, markerfacecolor="white")
            inset.set_title("LMMSE detail", fontsize=9)
            inset.tick_params(labelsize=8)
            inset.grid(True, alpha=0.2)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    draw_ber(axis, detail=True)
    axis.set_title("BER comparison")
    fig.savefig(output / "ber_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    draw_ber(axes[0])
    for name in DETECTORS:
        selected = [row for row in report["results"] if row["detector"] == name]
        axes[1].semilogy([row["ebn0_db"] for row in selected], [row["mse"] for row in selected],
                         **CURVE_STYLES[name], markerfacecolor="white", label=CURVE_LABELS[name])
        axes[1].semilogy([row["ebn0_db"] for row in selected], [row["predicted_mse"] for row in selected],
                         ":", color=CURVE_STYLES[name]["color"], alpha=0.8)
    axes[1].set(xlabel="Information Eb/N0 (dB)", ylabel="Uncoded symbol MSE (dotted: theory)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.savefig(output / "performance.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, name in zip(axes, DETECTORS):
        estimate = constellation[name]
        points = constellation["ordinary_ideal_points" if name == "ordinary_ofdm_zf" else "half_ideal_points"]
        axis.scatter(estimate.real, estimate.imag, s=5, alpha=0.3)
        axis.scatter(points.real, points.imag, marker="x", color="red")
        axis.set(xlabel="I", ylabel="Q", aspect="equal", title=CURVE_LABELS[name])
        axis.grid(True, alpha=0.3)
    fig.savefig(output / "constellations.png", dpi=160)
    plt.close(fig)


def write_summary(config, report, output):
    lines = ["# LMMSE / OFDM 与 RS、LDPC 编码性能", "",
             f"半间隔系统使用 {2 * config.k} 个 {config.modulation}-QAM 子载波，普通 OFDM 使用 "
             f"{config.k} 个 {config.ordinary_modulation}-QAM 子载波；每块承载 {config.bits_per_block} 个调制比特。",
             f"有效时长 T={config.symbol_duration:g}，N={config.samples}，CP={config.effective_cp_length}。", "",
             "两种系统在同一编码配置下使用同一载荷、同一码率、同一 FIR 和时域噪声，实现相同净数据速率。"
             "编码后的净速率为未编码速率的 R 倍；不同码率之间不声称净速率相同。",
             "横轴 Eb/N0 按信息比特定义，不计 CP 能量。编码时 nu=1/(R*log2(M)*EbN0)，"
             "保持每个发送块的能量不变，在相同信息 Eb/N0 下按码率调整噪声。", "",
             "半间隔接收机仍为原有理论 LMMSE，完整保留已知耦合矩阵和有色匹配噪声模型；"
             "普通 OFDM 使用 FFT 和单抽头相干均衡。显式求逆不再参与性能仿真或绘图。", "",
             "RS 仅使用最终 QAM 硬判决，不使用擦除或 Chase。LDPC 使用 max-log QAM LLR 和归一化 min-sum。"
             "LMMSE 软信息使用逐子载波有效增益及复残余方差 gain*(1-gain)，"
             "残余干扰采用标量高斯近似，未建模跨子载波相关性，译码结果不反馈给检测器。", "",
             f"理论匹配模型验证通过：{report['verification']['passed']}。实际 CP 与连续 FIR 链路保留，不执行 CP 验证。", ""]
    for name, data in report.get("coding", {}).items():
        lines.append(f"- {name.upper()}：({data['n_bits']},{data['k_bits']}) bit，R={data['rate']:.6g}，"
                     f"译码方式 {data['decoder']}；每个编码单元 {data['frames_per_interleaver_unit']} 个 OFDM 块，"
                     f"交织 {data['interleaved']}，实际发送 {data['actual_frames']} 块。")
    if report.get("results"):
        curves = list(dict.fromkeys(row["detector"] for row in report["results"]))
        lines += ["", "| Eb/N0 (dB) | " + " | ".join(CURVE_LABELS[name] for name in curves) + " |",
                  "| --- | " + " | ".join("---" for _ in curves) + " |"]
        for db in config.ebn0_db:
            selected = {row["detector"]: row for row in report["results"] if row["ebn0_db"] == db}
            lines.append(f"| {db:g} | " + " | ".join(f"{selected[name]['ber']:.8g}" for name in curves) + " |")
        lines += ["", "编码 BER 的分母仅包含译码后信息比特；raw_ber 为译码前编码比特 BER。"
                  "完整传输码字，按码长与 OFDM 块长的最小公倍数组织，块数向上取整，没有补零或丢弃失败码字。"
                  "译码失败也计入 BER；information_codeword_fer 根据真实信息比特计算，包含未检出的误译码。",
                  "编译码直接调用 utils.Coder.RSCoder/LDPCCoder 的公开接口。"
                  "RSCoder.decode 未返回失败计数，因此 RS 的 decoder_failed_codewords 留空，不假定为零。",
                  "默认随机交织在各编码单元内进行，两种系统共用同一排列并正确解交织。"
                  "误码统计标准误按独立编码/交织单元计算；未编码按 OFDM 块计算。", "",
                  "BER=0 仅表示有限样本未观察到误码。箭头标识的点放在 1/Nbits 处，"
                  "仅为对数坐标中的显示标记，不是置信界。",
                  "未编码 MSE 与各自线性理论公式核对；编码行的 MSE 是译码前符号误差，"
                  "预测仍基于单位输入协方差，不作为编码 BER 的理论保证。", "",
                  "结果：metrics.csv、report.json、summary.md、constellations.npz；图像："
                  "ber_comparison.png、performance.png、constellations.png。"]
    if "high_snr_analysis" in report:
        lines += ["", "高信噪比验证见 high_snr_analysis/summary.md 与 lmmse_high_snr.png。"
                  "完整规模的无平台结论来自满秩和 MSE/BER 上界；高精度数值参考是较小规模的等价特征信道。"
                  "不能把有限样本零误码或双精度时域极高 SNR 外推当作证明。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--k", type=int, default=256, help="Tones per half-spacing group; ordinary OFDM uses K tones")
    sizes.add_argument("--subcarriers", type=int, help="Total half-spacing tones (positive even number)")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--modulation", type=int, default=4, help="Half-spacing QAM order; ordinary OFDM uses its square")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ebn0-db", type=float, nargs="+", default=list(DEFAULT_EBN0_DB))
    parser.add_argument("--symbol-duration", type=float, default=1.0)
    parser.add_argument("--block-energy", type=float, default=1.0)
    parser.add_argument("--channel", choices=("awgn", "multipath"), default="multipath")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-trials", type=int, default=12000)
    parser.add_argument("--cp-length", type=int, default=None, help="Defaults to the largest nonzero FIR delay")
    parser.add_argument("--tap-delays", type=int, nargs="+", default=None)
    parser.add_argument("--tap-gains", nargs="+", default=["1", "0.45+0.2j", "0.2-0.15j"])
    parser.add_argument("--codes", nargs="+", choices=("rs", "ldpc", "none"), default=["rs", "ldpc"])
    parser.add_argument("--rs-c-exp", type=int, default=4)
    parser.add_argument("--rs-packet-size", type=int, default=11)
    parser.add_argument("--rs-nsym", type=int, default=4)
    parser.add_argument("--ldpc-rate", choices=("11/15", "14/15"), default="11/15")
    parser.add_argument("--ldpc-max-iter", type=int, default=30)
    parser.add_argument("--ldpc-min-sum-alpha", type=float, default=0.8)
    parser.add_argument("--interleave", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coding-frames", type=int, default=None, help="Optional coded-frame budget; defaults to --frames")
    parser.add_argument("--verify-only", action="store_true", help="Only verify matched-filter algebra (no CP experiments)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--high-snr-analysis", action="store_true", help="High-precision uncoded LMMSE reference and full-size no-floor bound")
    parser.add_argument("--analysis-k", type=int, nargs="+", default=[4, 8, 16, 32], help="Per-group K for high-precision reference, 1..32")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results/half_spacing_ofdm_comparison"))
    args = parser.parse_args()
    if any(k < 1 or k > 32 for k in args.analysis_k):
        parser.error("analysis-k values must be in [1,32].")
    if args.subcarriers is not None:
        if args.subcarriers < 2 or args.subcarriers % 2:
            parser.error("subcarriers must be an even number >=2.")
        args.k = args.subcarriers // 2
    if "none" in args.codes:
        if len(args.codes) != 1:
            parser.error("--codes none cannot be combined with rs/ldpc.")
        args.codes = []
    tuple_fields = ("ebn0_db", "tap_delays", "tap_gains", "codes")
    config = Config(**{name: tuple(getattr(args, name)) if name in tuple_fields and getattr(args, name) is not None
                       else getattr(args, name) for name in Config.__dataclass_fields__})
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return args, config


def main():
    args, config = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Equal-rate comparison: {2 * config.k} x {config.modulation}-QAM vs "
          f"{config.k} x {config.ordinary_modulation}-QAM | N={config.samples}, "
          f"CP={config.effective_cp_length}, {config.bits_per_block} bits/block", flush=True)
    report = {"config": asdict(config), "detectors": list(DETECTORS),
              "comparison": {"half_subcarriers": 2 * config.k, "ordinary_subcarriers": config.k,
                             "half_modulation": config.modulation, "ordinary_modulation": config.ordinary_modulation,
                             "bits_per_block": config.bits_per_block, "block_duration": config.block_duration,
                             "uncoded_payload_bit_rate": config.bits_per_block / config.block_duration,
                             "same_payload_bits": True, "same_time_domain_noise": True,
                             "same_expected_transmit_energy": True, "ebn0_includes_cp_energy": False,
                             "expected_energy_including_cp": config.block_energy * (1 + config.effective_cp_length / config.samples),
                             "nominal_bandwidth": config.k / config.symbol_duration,
                             "ordinary_equalizer": "FFT + coherent one-tap ZF (gain-compensated scalar MMSE hard decisions)"},
              "detection_notes": {"half_cp_scheme": "per_group_cp_then_half_shift", "ordinary_cp_scheme": "ordinary_cyclic_copy",
                                  "CP_samples_used_for_detection": False, "finite_constellation_prior_used": False,
                                  "regularization": "lambda=nu; no parameter tuning", "CP_validation_executed": False},
              "coding_notes": {"ebn0_reference": "information bit; CP energy excluded", "noise_rate_correction": "nu=1/(R*b*EbN0)",
                               "rs_decoder": "hard only; no erasures or Chase", "ldpc_llrs": "max-log QAM; positive means bit 0",
                               "lmmse_llr_model": "gain=diag(WB); complex residual variance=gain*(1-gain); scalar Gaussian approximation",
                               "lmmse_llr_cross_tone_correlation_modeled": False, "decoder_feedback_to_lmmse": False,
                               "padding_policy": "LCM codeword/OFDM framing; no padding or discarded codewords"},
              "verification": verify_theory(config)}
    if not args.verify_only:
        coding_analysis = {}
        rows, constellation = simulate(config, coding_analysis)
        report["coding"] = coding_analysis
        report["curves"] = [row["detector"] for row in rows if row["ebn0_db"] == config.ebn0_db[0]]
        report["results"] = rows
        write_csv(output / "metrics.csv", rows)
        np.savez_compressed(output / "constellations.npz", **constellation)
        if not args.no_plots:
            plot_results(config, report, constellation, output)
    if args.high_snr_analysis:
        from simulation.ofdm_lmmse_high_snr import run_analysis
        report["high_snr_analysis"] = run_analysis(config, args.analysis_k, output / "high_snr_analysis", args.no_plots)
    report["simulation_passed"] = all(row["mse_theory_passed"] is not False for row in report.get("results", []))
    report["simulation_passed"] = report["simulation_passed"] and report.get("high_snr_analysis", {}).get("passed", True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_summary(config, report, output)
    print(f"Results: {output}", flush=True)
    if not report["simulation_passed"]:
        raise RuntimeError("Measured MSE differs from the linear-model prediction; inspect report.json.")


if __name__ == "__main__":
    main()
