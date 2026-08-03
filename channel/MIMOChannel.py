"""频率选择性 MIMO 基带信道。"""

from __future__ import annotations

import numpy as np

from utils.MIMOUtils import ensure_antenna_axis, validate_mimo_params


class MIMOChannel:
    """实现 y_r[n] = sum_t h_rt[n] * x_t[n] + w_r[n]。"""

    def __init__(self, params, channel_impulse_response=None):
        self.params = params
        self.num_tx, self.num_rx, _ = validate_mimo_params(params)
        self.nfft = int(params.get("subwave_num"))
        self.random_seed = params.get("mimo_channel_seed")
        if self.random_seed is None:
            self.random_seed = params.get("random_seed")
        self.rng = np.random.default_rng(self.random_seed)
        self.measurement_diagnostics = None
        self.channel_impulse_response = (
            self._generate_channel()
            if channel_impulse_response is None
            else self._validate_channel(channel_impulse_response)
        )
        self.channel_frequency_response = np.fft.fft(
            self.channel_impulse_response, n=self.nfft, axis=-1
        )
        self.noise_power = 0.0

    def _validate_channel(self, taps):
        taps = np.asarray(taps, dtype=np.complex128)
        if taps.ndim == 2:
            taps = taps[..., np.newaxis]
        expected = (self.num_rx, self.num_tx)
        if taps.ndim != 3 or taps.shape[:2] != expected:
            raise ValueError(
                "MIMO 信道冲激响应形状必须为 "
                f"(num_rx, num_tx, num_taps)，实际为 {taps.shape}"
            )
        return taps.copy()

    def _generate_channel(self):
        if str(self.params.get("multipath_source", "simulated")).lower() == "measured":
            if (self.num_rx, self.num_tx) != (2, 2):
                raise ValueError("完整 THz-MIMO 实测信道仅支持 num_rx=num_tx=2")
            from channel.MeasuredChannel import MeasuredChannel

            measured = MeasuredChannel(self.params)
            taps = measured.load_taps(full_mimo=True)
            if taps.shape[-1] - 1 > int(self.params.get("gi_length")):
                raise ValueError("实测 MIMO 信道有效长度超过 gi_length")
            self.measurement_diagnostics = dict(measured.diagnostics)
            return taps

        model = str(self.params.get("mimo_channel_model", "iid_rayleigh")).lower()
        num_taps = int(self.params.get("mimo_num_taps", 4))
        if not self.params.get("enable_multipath", False):
            num_taps = 1
        if num_taps < 1:
            raise ValueError("mimo_num_taps 必须为正整数")
        if num_taps - 1 > int(self.params.get("gi_length")):
            raise ValueError("MIMO 信道最大整数时延不能超过 gi_length")

        if model == "identity":
            taps = np.zeros((self.num_rx, self.num_tx, num_taps), dtype=np.complex128)
            for index in range(min(self.num_rx, self.num_tx)):
                taps[index, index, 0] = 1.0
            return taps

        powers_db = self.params.get("mimo_path_powers_db")
        if powers_db is None:
            powers = np.exp(-np.arange(num_taps, dtype=float))
        else:
            powers_db = np.asarray(powers_db, dtype=float).reshape(-1)
            if len(powers_db) != num_taps:
                raise ValueError("mimo_path_powers_db 长度必须等于 mimo_num_taps")
            powers = 10.0 ** (powers_db / 10.0)
        powers /= np.sum(powers)

        iid = (
            self.rng.standard_normal((self.num_rx, self.num_tx, num_taps))
            + 1j * self.rng.standard_normal((self.num_rx, self.num_tx, num_taps))
        ) / np.sqrt(2.0)
        iid *= np.sqrt(powers)[np.newaxis, np.newaxis, :]

        if model == "rician":
            k_linear = 10.0 ** (float(self.params.get("mimo_rician_k_db", 0.0)) / 10.0)
            los = np.zeros_like(iid)
            los[:, :, 0] = 1.0 / np.sqrt(self.num_tx)
            iid = np.sqrt(1.0 / (k_linear + 1.0)) * iid + np.sqrt(
                k_linear / (k_linear + 1.0)
            ) * los
        elif model != "iid_rayleigh":
            raise ValueError(f"不支持的 mimo_channel_model: {model}")

        return self._apply_spatial_correlation(iid)

    def _apply_spatial_correlation(self, taps):
        tx_rho = float(self.params.get("mimo_tx_correlation", 0.0))
        rx_rho = float(self.params.get("mimo_rx_correlation", 0.0))
        if abs(tx_rho) >= 1.0 or abs(rx_rho) >= 1.0:
            raise ValueError("MIMO 空间相关系数绝对值必须小于 1")

        def correlation_matrix(size, rho):
            indices = np.arange(size)
            return rho ** np.abs(indices[:, None] - indices[None, :])

        def matrix_sqrt(matrix):
            values, vectors = np.linalg.eigh(matrix)
            return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.conj().T

        rt = matrix_sqrt(correlation_matrix(self.num_tx, tx_rho))
        rr = matrix_sqrt(correlation_matrix(self.num_rx, rx_rho))
        correlated = np.empty_like(taps)
        for tap_index in range(taps.shape[-1]):
            correlated[:, :, tap_index] = rr @ taps[:, :, tap_index] @ rt
        return correlated

    @staticmethod
    def _apply_iq(signal, gain_db, phase_deg):
        gain = 10.0 ** (float(gain_db) / 20.0)
        phase = np.deg2rad(float(phase_deg))
        alpha = 0.5 * (1.0 + gain * np.exp(-1j * phase))
        beta = 0.5 * (1.0 - gain * np.exp(1j * phase))
        return alpha * signal + beta * np.conj(signal)

    def _apply_cfo_and_phase_noise(self, signal, sample_rate):
        if not self.params.get("enable_cfo", False) and not self.params.get("enable_phase_noise", False):
            return signal
        num_samples = signal.shape[1]
        phase = np.zeros(num_samples, dtype=float)
        if self.params.get("enable_cfo", False):
            frequency_offset = float(self.params.get("ppm")) * 1e-6 * float(self.params.get("fc"))
            phase += 2.0 * np.pi * frequency_offset * np.arange(num_samples) / sample_rate
        if self.params.get("enable_phase_noise", False):
            phase += np.cumsum(
                self.rng.normal(0.0, float(self.params.get("phase_noise_std")), num_samples)
            )
        return signal * np.exp(1j * phase)[np.newaxis, :]

    def apply(self, signal_dict):
        tx = ensure_antenna_axis(
            signal_dict["signal_stream"], self.num_tx, "发射 signal_stream"
        ).astype(np.complex128, copy=True)
        if self.params.get("enable_iq_imbalance", False) and str(
            self.params.get("iq_imbalance_position", "rx")
        ).lower() in {"tx", "both"}:
            tx = self._apply_iq(
                tx,
                self.params.get("tx_iq_gain_imbalance_db"),
                self.params.get("tx_iq_phase_imbalance_deg"),
            )

        num_samples = tx.shape[1]
        rx = np.zeros((self.num_rx, num_samples), dtype=np.complex128)
        for rx_index in range(self.num_rx):
            for tx_index in range(self.num_tx):
                convolved = np.convolve(
                    tx[tx_index], self.channel_impulse_response[rx_index, tx_index], mode="full"
                )
                rx[rx_index] += convolved[:num_samples]

        signal_power = float(np.mean(np.abs(rx) ** 2))
        if self.params.get("enable_awgn", False):
            snr_linear = 10.0 ** (float(self.params.get("SNRdB")) / 10.0)
            self.noise_power = signal_power / snr_linear
            noise = np.sqrt(self.noise_power / 2.0) * (
                self.rng.standard_normal(rx.shape) + 1j * self.rng.standard_normal(rx.shape)
            )
            rx += noise

        sample_rate = float(signal_dict["sample_rate_Hz"])
        rx = self._apply_cfo_and_phase_noise(rx, sample_rate)
        if self.params.get("enable_iq_imbalance", False) and str(
            self.params.get("iq_imbalance_position", "rx")
        ).lower() in {"rx", "both"}:
            rx = self._apply_iq(
                rx,
                self.params.get("rx_iq_gain_imbalance_db"),
                self.params.get("rx_iq_phase_imbalance_deg"),
            )

        result = dict(signal_dict)
        result.update(
            signal_stream=rx,
            signal_length=num_samples,
            duration_seconds=num_samples / sample_rate,
            num_rx=self.num_rx,
            antenna_axis=0,
            signal_power=signal_power,
            noise_power=self.noise_power,
            mimo_channel_impulse_response=self.channel_impulse_response.copy(),
            mimo_channel_frequency_response=self.channel_frequency_response.copy(),
        )
        if self.measurement_diagnostics is not None:
            result["measured_channel_diagnostics"] = dict(
                self.measurement_diagnostics
            )
        return result
