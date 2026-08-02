"""MIMO-OFDM 接收前端、信道估计与空间检测流水线。"""

from __future__ import annotations

import numpy as np

from receiver.MIMOChannelEstimator import MIMOChannelEstimator
from receiver.MIMODetector import MIMODetector
from receiver.MIMOLayerDemapper import MIMOLayerDemapper
from utils.MIMOUtils import ensure_antenna_axis, validate_mimo_params


class MIMOReceiverProcessor:
    def __init__(self, params, transmitter):
        self.params = params
        self.transmitter = transmitter
        self.num_tx, self.num_rx, self.num_streams = validate_mimo_params(params)
        self.frame = transmitter.tx_signal_dict["mimo_frame"]
        self.nfft = int(self.frame["nfft"])
        self.cp_length = int(self.frame["cp_length"])
        self.block_length = int(self.frame["block_length"])
        self.detector = MIMODetector(params.get("mimo_detector", "mmse"))

    def _sync_template(self):
        time = np.fft.ifft(self.frame["sync_frequency"])
        if self.cp_length:
            time = np.concatenate((time[-self.cp_length:], time))
        return time

    def _compensate_known_rx_iq(self, received):
        """使用配置的 RX IQ 参数执行逐天线宽线性逆变换。"""
        position = str(self.params.get("iq_imbalance_position", "rx")).lower()
        if not (
            self.params.get("enable_iq_imbalance", False)
            and self.params.get("enable_iq_compensation", False)
            and position in {"rx", "both"}
        ):
            return received
        gain = 10.0 ** (float(self.params.get("rx_iq_gain_imbalance_db")) / 20.0)
        phase = np.deg2rad(float(self.params.get("rx_iq_phase_imbalance_deg")))
        alpha = 0.5 * (1.0 + gain * np.exp(-1j * phase))
        beta = 0.5 * (1.0 - gain * np.exp(1j * phase))
        denominator = abs(alpha) ** 2 - abs(beta) ** 2
        if abs(denominator) < 1e-12:
            raise ValueError("RX IQ 不平衡参数导致补偿矩阵不可逆")
        return (np.conj(alpha) * received - beta * np.conj(received)) / denominator

    def _detect_sync(self, received):
        template = self._sync_template()
        maximum = min(
            int(self.params.get("mimo_sync_search_samples", self.block_length)),
            received.shape[1] - len(template),
        )
        if maximum < 0:
            raise ValueError("接收信号短于 MIMO 同步符号")
        metrics = np.zeros(maximum + 1, dtype=float)
        for offset in range(maximum + 1):
            segment = received[:, offset:offset + len(template)]
            correlations = segment @ np.conj(template)
            metrics[offset] = np.sum(np.abs(correlations) ** 2)
        return int(np.argmax(metrics)), metrics

    def _estimate_and_compensate_cfo(self, received, sync_offset, sample_rate):
        first_start = sync_offset
        second_start = sync_offset + self.block_length
        first = received[:, first_start:first_start + self.block_length]
        second = received[:, second_start:second_start + self.block_length]
        if first.shape[1] != self.block_length or second.shape[1] != self.block_length:
            raise ValueError("接收信号不包含两个完整的重复同步符号")
        phase = np.angle(np.vdot(first, second))
        frequency_offset = phase * sample_rate / (2.0 * np.pi * self.block_length)
        if not self.params.get("enable_cfo_compensation", False):
            return received.copy(), float(frequency_offset)
        sample_index = np.arange(received.shape[1])
        correction = np.exp(-1j * 2.0 * np.pi * frequency_offset * sample_index / sample_rate)
        return received * correction[np.newaxis, :], float(frequency_offset)

    def _extract_fft_blocks(self, received, start, count):
        end = start + count * self.block_length
        if end > received.shape[1]:
            raise ValueError(
                f"MIMO 帧不完整：需要 {end} 个采样，实际仅 {received.shape[1]} 个"
            )
        blocks = received[:, start:end].reshape(
            self.num_rx, count, self.block_length
        )
        without_cp = blocks[..., self.cp_length:]
        return np.fft.fft(without_cp, axis=-1)

    def process(self, signal_dict):
        received = ensure_antenna_axis(
            signal_dict["signal_stream"], self.num_rx, "接收 signal_stream"
        ).astype(np.complex128, copy=False)
        received = self._compensate_known_rx_iq(received)
        sample_rate = float(signal_dict["sample_rate_Hz"])

        sync_offset, sync_metric = self._detect_sync(received)
        compensated, estimated_cfo = self._estimate_and_compensate_cfo(
            received, sync_offset, sample_rate
        )
        sync_frequency = self._extract_fft_blocks(compensated, sync_offset, 2)
        estimated_noise_frequency = float(
            np.mean(np.abs(sync_frequency[:, 0] - sync_frequency[:, 1]) ** 2) / 2.0
        )

        training_start = sync_offset + 2 * self.block_length
        training = self._extract_fft_blocks(compensated, training_start, self.num_tx)
        estimated_channel = MIMOChannelEstimator.estimate(
            training, self.frame["pilot_frequency"]
        )

        true_channel = signal_dict.get("mimo_channel_frequency_response")
        channel_nmse = None
        if true_channel is not None:
            channel_nmse = MIMOChannelEstimator.nmse(estimated_channel, true_channel)

        csi_mode = str(self.params.get("mimo_csi_mode", "estimated")).lower()
        if csi_mode == "ideal":
            if true_channel is None:
                raise ValueError("mimo_csi_mode='ideal' 需要信道提供真实频率响应")
            detection_channel = np.asarray(true_channel)
        elif csi_mode == "estimated":
            detection_channel = estimated_channel
        else:
            raise ValueError("mimo_csi_mode 仅支持 'estimated' 或 'ideal'")

        data_start = training_start + self.num_tx * self.block_length
        received_data = self._extract_fft_blocks(
            compensated, data_start, int(self.frame["num_data_symbols"])
        )
        noise_frequency = signal_dict.get("noise_power")
        if noise_frequency is None:
            noise_frequency = estimated_noise_frequency
        else:
            noise_frequency = float(noise_frequency) * self.nfft
        if str(self.params.get("mimo_detector", "mmse")).lower() == "zf":
            detector_noise = 0.0
        else:
            detector_noise = max(float(noise_frequency), 0.0)

        detected, detector_diagnostics = self.detector.detect(
            received_data, detection_channel, detector_noise
        )
        tx_scale = float(self.frame["tx_scale"])
        layers = []
        for stream_index, length in enumerate(self.frame["layer_lengths"]):
            layer = detected[stream_index].reshape(-1)[:int(length)] / tx_scale
            layers.append(layer)
        recovered_symbols = MIMOLayerDemapper.demap(
            layers, self.frame["original_symbol_count"]
        )

        post_variance = detector_diagnostics["post_noise_variance"]
        if tx_scale != 0:
            post_variance = post_variance / (tx_scale ** 2)
        mean_post_variance = float(np.mean(post_variance)) if post_variance.size else 0.0

        output = {
            "signal_stream": recovered_symbols,
            "sample_rate_Hz": sample_rate,
            "duration_seconds": len(recovered_symbols) / sample_rate,
            "signal_length": len(recovered_symbols),
            "padding_bit_num": signal_dict.get("padding_bit_num", 0),
            "channel_freq_response": estimated_channel,
            "mimo_channel_estimate": estimated_channel,
            "mimo_channel_nmse": channel_nmse,
            "mimo_sync_offset": sync_offset,
            "mimo_sync_metric": sync_metric,
            "estimated_cfo_Hz": estimated_cfo,
            "noise_var": mean_post_variance,
            "detector_diagnostics": detector_diagnostics,
        }
        self.equalized = output
        self.compensated_signal = dict(signal_dict)
        self.compensated_signal["signal_stream"] = compensated
        self.compensated_signal["mimo_sync_offset"] = sync_offset
        return output
