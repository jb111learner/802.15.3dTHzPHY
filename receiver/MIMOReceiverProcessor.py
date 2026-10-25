"""MIMO-OFDM 接收前端、信道估计与空间检测流水线。"""

from __future__ import annotations

from receiver.FixedReceiverSupport import FixedReceiverMixin

import numpy as np

from receiver.MIMOChannelEstimator import MIMOChannelEstimator
from receiver.MIMODetector import MIMODetector
from receiver.MIMOLayerDemapper import MIMOLayerDemapper
from utils.MIMOUtils import ensure_antenna_axis, validate_mimo_params


class MIMOReceiverProcessor(FixedReceiverMixin):
    def __init__(self, params, transmitter):
        self._init_fixed(params, "mimo.rx.processor")
        self.params = params
        self.transmitter = transmitter
        self.num_tx, self.num_rx, self.num_streams = validate_mimo_params(params)
        self.frame = transmitter.tx_signal_dict["mimo_frame"]
        self.nfft = int(self.frame["nfft"])
        self.cp_length = int(self.frame["cp_length"])
        self.block_length = int(self.frame["block_length"])
        self.detector = MIMODetector(params.get("mimo_detector", "mmse"), params=params)
        self.channel_estimator = MIMOChannelEstimator(params)
        self.layer_demapper = MIMOLayerDemapper(params)

    def _sync_template(self):
        mid = "mimo.rx.synchronization"
        if self._fixed_enabled(mid):
            time = self.fixed_context.fft(self.frame["sync_frequency"], mid, inverse=True)
        else:
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
        mid = "mimo.rx.iq"
        if self._fixed_enabled(mid):
            ctx = self.fixed_context
            direct = ctx.mul(np.conj(alpha), received, mid, role="product", a_role="coefficient")
            image = ctx.mul(beta, np.conj(received), mid, role="product", a_role="coefficient")
            numerator = ctx.add(direct, -image, mid, role="accumulator", a_role="product", b_role="product")
            return ctx.div(numerator, denominator, mid, a_role="accumulator", b_role="denominator")
        return (np.conj(alpha) * received - beta * np.conj(received)) / denominator

    def _detect_sync(self, received):
        mid = "mimo.rx.synchronization"
        received = self.fixed_context.quantize(received, mid, role="input")
        template = self._sync_template()
        template = self.fixed_context.quantize(template, mid, role="coefficient")
        maximum = min(
            int(self.params.get("mimo_sync_search_samples", self.block_length)),
            received.shape[1] - len(template),
        )
        if maximum < 0:
            raise ValueError("接收信号短于 MIMO 同步符号")
        if self._fixed_enabled(mid) and self.fixed_context.arithmetic_level(mid) == "operation":
            ctx = self.fixed_context
            correlations = np.asarray([
                ctx.correlate(antenna[:maximum + len(template)], template, mid, mode="valid")
                for antenna in received
            ])
            power = np.real(ctx.mul(correlations, np.conj(correlations), mid,
                                    role="product", a_role="output", b_role="output"))
            metrics = ctx.sum(power, mid, axis=0, input_role="product")
            ctx.record_coverage(mid, "operation", "template IFFT/antenna correlations/power accumulation")
            return int(np.argmax(metrics)), metrics
        metrics = np.zeros(maximum + 1, dtype=float)
        for offset in range(maximum + 1):
            segment = received[:, offset:offset + len(template)]
            correlations = segment @ np.conj(template)
            metrics[offset] = np.sum(np.abs(correlations) ** 2)
        metrics = self.fixed_context.quantize(metrics, mid, role="accumulator")
        return int(np.argmax(metrics)), metrics

    def _estimate_and_compensate_cfo(self, received, sync_offset, sample_rate):
        mid = "mimo.rx.cfo"
        first_start = sync_offset
        second_start = sync_offset + self.block_length
        first = received[:, first_start:first_start + self.block_length]
        second = received[:, second_start:second_start + self.block_length]
        if first.shape[1] != self.block_length or second.shape[1] != self.block_length:
            raise ValueError("接收信号不包含两个完整的重复同步符号")
        if self._fixed_enabled(mid):
            ctx = self.fixed_context
            product = ctx.mul(np.conj(first), second, mid, role="product")
            phase = ctx.angle(ctx.sum(product, mid, input_role="product"), mid)
        else:
            phase = np.angle(np.vdot(first, second))
        frequency_offset = phase * sample_rate / (2.0 * np.pi * self.block_length)
        if not self.params.get("enable_cfo_compensation", False):
            return received.copy(), float(frequency_offset)
        sample_index = np.arange(received.shape[1])
        if self._fixed_enabled(mid):
            phase_grid = (-2.0 * np.pi * frequency_offset * sample_index / sample_rate + np.pi) % (2 * np.pi) - np.pi
            correction = ctx.phasor(phase_grid, mid)
            compensated = ctx.mul(received, correction[np.newaxis, :], mid, b_role="coefficient")
            ctx.record_coverage(mid, ctx.arithmetic_level(mid), "repeated-block correlation/CORDIC/rotation; physical phase-grid metadata")
            return compensated, float(frequency_offset)
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
        if self._fixed_enabled("mimo.rx.fft"):
            return self.fixed_context.fft(without_cp, "mimo.rx.fft", axis=-1)
        result = np.fft.fft(without_cp, axis=-1)
        self._watch(without_cp, "mimo.rx.fft.input")
        self._watch(result, "mimo.rx.fft.output")
        return result

    def process(self, signal_dict):
        received = ensure_antenna_axis(
            signal_dict["signal_stream"], self.num_rx, "接收 signal_stream"
        ).astype(np.complex128, copy=False)
        received = self._compensate_known_rx_iq(received)
        sample_rate = float(signal_dict["sample_rate_Hz"])

        # 帧结构：发射端逐帧组帧（每帧 = 保护 + 2 SYNC + 训练 + 数据块）
        frame_num = int(self.frame.get("frame_num", 1))
        num_data_per_frame = int(self.frame["num_data_symbols"])
        guard_length = int(self.frame.get("guard_length", 0))
        frame_length = (
            guard_length
            + (2 + self.num_tx + num_data_per_frame) * self.block_length
        )
        tx_scale = float(self.frame["tx_scale"])
        csi_mode = str(self.params.get("mimo_csi_mode", "estimated")).lower()
        if csi_mode not in ("estimated", "ideal"):
            raise ValueError("mimo_csi_mode 仅支持 'estimated' 或 'ideal'")

        # 首帧同步（其余帧按等长帧结构顺延定位）
        sync_offset, sync_metric = self._detect_sync(received)
        timing_shift = sync_offset - guard_length

        true_channel = signal_dict.get("mimo_channel_frequency_response")
        aligned_true_channel = None
        if true_channel is not None:
            aligned_true_channel = np.asarray(true_channel)
            if timing_shift:
                phase = np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * np.arange(self.nfft, dtype=float)
                    * timing_shift
                    / self.nfft
                )
                aligned_true_channel = aligned_true_channel * phase[None, None, :]

        noise_frequency_known = signal_dict.get("noise_power")
        zf_mode = str(self.params.get("mimo_detector", "mmse")).lower() == "zf"

        # ── 逐帧处理：CFO / 噪声 / 信道估计 / 检测 / 层解映射 ──
        # 注意：_detect_sync 定位的是 SYNC 块（含 CP）起点，帧起点在其前
        # guard_length 个样点处；后续帧按等长帧结构顺延。
        recovered_frames = []
        post_vars = []
        cfo_list = []
        channel_estimates = []
        detector_diagnostics = None
        frame0_start = sync_offset - guard_length
        for frame_index in range(frame_num):
            frame_start = frame0_start + frame_index * frame_length
            frame_end = frame_start + frame_length
            if frame_end > received.shape[1]:
                raise ValueError(
                    f"第 {frame_index} 帧不完整：需要 {frame_end} 个采样，"
                    f"实际仅 {received.shape[1]} 个"
                )
            frame_signal = received[:, frame_start:frame_end]

            # 帧内 CFO（两个重复 SYNC 的相位差）
            compensated, cfo_frame = self._estimate_and_compensate_cfo(
                frame_signal, guard_length, sample_rate
            )
            cfo_list.append(cfo_frame)

            # 帧内噪声方差（SYNC 重复块差分）
            sync_frequency = self._extract_fft_blocks(
                compensated, guard_length, 2
            )
            noise_mid = "mimo.rx.noise_estimator"
            if self._fixed_enabled(noise_mid):
                ctx = self.fixed_context
                difference = ctx.add(sync_frequency[:, 0], -sync_frequency[:, 1], noise_mid,
                                     a_role="output", b_role="output")
                power = np.real(ctx.mul(difference, np.conj(difference), noise_mid, role="product",
                                        a_role="output", b_role="output"))
                total = ctx.sum(power, noise_mid, input_role="product")
                noise_frequency = float(ctx.div(total, 2 * power.size, noise_mid,
                                                role="noise", a_role="accumulator", b_role="index"))
            else:
                noise_frequency = float(
                    np.mean(np.abs(sync_frequency[:, 0] - sync_frequency[:, 1]) ** 2) / 2.0
                )

            # 帧内训练符号 → LS 信道估计
            training_start = guard_length + 2 * self.block_length
            training = self._extract_fft_blocks(
                compensated, training_start, self.num_tx
            )
            estimated_channel = self.channel_estimator.estimate_with_config(
                training, self.frame["pilot_frequency"]
            )
            channel_estimates.append(estimated_channel)

            if csi_mode == "ideal":
                if true_channel is None:
                    raise ValueError("mimo_csi_mode='ideal' 需要信道提供真实频率响应")
                detection_channel = aligned_true_channel
            else:
                detection_channel = estimated_channel

            data_start = training_start + self.num_tx * self.block_length
            received_data = self._extract_fft_blocks(
                compensated, data_start, num_data_per_frame
            )
            if noise_frequency_known is not None:
                detector_noise = max(
                    float(noise_frequency_known) * self.nfft, 0.0
                ) if not zf_mode else 0.0
                if not zf_mode and self._fixed_enabled(noise_mid):
                    detector_noise = float(self.fixed_context.mul(
                        noise_frequency_known, self.nfft, noise_mid,
                        role="noise", a_role="noise", b_role="index", node=noise_mid + ".known_frequency_noise"
                    ))
            else:
                detector_noise = 0.0 if zf_mode else max(noise_frequency, 0.0)

            detected, detector_diagnostics = self.detector.detect(
                received_data, detection_channel, detector_noise
            )
            layers = []
            for stream_index, length in enumerate(self.frame["layer_lengths"]):
                layer = detected[stream_index].reshape(-1)[:int(length)] / tx_scale
                if self._fixed_enabled("mimo.rx.layer_scale"):
                    layer = self.fixed_context.div(detected[stream_index].reshape(-1)[:int(length)], tx_scale,
                                                   "mimo.rx.layer_scale", a_role="output", b_role="coefficient")
                layers.append(layer)
            recovered_frames.append(
                self.layer_demapper.demap(
                    layers, self.frame["original_symbol_count"]
                )
            )
            if self._fixed_enabled("mimo.rx.layer_demap"):
                self.fixed_context.record_coverage("mimo.rx.layer_demap", "bit_exact", "interleaved layer sample selection")

            post_variance = detector_diagnostics["post_noise_variance"]
            if tx_scale != 0:
                if self._fixed_enabled("mimo.rx.layer_scale"):
                    ctx, scale_mid = self.fixed_context, "mimo.rx.layer_scale"
                    scale_power = ctx.mul(tx_scale, tx_scale, scale_mid, role="coefficient",
                                          a_role="coefficient", b_role="coefficient")
                    post_variance = ctx.div(post_variance, scale_power, scale_mid, role="noise",
                                            a_role="noise", b_role="coefficient", node=scale_mid + ".post_noise")
                else:
                    post_variance = post_variance / (tx_scale ** 2)
            if post_variance.size and self._fixed_enabled(noise_mid):
                mean_variance = self.fixed_context.div(
                    self.fixed_context.sum(post_variance, noise_mid, input_role="noise"),
                    post_variance.size, noise_mid, role="noise", a_role="accumulator", b_role="index"
                )
                post_vars.append(float(mean_variance))
            else:
                post_vars.append(float(np.mean(post_variance)) if post_variance.size else 0.0)

        recovered_symbols = (
            np.concatenate(recovered_frames)
            if recovered_frames
            else np.zeros(0, dtype=np.complex128)
        )
        # 多帧信道估计取平均（静态信道假设下降低估计噪声）
        if self._fixed_enabled("mimo.rx.channel_estimation"):
            ctx, est_mid = self.fixed_context, "mimo.rx.channel_estimation"
            mean_channel_estimate = ctx.div(
                ctx.sum(np.asarray(channel_estimates), est_mid, axis=0, input_role="output"),
                len(channel_estimates), est_mid, a_role="accumulator", b_role="index"
            )
        else:
            mean_channel_estimate = np.mean(np.asarray(channel_estimates), axis=0)
        channel_nmse = None
        if aligned_true_channel is not None:
            channel_nmse = MIMOChannelEstimator.nmse(
                mean_channel_estimate, aligned_true_channel
            )
        estimated_cfo = float(np.mean(cfo_list)) if cfo_list else 0.0
        if post_vars and self._fixed_enabled("mimo.rx.noise_estimator"):
            ctx, noise_mid = self.fixed_context, "mimo.rx.noise_estimator"
            mean_post_variance = float(ctx.div(
                ctx.sum(post_vars, noise_mid, input_role="noise"), len(post_vars), noise_mid,
                role="noise", a_role="accumulator", b_role="index"
            ))
        else:
            mean_post_variance = float(np.mean(post_vars)) if post_vars else 0.0

        output = {
            "signal_stream": recovered_symbols,
            "sample_rate_Hz": sample_rate,
            "duration_seconds": len(recovered_symbols) / sample_rate,
            "signal_length": len(recovered_symbols),
            "padding_bit_num": signal_dict.get("padding_bit_num", 0),
            "channel_freq_response": mean_channel_estimate,
            "mimo_channel_estimate": mean_channel_estimate,
            "mimo_channel_nmse": channel_nmse,
            "mimo_sync_offset": sync_offset,
            "mimo_timing_shift_samples": timing_shift,
            "mimo_sync_metric": sync_metric,
            "estimated_cfo_Hz": estimated_cfo,
            "noise_var": mean_post_variance,
            "detector_diagnostics": detector_diagnostics,
        }
        self.equalized = output
        self.compensated_signal = dict(signal_dict)
        self.compensated_signal["signal_stream"] = received
        self.compensated_signal["mimo_sync_offset"] = sync_offset
        return output
