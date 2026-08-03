"""SISO-OFDM 专用宽线性 IQ 不平衡补偿器。"""

from __future__ import annotations

import numpy as np

from channel.IQImbalance import IQImbalance


def resolve_iq_compensation_method(params, link_mode):
    """解析旧配置并为 OFDM/SC-FDE 选择合法的补偿方法。"""
    if not params.get("enable_iq_compensation", False):
        return "none"
    method = str(params.get("iq_compensation_method", "auto")).strip().lower()
    aliases = {
        "ofdm": "ofdm_widely_linear",
        "widely_linear": "ofdm_widely_linear",
        "known": "configured_inverse",
        "configured": "configured_inverse",
        "dd": "decision_directed",
    }
    method = aliases.get(method, method)
    if method == "auto":
        return "ofdm_widely_linear" if link_mode == "ofdm" else "decision_directed"
    if link_mode == "ofdm" and method == "decision_directed":
        return "ofdm_widely_linear"
    valid = {
        "configured_inverse", "ofdm_widely_linear",
        "decision_directed", "ces",
    }
    if method not in valid:
        raise ValueError(f"不支持的 IQ 补偿方法：{method}")
    return method


class ConfiguredRXIQCompensator:
    """使用仿真配置参数精确反演频率无关的 RX IQ 不平衡。"""

    def __init__(self, params):
        self.params = params
        self.diagnostics = None

    def is_applicable(self) -> bool:
        position = str(self.params.get("iq_imbalance_position", "rx")).lower()
        model = str(self.params.get("iq_imbalance_model", "fid")).lower()
        return bool(
            self.params.get("enable_iq_imbalance", False)
            and position in {"rx", "both"}
            and model == "fid"
        )

    def compensate(self, data_dict):
        if not self.is_applicable():
            return data_dict

        signal = np.asarray(data_dict["signal_stream"], dtype=np.complex128)
        _, _, mu, nu = IQImbalance.compute_fid_coefficients(
            self.params.get("rx_iq_gain_imbalance_db", 0.0),
            self.params.get("rx_iq_phase_imbalance_deg", 0.0),
        )

        if self.params.get("iq_power_normalize", False):
            power_gain = np.sqrt(abs(mu) ** 2 + abs(nu) ** 2)
            if power_gain > 0:
                mu /= power_gain
                nu /= power_gain

        denominator = abs(mu) ** 2 - abs(nu) ** 2
        if abs(denominator) < 1e-12:
            raise ValueError("RX IQ 不平衡参数导致宽线性补偿矩阵不可逆")

        compensated = (
            np.conj(mu) * signal - nu * np.conj(signal)
        ) / denominator
        self.diagnostics = {
            "mu": mu,
            "nu": nu,
            "denominator": float(np.real(denominator)),
        }

        result = dict(data_dict)
        result.update({
            "signal_stream": compensated,
            "signal_length": len(compensated),
            "duration_seconds": len(compensated) / data_dict["sample_rate_Hz"],
            "iq_compensation_method": "configured_inverse",
            "iq_configured_diagnostics": self.diagnostics,
        })
        return result


class BlindRXIQWhiteningCompensator:
    """在 CFO 补偿前用二阶统计量盲校正 RX I/Q 正交性。

    该步骤不使用配置中的增益/相位数值，只假设输入复包络近似 proper。
    频率相关残差由后续 OFDM 宽线性均衡继续处理。
    """

    def __init__(self, params):
        self.params = params
        self.diagnostics = None

    def is_applicable(self) -> bool:
        position = str(self.params.get("iq_imbalance_position", "rx")).lower()
        return bool(
            self.params.get("enable_iq_imbalance", False)
            and position in {"rx", "both"}
        )

    def compensate(self, data_dict):
        if not self.is_applicable():
            return data_dict
        signal = np.asarray(data_dict["signal_stream"], dtype=np.complex128)
        if signal.ndim != 1 or len(signal) < 16:
            raise ValueError("盲 RX IQ 校正至少需要 16 个一维复采样点")

        components = np.vstack((signal.real, signal.imag))
        trim_fraction = float(
            self.params.get("iq_blind_covariance_trim_fraction", 0.2)
        )
        trim_fraction = min(max(trim_fraction, 0.0), 0.8)
        covariance_start = int(trim_fraction * components.shape[1])
        covariance_samples = components[:, covariance_start:]
        covariance_mean = np.mean(covariance_samples, axis=1, keepdims=True)
        covariance_centered = covariance_samples - covariance_mean
        covariance = (
            covariance_centered @ covariance_centered.T
            / covariance_centered.shape[1]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        maximum = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
        floor = maximum * 1e-10
        eigenvalues = np.maximum(eigenvalues, floor)
        target_variance = float(np.mean(eigenvalues))
        whitening = (
            eigenvectors
            @ np.diag(np.sqrt(target_variance / eigenvalues))
            @ eigenvectors.T
        )
        signal_mean = np.mean(components, axis=1, keepdims=True)
        corrected_components = whitening @ (components - signal_mean)
        compensated = corrected_components[0] + 1j * corrected_components[1]

        corrected_covariance = (
            corrected_components @ corrected_components.T
            / corrected_components.shape[1]
        )
        self.diagnostics = {
            "input_covariance": covariance,
            "output_covariance": corrected_covariance,
            "whitening_matrix": whitening,
            "input_condition_number": float(np.linalg.cond(covariance)),
            "covariance_trim_fraction": trim_fraction,
        }
        result = dict(data_dict)
        result.update({
            "signal_stream": compensated,
            "signal_length": len(compensated),
            "duration_seconds": len(compensated) / data_dict["sample_rate_Hz"],
            "iq_frontend_compensation_method": "blind_rx_whitening",
            "iq_blind_diagnostics": self.diagnostics,
        })
        return result


class OFDMWidelyLinearIQCompensator:
    """基于正交块导频的逐子载波宽线性信道估计与均衡。

    对每个子载波估计

        Y[k] = A[k] X[k] + B[k] conj(X[-k])

    其中 A/B 是传播信道与 TX/RX IQ 不平衡的联合有效响应。
    """

    def __init__(self, params, pilot_indexes, pilot_sequences, n_subcarriers):
        self.params = params
        self.pilot_indexes = [int(index) for index in pilot_indexes]
        self.pilot_sequences = np.asarray(
            pilot_sequences, dtype=np.complex128
        )
        self.n_subcarriers = int(n_subcarriers)
        self.estimation_ridge = float(
            self.params.get("iq_ofdm_estimation_ridge", 1e-8)
        )
        self.equalizer_ridge = float(
            self.params.get("iq_ofdm_equalizer_ridge", 1e-8)
        )
        self.condition_limit = float(
            self.params.get("iq_ofdm_condition_limit", 1e8)
        )
        configured_taps = self.params.get("iq_ofdm_response_taps", None)
        if configured_taps is None:
            configured_taps = self.params.get("gi_length", 0)
        self.response_taps = int(configured_taps or 0)
        self.compensation_mode = str(
            self.params.get("iq_compensation_mode", "per_frame")
        ).lower()
        self.diagnostics = None

        if self.pilot_sequences.shape != (
            len(self.pilot_indexes), self.n_subcarriers
        ):
            raise ValueError(
                "OFDM IQ 导频参考维度必须为 "
                f"({len(self.pilot_indexes)}, {self.n_subcarriers})"
            )
        if len(self.pilot_indexes) < 2:
            raise ValueError("OFDM 宽线性 IQ 估计至少需要两个块导频")
        if len(set(self.pilot_indexes)) != len(self.pilot_indexes):
            raise ValueError("OFDM IQ 导频索引不能重复")
        if min(self.pilot_indexes) < 0:
            raise ValueError("OFDM IQ 导频索引不能为负")
        if self.estimation_ridge < 0 or self.equalizer_ridge < 0:
            raise ValueError("OFDM IQ 正则系数不能为负")
        if self.condition_limit <= 0:
            raise ValueError("iq_ofdm_condition_limit 必须为正数")
        if self.compensation_mode not in {"per_frame", "first_frame"}:
            raise ValueError("iq_compensation_mode 仅支持 per_frame / first_frame")

        mirror = self.mirror_indexes
        for k in range(self.n_subcarriers):
            design = np.column_stack((
                self.pilot_sequences[:, k],
                np.conj(self.pilot_sequences[:, mirror[k]]),
            ))
            if np.linalg.matrix_rank(design) < 2:
                raise ValueError(
                    f"OFDM IQ 导频在子载波 {k} 上不可辨识，请使用满秩相位编码"
                )

    @property
    def mirror_indexes(self):
        indexes = np.arange(self.n_subcarriers)
        return (-indexes) % self.n_subcarriers

    def _smooth_response(self, response):
        taps = min(max(self.response_taps, 0), self.n_subcarriers)
        if taps == 0 or taps == self.n_subcarriers:
            return response
        impulse = np.fft.ifft(response)
        # 匹配滤波与定时对齐会使有效冲激响应跨越循环边界，不能只保留
        # impulse[:taps]。选择能量最大的循环连续窗口，可同时保留前游标、
        # 后游标和因多径产生的有效抽头。
        energy = np.abs(impulse) ** 2
        extended_energy = np.concatenate((energy, energy))
        window_energy = np.convolve(
            extended_energy, np.ones(taps, dtype=float), mode="valid"
        )[:self.n_subcarriers]
        start = int(np.argmax(window_energy))
        keep = (start + np.arange(taps)) % self.n_subcarriers
        truncated = np.zeros_like(impulse)
        truncated[keep] = impulse[keep]
        return np.fft.fft(truncated)

    def _estimate_frame(self, freq_grid, frame_index, frame_num):
        mirror = self.mirror_indexes
        symbols_per_frame = freq_grid.shape[1] // frame_num
        received_pilots = np.stack([
            freq_grid[:, frame_index * symbols_per_frame + pilot_index]
            for pilot_index in self.pilot_indexes
        ])

        direct = np.empty(self.n_subcarriers, dtype=np.complex128)
        image = np.empty(self.n_subcarriers, dtype=np.complex128)
        design_conditions = np.empty(self.n_subcarriers, dtype=float)
        pilot_residuals = np.empty(self.n_subcarriers, dtype=float)
        identity = np.eye(2, dtype=np.complex128)

        for k in range(self.n_subcarriers):
            design = np.column_stack((
                self.pilot_sequences[:, k],
                np.conj(self.pilot_sequences[:, mirror[k]]),
            ))
            gram = design.conj().T @ design
            design_conditions[k] = np.linalg.cond(gram)
            rhs = design.conj().T @ received_pilots[:, k]
            coefficients = np.linalg.solve(
                gram + self.estimation_ridge * identity, rhs
            )
            direct[k], image[k] = coefficients
            residual = received_pilots[:, k] - design @ coefficients
            pilot_residuals[k] = np.mean(np.abs(residual) ** 2)

        direct = self._smooth_response(direct)
        image = self._smooth_response(image)
        return direct, image, {
            "design_condition_max": float(np.max(design_conditions)),
            "design_condition_mean": float(np.mean(design_conditions)),
            "pilot_residual_power": float(np.mean(pilot_residuals)),
        }

    def estimate_effective_channels(self, freq_grid, frame_num):
        freq_grid = np.asarray(freq_grid, dtype=np.complex128)
        frame_num = int(frame_num)
        if frame_num <= 0:
            raise ValueError("OFDM IQ 补偿帧数必须为正整数")
        if freq_grid.ndim != 2 or freq_grid.shape[0] != self.n_subcarriers:
            raise ValueError("OFDM IQ 补偿输入必须是 (子载波, OFDM块) 二维网格")
        if freq_grid.shape[1] % frame_num != 0:
            raise ValueError("OFDM 频域网格列数必须是帧数的整数倍")
        symbols_per_frame = freq_grid.shape[1] // frame_num
        if max(self.pilot_indexes) >= symbols_per_frame:
            raise ValueError("OFDM IQ 导频索引超出每帧 OFDM 符号范围")

        direct_list = []
        image_list = []
        estimation_info = []
        estimate_count = 1 if self.compensation_mode == "first_frame" else frame_num
        for frame_index in range(estimate_count):
            direct, image, info = self._estimate_frame(
                freq_grid, frame_index, frame_num
            )
            direct_list.append(direct)
            image_list.append(image)
            estimation_info.append(info)

        if estimate_count == 1 and frame_num > 1:
            direct_list *= frame_num
            image_list *= frame_num
            estimation_info *= frame_num
        return direct_list, image_list, estimation_info

    def _equalize_frame(self, received, direct, image):
        mirror = self.mirror_indexes
        equalized = np.zeros_like(received)
        processed = np.zeros(self.n_subcarriers, dtype=bool)
        pair_conditions = []
        fallback_count = 0
        identity = np.eye(2, dtype=np.complex128)

        for k in range(self.n_subcarriers):
            if processed[k]:
                continue
            m = int(mirror[k])
            matrix = np.array([
                [direct[k], image[k]],
                [np.conj(image[m]), np.conj(direct[m])],
            ], dtype=np.complex128)
            condition = float(np.linalg.cond(matrix))
            pair_conditions.append(condition)

            if not np.isfinite(condition) or condition > self.condition_limit:
                fallback_count += 1
                denominator_k = direct[k] if abs(direct[k]) > 1e-12 else 1.0
                equalized[k] = received[k] / denominator_k
                if m != k:
                    denominator_m = direct[m] if abs(direct[m]) > 1e-12 else 1.0
                    equalized[m] = received[m] / denominator_m
            else:
                observation = np.vstack((received[k], np.conj(received[m])))
                gram = matrix.conj().T @ matrix
                inverse = np.linalg.solve(
                    gram + self.equalizer_ridge * identity,
                    matrix.conj().T,
                )
                recovered = inverse @ observation
                equalized[k] = recovered[0]
                if m != k:
                    equalized[m] = np.conj(recovered[1])

            processed[k] = True
            processed[m] = True

        return equalized, {
            "pair_condition_max": float(np.max(pair_conditions)),
            "pair_condition_mean": float(np.mean(pair_conditions)),
            "fallback_pair_count": int(fallback_count),
        }

    def compensate(self, freq_grid, frame_num):
        """估计有效直通/镜像响应，并返回联合均衡后的频域网格。"""
        freq_grid = np.asarray(freq_grid, dtype=np.complex128)
        direct_list, image_list, estimation_info = (
            self.estimate_effective_channels(freq_grid, frame_num)
        )
        equalized = np.zeros_like(freq_grid)
        frame_info = []
        for frame_index in range(frame_num):
            symbols_per_frame = freq_grid.shape[1] // frame_num
            start = frame_index * symbols_per_frame
            columns = np.arange(start, start + symbols_per_frame)
            frame_equalized, equalizer_info = self._equalize_frame(
                freq_grid[:, columns],
                direct_list[frame_index],
                image_list[frame_index],
            )
            equalized[:, columns] = frame_equalized
            direct_power = np.mean(np.abs(direct_list[frame_index]) ** 2)
            image_power = np.mean(np.abs(image_list[frame_index]) ** 2)
            irr_db = 10.0 * np.log10(
                (direct_power + np.finfo(float).tiny)
                / (image_power + np.finfo(float).tiny)
            )
            frame_info.append({
                **estimation_info[frame_index],
                **equalizer_info,
                "estimated_irr_db": float(irr_db),
            })

        self.diagnostics = {
            "method": "ofdm_widely_linear",
            "frames": frame_info,
            "direct_response": direct_list,
            "image_response": image_list,
        }
        return equalized, direct_list, image_list, self.diagnostics
