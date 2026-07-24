import numpy as np

from receiver.IQRealParallelCalibrator import IQRealParallelCalibrator


class DecisionDirectedIQCompensator:
    """均衡后基于硬判决符号的残余 RX IQ 不平衡补偿器。"""

    _QAM_SCALE = {1: 1.0, 2: np.sqrt(2), 4: np.sqrt(10),
                  6: np.sqrt(42), 8: np.sqrt(170)}

    def __init__(self, params, filter_len=5, ridge_lambda=0.0, iterations=1):
        self.params = params
        self.filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        self.ridge_lambda = float(ridge_lambda)
        self.iterations = int(iterations)
        if self.iterations <= 0:
            raise ValueError("iq_comp_dd_iterations 必须为正整数")
        self.mcs = self.params.get("MCS")
        self.ncbps = int(self.params.get("NCBPS"))
        self.constellation = self._build_constellation()
        self.diagnostics = None

    def _build_constellation(self):
        if self.mcs != 1:
            raise ValueError("判决导向 IQ 补偿当前仅支持 MCS=1 的 PSK/QAM")
        if self.ncbps == 2:
            points = np.array([
                (i + 1j * q) * np.exp(-1j * np.pi / 4) / np.sqrt(2)
                for i in (-1, 1) for q in (-1, 1)
            ])
        elif self.ncbps == 3:
            points = np.exp(1j * np.pi * np.arange(8) / 4)
        elif self.ncbps in (1, 4, 6, 8):
            side = 2 ** (self.ncbps // 2) if self.ncbps > 1 else 2
            levels = np.array([-1.0, 1.0]) if self.ncbps == 1 else (
                2 * np.arange(side) - (side - 1)
            ) / self._QAM_SCALE[self.ncbps]
            if self.ncbps == 1:
                points = levels.astype(complex)
            else:
                points = np.array([i + 1j * q for i in levels for q in levels])
        else:
            raise ValueError(f"判决导向 IQ 补偿不支持 NCBPS={self.ncbps}")
        return np.asarray(points, dtype=np.complex128)

    def _slice_symbols(self, symbols):
        symbols = np.asarray(symbols, dtype=np.complex128)
        phase = np.exp(1j * np.pi * np.arange(len(symbols)) / 2)
        baseband = symbols / phase
        decisions = np.empty_like(baseband)
        distances = np.empty(len(baseband), dtype=float)
        chunk_size = 8192
        for start in range(0, len(baseband), chunk_size):
            stop = min(start + chunk_size, len(baseband))
            distance_matrix = np.abs(
                baseband[start:stop, None] - self.constellation[None, :]
            ) ** 2
            indices = np.argmin(distance_matrix, axis=1)
            decisions[start:stop] = self.constellation[indices]
            distances[start:stop] = distance_matrix[np.arange(stop - start), indices]
        return decisions * phase, distances

    @staticmethod
    def _evm(reference, measured):
        return float(np.sqrt(
            np.mean(np.abs(measured - reference) ** 2)
            / np.mean(np.abs(reference) ** 2)
        ))

    def compensate(self, data_dict):
        required_keys = ["signal_stream", "sample_rate_Hz", "duration_seconds",
                         "signal_length", "padding_bit_num"]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
        signal = np.asarray(data_dict["signal_stream"], dtype=np.complex128)
        if signal.ndim != 1 or len(signal) <= 2 * self.filter_len + 1:
            raise ValueError("判决导向 IQ 补偿输入必须是一维且长度足以估计 FIR")
        if len(signal) != data_dict["signal_length"]:
            raise ValueError("signal_stream 长度与 signal_length 不匹配")

        compensated = signal
        iteration_info = []
        for _ in range(self.iterations):
            decisions, decision_distances = self._slice_symbols(compensated)
            estimate = IQRealParallelCalibrator.estimate_rx_impairment_filters(
                decisions, compensated, self.filter_len,
                ridge_lambda=self.ridge_lambda,
            )
            filters = IQRealParallelCalibrator.design_postcompensation_filters(
                estimate["e1"], estimate["e2"], estimate["e3"], estimate["e4"],
                self.filter_len,
            )
            dc = IQRealParallelCalibrator.design_postcompensation_dc_offset(
                filters["f1"], filters["f2"], filters["f3"], filters["f4"],
                estimate["dI"], estimate["dQ"],
            )
            before_evm = self._evm(decisions, compensated)
            compensated = IQRealParallelCalibrator.apply_postcompensation(
                compensated,
                filters["f1"], filters["f2"], filters["f3"], filters["f4"],
                dc["cI"], dc["cQ"],
            )
            after_evm = self._evm(decisions, compensated)
            iteration_info.append({
                "decision_evm_before": before_evm,
                "decision_evm_after": after_evm,
                "decision_distance_mean": float(np.mean(decision_distances)),
                "design_condition_number": estimate["design_condition_number"],
                "post_condition_number": filters["condition_number"],
            })

        self.diagnostics = iteration_info
        result = dict(data_dict)
        result.update({
            "signal_stream": compensated,
            "signal_length": len(compensated),
            "duration_seconds": len(compensated) / data_dict["sample_rate_Hz"],
            "iq_compensation_method": "decision_directed",
            "iq_dd_diagnostics": iteration_info,
        })
        return result
