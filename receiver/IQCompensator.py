import warnings

import numpy as np


class IQRealParallelCalibrator:
    """
    Real-based parallel LS estimator and TX precompensator for IQ imbalance.

    The model is:
        xI = h1 * sI + h2 * sQ + bI
        xQ = h3 * sI + h4 * sQ + bQ

    All FIR convolutions are causal. Same-length signal outputs are produced by
    truncating the full causal convolution to the input sequence length.
    """

    _COND_WARNING_THRESHOLD = 1e10

    @staticmethod
    def _require_identifiable(matrix, name):
        """Reject rank-deficient or numerically singular calibration systems."""
        rank = int(np.linalg.matrix_rank(matrix))
        required_rank = int(matrix.shape[1])
        condition_number = float(np.linalg.cond(matrix))
        if (
            rank < required_rank
            or not np.isfinite(condition_number)
            or condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
        ):
            raise np.linalg.LinAlgError(
                f"{name} is not identifiable: rank={rank}/{required_rank}, "
                f"condition_number={condition_number:.6e}"
            )
        return rank, condition_number

    @staticmethod
    def _as_1d_array(values, dtype=float, name="array"):
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        return array

    @staticmethod
    def _validate_filter_len(filter_len):
        try:
            filter_len = int(filter_len)
        except (TypeError, ValueError) as exc:
            raise ValueError("filter_len must be a positive integer") from exc

        if filter_len <= 0:
            raise ValueError("filter_len must be a positive integer")
        return filter_len

    @staticmethod
    def _causal_fir_same_length(signal, taps):
        signal = IQRealParallelCalibrator._as_1d_array(
            signal, dtype=float, name="signal"
        )
        taps = IQRealParallelCalibrator._as_1d_array(taps, dtype=float, name="taps")
        if taps.size == 0:
            raise ValueError("taps must not be empty")
        if signal.size == 0:
            return np.zeros(0, dtype=float)
        return np.convolve(signal, taps, mode="full")[: signal.size]

    @staticmethod
    def _build_full_convolution_matrix(taps, input_len, output_len=None):
        taps = IQRealParallelCalibrator._as_1d_array(taps, dtype=float, name="taps")
        if taps.size == 0:
            raise ValueError("taps must not be empty")

        input_len = IQRealParallelCalibrator._validate_filter_len(input_len)
        if output_len is None:
            output_len = taps.size + input_len - 1
        output_len = IQRealParallelCalibrator._validate_filter_len(output_len)

        matrix = np.zeros((output_len, input_len), dtype=float)
        for col in range(input_len):
            row_start = col
            row_stop = min(col + taps.size, output_len)
            if row_stop > row_start:
                matrix[row_start:row_stop, col] = taps[: row_stop - row_start]
        return matrix

    @staticmethod
    def build_toeplitz(signal, filter_len):
        """
        Build the same-length causal FIR convolution matrix.

        If T = build_toeplitz(x, L), then T @ h equals
        np.convolve(x, h, mode="full")[:len(x)] for len(h) == L.
        """
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        signal = IQRealParallelCalibrator._as_1d_array(
            signal, dtype=float, name="signal"
        )
        if signal.size == 0:
            raise ValueError("signal must not be empty")

        toeplitz = np.zeros((signal.size, filter_len), dtype=float)
        for tap_idx in range(min(filter_len, signal.size)):
            toeplitz[tap_idx:, tap_idx] = signal[: signal.size - tap_idx]
        return toeplitz

    @staticmethod
    def estimate_impairment_filters(s, x, filter_len, ridge_lambda=0.0):
        """Estimate h1, h2, h3, h4 and I/Q DC offsets by LS."""
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        ridge_lambda = float(ridge_lambda)
        if not np.isfinite(ridge_lambda):
            raise ValueError("ridge_lambda must be finite")

        s = IQRealParallelCalibrator._as_1d_array(s, dtype=complex, name="s")
        x = IQRealParallelCalibrator._as_1d_array(x, dtype=complex, name="x")

        if s.size != x.size:
            raise ValueError("s and x must have the same length")
        if s.size <= filter_len:
            raise ValueError("input length must be greater than filter_len")

        sI = np.real(s)
        sQ = np.imag(s)
        xI = np.real(x)
        xQ = np.imag(x)

        design_matrix = np.column_stack(
            (
                np.ones(s.size, dtype=float),
                IQRealParallelCalibrator.build_toeplitz(sI, filter_len),
                IQRealParallelCalibrator.build_toeplitz(sQ, filter_len),
            )
        )
        _, design_condition_number = IQRealParallelCalibrator._require_identifiable(
            design_matrix, "IQ impairment estimation matrix"
        )
        warning_message = None

        if ridge_lambda <= 0.0:
            estimation_method = "pinv"
            design_pinv = np.linalg.pinv(design_matrix)
            coeff_I = design_pinv @ xI
            coeff_Q = design_pinv @ xQ
        else:
            estimation_method = "ridge"
            regularizer = np.eye(design_matrix.shape[1], dtype=float)
            regularizer[0, 0] = 0.0
            lhs = design_matrix.T @ design_matrix + ridge_lambda * regularizer
            rhs_I = design_matrix.T @ xI
            rhs_Q = design_matrix.T @ xQ
            lhs_condition_number = float(np.linalg.cond(lhs))
            use_pinv = (
                not np.isfinite(lhs_condition_number)
                or lhs_condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
            )

            if use_pinv:
                warning_message = (
                    "ridge normal-equation matrix is ill-conditioned; "
                    f"condition_number={lhs_condition_number}. "
                    "Using pseudoinverse fallback."
                )
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                lhs_pinv = np.linalg.pinv(lhs)
                coeff_I = lhs_pinv @ rhs_I
                coeff_Q = lhs_pinv @ rhs_Q
            else:
                try:
                    coeff_I = np.linalg.solve(lhs, rhs_I)
                    coeff_Q = np.linalg.solve(lhs, rhs_Q)
                except np.linalg.LinAlgError:
                    warning_message = (
                        "ridge normal-equation solve failed. "
                        "Using pseudoinverse fallback."
                    )
                    warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                    lhs_pinv = np.linalg.pinv(lhs)
                    coeff_I = lhs_pinv @ rhs_I
                    coeff_Q = lhs_pinv @ rhs_Q

        if not (np.all(np.isfinite(coeff_I)) and np.all(np.isfinite(coeff_Q))):
            raise np.linalg.LinAlgError(
                "impairment filter estimation produced non-finite coefficients"
            )

        bI = float(coeff_I[0])
        h1 = coeff_I[1 : 1 + filter_len].astype(float)
        h2 = coeff_I[1 + filter_len :].astype(float)
        bQ = float(coeff_Q[0])
        h3 = coeff_Q[1 : 1 + filter_len].astype(float)
        h4 = coeff_Q[1 + filter_len :].astype(float)

        residual_I = design_matrix @ coeff_I - xI
        residual_Q = design_matrix @ coeff_Q - xQ
        result = {
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "bI": bI,
            "bQ": bQ,
            "residual_norm_I": float(np.linalg.norm(residual_I)),
            "residual_norm_Q": float(np.linalg.norm(residual_Q)),
            "ridge_lambda": ridge_lambda,
            "design_condition_number": design_condition_number,
            "estimation_method": estimation_method,
        }
        if warning_message is not None:
            result["warning"] = warning_message
        return result

    @staticmethod
    def estimate_rx_impairment_filters(y, r, filter_len, ridge_lambda=0.0):
        """Estimate RX-side real-based impairment filters."""
        estimate = IQRealParallelCalibrator.estimate_impairment_filters(
            y, r, filter_len, ridge_lambda=ridge_lambda
        )
        rx_estimate = {
            "e1": estimate["h1"],
            "e2": estimate["h2"],
            "e3": estimate["h3"],
            "e4": estimate["h4"],
            "dI": estimate["bI"],
            "dQ": estimate["bQ"],
            "residual_norm_I": estimate["residual_norm_I"],
            "residual_norm_Q": estimate["residual_norm_Q"],
            "ridge_lambda": estimate["ridge_lambda"],
            "design_condition_number": estimate["design_condition_number"],
            "estimation_method": estimate["estimation_method"],
        }
        if "warning" in estimate:
            rx_estimate["warning"] = estimate["warning"]
        return rx_estimate

    @staticmethod
    def design_precompensation_filters(h1, h2, h3, h4, filter_len):
        """Design real FIR TX precompensation filters g1, g2, g3, g4."""
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        h1 = IQRealParallelCalibrator._as_1d_array(h1, dtype=float, name="h1")
        h2 = IQRealParallelCalibrator._as_1d_array(h2, dtype=float, name="h2")
        h3 = IQRealParallelCalibrator._as_1d_array(h3, dtype=float, name="h3")
        h4 = IQRealParallelCalibrator._as_1d_array(h4, dtype=float, name="h4")

        if min(h1.size, h2.size, h3.size, h4.size) == 0:
            raise ValueError("h1, h2, h3, and h4 must not be empty")

        output_len = max(h1.size, h2.size, h3.size, h4.size) + filter_len - 1
        H1 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h1, filter_len, output_len=output_len
        )
        H2 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h2, filter_len, output_len=output_len
        )
        H3 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h3, filter_len, output_len=output_len
        )
        H4 = IQRealParallelCalibrator._build_full_convolution_matrix(
            h4, filter_len, output_len=output_len
        )

        block_matrix = np.block([[H1, H2], [H3, H4]])
        delta = np.zeros(output_len, dtype=float)
        delta[0] = 1.0
        zero = np.zeros(output_len, dtype=float)
        condition_number = float(np.linalg.cond(block_matrix))
        warning_message = None
        if (
            not np.isfinite(condition_number)
            or condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
        ):
            warning_message = (
                "precompensation LS matrix is ill-conditioned; "
                f"condition_number={condition_number}"
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

        block_pinv = np.linalg.pinv(block_matrix)
        solution_col_1 = block_pinv @ np.concatenate((delta, zero))
        solution_col_2 = block_pinv @ np.concatenate((zero, delta))
        if not (
            np.all(np.isfinite(solution_col_1))
            and np.all(np.isfinite(solution_col_2))
        ):
            raise np.linalg.LinAlgError(
                "precompensation filter design produced non-finite coefficients"
            )

        result = {
            "g1": solution_col_1[:filter_len].astype(float),
            "g3": solution_col_1[filter_len:].astype(float),
            "g2": solution_col_2[:filter_len].astype(float),
            "g4": solution_col_2[filter_len:].astype(float),
            "condition_number": condition_number,
        }
        if warning_message is not None:
            result["warning"] = warning_message
        return result

    @staticmethod
    def design_postcompensation_filters(e1, e2, e3, e4, filter_len):
        """Design real FIR RX postcompensation filters f1, f2, f3, f4."""
        filter_len = IQRealParallelCalibrator._validate_filter_len(filter_len)
        e1 = IQRealParallelCalibrator._as_1d_array(e1, dtype=float, name="e1")
        e2 = IQRealParallelCalibrator._as_1d_array(e2, dtype=float, name="e2")
        e3 = IQRealParallelCalibrator._as_1d_array(e3, dtype=float, name="e3")
        e4 = IQRealParallelCalibrator._as_1d_array(e4, dtype=float, name="e4")

        if min(e1.size, e2.size, e3.size, e4.size) == 0:
            raise ValueError("e1, e2, e3, and e4 must not be empty")

        output_len = max(e1.size, e2.size, e3.size, e4.size) + filter_len - 1
        E1 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e1, filter_len, output_len=output_len
        )
        E2 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e2, filter_len, output_len=output_len
        )
        E3 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e3, filter_len, output_len=output_len
        )
        E4 = IQRealParallelCalibrator._build_full_convolution_matrix(
            e4, filter_len, output_len=output_len
        )

        block_matrix = np.block([[E1, E3], [E2, E4]])
        delta = np.zeros(output_len, dtype=float)
        delta[0] = 1.0
        zero = np.zeros(output_len, dtype=float)
        _, condition_number = IQRealParallelCalibrator._require_identifiable(
            block_matrix, "IQ postcompensation matrix"
        )
        warning_message = None

        block_pinv = np.linalg.pinv(block_matrix)
        solution_row_1 = block_pinv @ np.concatenate((delta, zero))
        solution_row_2 = block_pinv @ np.concatenate((zero, delta))
        if not (
            np.all(np.isfinite(solution_row_1))
            and np.all(np.isfinite(solution_row_2))
        ):
            raise np.linalg.LinAlgError(
                "postcompensation filter design produced non-finite coefficients"
            )

        result = {
            "f1": solution_row_1[:filter_len].astype(float),
            "f2": solution_row_1[filter_len:].astype(float),
            "f3": solution_row_2[:filter_len].astype(float),
            "f4": solution_row_2[filter_len:].astype(float),
            "condition_number": condition_number,
        }
        if warning_message is not None:
            result["warning"] = warning_message
        return result

    @staticmethod
    def design_precompensation_dc_offset(h1, h2, h3, h4, bI, bQ):
        """Design constant I/Q TX offsets that cancel real-based RF DC offsets."""
        h1 = IQRealParallelCalibrator._as_1d_array(h1, dtype=float, name="h1")
        h2 = IQRealParallelCalibrator._as_1d_array(h2, dtype=float, name="h2")
        h3 = IQRealParallelCalibrator._as_1d_array(h3, dtype=float, name="h3")
        h4 = IQRealParallelCalibrator._as_1d_array(h4, dtype=float, name="h4")

        if min(h1.size, h2.size, h3.size, h4.size) == 0:
            raise ValueError("h1, h2, h3, and h4 must not be empty")

        bI = float(bI)
        bQ = float(bQ)
        dc_matrix = np.array(
            [[np.sum(h1), np.sum(h2)], [np.sum(h3), np.sum(h4)]],
            dtype=float,
        )
        rhs = np.array([-bI, -bQ], dtype=float)
        condition_number = float(np.linalg.cond(dc_matrix))
        use_pinv = (
            not np.isfinite(condition_number)
            or condition_number > IQRealParallelCalibrator._COND_WARNING_THRESHOLD
        )

        if use_pinv:
            warnings.warn(
                "DC offset precompensation matrix is ill-conditioned; "
                f"condition_number={condition_number}. Using pseudoinverse LS.",
                RuntimeWarning,
                stacklevel=2,
            )
            solution = np.linalg.pinv(dc_matrix) @ rhs
        else:
            try:
                solution = np.linalg.solve(dc_matrix, rhs)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "DC offset precompensation matrix is singular. "
                    "Using pseudoinverse LS.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                solution = np.linalg.pinv(dc_matrix) @ rhs

        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                "DC offset precompensation produced non-finite coefficients"
            )
        return {
            "aI": float(solution[0]),
            "aQ": float(solution[1]),
            "dc_matrix_condition_number": condition_number,
        }

    @staticmethod
    def design_postcompensation_dc_offset(f1, f2, f3, f4, dI, dQ):
        """Design RX postcompensation DC offsets after f1-f4 filtering."""
        f1 = IQRealParallelCalibrator._as_1d_array(f1, dtype=float, name="f1")
        f2 = IQRealParallelCalibrator._as_1d_array(f2, dtype=float, name="f2")
        f3 = IQRealParallelCalibrator._as_1d_array(f3, dtype=float, name="f3")
        f4 = IQRealParallelCalibrator._as_1d_array(f4, dtype=float, name="f4")

        if min(f1.size, f2.size, f3.size, f4.size) == 0:
            raise ValueError("f1, f2, f3, and f4 must not be empty")

        dI = float(dI)
        dQ = float(dQ)
        cI = -(dI * np.sum(f1) + dQ * np.sum(f2))
        cQ = -(dI * np.sum(f3) + dQ * np.sum(f4))
        if not (np.isfinite(cI) and np.isfinite(cQ)):
            raise np.linalg.LinAlgError(
                "RX postcompensation DC offset produced non-finite coefficients"
            )
        return {"cI": float(cI), "cQ": float(cQ)}

    @staticmethod
    def apply_precompensation(s, g1, g2, g3, g4, aI=0.0, aQ=0.0):
        """Apply real-based parallel TX precompensation to a complex signal."""
        s = IQRealParallelCalibrator._as_1d_array(s, dtype=complex, name="s")
        sI = np.real(s)
        sQ = np.imag(s)
        spI = (
            IQRealParallelCalibrator._causal_fir_same_length(sI, g1)
            + IQRealParallelCalibrator._causal_fir_same_length(sQ, g2)
            + float(aI)
        )
        spQ = (
            IQRealParallelCalibrator._causal_fir_same_length(sI, g3)
            + IQRealParallelCalibrator._causal_fir_same_length(sQ, g4)
            + float(aQ)
        )
        return spI + 1j * spQ

    @staticmethod
    def apply_postcompensation(r, f1, f2, f3, f4, cI=0.0, cQ=0.0):
        """Apply real-based parallel RX postcompensation to a complex signal."""
        r = IQRealParallelCalibrator._as_1d_array(r, dtype=complex, name="r")
        rI = np.real(r)
        rQ = np.imag(r)
        rcI = (
            IQRealParallelCalibrator._causal_fir_same_length(rI, f1)
            + IQRealParallelCalibrator._causal_fir_same_length(rQ, f2)
            + float(cI)
        )
        rcQ = (
            IQRealParallelCalibrator._causal_fir_same_length(rI, f3)
            + IQRealParallelCalibrator._causal_fir_same_length(rQ, f4)
            + float(cQ)
        )
        return rcI + 1j * rcQ


class IQCompensator:
    """
    RX IQ 不平衡补偿器（符号率、多帧支持）

    利用每帧的 CES 字段（收发双方均已知）估计 RX 端 I/Q 损伤滤波器，
    设计后补偿滤波器并应用到整个帧（含前导码），使下游信道估计/均衡
    模块看到干净的 I/Q 信号。

    算法核心（来自 IQRealParallelCalibrator）：
        rI = e1·yI + e2·yQ + dI       ← RX 损伤模型
        rQ = e3·yI + e4·yQ + dQ
        ycI = f1·rI + f2·rQ + cI      ← 后补偿模型
        ycQ = f3·rI + f4·rQ + cQ

    在 THzReceiver 流程中的推荐位置：
        fine CFO 补偿之后、信道估计之前
    """

    def __init__(self, transmitter, filter_len=5, ridge_lambda=0.0,
                 compensation_mode='per_frame'):
        """
        :param transmitter: THzTransmitter 实例（提供 preamble/CES 结构信息）
        :param filter_len: IQ 损伤 FIR 滤波器阶数（≥1，默认 5）
        :param ridge_lambda: LS 估计的岭回归正则化系数（0 表示伪逆，>0 启用岭回归）
        :param compensation_mode: 'per_frame' 逐帧独立估计/补偿，
                                  'first_frame' 仅第一帧估计，其余复用
        """
        self.params = transmitter.params
        self.transmitter = transmitter

        # IQ 补偿超参数
        self.filter_len = self._validate_filter_len(filter_len)
        self.ridge_lambda = float(ridge_lambda)
        if not np.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda 必须为非负有限值")
        if compensation_mode not in ('per_frame', 'first_frame'):
            raise ValueError("compensation_mode 必须为 'per_frame' 或 'first_frame'")
        self.compensation_mode = compensation_mode

        # 从前导码结构中提取关键长度信息
        self.sync_len = len(transmitter.sync)       # SYNC 符号数
        self.sfd_len = len(transmitter.sfd)         # SFD 符号数
        self.tx_ces = transmitter.ces               # 本地已知 CES 序列（符号级）
        self.ces_len = len(self.tx_ces)             # CES 符号数
        self.preamble_len = len(transmitter.preamble)  # 前导码总符号数

        # 直接复用发射机实际生成的帧元数据。OFDM 帧还包含块状导频，
        # 不能使用 subframe_num 重新推导，否则多帧切分会发生错位。
        frame_dict = getattr(transmitter, "data_with_preamble_dict", None)
        if not isinstance(frame_dict, dict):
            raise ValueError("初始化 IQCompensator 前必须先执行 transmitter.run()")
        for key in ("frame_symbol_num", "frame_num"):
            if key not in frame_dict:
                raise KeyError(f"发射机帧数据缺少必要键值：{key}")
        self.frame_symbol_num = int(frame_dict["frame_symbol_num"])
        self.expected_frame_num = int(frame_dict["frame_num"])
        if self.frame_symbol_num <= 0 or self.expected_frame_num <= 0:
            raise ValueError("发射机帧长度和帧数必须为正整数")
        if self.frame_symbol_num < self.preamble_len:
            raise ValueError("发射机单帧长度不能小于前导码长度")
        expected_signal_length = self.frame_symbol_num * self.expected_frame_num
        if int(frame_dict.get("signal_length", -1)) != expected_signal_length:
            raise ValueError(
                "发射机帧元数据不一致："
                f"frame_symbol_num * frame_num={expected_signal_length}, "
                f"signal_length={frame_dict.get('signal_length')}"
            )

        # 输出缓存
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0
        self.compensation_filters = None   # 每帧补偿滤波器列表

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_filter_len(filter_len):
        """校验并转换 filter_len 为整数"""
        try:
            filter_len = int(filter_len)
        except (TypeError, ValueError) as exc:
            raise ValueError("filter_len 必须为正整数") from exc
        if filter_len <= 0:
            raise ValueError("filter_len 必须为正整数")
        return filter_len

    def _verification_data(self, data_dict):
        """校验输入数据字典的完整性"""
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        signal = np.asarray(data_dict["signal_stream"])
        if signal.ndim != 1:
            raise ValueError("IQ 补偿输入 signal_stream 必须是一维信号")
        if len(signal) != int(data_dict["signal_length"]):
            raise ValueError(
                "IQ 补偿输入信号长度不匹配："
                f"len(signal_stream)={len(signal)}, signal_length={data_dict['signal_length']}"
            )

        # 采样率 × 时长 ≅ 信号长度 的一致性校验
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")

        self.sample_rate = data_dict["sample_rate_Hz"]
        self.signal_length = data_dict["signal_length"]
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]

    def _split_into_frames(self, rx_signal, frame_len, tail_mode='error'):
        """
        将一维信号分割为帧列表

        :param rx_signal: 1D 复信号
        :param frame_len: 每帧符号数
        :param tail_mode: 'error' 要求严格整帧，'discard' 丢弃不足一帧的尾部，
                          'zero_pad' 补零至整帧
        :return: (frames_list, num_frames, processed_len)
        """
        total_len = len(rx_signal)
        if frame_len <= 0:
            raise ValueError("帧长度必须大于 0")

        num_frames = total_len // frame_len
        remainder = total_len % frame_len
        if total_len == 0:
            return [], 0, 0

        if remainder == 0:
            processed_signal = rx_signal
        else:
            if tail_mode == 'error':
                raise ValueError(
                    f"IQ 补偿输入长度 {total_len} 不是帧长 {frame_len} 的整数倍，"
                    f"尾部剩余 {remainder} 个符号"
                )
            elif tail_mode == 'discard':
                processed_signal = rx_signal[:num_frames * frame_len]
                print(f"[IQCompensator] 丢弃尾部 {remainder} 个采样点（不足一帧）")
            elif tail_mode == 'zero_pad':
                pad_len = frame_len - remainder
                if rx_signal.ndim == 1:
                    pad = np.zeros(pad_len, dtype=rx_signal.dtype)
                else:
                    pad = np.zeros((pad_len, rx_signal.shape[1]), dtype=rx_signal.dtype)
                processed_signal = np.concatenate([rx_signal, pad], axis=0)
                num_frames += 1
                print(f"[IQCompensator] 尾部补零 {pad_len} 个采样点")
            else:
                raise ValueError("tail_mode 必须为 'error'、'discard' 或 'zero_pad'")

        if num_frames == 0:
            return [], 0, len(processed_signal)
        frames = np.split(processed_signal, num_frames, axis=0)
        return frames, num_frames, len(processed_signal)

    def _estimate_and_design_filters(self, rx_ces):
        """
        从单帧的 RX CES 和本地 TX CES 估计损伤并设计后补偿滤波器

        :param rx_ces: 接收端受损 CES 序列（1D 复信号）
        :return: dict {f1, f2, f3, f4, cI, cQ} 补偿滤波器系数
        """
        # 步骤1：估计 RX 端损伤滤波器 e1~e4 及 DC 偏移 dI/dQ
        rx_est = IQRealParallelCalibrator.estimate_rx_impairment_filters(
            y=self.tx_ces,
            r=rx_ces,
            filter_len=self.filter_len,
            ridge_lambda=self.ridge_lambda
        )

        # 步骤2：设计后补偿 FIR 滤波器 f1~f4
        post_filters = IQRealParallelCalibrator.design_postcompensation_filters(
            rx_est['e1'], rx_est['e2'], rx_est['e3'], rx_est['e4'],
            filter_len=self.filter_len
        )

        # 步骤3：设计后补偿 DC 偏移 cI/cQ
        dc_offset = IQRealParallelCalibrator.design_postcompensation_dc_offset(
            post_filters['f1'], post_filters['f2'],
            post_filters['f3'], post_filters['f4'],
            rx_est['dI'], rx_est['dQ']
        )

        return {
            'f1': post_filters['f1'],
            'f2': post_filters['f2'],
            'f3': post_filters['f3'],
            'f4': post_filters['f4'],
            'cI': dc_offset['cI'],
            'cQ': dc_offset['cQ'],
            # 附加诊断信息
            'residual_norm_I': rx_est['residual_norm_I'],
            'residual_norm_Q': rx_est['residual_norm_Q'],
            'design_condition_number': rx_est['design_condition_number'],
        }

    # ------------------------------------------------------------------
    # 主入口：IQ 补偿
    # ------------------------------------------------------------------

    def compensate(self, data_dict, tail_mode='error'):
        """
        多帧 IQ 不平衡补偿入口

        流程：
          1. 将输入信号按帧分割
          2. 逐帧提取 CES → 估计损伤 → 设计补偿滤波器
          3. 对整帧应用后补偿（含前导码 + 数据部分）
          4. 拼接所有帧输出

        :param data_dict: 输入信号字典（符号率，细 CFO 补偿之后）
        :param tail_mode: 尾部处理方式 'error' / 'discard' / 'zero_pad'
        :return: result_dict 包含补偿后的信号流及补偿滤波器信息
        """
        self._verification_data(data_dict)
        rx_signal = data_dict["signal_stream"]
        fs = self.sample_rate

        # 分帧
        frames, num_frames, processed_len = self._split_into_frames(
            rx_signal, self.frame_symbol_num, tail_mode
        )
        if num_frames == 0:
            raise ValueError("[IQCompensator] 信号不包含完整帧，无法进行 IQ 补偿")
        if tail_mode == 'error' and num_frames != self.expected_frame_num:
            raise ValueError(
                f"IQ 补偿输入帧数不匹配：预期 {self.expected_frame_num}，实际 {num_frames}"
            )

        compensated_frames = []
        filter_info_list = []

        # 用于 'first_frame' 模式：缓存第一帧的滤波器
        first_filters = None

        for i, frame in enumerate(frames):
            # 提取接收帧中的 CES 字段
            ces_start = self.sync_len + self.sfd_len
            rx_ces = frame[ces_start : ces_start + self.ces_len]

            # 根据模式决定是重新估计还是复用
            if self.compensation_mode == 'per_frame' or (
                self.compensation_mode == 'first_frame' and i == 0
            ):
                filters = self._estimate_and_design_filters(rx_ces)
                if self.compensation_mode == 'first_frame':
                    first_filters = filters
            else:
                filters = first_filters

            # 对整帧应用后补偿（前导码 + 数据全部补偿，
            # 使下游信道估计模块看到 IQ 干净的 CES）
            compensated_frame = IQRealParallelCalibrator.apply_postcompensation(
                frame,
                filters['f1'], filters['f2'],
                filters['f3'], filters['f4'],
                filters['cI'], filters['cQ']
            )

            compensated_frames.append(compensated_frame)
            filter_info_list.append(filters)

        # 缓存补偿滤波器
        self.compensation_filters = filter_info_list

        # 拼接所有帧
        rx_compensated = np.concatenate(compensated_frames)

        # 更新时长和长度
        self.duration = len(rx_compensated) / fs
        self.signal_length = len(rx_compensated)

        result_dict = dict(data_dict)
        result_dict.update({
            "signal_stream": rx_compensated,
            "sample_rate_Hz": fs,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
            "frame_symbol_num": self.frame_symbol_num,
            "frame_num": num_frames,
            "iq_compensation_filters": filter_info_list,
        })
        return result_dict


class DecisionDirectedIQCompensator:
    """均衡后基于硬判决符号的残余 RX IQ 不平衡补偿器。"""

    _QAM_SCALE = {
        1: 1.0,
        2: np.sqrt(2),
        4: np.sqrt(10),
        6: np.sqrt(42),
        8: np.sqrt(170),
    }

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
                for i in (-1, 1)
                for q in (-1, 1)
            ])
        elif self.ncbps == 3:
            points = np.exp(1j * np.pi * np.arange(8) / 4)
        elif self.ncbps in (1, 4, 6, 8):
            side = 2 ** (self.ncbps // 2) if self.ncbps > 1 else 2
            levels = (
                np.array([-1.0, 1.0])
                if self.ncbps == 1
                else (2 * np.arange(side) - (side - 1))
                / self._QAM_SCALE[self.ncbps]
            )
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
            distances[start:stop] = distance_matrix[
                np.arange(stop - start), indices
            ]
        return decisions * phase, distances

    @staticmethod
    def _evm(reference, measured):
        return float(
            np.sqrt(
                np.mean(np.abs(measured - reference) ** 2)
                / np.mean(np.abs(reference) ** 2)
            )
        )

    def compensate(self, data_dict):
        required_keys = [
            "signal_stream",
            "sample_rate_Hz",
            "duration_seconds",
            "signal_length",
            "padding_bit_num",
        ]
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
                decisions,
                compensated,
                self.filter_len,
                ridge_lambda=self.ridge_lambda,
            )
            filters = IQRealParallelCalibrator.design_postcompensation_filters(
                estimate["e1"],
                estimate["e2"],
                estimate["e3"],
                estimate["e4"],
                self.filter_len,
            )
            dc = IQRealParallelCalibrator.design_postcompensation_dc_offset(
                filters["f1"],
                filters["f2"],
                filters["f3"],
                filters["f4"],
                estimate["dI"],
                estimate["dQ"],
            )
            before_evm = self._evm(decisions, compensated)
            compensated = IQRealParallelCalibrator.apply_postcompensation(
                compensated,
                filters["f1"],
                filters["f2"],
                filters["f3"],
                filters["f4"],
                dc["cI"],
                dc["cQ"],
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


# ======================================================================
# 测试代码：验证 IQ 补偿在接收链路中的效果
# ======================================================================
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter
    from channel.THzChannel import THzChannel
    from receiver.CFOEstimator import CFOEstimator
    from receiver.sync.FineSync import FineSync
    from receiver.ChannelEstimator import ChannelEstimator
    from receiver.NoiseEstimator import NoiseEstimator
    from receiver.Equalizer import FreqDomainEqualizer
    from receiver.MatchedFilter import RxMatchedFilter
    from receiver.Downsampler import Downsampler
    import matplotlib.pyplot as plt

    # ---- 1. 配置参数（开启 RX 端 IQ 不平衡） ----
    params = PHYParams()
    params.apply_dict({
        "enable_iq_imbalance": True,
        "iq_imbalance_position": "rx",
        "iq_imbalance_model": "fd",           # 频率相关 IQ 不平衡
        "iq_gain_imbalance_db": 2.0,
        "iq_phase_imbalance_deg": 5.0,
        "iq_gI_taps": [1.0, 0.08, -0.03],
        "iq_gQ_taps": [1.0, -0.12, 0.04],
        "iq_power_normalize": False,
    })

    # ---- 2. 发射 → 信道 → 接收前端（匹配滤波 + 同步 + CFO） ----
    transmitter = THzTransmitter(params)
    tx_signal_dict = transmitter.run()

    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    print(f"发射信号长度: {len(tx_signal_dict['signal_stream'])}, "
          f"接收信号长度: {len(rx_signal_dict['signal_stream'])}")

    # 匹配滤波
    rx_matched_filter = RxMatchedFilter(transmitter)
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)

    # 帧定时同步
    fine_sync = FineSync(transmitter)
    rx_sampled_signal_dict = fine_sync.fine_sync(rx_matched_signal_dict)
    print(f"同步偏移: {rx_sampled_signal_dict['sync_offset']}")

    # 粗频偏估计与补偿
    cfo_estimator = CFOEstimator(transmitter)
    coarse_result = cfo_estimator.estimate_and_compensate_cfo_coarse(rx_sampled_signal_dict)
    print(f"粗频偏估计: {coarse_result['cfo_estimates_Hz'][0]:.2f} Hz")

    # 下采样
    downsampler = Downsampler(transmitter)
    rx_downsampled_dict = downsampler.recover_symbol(coarse_result)

    # 细频偏估计与补偿
    fine_cfo_dict = cfo_estimator.estimate_and_compensate_cfo_fine(rx_downsampled_dict)
    print(f"细频偏估计: {fine_cfo_dict['cfo_estimates_Hz'][0]:.2f} Hz")

    # ---- 3. ★ IQ 补偿（在细 CFO 之后、信道估计之前）★ ----
    iq_compensator = IQCompensator(
        transmitter,
        filter_len=5,
        ridge_lambda=0.0,
        compensation_mode='per_frame'
    )
    iq_compensated_dict = iq_compensator.compensate(fine_cfo_dict)

    # 打印诊断信息
    for i, filt in enumerate(iq_compensated_dict['iq_compensation_filters']):
        print(f"\n--- 第 {i+1} 帧 IQ 补偿滤波器 ---")
        print(f"  f1: {filt['f1']}")
        print(f"  f2: {filt['f2']}")
        print(f"  f3: {filt['f3']}")
        print(f"  f4: {filt['f4']}")
        print(f"  DC (cI, cQ): ({filt['cI']:.4f}, {filt['cQ']:.4f})")
        print(f"  残差范数 (I/Q): ({filt['residual_norm_I']:.4f}, {filt['residual_norm_Q']:.4f})")

    # ---- 4. 信道估计（在 IQ 补偿后） ----
    channel_estimator = ChannelEstimator(transmitter)
    rx_estimated_dict = channel_estimator.channel_estimate(iq_compensated_dict)
    H_est_list = rx_estimated_dict['channel_freq_response']

    # ---- 5. 噪声估计 ----
    noise_estimator = NoiseEstimator(transmitter)
    noise_result = noise_estimator.noise_estimate(rx_estimated_dict)

    # ---- 6. 均衡 ----
    equalizer = FreqDomainEqualizer(transmitter)
    equalized_dict = equalizer.equalize(noise_result)

    # ---- 7. 可视化对比 ----
    num_points = min(2048, len(equalized_dict['signal_stream']))
    eq_signal = equalized_dict['signal_stream'][:num_points]

    # IQ 补偿前后对比结果
    rx_iq_compensated = iq_compensated_dict['signal_stream'][:num_points]
    rx_no_iq = fine_cfo_dict['signal_stream'][:num_points]

    # 参考：无 IQ 补偿的均衡结果
    # （跳过 IQ 补偿，直接用细 CFO 结果走信道估计 → 均衡）
    rx_no_iq_est = channel_estimator.channel_estimate(fine_cfo_dict)
    rx_no_iq_noise = noise_estimator.noise_estimate(rx_no_iq_est)
    eq_no_iq_dict = equalizer.equalize(rx_no_iq_noise)
    eq_no_iq_signal = eq_no_iq_dict['signal_stream'][:num_points]

    # 原始发射星座点
    tx_data = transmitter.modulated_data_dict['signal_stream'][:num_points]

    fig, axes = plt.subplots(1, 5, figsize=(30, 5))

    axes[0].plot(tx_data.real, tx_data.imag, 'x', markersize=3, alpha=0.6)
    axes[0].set_title("TX 原始星座图")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
    axes[0].axis('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eq_no_iq_signal.real, eq_no_iq_signal.imag, '.', markersize=2, alpha=0.6)
    axes[1].set_title("无 IQ 补偿 → 均衡后星座图")
    axes[1].set_xlabel("I"); axes[1].set_ylabel("Q")
    axes[1].axis('equal'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eq_signal.real, eq_signal.imag, '.', markersize=2, alpha=0.6)
    axes[2].set_title("IQ 补偿后 → 均衡后星座图")
    axes[2].set_xlabel("I"); axes[2].set_ylabel("Q")
    axes[2].axis('equal'); axes[2].grid(True, alpha=0.3)

    axes[3].plot(rx_no_iq.real, rx_no_iq.imag, 'r.', markersize=2, alpha=0.6)
    axes[3].set_title("IQ补偿前接收信号星座图")
    axes[3].set_xlabel("I"); axes[3].set_ylabel("Q")
    axes[3].axis('equal'); axes[3].grid(True, alpha=0.3)

    axes[4].plot(rx_iq_compensated.real, rx_iq_compensated.imag, 'g.', markersize=2, alpha=0.6)
    axes[4].set_title("IQ补偿后接收信号星座图")
    axes[4].set_xlabel("I"); axes[4].set_ylabel("Q")
    axes[4].axis('equal'); axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 频域 CSI 对比（第一帧）
    if H_est_list:
        H_iq = H_est_list[0]
        H_no_iq = rx_no_iq_est['channel_freq_response']
        if isinstance(H_no_iq, list):
            H_no_iq = H_no_iq[0]

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        nfft = len(H_iq)
        axes2[0].plot(np.abs(H_iq), 'b-', label='w/ IQ补偿')
        axes2[0].plot(np.abs(H_no_iq), 'r--', label='w/o IQ补偿')
        axes2[0].set_title(f"频域 CSI 幅值对比 ({nfft}-FFT)")
        axes2[0].set_xlabel("子载波索引"); axes2[0].set_ylabel("|H|")
        axes2[0].legend(); axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(np.angle(H_iq), 'b-', label='w/ IQ补偿')
        axes2[1].plot(np.angle(H_no_iq), 'r--', label='w/o IQ补偿')
        axes2[1].set_title(f"频域 CSI 相位对比 ({nfft}-FFT)")
        axes2[1].set_xlabel("子载波索引"); axes2[1].set_ylabel("∠H (rad)")
        axes2[1].legend(); axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("\n===== IQ 补偿测试完成 =====")
