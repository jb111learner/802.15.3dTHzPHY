import numpy as np

class IQImbalance:
    """
    频率无关（FID）和频率相关（FD）的 IQ 不平衡模型。

    等效复基带模型：
        y = mu * x + nu * conj(x)

    对于 FID，系数由增益/相位不平衡推导；
    对于 FD，额外对 I/Q 支路施加不同的 FIR 滤波器。
    """

    VALID_POSITIONS = {"tx", "rx", "both"}
    SUPPORTED_MODELS = {"fid", "fd"}
    STAGES = {"tx", "rx"}

    def __init__(self, params):
        """
        参数：
            params : dict
                必须包含以下可选键：
                - enable_iq_imbalance (bool)        : 是否启用，默认 False
                - iq_imbalance_position (str)       : 'tx', 'rx', 'both'，默认 'rx'
                - iq_imbalance_model (str)          : 'fid' 或 'fd'，默认 'fid'
                - iq_power_normalize (bool)         : 是否归一化输出功率，默认 False

                - iq_gain_imbalance_db (float)      : 全局增益不平衡（dB），默认 0.0
                - iq_phase_imbalance_deg (float)    : 全局相位不平衡（度），默认 0.0
                - iq_gI_taps (list)                 : 全局 I 支路 FIR 抽头，默认 [1.0]
                - iq_gQ_taps (list)                 : 全局 Q 支路 FIR 抽头，默认 [1.0]

                - tx_iq_gain_imbalance_db (float)   : 发射端专用增益
                - tx_iq_phase_imbalance_deg (float) : 发射端专用相位
                - tx_iq_gI_taps (list)              : 发射端 I 支路抽头
                - tx_iq_gQ_taps (list)              : 发射端 Q 支路抽头

                - rx_iq_gain_imbalance_db (float)   : 接收端专用增益
                - rx_iq_phase_imbalance_deg (float) : 接收端专用相位
                - rx_iq_gI_taps (list)              : 接收端 I 支路抽头
                - rx_iq_gQ_taps (list)              : 接收端 Q 支路抽头

            阶段特定参数若未提供，则回退到全局对应参数。
        """
        self.params = params

        # 基本开关
        self.enable = params.get("enable_iq_imbalance")
        self.position = params.get("iq_imbalance_position").lower()
        self.model = params.get("iq_imbalance_model").lower()
        self.normalize = params.get("iq_power_normalize")

        # 全局增益/相位
        self.gain_db = float(params.get("iq_gain_imbalance_db"))
        self.phase_deg = float(params.get("iq_phase_imbalance_deg"))

        # 全局抽头
        self.gI_global = self._normalize_taps(params.get("iq_gI_taps"))
        self.gQ_global = self._normalize_taps(params.get("iq_gQ_taps"))

        # ---- 解析各阶段参数（阶段特定优先，否则回退到全局） ----
        # 发射端
        tx_gain = params.get("tx_iq_gain_imbalance_db")
        self.tx_gain_db = float(tx_gain) if tx_gain is not None else self.gain_db
        tx_phase = params.get("tx_iq_phase_imbalance_deg")
        self.tx_phase_deg = float(tx_phase) if tx_phase is not None else self.phase_deg

        tx_gI = self._normalize_taps(params.get("tx_iq_gI_taps"))
        self.tx_gI = tx_gI if not self._is_identity(tx_gI) else self.gI_global
        tx_gQ = self._normalize_taps(params.get("tx_iq_gQ_taps"))
        self.tx_gQ = tx_gQ if not self._is_identity(tx_gQ) else self.gQ_global

        # 接收端
        rx_gain = params.get("rx_iq_gain_imbalance_db")
        self.rx_gain_db = float(rx_gain) if rx_gain is not None else self.gain_db
        rx_phase = params.get("rx_iq_phase_imbalance_deg")
        self.rx_phase_deg = float(rx_phase) if rx_phase is not None else self.phase_deg

        rx_gI = self._normalize_taps(params.get("rx_iq_gI_taps"))
        self.rx_gI = rx_gI if not self._is_identity(rx_gI) else self.gI_global
        rx_gQ = self._normalize_taps(params.get("rx_iq_gQ_taps"))
        self.rx_gQ = rx_gQ if not self._is_identity(rx_gQ) else self.gQ_global

        # 存储输入字段（校验后填充）
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0

    # ---------- 静态工具方法 ----------
    @staticmethod
    def _normalize_taps(taps):
        """将抽头转换为 1D float numpy 数组，空或 None 则返回 [1.0]"""
        if taps is None:
            taps = [1.0]
        arr = np.asarray(taps, dtype=float)
        if arr.ndim != 1:
            raise ValueError("IQ imbalance FIR taps must be one-dimensional")
        return arr if arr.size > 0 else np.array([1.0], dtype=float)

    @staticmethod
    def _is_identity(taps):
        """判断抽头是否为单位脉冲 [1.0]"""
        return len(taps) == 1 and np.isclose(taps[0], 1.0)

    @staticmethod
    def compute_fid_coefficients(gain_db, theta_deg):
        """
        根据增益/相位计算 FID 模型的 mu, nu 以及辅助参数 eps, theta。
        返回 (eps, theta, mu, nu)
        """
        rho = 10 ** (float(gain_db) / 20.0)
        eps = (rho - 1.0) / (rho + 1.0)
        theta = np.deg2rad(float(theta_deg))
        mu = np.cos(theta) + 1j * eps * np.sin(theta)
        nu = -eps * np.cos(theta) - 1j * np.sin(theta)
        return eps, theta, mu, nu

    @staticmethod
    def _apply_fir_same_length(x, taps):
        """对信号 x 应用 FIR 滤波器，输出长度与 x 相同（截断）"""
        taps = IQImbalance._normalize_taps(taps)
        x = np.asarray(x)
        if len(x) == 0:
            return np.zeros_like(x, dtype=float)
        return np.convolve(x, taps, mode="full")[:len(x)]

    @staticmethod
    def _compute_fd_equivalent_taps(gI_taps, gQ_taps, eps, theta):
        """计算 FD 模型的等效 mu_taps 和 nu_taps"""
        max_len = max(len(gI_taps), len(gQ_taps))
        gI = np.zeros(max_len, dtype=float)
        gQ = np.zeros(max_len, dtype=float)
        gI[:len(gI_taps)] = gI_taps
        gQ[:len(gQ_taps)] = gQ_taps

        i_branch = (1.0 - eps) * (np.cos(theta) - 1j * np.sin(theta))
        q_branch = (1.0 + eps) * (-np.sin(theta) + 1j * np.cos(theta))
        q_to_complex = q_branch / (2.0j)

        mu_taps = 0.5 * i_branch * gI + q_to_complex * gQ
        nu_taps = 0.5 * i_branch * gI - q_to_complex * gQ
        return mu_taps, nu_taps

    @staticmethod
    def analyze_fd_response(gI_taps, gQ_taps, gain_db=0.0, theta_deg=0.0, nfft=512):
        """
        分析 FD 模型的频率响应，返回各频点的 IRR。
        返回字典包含频率、传递函数、IRR 等（静态工具，不影响主流程）。
        """
        nfft = int(nfft)
        if nfft <= 0:
            raise ValueError("nfft must be positive")
        gI_taps = IQImbalance._normalize_taps(gI_taps)
        gQ_taps = IQImbalance._normalize_taps(gQ_taps)
        eps, theta, _, _ = IQImbalance.compute_fid_coefficients(gain_db, theta_deg)
        mu_taps, nu_taps = IQImbalance._compute_fd_equivalent_taps(
            gI_taps, gQ_taps, eps, theta
        )
        H_mu = np.fft.fft(mu_taps, n=nfft)
        H_nu = np.fft.fft(nu_taps, n=nfft)
        idx = np.arange(nfft)
        H_nu_image = H_nu[(-idx) % nfft]
        tiny = np.finfo(float).tiny
        irr_tone = 10 * np.log10((np.abs(H_mu)**2 + tiny) / (np.abs(H_nu_image)**2 + tiny))
        return {
            "freq_normalized": np.fft.fftfreq(nfft, d=1.0),
            "H_mu": H_mu,
            "H_nu": H_nu,
            "H_nu_image": H_nu_image,
            "irr_tone_db": irr_tone,
            "irr_db": irr_tone,
            "mu_taps": mu_taps,
            "nu_taps": nu_taps,
            "irr_definition": "tone/image IRR: |H_mu(f)|^2 / |H_nu(-f)|^2"
        }

    # ---------- 内部应用辅助 ----------
    def _verification_data(self, data_dict):
        """
        校验输入数据字典的合法性，提取必要字段。
        与 CFO 类风格一致，要求包含所有键。
        """
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        # 存储到实例属性
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.duration = data_dict["duration_seconds"]
        self.signal_length = data_dict["signal_length"]
        self.padding_bit_num = data_dict["padding_bit_num"]

        # 可选：校验采样率与时长是否匹配（与 CFO 一致）
        if round(self.sample_rate * self.duration) != self.signal_length:
            raise ValueError("采样率与时长不匹配信号长度")

    def _stage_is_enabled(self, stage):
        if not self.enable:
            return False
        return self.position == "both" or self.position == stage

    def _get_stage_gain_phase(self, stage):
        """返回指定阶段的 (gain_db, phase_deg)"""
        if stage == "tx":
            return self.tx_gain_db, self.tx_phase_deg
        else:  # rx
            return self.rx_gain_db, self.rx_phase_deg

    def _get_stage_taps(self, stage):
        """返回指定阶段的 (gI_taps, gQ_taps)"""
        if stage == "tx":
            return self.tx_gI, self.tx_gQ
        else:
            return self.rx_gI, self.rx_gQ

    def _apply_fid(self, x, gain_db, phase_deg):
        """FID 模型应用，返回 (y, norm_gain)"""
        eps, theta, mu, nu = self.compute_fid_coefficients(gain_db, phase_deg)
        y_raw = mu * x + nu * np.conj(x)
        norm_gain = 1.0
        if self.normalize:
            norm = np.sqrt(abs(mu)**2 + abs(nu)**2)
            if norm > 0:
                norm_gain = 1.0 / norm
                y_raw = y_raw * norm_gain
        return y_raw, norm_gain

    def _apply_fd(self, x, gain_db, phase_deg, gI_taps, gQ_taps):
        """FD 模型应用，返回 (y, norm_gain)"""
        eps, theta, _, _ = self.compute_fid_coefficients(gain_db, phase_deg)
        I = np.real(x)
        Q = np.imag(x)
        I_f = self._apply_fir_same_length(I, gI_taps)
        Q_f = self._apply_fir_same_length(Q, gQ_taps)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        I_out = (1.0 - eps) * I_f * cos_t - (1.0 + eps) * Q_f * sin_t
        Q_out = -(1.0 - eps) * I_f * sin_t + (1.0 + eps) * Q_f * cos_t
        y_raw = I_out + 1j * Q_out

        norm_gain = 1.0
        if self.normalize and len(x) > 0:
            inp_pwr = float(np.mean(np.abs(x)**2))
            out_pwr = float(np.mean(np.abs(y_raw)**2))
            if inp_pwr > 0 and out_pwr > 0:
                norm_gain = np.sqrt(inp_pwr / out_pwr)
                y_raw = y_raw * norm_gain
        return y_raw, norm_gain

    # ---------- 主应用接口 ----------
    def apply(self, signal_dict, stage="rx"):
        """
        对信号字典应用 IQ 不平衡，返回固定格式的输出字典。
        signal_dict 必须包含：signal_stream, sample_rate_Hz, duration_seconds,
                             signal_length, padding_bit_num。
        stage 指定应用阶段 ('tx' 或 'rx')。
        """
        stage = stage.lower()
        if stage not in self.STAGES:
            raise ValueError("stage must be 'tx' or 'rx'")

        # 校验输入并存储字段
        self._verification_data(signal_dict)

        # 若未启用，直接返回原信号（但输出格式固定）
        if not self._stage_is_enabled(stage):
            result_signal = np.asarray(signal_dict["signal_stream"], dtype=complex)
        else:
            x = np.asarray(signal_dict["signal_stream"], dtype=complex)
            gain_db, phase_deg = self._get_stage_gain_phase(stage)

            if self.model == "fid":
                y, _ = self._apply_fid(x, gain_db, phase_deg)
            elif self.model == "fd":
                gI_taps, gQ_taps = self._get_stage_taps(stage)
                y, _ = self._apply_fd(x, gain_db, phase_deg, gI_taps, gQ_taps)
            else:
                raise ValueError(f"Unsupported model '{self.model}'")
            result_signal = y

        # 校验输出长度（可选）
        if len(result_signal) != self.signal_length:
            raise ValueError(f"输出信号长度不匹配: 预期 {self.signal_length}, 实际 {len(result_signal)}")

        # 构建固定输出字典
        results_dict = {
            "signal_stream": result_signal,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return results_dict
