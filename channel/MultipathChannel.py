import copy
import numpy as np


# 3GPP TR 38.901 V19.4.0, Table 7.7.2-1 through Table 7.7.2-5.
# Delays are normalized to the requested RMS delay spread at runtime.
_TDL_PROFILES = {
    "TDL-A": {
        "kind": "rayleigh",
        "taps": (
            (0.0000, -13.4), (0.3819, 0.0), (0.4025, -2.2),
            (0.5868, -4.0), (0.4610, -6.0), (0.5375, -8.2),
            (0.6708, -9.9), (0.5750, -10.5), (0.7618, -7.5),
            (1.5375, -15.9), (1.8978, -6.6), (2.2242, -16.7),
            (2.1718, -12.4), (2.4942, -15.2), (2.5119, -10.8),
            (3.0582, -11.3), (4.0810, -12.7), (4.4579, -16.2),
            (4.5695, -18.3), (4.7966, -18.9), (5.0066, -16.6),
            (5.3043, -19.9), (9.6586, -29.7),
        ),
    },
    "TDL-B": {
        "kind": "rayleigh",
        "taps": (
            (0.0000, 0.0), (0.1072, -2.2), (0.2155, -4.0),
            (0.2095, -3.2), (0.2870, -9.8), (0.2986, -1.2),
            (0.3752, -3.4), (0.5055, -5.2), (0.3681, -7.6),
            (0.3697, -3.0), (0.5700, -8.9), (0.5283, -9.0),
            (1.1021, -4.8), (1.2756, -5.7), (1.5474, -7.5),
            (1.7842, -1.9), (2.0169, -7.6), (2.8294, -12.2),
            (3.0219, -9.8), (3.6187, -11.4), (4.1067, -14.9),
            (4.2790, -9.2), (4.7834, -11.3),
        ),
    },
    "TDL-C": {
        "kind": "rayleigh",
        "taps": (
            (0.0000, -4.4), (0.2099, -1.2), (0.2219, -3.5),
            (0.2329, -5.2), (0.2176, -2.5), (0.6366, 0.0),
            (0.6448, -2.2), (0.6560, -3.9), (0.6584, -7.4),
            (0.7935, -7.1), (0.8213, -10.7), (0.9336, -11.1),
            (1.2285, -5.1), (1.3083, -6.8), (2.1704, -8.7),
            (2.7105, -13.2), (4.2589, -13.9), (4.6003, -13.9),
            (5.4902, -15.8), (5.6077, -17.1), (6.3065, -16.0),
            (6.6374, -15.7), (7.0427, -21.6), (8.6523, -22.8),
        ),
    },
    "TDL-D": {
        "kind": "rician",
        "los_first": (-0.2, -13.5, 13.3),
        "taps": (
            (0.0350, -18.8), (0.6120, -21.0), (1.3630, -22.8),
            (1.4050, -17.9), (1.8040, -20.1), (2.5960, -21.9),
            (1.7750, -22.9), (4.0420, -27.8), (7.9370, -23.6),
            (9.4240, -24.8), (9.7080, -30.0), (12.5250, -27.7),
        ),
    },
    "TDL-E": {
        "kind": "rician",
        "los_first": (-0.03, -22.03, 22.0),
        "taps": (
            (0.5133, -15.8), (0.5440, -18.1), (0.5630, -19.8),
            (0.5440, -22.9), (0.7112, -22.4), (1.9092, -18.6),
            (1.9293, -20.8), (1.9589, -22.6), (2.6426, -22.3),
            (3.7136, -25.6), (5.4524, -20.2), (12.0034, -29.8),
            (20.6519, -29.2),
        ),
    },
}


class MultipathChannel:
    """
    多径信道模型：支持 3GPP TDL-A~E 和旧版自定义/PDP 回退。

    设计目标：
    1. 接收并返回当前平台统一使用的 signal_dict，而不是裸数组。
    2. 支持可配置的静态多径：delay_samples / tau + static complex gain。
    3. 支持可选分数延迟滤波，输出长度严格保持与输入一致。
    4. 不在本模块中添加 AWGN，噪声仍由 channel/AWGN.py 负责。
    5. TDL 模式使用固定的 Rayleigh/Rician 信道实现；旧接口保留原有衰落选项。
    """

    _EPS = 1e-12
    _LIGHT_SPEED_MPS = 299792458.0
    _DEFAULT_JAKES_SINUSOIDS = 48

    def __init__(self, params):
        self.params = params

        self.tdl_model = self._normalize_tdl_model(self._get_param("tdl_model", None))
        self.tdl_delay_spread_ns = self._get_tdl_delay_spread_ns()
        self.tdl_velocity_mps = self._get_tdl_velocity_mps()
        self.tdl_fc_hz = self._get_tdl_fc_hz()
        self.tdl_max_doppler_hz = (
            abs(self.tdl_velocity_mps) * self.tdl_fc_hz / self._LIGHT_SPEED_MPS
            if self.tdl_model is not None else 0.0
        )
        self.tdl_num_sinusoids = int(
            self._get_param("tdl_jakes_num_sinusoids", self._DEFAULT_JAKES_SINUSOIDS)
        )
        if self.tdl_num_sinusoids < 4 or self.tdl_num_sinusoids % 2:
            raise ValueError("tdl_jakes_num_sinusoids 必须为不小于 4 的偶数")
        # TDL profiles use the protocol's static realization and integer delays.
        self.use_frac_delay = (
            False
            if self.tdl_model is not None
            else bool(self._get_param("use_frac_delay", False))
        )
        self.frac_filter_half_len = int(self._get_param("frac_filter_half_len", 12))
        self.frac_filter_window = self._get_param("frac_filter_window", "hann")
        
        self.static_channel = (
            True if self.tdl_model is not None
            else bool(self._get_param("static_channel", True))
        )
        self.normalize_channel_power = (
            False if self.tdl_model is not None
            else bool(self._get_param("normalize_channel_power", False))
        )

        # 是否启用帧间输入历史缓冲区。第一版只用于 static 多径。
        self.enable_multipath_memory = bool(self._get_param("enable_multipath_memory", False))

        # 衰落模型控制。当前版本支持 static 和 frame。
        # frame 表示每次 apply() 时，每条 Rayleigh/Rician 路径生成一个帧内固定衰落系数。
        self.fading_model = (
            "static" if self.tdl_model is not None
            else str(self._get_param("fading_model", "static")).lower()
        )
        self.fading_seed = self._get_param(
            "fading_seed", self._get_param("random_seed", None)
        )
        self.block_length = self._get_param("block_length", 256)
        self.rng = np.random.default_rng(self.fading_seed)

        if self.fading_model not in ("static", "frame", "block"):
            raise ValueError(
                f"当前 MultipathChannel 支持 fading_model='static'/'frame'/'block'，"
                f"但收到 {self.fading_model}"
            )

        if self.frac_filter_half_len < 1:
            raise ValueError("frac_filter_half_len 必须为正整数")

        self.active_paths = None
        self.active_blocks = []
        self.resolved_paths = None
        self.chan_impulse = None
        self.channel_power_gain = None
        # PDP 自动路径生成缓存。PDP 只负责生成路径几何和平均功率，不参与每帧随机衰落。
        self._pdp_paths_cache = None
        self._tdl_paths_cache = None
        self._tdl_jakes_state = None
        self._last_tdl_gain_traces = None

        # 大尺度衰落 / 路径损耗缓存。默认不启用，启用后在当前信道实例内保持不变。
        self.enable_large_scale_fading = bool(self._get_param("enable_large_scale_fading", False))
        self.large_scale_model = str(self._get_param("large_scale_model", "none")).lower()
        self.large_scale_seed = self._get_param("large_scale_seed", None)
        self.large_scale_rng = np.random.default_rng(self.large_scale_seed)
        self._large_scale_info_cache = None

        if self.large_scale_model not in ("none", "fspl", "log_distance"):
            raise ValueError(
                f"当前支持 large_scale_model='none'/'fspl'/'log_distance'，"
                f"但收到 {self.large_scale_model}"
            )
        
        # 帧间连续性状态：保存上一帧末尾的输入样本，而不是输出样本。
        self._input_history = np.zeros(0, dtype=complex)
        self._history_length = 0
        self._processed_samples = 0

    def _get_param(self, key, default=None):
        """兼容 BaseParams.get()：参数不存在时返回默认值。"""
        try:
            return self.params.get(key)
        except KeyError:
            return default

    @staticmethod
    def _normalize_tdl_model(value):
        """Normalize the public selector while preserving None for legacy mode."""
        if value is None:
            return None
        value = str(value).strip().upper()
        if value in ("", "NONE", "OFF", "DISABLED"):
            return None
        if value in ("A", "B", "C", "D", "E"):
            value = f"TDL-{value}"
        if value not in _TDL_PROFILES:
            raise ValueError(
                f"tdl_model 必须为 None 或 TDL-A/TDL-B/TDL-C/TDL-D/TDL-E，收到 {value}"
            )
        return value

    def _get_tdl_delay_spread_ns(self):
        if self.tdl_model is None:
            return None
        value = float(self._get_param("tdl_delay_spread_ns", 100.0))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("tdl_delay_spread_ns 必须为有限正数")
        return value

    def _get_tdl_velocity_mps(self):
        if self.tdl_model is None:
            return 0.0
        value = float(self._get_param("tdl_velocity_mps", 0.0))
        if not np.isfinite(value):
            raise ValueError("tdl_velocity_mps 必须为有限数")
        return value

    def _get_tdl_fc_hz(self):
        if self.tdl_model is None:
            return 0.0
        value = self._get_param("fc", None)
        if value is None:
            value = self._get_param("carrier_frequency_Hz", 0.0)
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("TDL Doppler 计算要求 fc 为有限正数")
        return value

    def _get_pdp_param(self, key, default=None):
        """读取 PDP 参数，和 _get_param 分开只是为了代码语义更清晰。"""
        return self._get_param(key, default)

    def reset_state(self):
        """
        重置多径信道的帧间状态。

        注意：
        如果设置了 fading_seed，这里会同时重置随机数生成器，
        便于复现实验。
        """
        self._input_history = np.zeros(0, dtype=complex)
        self._history_length = 0
        self._processed_samples = 0
        self.rng = np.random.default_rng(self.fading_seed)
        if self.tdl_model is not None:
            self._tdl_paths_cache = None
            self._tdl_jakes_state = None
            self._last_tdl_gain_traces = None

    def _get_configured_paths(self):
        """
        获取路径配置。

        优先级：
        1. use_pdp=True 或 pdp_model != "manual" 时，使用 PDP 自动生成路径；
        2. 否则使用新版 multipath_paths；
        3. 否则兼容旧版 chan_delays / chan_gains；
        4. 都没有时，退化为一条直达径。
        """
        if self.tdl_model is not None:
            return self._get_tdl_paths_cached()

        use_pdp = bool(self._get_param("use_pdp", False))
        pdp_model = str(self._get_param("pdp_model", "manual")).lower()

        if use_pdp or pdp_model != "manual":
            return self._get_pdp_paths_cached()

        paths = self._get_param("multipath_paths", None)
        if paths:
            return copy.deepcopy(paths)

        chan_delays = self._get_param("chan_delays", None)
        chan_gains = self._get_param("chan_gains", None)
        if chan_delays is not None and chan_gains is not None:
            if len(chan_delays) != len(chan_gains):
                raise ValueError("chan_delays 与 chan_gains 长度必须一致")
            return [
                {
                    "delay_samples": float(delay),
                    "gain_type": "static",
                    "gain": complex(gain),
                }
                for delay, gain in zip(chan_delays, chan_gains)
            ]

        return [{"delay_samples": 0.0, "gain_type": "static", "gain": 1.0 + 0.0j}]

    def _get_large_scale_info_cached(self):
        """
        获取当前信道实例的大尺度衰落信息。

        大尺度衰落通常表示当前链路场景，如距离、路径损耗和阴影衰落。
        因此在一个 MultipathChannel 实例内保持不变，不随每帧小尺度衰落更新。
        """
        if self._large_scale_info_cache is None:
            self._large_scale_info_cache = self._compute_large_scale_info()

        return copy.deepcopy(self._large_scale_info_cache)


    def _compute_large_scale_info(self):
        """
        计算大尺度路径损耗和线性功率增益。

        返回：
            path_loss_db:
                路径损耗，单位 dB。

            power_gain_linear:
                线性功率增益，等于 10^(-PL/10)。

            amplitude_gain_linear:
                线性幅度增益，等于 sqrt(power_gain_linear)。
        """
        c = 299792458.0

        enabled = bool(self.enable_large_scale_fading)
        model = str(self.large_scale_model).lower()

        if (not enabled) or model == "none":
            return {
                "enabled": False,
                "model": model,
                "path_loss_db": 0.0,
                "shadowing_db": 0.0,
                "power_gain_linear": 1.0,
                "amplitude_gain_linear": 1.0,
            }

        d = float(self._get_param("link_distance_m", 1.0))
        f = float(self._get_param("fc", 300e9))
        d0 = float(self._get_param("reference_distance_m", 1.0))
        n = float(self._get_param("path_loss_exponent", 2.0))
        shadowing_std_db = float(self._get_param("shadowing_std_db", 0.0))
        reference_path_loss_db = self._get_param("reference_path_loss_db", None)

        if d <= 0:
            raise ValueError("link_distance_m 必须为正数")
        if f <= 0:
            raise ValueError("fc 必须为正数")
        if d0 <= 0:
            raise ValueError("reference_distance_m 必须为正数")
        if n < 0:
            raise ValueError("path_loss_exponent 不能为负数")
        if shadowing_std_db < 0:
            raise ValueError("shadowing_std_db 不能为负数")

        if model == "fspl":
            path_loss_db = 20.0 * np.log10(4.0 * np.pi * d * f / c)

        elif model == "log_distance":
            if reference_path_loss_db is None:
                pl_d0_db = 20.0 * np.log10(4.0 * np.pi * d0 * f / c)
            else:
                pl_d0_db = float(reference_path_loss_db)

            path_loss_db = pl_d0_db + 10.0 * n * np.log10(d / d0)

        else:
            raise ValueError(f"未知 large_scale_model: {model}")

        if shadowing_std_db > 0:
            shadowing_db = float(self.large_scale_rng.normal(loc=0.0, scale=shadowing_std_db))
        else:
            shadowing_db = 0.0

        # 正的 shadowing_db 表示额外损耗，负的表示阴影条件较好。
        path_loss_db = float(path_loss_db + shadowing_db)

        power_gain_linear = float(10.0 ** (-path_loss_db / 10.0))
        amplitude_gain_linear = float(np.sqrt(power_gain_linear))

        return {
            "enabled": True,
            "model": model,
            "link_distance_m": d,
            "fc": f,
            "reference_distance_m": d0,
            "reference_path_loss_db": reference_path_loss_db,
            "path_loss_exponent": n,
            "shadowing_std_db": shadowing_std_db,
            "shadowing_db": shadowing_db,
            "path_loss_db": path_loss_db,
            "power_gain_linear": power_gain_linear,
            "amplitude_gain_linear": amplitude_gain_linear,
        }


    def _apply_large_scale_fading_to_paths(self, resolved_paths):
        """
        将大尺度衰落作用到路径平均功率上。

        对 Rayleigh / Rician：
            power *= large_scale_power_gain

        对 static：
            gain *= sqrt(large_scale_power_gain)
            power *= large_scale_power_gain

        注意：
            这里缩放的是平均路径功率，后续小尺度衰落仍然正常生成。
        """
        large_scale_info = self._get_large_scale_info_cached()
        power_gain = float(large_scale_info["power_gain_linear"])
        amp_gain = float(large_scale_info["amplitude_gain_linear"])

        scaled_paths = []

        for path in resolved_paths:
            p = path.copy()

            if "power" in p and p["power"] is not None:
                p["power"] = float(p["power"]) * power_gain

            if "gain" in p and p["gain"] is not None:
                p["gain"] = complex(p["gain"]) * amp_gain

            p["large_scale_power_gain"] = power_gain
            p["large_scale_amplitude_gain"] = amp_gain
            p["large_scale_path_loss_db"] = large_scale_info["path_loss_db"]
            p["large_scale_model"] = large_scale_info["model"]

            scaled_paths.append(p)

        return scaled_paths

    def _parse_paths(self, fs):
        """
        将用户配置的路径转换为内部标准格式。

        支持两种时延写法：
        - delay_samples：直接给出采样点时延，可为浮点数。
        - tau：连续时间时延，单位秒，内部用 tau * fs 转换为采样点时延。

        优先级：delay_samples > tau。
        """
        if fs <= 0:
            raise ValueError("sample_rate_Hz 必须为正数")

        configured_paths = self._get_configured_paths()
        resolved_paths = []

        for idx, path in enumerate(configured_paths):
            if not isinstance(path, dict):
                raise TypeError(f"第 {idx} 条多径配置必须为 dict")

            if "delay_samples" in path and path["delay_samples"] is not None:
                delay_samples = float(path["delay_samples"])
            elif "tau" in path and path["tau"] is not None:
                delay_samples = float(path["tau"]) * fs
            else:
                delay_samples = 0.0

            if delay_samples < -self._EPS:
                raise ValueError(f"第 {idx} 条路径的时延不能为负数: {delay_samples}")
            delay_samples = max(0.0, delay_samples)

            gain_type = str(path.get("gain_type", "static")).lower()
            if gain_type not in ("static", "rayleigh", "rician"):
                raise ValueError(
                    f"第 {idx} 条路径的 gain_type={gain_type} 不支持，"
                    f"当前支持 static / rayleigh / rician"
                )

            if self.static_channel:
                gain_type = "static"

            gain = path.get("gain", None)
            power = path.get("power", None)

            if power is None:
                if gain is not None:
                    power = abs(complex(gain)) ** 2
                else:
                    power = 1.0

            power = float(power)
            if power < 0:
                raise ValueError(f"第 {idx} 条路径的 power 不能为负数")

            if gain is None:
                static_gain = np.sqrt(power) + 0.0j
            else:
                static_gain = complex(gain)

            K = float(path.get("K", 0.0))
            if K < 0:
                raise ValueError(f"第 {idx} 条路径的 Rician K 因子不能为负数")

            fd = float(path.get("fd", 0.0))
            f_los = float(path.get("f_los", 0.0))
            los_phase = float(path.get("los_phase", 0.0))

            resolved_path = {
                "index": idx,
                "delay_samples": delay_samples,
                "gain_type": gain_type,

                # static 路径直接使用 gain；
                # rayleigh / rician 路径使用 power 生成帧内随机增益。
                "gain": static_gain,
                "power": power,

                # 预留给后续多普勒连续模型
                "fd": fd,
                "K": K,
                "f_los": f_los,
                "los_phase": los_phase,

                "raw": path,
            }
            for key in (
                "tdl_model", "tdl_fading_type", "tdl_normalized_delay",
                "tdl_power_normalized", "tdl_profile_index",
            ):
                if key in path:
                    resolved_path[key] = path[key]
            resolved_paths.append(resolved_path)

        if not resolved_paths:
            resolved_paths = [
                {
                    "index": 0,
                    "delay_samples": 0.0,
                    "gain_type": "static",
                    "gain": 1.0 + 0.0j,
                    "power": 1.0,
                    "raw": {"delay_samples": 0.0, "gain_type": "static", "gain": 1.0 + 0.0j},
                }
            ]

        return resolved_paths

    def _get_pdp_paths_cached(self):
        """
        获取 PDP 生成的路径。

        对于一个 MultipathChannel 实例，PDP 生成的是路径几何和平均功率，
        应该在仿真过程中保持不变；Rayleigh/Rician 的随机增益仍然在 frame/block 中更新。
        """
        if self._pdp_paths_cache is None:
            self._pdp_paths_cache = self._generate_pdp_paths()

        return copy.deepcopy(self._pdp_paths_cache)

    def _get_tdl_paths_cached(self):
        """Build one static 3GPP TDL realization and reuse it across frames."""
        if self._tdl_paths_cache is None:
            profile = _TDL_PROFILES[self.tdl_model]
            rows = []

            if profile["kind"] == "rician":
                los_db, scatter_db, k_db = profile["los_first"]
                los_power = 10.0 ** (los_db / 10.0)
                scatter_power = 10.0 ** (scatter_db / 10.0)
                rows.append({
                    "normalized_delay": 0.0,
                    "power_raw": los_power + scatter_power,
                    "fading_type": "rician",
                    "K": 10.0 ** (k_db / 10.0),
                    "los_power_db": los_db,
                    "scatter_power_db": scatter_db,
                })

            for normalized_delay, power_db in profile["taps"]:
                rows.append({
                    "normalized_delay": float(normalized_delay),
                    "power_raw": 10.0 ** (float(power_db) / 10.0),
                    "fading_type": "rayleigh",
                    "power_db": float(power_db),
                })

            total_power = sum(row["power_raw"] for row in rows)
            if not np.isfinite(total_power) or total_power <= self._EPS:
                raise ValueError(f"{self.tdl_model} profile has invalid total power")

            paths = []
            for index, row in enumerate(sorted(rows, key=lambda item: item["normalized_delay"])):
                power = row["power_raw"] / total_power
                if self.tdl_max_doppler_hz > 0.0:
                    # Time-varying gains are generated continuously in apply().
                    gain = 0.0 + 0.0j
                elif row["fading_type"] == "rician":
                    gain = self._draw_rician_gain(
                        power=power,
                        K=row["K"],
                        f_los=0.0,
                        los_phase=0.0,
                        fs=1.0,
                    )
                else:
                    gain = self._draw_rayleigh_gain(power)

                paths.append({
                    "tau": row["normalized_delay"] * self.tdl_delay_spread_ns * 1e-9,
                    "gain_type": "static",
                    "gain": gain,
                    "power": power,
                    "K": row.get("K", 0.0),
                    "tdl_model": self.tdl_model,
                    "tdl_fading_type": row["fading_type"],
                    "tdl_normalized_delay": row["normalized_delay"],
                    "tdl_power_normalized": power,
                    "tdl_profile_index": index,
                    "raw": row,
                })

            self._tdl_paths_cache = paths

        return copy.deepcopy(self._tdl_paths_cache)

    def _ensure_tdl_jakes_state(self, num_paths):
        """Create deterministic per-path Jakes phases once per channel realization."""
        if self._tdl_jakes_state is not None:
            return
        sinusoid_index = np.arange(1, self.tdl_num_sinusoids + 1, dtype=float)
        angles = np.pi * (sinusoid_index - 0.5) / self.tdl_num_sinusoids
        doppler_frequencies = self.tdl_max_doppler_hz * np.cos(angles)
        phase_offsets = self.rng.uniform(
            0.0, 2.0 * np.pi, size=(num_paths, self.tdl_num_sinusoids)
        )
        los_phases = self.rng.uniform(0.0, 2.0 * np.pi, size=num_paths)
        self._tdl_jakes_state = {
            "doppler_frequencies_hz": doppler_frequencies,
            "phase_offsets_rad": phase_offsets,
            "los_phases_rad": los_phases,
        }

    def _generate_tdl_gain_traces(self, resolved_paths, num_samples, fs):
        """Generate continuous per-sample TDL gains using a Jakes SOS process."""
        if self.tdl_max_doppler_hz <= 0.0:
            return None
        if fs <= 0.0:
            raise ValueError("sample_rate_Hz 必须为正数")

        self._ensure_tdl_jakes_state(len(resolved_paths))
        state = self._tdl_jakes_state
        sample_indices = self._processed_samples + np.arange(num_samples, dtype=float)
        time_seconds = sample_indices / fs
        phase_matrix = (
            time_seconds[:, None] * 2.0 * np.pi
            * state["doppler_frequencies_hz"][None, :]
        )
        gain_traces = []

        for path_index, path in enumerate(resolved_paths):
            scattering = np.exp(
                1j * (
                    phase_matrix
                    + state["phase_offsets_rad"][path_index][None, :]
                )
            ).sum(axis=1) / np.sqrt(self.tdl_num_sinusoids)
            power = float(path["power"])
            if path.get("tdl_fading_type") == "rician":
                K = float(path.get("K", 0.0))
                los = np.sqrt(K / (K + 1.0)) * np.exp(
                    1j * (
                        state["los_phases_rad"][path_index]
                        + 2.0 * np.pi * 0.7 * self.tdl_max_doppler_hz * time_seconds
                    )
                )
                gain = np.sqrt(power) * (
                    los + np.sqrt(1.0 / (K + 1.0)) * scattering
                )
            else:
                gain = np.sqrt(power) * scattering
            gain_traces.append(gain)

        return gain_traces

    def _activate_tdl_paths(self, resolved_paths, num_samples, fs):
        """Return diagnostics plus continuous gain traces for the current signal."""
        if self.tdl_max_doppler_hz <= 0.0:
            active_paths = self._activate_path_gains_for_frame(resolved_paths, fs)
            self._last_tdl_gain_traces = None
            return active_paths, None

        gain_traces = self._generate_tdl_gain_traces(resolved_paths, num_samples, fs)
        self._last_tdl_gain_traces = [trace.copy() for trace in gain_traces]
        active_paths = []
        for path, gains in zip(resolved_paths, gain_traces):
            active_path = path.copy()
            active_path["gain"] = complex(gains[0]) if len(gains) else 0.0 + 0.0j
            active_path["frame_gain"] = active_path["gain"]
            active_path["tdl_gain_trace_length"] = len(gains)
            active_paths.append(active_path)
        return active_paths, gain_traces


    def _generate_pdp_delays(self, num_paths, use_tau_domain):
        """
        生成 PDP 路径时延。

        use_tau_domain=False:
            返回 delay_samples 列表。

        use_tau_domain=True:
            返回 tau 列表。
        """
        random_delays = bool(self._get_pdp_param("pdp_random_delays", False))
        pdp_seed = self._get_pdp_param("pdp_seed", self.fading_seed)
        rng = np.random.default_rng(pdp_seed)

        if use_tau_domain:
            max_delay = self._get_pdp_param("pdp_max_tau", None)
            if max_delay is None:
                raise ValueError("使用 tau 域 PDP 时，必须配置 pdp_max_tau")
            max_delay = float(max_delay)
        else:
            max_delay = self._get_pdp_param("pdp_max_delay_samples", 32.0)
            if max_delay is None:
                raise ValueError("使用 delay_samples 域 PDP 时，必须配置 pdp_max_delay_samples")
            max_delay = float(max_delay)

        if max_delay < 0:
            raise ValueError("PDP 最大时延不能为负数")

        if num_paths == 1:
            return np.array([0.0], dtype=float)

        if random_delays:
            # 保证包含 0 时延直达/首径，其余路径随机分布在 (0, max_delay]。
            delays = np.concatenate([
                np.array([0.0]),
                rng.uniform(low=0.0, high=max_delay, size=num_paths - 1)
            ])
            delays = np.sort(delays)
        else:
            delays = np.linspace(0.0, max_delay, num_paths)

        return delays.astype(float)


    def _normalize_pdp_powers(self, powers):
        """
        根据 pdp_normalize_power 决定是否归一化 PDP 平均路径功率。
        """
        powers = np.asarray(powers, dtype=float)

        if np.any(powers < 0):
            raise ValueError("PDP 路径功率不能为负数")

        total_power = float(self._get_pdp_param("pdp_total_power", 1.0))
        if total_power < 0:
            raise ValueError("pdp_total_power 不能为负数")

        if bool(self._get_pdp_param("pdp_normalize_power", True)):
            power_sum = np.sum(powers)
            if power_sum > self._EPS:
                powers = powers / power_sum * total_power

        return powers


    def _generate_pdp_powers(self, delays, use_tau_domain):
        """
        根据 PDP 模型生成路径平均功率。
        """
        pdp_model = str(self._get_pdp_param("pdp_model", "manual")).lower()

        if pdp_model == "uniform":
            powers = np.ones(len(delays), dtype=float)

        elif pdp_model == "exponential":
            if use_tau_domain:
                decay = self._get_pdp_param("pdp_rms_tau", None)
                if decay is None:
                    max_tau = max(float(np.max(delays)), self._EPS)
                    decay = max_tau / 3.0
            else:
                decay = self._get_pdp_param("pdp_rms_delay_samples", None)
                if decay is None:
                    max_delay = max(float(np.max(delays)), self._EPS)
                    decay = max_delay / 3.0

            decay = float(decay)
            if decay <= 0:
                raise ValueError("指数 PDP 的衰减尺度必须为正数")

            powers = np.exp(-delays / decay)

        elif pdp_model == "custom":
            custom_powers = self._get_pdp_param("pdp_custom_powers", None)
            if custom_powers is None:
                raise ValueError("pdp_model='custom' 时必须配置 pdp_custom_powers")
            powers = np.asarray(custom_powers, dtype=float)
            if len(powers) != len(delays):
                raise ValueError("pdp_custom_powers 长度必须与自定义时延长度一致")

        else:
            raise ValueError(
                f"不支持的 pdp_model={pdp_model}，当前支持 manual / uniform / exponential / custom"
            )

        return self._normalize_pdp_powers(powers)


    def _generate_pdp_paths(self):
        """
        根据 PDP 参数自动生成 multipath_paths。

        PDP 只生成：
        - delay_samples 或 tau
        - power
        - gain_type / K / f_los 等路径配置

        实际 Rayleigh/Rician 复增益仍然由 _activate_path_gains_for_frame() 生成。
        """
        pdp_model = str(self._get_pdp_param("pdp_model", "manual")).lower()

        if pdp_model == "manual":
            paths = self._get_param("multipath_paths", None)
            if paths:
                return copy.deepcopy(paths)
            return [{"delay_samples": 0.0, "gain_type": "static", "gain": 1.0 + 0.0j}]

        if pdp_model == "custom":
            custom_delay_samples = self._get_pdp_param("pdp_custom_delay_samples", None)
            custom_taus = self._get_pdp_param("pdp_custom_taus", None)

            if custom_delay_samples is not None:
                delays = np.asarray(custom_delay_samples, dtype=float)
                delay_key = "delay_samples"
                use_tau_domain = False
            elif custom_taus is not None:
                delays = np.asarray(custom_taus, dtype=float)
                delay_key = "tau"
                use_tau_domain = True
            else:
                raise ValueError(
                    "pdp_model='custom' 时必须配置 pdp_custom_delay_samples 或 pdp_custom_taus"
                )

            if len(delays) == 0:
                raise ValueError("custom PDP 至少需要一条路径")

        else:
            num_paths = int(self._get_pdp_param("pdp_num_paths", 6))
            if num_paths <= 0:
                raise ValueError("pdp_num_paths 必须为正整数")

            # 优先使用 delay_samples 域。如果 pdp_max_delay_samples=None 且 pdp_max_tau 不为 None，则用 tau 域。
            if self._get_pdp_param("pdp_max_delay_samples", 32.0) is not None:
                use_tau_domain = False
                delay_key = "delay_samples"
            else:
                use_tau_domain = True
                delay_key = "tau"

            delays = self._generate_pdp_delays(num_paths, use_tau_domain=use_tau_domain)

        powers = self._generate_pdp_powers(delays, use_tau_domain=use_tau_domain)

        pdp_gain_type = str(self._get_pdp_param("pdp_gain_type", "rayleigh")).lower()
        pdp_los_gain_type = str(self._get_pdp_param("pdp_los_gain_type", "rician")).lower()
        include_los = bool(self._get_pdp_param("pdp_include_los", True))
        K = float(self._get_pdp_param("pdp_rician_K", 10.0))
        f_los = float(self._get_pdp_param("pdp_f_los", 0.0))
        los_phase = float(self._get_pdp_param("pdp_los_phase", 0.0))

        paths = []

        for idx, (delay, power) in enumerate(zip(delays, powers)):
            if include_los and idx == 0:
                gain_type = pdp_los_gain_type
            else:
                gain_type = pdp_gain_type

            path = {
                delay_key: float(delay),
                "gain_type": gain_type,
                "power": float(power),
                "source": "pdp",
                "pdp_model": pdp_model,
            }

            if gain_type == "static":
                path["gain"] = np.sqrt(float(power)) + 0.0j

            if gain_type == "rician":
                path["K"] = K
                path["f_los"] = f_los
                path["los_phase"] = los_phase

            paths.append(path)

        return paths

    def _draw_rayleigh_gain(self, power):
        """
        生成一条 Rayleigh 路径的帧内固定复衰落系数。

        h ~ CN(0, power)
        即 E[|h|^2] = power
        """
        power = float(power)
        if power <= self._EPS:
            return 0.0 + 0.0j

        return np.sqrt(power / 2.0) * (
            self.rng.standard_normal() + 1j * self.rng.standard_normal()
        )


    def _draw_rician_gain(self, power, K, f_los=0.0, los_phase=0.0, fs=1.0):
        """
        生成一条 Rician 路径的帧内固定复衰落系数。

        模型：
            h = LOS + Scatter

        其中：
            LOS 功率占比 K / (K + 1)
            散射功率占比 1 / (K + 1)

        power 为该路径平均功率，即 E[|h|^2] = power。
        """
        power = float(power)
        K = float(K)

        if power <= self._EPS:
            return 0.0 + 0.0j

        if K < 0:
            raise ValueError("Rician K 因子不能为负数")

        # 当前是 frame-level fading，用当前帧起点时间近似 LOS 相位。
        t0 = self._processed_samples / fs if fs > 0 else 0.0
        phase = los_phase + 2.0 * np.pi * f_los * t0

        los_amp = np.sqrt(power * K / (K + 1.0)) if K > 0 else 0.0
        scatter_amp = np.sqrt(power / (K + 1.0))

        los = los_amp * np.exp(1j * phase)
        scatter = np.sqrt(0.5) * scatter_amp * (
            self.rng.standard_normal() + 1j * self.rng.standard_normal()
        )

        return los + scatter

    def _normalize_active_path_gains(self, active_paths):
        """
        对当前时间段的 active gains 做可选归一化。

        设计原则：
        - 不归一化到 1；
        - 而是归一化到当前路径平均功率之和；
        - 这样不会抹掉 PDP 总功率和大尺度路径损耗。

        仅当 normalize_channel_power=True 时生效。
        """
        if not self.normalize_channel_power:
            return active_paths

        current_power = 0.0
        target_power = 0.0

        for path in active_paths:
            current_power += abs(path["gain"]) ** 2

            if "power" in path and path["power"] is not None:
                target_power += float(path["power"])
            else:
                target_power += abs(path["gain"]) ** 2

        if current_power <= self._EPS:
            return active_paths

        if target_power < 0:
            raise ValueError("归一化目标功率不能为负数")

        scale = np.sqrt(target_power / current_power) if target_power > self._EPS else 0.0

        for path in active_paths:
            path["gain_before_normalization"] = path["gain"]
            path["gain"] = path["gain"] * scale
            path["frame_gain"] = path["gain"]
            path["normalization_scale"] = scale
            path["normalization_target_power"] = target_power
            path["normalization_power_before"] = current_power
            path["normalization_power_after"] = target_power

        return active_paths

    def _activate_path_gains_for_frame(self, resolved_paths, fs):
        """
        根据 gain_type 为当前帧生成实际使用的路径增益。

        static:
            使用固定复增益 gain。

        rayleigh:
            每次 apply() 为该路径生成一个帧内固定 Rayleigh 增益。

        rician:
            每次 apply() 为该路径生成一个帧内固定 Rician 增益。

        这个函数名字虽然叫 _activate_path_gains_for_frame()，但 block 模式下我们也可以复用它，含义就是“为当前时间段生成 active gains”

            理论上，Rayleigh 路径满足 E[h] = 0, E[|h|^2] = power。
            Rician 路径满足 E[|h|^2] = power, E[h] 的幅度为 sqrt(power * K / (K + 1))，相位由 f_los 和 los_phase 决定。

        """
        active_paths = []

        if self.fading_model not in ("static", "frame", "block"):
            raise NotImplementedError(
                f"当前版本仅支持 fading_model='static'、'frame' 或 'block'，"
                f"当前配置为 {self.fading_model}。sample/Jakes 后续再实现。"
            )

        for path in resolved_paths:
            active_path = path.copy()
            gain_type = path["gain_type"]

            if self.fading_model == "static" or gain_type == "static":
                active_gain = path["gain"]

            elif gain_type == "rayleigh":
                active_gain = self._draw_rayleigh_gain(path["power"])

            elif gain_type == "rician":
                active_gain = self._draw_rician_gain(
                    power=path["power"],
                    K=path["K"],
                    f_los=path["f_los"],
                    los_phase=path["los_phase"],
                    fs=fs,
                )

            else:
                raise ValueError(f"未知 gain_type: {gain_type}")

            active_path["gain"] = active_gain
            active_path["frame_gain"] = active_gain
            active_paths.append(active_path)
        active_paths = self._normalize_active_path_gains(active_paths)

        return active_paths

    def _window(self, length):
        """生成分数延迟滤波器窗函数。"""
        window_type = str(self.frac_filter_window).lower()
        if window_type in ("hann", "hanning"):
            return np.hanning(length)
        if window_type == "hamming":
            return np.hamming(length)
        if window_type in ("rect", "rectangular", "boxcar", "none"):
            return np.ones(length)
        raise ValueError(f"暂不支持的分数延迟窗函数类型: {self.frac_filter_window}")

    def design_frac_delay_filter(self, d, half_len=None, window=None):
        """
        设计分数延迟 FIR 滤波器。

        参数：
            d: 分数延迟，范围 [0, 1)。
            half_len: FIR 半长，实际滤波器长度为 2 * half_len + 1。
            window: 窗函数类型。

        返回：
            h: 分数延迟滤波器系数。
        """
        half_len = self.frac_filter_half_len if half_len is None else int(half_len)
        old_window = self.frac_filter_window
        if window is not None:
            self.frac_filter_window = window

        try:
            d = float(d)
            if d < -self._EPS or d >= 1.0 + self._EPS:
                raise ValueError("分数延迟 d 必须满足 0 <= d < 1")
            d = min(max(d, 0.0), 1.0 - self._EPS)

            n = np.arange(-half_len, half_len + 1, dtype=float)
            h = np.sinc(n - d)
            h *= self._window(len(h))

            h_sum = np.sum(h)
            if abs(h_sum) > self._EPS:
                h = h / h_sum

            return h.astype(float)
        finally:
            self.frac_filter_window = old_window

    def apply_fractional_delay(self, x, d):
        """
        对输入信号施加 d 个采样点的分数延迟，并保持输出长度不变。
        """
        h = self.design_frac_delay_filter(d)
        half_len = self.frac_filter_half_len
        z = np.convolve(x, h, mode="full")
        return z[half_len: half_len + len(x)]

    @staticmethod
    def _apply_integer_delay(x, integer_delay):
        """施加非负整数采样延迟，输出长度与输入一致。"""
        n = len(x)
        integer_delay = int(integer_delay)

        if integer_delay <= 0:
            return x.copy()

        y = np.zeros(n, dtype=complex)
        if integer_delay < n:
            y[integer_delay:] = x[: n - integer_delay]
        return y

    def _apply_delay(self, x, delay_samples):
        """
        根据 delay_samples 对信号施加因果延迟。

        若 use_frac_delay=True，则 delay_samples = D + d，其中 D=floor(delay)，d 为分数部分；
        否则直接 round(delay_samples) 为整数采样延迟。
        """
        delay_samples = float(delay_samples)

        if self.use_frac_delay:
            integer_delay = int(np.floor(delay_samples))
            frac_delay = delay_samples - integer_delay

            if frac_delay > self._EPS:
                delayed = self.apply_fractional_delay(x, frac_delay)
            else:
                delayed = x.copy()

            return self._apply_integer_delay(delayed, integer_delay)

        integer_delay = int(np.round(delay_samples))
        return self._apply_integer_delay(x, integer_delay)

    def _apply_static_fir_no_history(self, x, h):
        """
        不启用帧间历史时的静态 FIR 多径处理。

        输出长度保持与输入 x 相同。
        """
        if len(h) == 0:
            return np.zeros_like(x, dtype=complex)

        y_full = np.convolve(x, h, mode="full")
        return y_full[: len(x)]

    def _apply_tdl_path_sum(self, x, active_paths, gain_traces=None):
        """Apply integer-delay TDL paths without forming a dense long FIR convolution."""
        x = np.asarray(x, dtype=complex)
        if not active_paths:
            return np.zeros_like(x, dtype=complex)

        max_delay = max(int(np.round(path["delay_samples"])) for path in active_paths)
        if self.enable_multipath_memory:
            self._ensure_history_length(max_delay)
            x_ext = np.concatenate([self._input_history, x])
            y_ext = np.zeros_like(x_ext, dtype=complex)
            for path_index, path in enumerate(active_paths):
                delayed = self._apply_integer_delay(
                    x_ext, int(np.round(path["delay_samples"]))
                )
                if gain_traces is None:
                    y_ext += path["gain"] * delayed
                else:
                    trace = np.asarray(gain_traces[path_index], dtype=complex)
                    y_ext[max_delay:max_delay + len(x)] += trace * delayed[
                        max_delay:max_delay + len(x)
                    ]
            y = y_ext[max_delay:max_delay + len(x)]
            self._input_history = x_ext[-max_delay:].copy() if max_delay else np.zeros(0, dtype=complex)
            self._history_length = len(self._input_history)
        else:
            y = np.zeros_like(x, dtype=complex)
            for path_index, path in enumerate(active_paths):
                delayed = self._apply_integer_delay(
                    x, int(np.round(path["delay_samples"]))
                )
                if gain_traces is None:
                    y += path["gain"] * delayed
                else:
                    y += np.asarray(gain_traces[path_index], dtype=complex) * delayed

        self._processed_samples += len(x)
        return y

    def _apply_fir_with_history_state(self, x, h, input_history):
        """
        使用给定输入历史执行 FIR 滤波。

        这个函数不直接修改 self._input_history，适合 block fading 中局部维护历史。
        """
        x = np.asarray(x, dtype=complex)
        h = np.asarray(h, dtype=complex)

        if len(h) == 0:
            return np.zeros_like(x, dtype=complex), np.zeros(0, dtype=complex)

        history_length = max(len(h) - 1, 0)

        if history_length == 0:
            return h[0] * x, np.zeros(0, dtype=complex)

        if input_history is None or len(input_history) != history_length:
            input_history = np.zeros(history_length, dtype=complex)

        x_ext = np.concatenate([input_history, x])
        y_ext = np.convolve(x_ext, h, mode="full")

        y = y_ext[history_length: history_length + len(x)]
        new_history = x_ext[-history_length:].copy()

        return y, new_history

    def _ensure_history_length(self, history_length):
        """
        确保输入历史缓冲区长度正确。

        如果 FIR 长度变化，说明信道记忆长度变化，此时旧历史不再可靠，
        直接清零历史缓冲区。
        """
        history_length = int(max(0, history_length))

        if self._history_length != history_length or len(self._input_history) != history_length:
            self._history_length = history_length
            self._input_history = np.zeros(history_length, dtype=complex)


    def _apply_static_fir_with_history(self, x, h):
        """
        启用帧间输入历史时的静态 FIR 多径处理。

        对于长度为 L 的 FIR 信道，需要保存上一帧末尾 L-1 个输入样本。
        当前帧输出的第 n 个采样点由当前帧输入和历史输入共同决定。

        输入：
            x: 当前帧输入信号，长度为 N
            h: 等效多径 FIR 冲激响应，长度为 L

        输出：
            y: 当前帧对应输出，长度仍为 N
        """
        h = np.asarray(h, dtype=complex)

        if len(h) == 0:
            self._processed_samples += len(x)
            return np.zeros_like(x, dtype=complex)

        history_length = max(len(h) - 1, 0)
        self._ensure_history_length(history_length)

        y, new_history = self._apply_fir_with_history_state(x, h, self._input_history)

        self._input_history = new_history
        self._history_length = len(new_history)
        self._processed_samples += len(x)

        return y

    def _get_effective_block_length(self, num_samples):
        """
        获取 block fading 的实际 block 长度。

        如果 block_length 为 None，则退化为整帧一个 block。
        """
        if self.block_length is None:
            return max(1, int(num_samples))

        block_length = int(self.block_length)

        if block_length <= 0:
            raise ValueError("block_length 必须为正整数")

        return block_length

    def _apply_block_fading_fir(self, x, resolved_paths, fs):
        """
        block fading FIR 处理。

        每 block_length 个采样点更新一次 Rayleigh/Rician 路径增益。
        block 内信道保持不变，block 间信道更新。

        注意：
        即使 enable_multipath_memory=False，当前帧内部的 block 之间也必须保留局部输入历史，
        否则 block 边界会丢失多径拖尾。

        如果 enable_multipath_memory=True，则最后一个 block 的输入历史会继续保留到下一帧。
        如果 enable_multipath_memory=False，则历史只在当前帧内部保留，帧结束后清空。
        """
        x = np.asarray(x, dtype=complex)
        num_samples = len(x)
        block_length = self._get_effective_block_length(num_samples)

        y = np.zeros_like(x, dtype=complex)
        block_records = []

        if self.enable_multipath_memory:
            current_history = self._input_history.copy()
        else:
            current_history = np.zeros(0, dtype=complex)

        last_active_paths = None
        last_h = None

        for block_index, start in enumerate(range(0, num_samples, block_length)):
            end = min(start + block_length, num_samples)
            x_block = x[start:end]

            global_start = self._processed_samples

            # 为当前 block 生成 active gains
            active_paths = self._activate_path_gains_for_frame(resolved_paths, fs)
            h_block = self._build_effective_impulse_response(active_paths)

            # 使用局部历史执行 FIR，保证 block 边界多径拖尾正确
            y_block, current_history = self._apply_fir_with_history_state(
                x_block, h_block, current_history
            )

            y[start:end] = y_block

            self._processed_samples += len(x_block)
            global_end = self._processed_samples

            block_records.append(
                {
                    "block_index": block_index,
                    "start": start,
                    "end": end,
                    "global_start": global_start,
                    "global_end": global_end,
                    "active_paths": active_paths,
                    "chan_impulse": h_block,
                }
            )

            last_active_paths = active_paths
            last_h = h_block

        if self.enable_multipath_memory:
            self._input_history = current_history.copy()
            self._history_length = len(self._input_history)
        else:
            # 不跨帧保留历史，但当前帧内部 block 之间已经保留过局部历史
            self._input_history = np.zeros(0, dtype=complex)
            self._history_length = 0

        if last_active_paths is None:
            last_active_paths = []
        if last_h is None:
            last_h = np.array([0.0 + 0.0j], dtype=complex)

        return y, last_active_paths, last_h, block_records

    def _build_effective_impulse_response(self, resolved_paths):
        """
        构造与 apply() 实际处理方式一致的等效离散冲激响应。

        注意：
        这里不要直接把完整的分数延迟滤波器 h_frac 从 integer_delay 位置写入，
        因为 apply_fractional_delay() 内部已经通过裁剪补偿了 half_len 的群延迟。

        为了保证调试用 chan_impulse 与真实信号处理完全一致，
        这里直接构造一个单位冲激，然后调用 _apply_delay() 生成每条路径的响应。
        """

        # 先估计需要的冲激响应长度
        max_len = 1

        for path in resolved_paths:
            delay_samples = float(path["delay_samples"])

            if self.use_frac_delay:
                integer_delay = int(np.floor(delay_samples))
                frac_delay = delay_samples - integer_delay

                if frac_delay > self._EPS:
                    # 分数延迟在当前实现中经过群延迟补偿后，主要响应长度约为 half_len + 1
                    path_len = integer_delay + self.frac_filter_half_len + 1
                else:
                    path_len = integer_delay + 1
            else:
                path_len = int(np.round(delay_samples)) + 1

            max_len = max(max_len, path_len)

        impulse = np.zeros(max_len, dtype=complex)
        impulse[0] = 1.0 + 0.0j

        h_eff = np.zeros(max_len, dtype=complex)

        for path in resolved_paths:
            delayed_impulse = self._apply_delay(impulse, path["delay_samples"])
            h_eff += path["gain"] * delayed_impulse

        # 归一化
        if self.normalize_channel_power:
            power = np.sum(np.abs(h_eff) ** 2)
            if power > self._EPS:
                h_eff = h_eff / np.sqrt(power)

        return h_eff

    def apply(self, signal_dict):
        """
        将输入 signal_dict 通过静态多径信道。

        输入 signal_dict 必须包含：
            signal_stream, sample_rate_Hz, duration_seconds, signal_length, padding_bit_num

        返回：
            新的 signal_dict，signal_stream 已替换为多径后的信号，长度保持不变。
        """
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in signal_dict:
                raise KeyError(f"输入 signal_dict 缺少必要键值：{key}")

        x = np.asarray(signal_dict["signal_stream"], dtype=complex)
        fs = float(signal_dict["sample_rate_Hz"])
        expected_length = int(signal_dict["signal_length"])

        if len(x) != expected_length:
            raise ValueError(f"signal_stream 长度与 signal_length 不一致: {len(x)} != {expected_length}")

        if round(fs * float(signal_dict["duration_seconds"])) != expected_length:
            raise ValueError("输入 signal_dict 中的采样率、时长与信号长度不匹配")

        resolved_paths_raw = self._parse_paths(fs)

        # 大尺度衰落在路径解析之后、小尺度衰落激活之前作用。
        # PDP 生成的是相对平均功率；这里根据距离/载频/阴影衰落对平均功率整体缩放。
        resolved_paths = self._apply_large_scale_fading_to_paths(resolved_paths_raw)
        self.resolved_paths = resolved_paths

        if self.fading_model == "block":
            y, active_paths, h_current, block_records = self._apply_block_fading_fir(
                x, resolved_paths, fs
            )

            self.active_paths = active_paths
            self.active_blocks = block_records
            self.chan_impulse = h_current

        else:
            if self.tdl_model is not None:
                active_paths, gain_traces = self._activate_tdl_paths(
                    resolved_paths, len(x), fs
                )
            else:
                active_paths = self._activate_path_gains_for_frame(resolved_paths, fs)
                gain_traces = None

            self.active_paths = active_paths
            self.active_blocks = [
                {
                    "block_index": 0,
                    "start": 0,
                    "end": len(x),
                    "global_start": self._processed_samples,
                    "global_end": self._processed_samples + len(x),
                    "active_paths": active_paths,
                }
            ]

            self.chan_impulse = self._build_effective_impulse_response(active_paths)

            if self.tdl_model is not None:
                y = self._apply_tdl_path_sum(x, active_paths, gain_traces)
            elif self.enable_multipath_memory:
                y = self._apply_static_fir_with_history(x, self.chan_impulse)
            else:
                y = self._apply_static_fir_no_history(x, self.chan_impulse)
                self._processed_samples += len(x)

        input_power = float(np.mean(np.abs(x) ** 2)) if len(x) > 0 else 0.0
        output_power = float(np.mean(np.abs(y) ** 2)) if len(y) > 0 else 0.0
        self.channel_power_gain = output_power / input_power if input_power > self._EPS else 0.0

        result_dict = signal_dict.copy()
        result_dict["signal_stream"] = y
        result_dict["signal_length"] = len(y)
        result_dict["duration_seconds"] = len(y) / fs
        result_dict["multipath_enabled"] = True
        result_dict["multipath_input_power"] = input_power
        result_dict["multipath_output_power"] = output_power
        result_dict["channel_power_gain"] = self.channel_power_gain
        result_dict["chan_impulse"] = self.chan_impulse
        result_dict["multipath_paths_resolved"] = resolved_paths
        result_dict["multipath_paths_active"] = active_paths
        result_dict["multipath_blocks_active"] = self.active_blocks
        result_dict["multipath_num_blocks"] = len(self.active_blocks)
        result_dict["multipath_block_length"] = (
            self._get_effective_block_length(len(x)) if self.fading_model == "block" else len(x)
        )
        result_dict["pdp_enabled"] = (
            bool(self._get_param("use_pdp", False))
            or str(self._get_param("pdp_model", "manual")).lower() != "manual"
        )
        result_dict["pdp_model"] = self._get_param("pdp_model", "manual")
        result_dict["pdp_paths_generated"] = copy.deepcopy(self._pdp_paths_cache)
        result_dict["large_scale_info"] = self._get_large_scale_info_cached()
        result_dict["large_scale_enabled"] = result_dict["large_scale_info"]["enabled"]
        result_dict["large_scale_model"] = result_dict["large_scale_info"]["model"]
        result_dict["large_scale_path_loss_db"] = result_dict["large_scale_info"]["path_loss_db"]
        result_dict["large_scale_power_gain"] = result_dict["large_scale_info"]["power_gain_linear"]
        result_dict["normalize_channel_power"] = self.normalize_channel_power
        result_dict["multipath_memory_enabled"] = self.enable_multipath_memory
        result_dict["multipath_fir_length"] = len(self.chan_impulse)
        result_dict["multipath_history_length"] = max(len(self.chan_impulse) - 1, 0)
        result_dict["multipath_processed_samples"] = self._processed_samples
        result_dict["tdl_enabled"] = self.tdl_model is not None
        result_dict["tdl_model"] = self.tdl_model
        result_dict["tdl_delay_spread_ns"] = self.tdl_delay_spread_ns
        result_dict["tdl_velocity_mps"] = self.tdl_velocity_mps
        result_dict["tdl_carrier_frequency_hz"] = self.tdl_fc_hz
        result_dict["tdl_max_doppler_hz"] = self.tdl_max_doppler_hz
        result_dict["tdl_los_doppler_hz"] = 0.7 * self.tdl_max_doppler_hz
        result_dict["tdl_fading_time_varying"] = self.tdl_max_doppler_hz > 0.0

        return result_dict

    def apply_multipath(self, signal):
        """
        兼容旧版裸数组接口。推荐新代码使用 apply(signal_dict)。

        注意：旧接口无法使用 tau，因为缺少 sample_rate_Hz，只能使用 delay_samples 或旧 chan_delays。
        """
        x = np.asarray(signal, dtype=complex)
        dummy_fs = float(self._get_param("sample_rate", 1.0))

        dummy_dict = {
            "signal_stream": x,
            "sample_rate_Hz": dummy_fs,
            "duration_seconds": len(x) / dummy_fs,
            "signal_length": len(x),
            "padding_bit_num": 0,
        }

        return self.apply(dummy_dict)["signal_stream"]

# import numpy as np
# from transmitter.THzTransmitter import THzTransmitter
# from params.PHYParams import PHYParams
# class MultipathChannel:
#     """
#     多径信道模型：生成信道冲激响应（CIR），实现信号的多径卷积
#     支持自定义多径时延、增益，或生成随机多径（如瑞利衰落）
#     """
#     def __init__(self, params):
#         self.params = params
#         self.chan_impulse = None  # 信道冲激响应（CIR）
#         self._init_channel()

#     def _init_channel(self):
#         """初始化信道冲激响应：基于参数配置的多径时延和增益"""
#         chan_delays = self.params.get("chan_delays")  # 多径时延（采样点单位）
#         chan_gains = self.params.get("chan_gains")    # 多径增益（复数值）
        
#         # 验证时延和增益长度一致
#         assert len(chan_delays) == len(chan_gains), "多径时延和增益长度必须一致"
        
#         # 生成信道冲激响应（索引对应时延，值对应增益）
#         max_delay = max(chan_delays) if len(chan_delays) > 0 else 0
#         self.chan_impulse = np.zeros(max_delay + 1, dtype=complex)
#         for delay, gain in zip(chan_delays, chan_gains):
#             self.chan_impulse[delay] = gain

#     def generate_rayleigh_fading(self, num_paths=4, max_delay=12):
#         """生成瑞利衰落多径信道（可选：随机多径）"""
#         # 随机生成多径时延（0~max_delay）
#         chan_delays = np.sort(np.random.choice(max_delay + 1, num_paths, replace=False))
#         # 瑞利衰落增益（幅度服从瑞利分布，相位均匀分布）
#         chan_gains = (np.random.randn(num_paths) + 1j * np.random.randn(num_paths)) / np.sqrt(2)
#         chan_gains /= np.sqrt(np.sum(np.abs(chan_gains)**2))
#         # 更新参数和冲激响应
#         self.params.update(chan_delays=chan_delays, chan_gains=chan_gains)
#         self._init_channel()

#     def apply_multipath(self, signal):
#         """应用多径效应：信号与信道冲激响应的线性卷积"""
#         # 卷积（保持输出长度与输入一致，mode="same"）
#         signal_with_multipath = np.convolve(signal, self.chan_impulse, mode="full")
#         return signal_with_multipath

# # 测试
# if __name__ == "__main__":
#     # 初始化参数和发射机
#     params = PHYParams()
#     transmitter = THzTransmitter(params)
#     tx_signal = transmitter.run()
#     multipath_chan = MultipathChannel(params)
#     signal_with_multipath = multipath_chan.apply_multipath(tx_signal)
#     print(f"信道冲激响应：{multipath_chan.chan_impulse}")
#     print(f"原始信号长度：{len(tx_signal)}, 多径后长度：{len(signal_with_multipath)}")
