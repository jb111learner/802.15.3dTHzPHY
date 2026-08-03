"""THz-MIMO 实测信道的确定性回放与 PDP-Rayleigh 随机化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MeasuredChannel:
    """加载固定的 2×2 实测 CIR，并生成可供 SISO/MIMO 使用的抽头。

    数据只有单个测量快照，因此 Rayleigh 是显式的工程建模假设；实测数据只
    用于确定有效时延窗、平均 PDP 和 MIMO 子链路的相对平均功率。
    """

    MODE_DETERMINISTIC = "deterministic"
    MODE_PDP_RAYLEIGH = "pdp_rayleigh"
    SOURCE_SAMPLE_RATE_HZ = 30e9
    DATA_KEY = "tap_cube"

    SCENARIOS = {
        "8cm": {
            "filename": "THz_MIMO_08cm_2.npz",
            "max_delay_ns": 1.6,
            "gi_length": 64,
        },
        "12cm": {
            "filename": "THz_MIMO_12cm_2.npz",
            "max_delay_ns": 1.2,
            "gi_length": 64,
        },
        "50cm": {
            "filename": "THz_MIMO_50cm_2.npz",
            "max_delay_ns": 0.5,
            "gi_length": 32,
        },
    }

    def __init__(self, params):
        self.params = params
        self.scenario = str(self._get("measured_channel_scenario", "8cm")).lower()
        if self.scenario not in self.SCENARIOS:
            raise ValueError(
                "measured_channel_scenario 仅支持 "
                f"{', '.join(self.SCENARIOS)}，实际为 {self.scenario!r}"
            )

        self.mode = str(
            self._get("measured_channel_mode", self.MODE_DETERMINISTIC)
        ).lower()
        if self.mode not in {self.MODE_DETERMINISTIC, self.MODE_PDP_RAYLEIGH}:
            raise ValueError(
                "measured_channel_mode 仅支持 deterministic / pdp_rayleigh"
            )

        self.retained_power = float(
            self._get("measured_channel_retained_power", 0.95)
        )
        if not 0.0 < self.retained_power <= 1.0:
            raise ValueError("measured_channel_retained_power 必须位于 (0, 1]")

        self.tx_index = int(self._get("measured_channel_tx_index", 0))
        self.rx_index = int(self._get("measured_channel_rx_index", 0))
        if self.tx_index not in {0, 1} or self.rx_index not in {0, 1}:
            raise ValueError("实测 SISO 天线索引必须为 0 或 1")

        seed = self._get("mimo_channel_seed", None)
        if seed is None:
            seed = self._get("random_seed", None)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            # 与 MIMOChannel 的噪声 RNG 使用不同子流，避免信道与噪声相关。
            self.rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 0x4D454153])
            )

        self.data_dir = Path(__file__).resolve().parent / "THz_MIMO_measurement"
        self.raw_taps: np.ndarray | None = None
        self.cropped_taps: np.ndarray | None = None
        self.noise_floor_per_link: np.ndarray | None = None
        self.denoised_power: np.ndarray | None = None
        self.aggregate_pdp: np.ndarray | None = None
        self.selected_pdp: np.ndarray | None = None
        self.selected_path_indexes: np.ndarray | None = None
        self.relative_link_power: np.ndarray | None = None
        self.channel_impulse_response: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

        self._prepare_measurement()

    def _get(self, key, default=None):
        if hasattr(self.params, "get"):
            return self.params.get(key, default)
        return self.params.get(key, default)

    @property
    def scenario_info(self) -> dict[str, Any]:
        return self.SCENARIOS[self.scenario]

    @property
    def source_path(self) -> Path:
        return self.data_dir / str(self.scenario_info["filename"])

    @property
    def num_effective_taps(self) -> int:
        delay_s = float(self.scenario_info["max_delay_ns"]) * 1e-9
        return int(round(delay_s * self.SOURCE_SAMPLE_RATE_HZ)) + 1

    def _load_raw_taps(self) -> np.ndarray:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"找不到实测信道文件：{self.source_path}")
        with np.load(self.source_path, allow_pickle=False) as data:
            if self.DATA_KEY not in data.files:
                raise KeyError(
                    f"实测信道文件缺少键 {self.DATA_KEY!r}：{self.source_path}"
                )
            taps = np.asarray(data[self.DATA_KEY], dtype=np.complex128)
        if taps.shape != (2, 2, 512):
            raise ValueError(
                "实测 tap_cube 形状必须为 (2, 2, 512)，"
                f"实际为 {taps.shape}"
            )
        if not np.all(np.isfinite(taps)):
            raise ValueError("实测 tap_cube 包含 NaN 或 Inf")
        return taps

    @staticmethod
    def _rms_delay_spread_ns(pdp: np.ndarray, sample_rate_hz: float) -> float:
        pdp = np.asarray(pdp, dtype=float)
        total = float(np.sum(pdp))
        if total <= 0.0:
            return 0.0
        delay_s = np.arange(len(pdp), dtype=float) / float(sample_rate_hz)
        mean = float(np.sum(pdp * delay_s) / total)
        variance = float(np.sum(pdp * (delay_s - mean) ** 2) / total)
        return np.sqrt(max(variance, 0.0)) * 1e9

    def _prepare_measurement(self) -> None:
        raw = self._load_raw_taps()
        aggregate_raw = np.sum(np.abs(raw) ** 2, axis=(0, 1))
        main_tap = int(np.argmax(aggregate_raw))

        # 所有 MIMO 子链路共同平移，保留链路之间的相对时延。
        aligned = raw[..., main_tap:]
        num_taps = self.num_effective_taps
        if aligned.shape[-1] < num_taps:
            raise ValueError("实测信道主径对齐后长度不足以覆盖物理时延窗")
        cropped = aligned[..., :num_taps].copy()

        # 用有效数据后半段估计每条链路的时延域噪声底。文件末端的精确零值
        # 不参与估计，避免把零填充误判为极低噪声。
        nonzero = np.flatnonzero(aggregate_raw > 0.0)
        tail_end = int(nonzero[-1] + 1) if len(nonzero) else raw.shape[-1]
        tail_start = max(main_tap + num_taps, tail_end // 2)
        if tail_start >= tail_end:
            tail_start = max(0, tail_end - max(16, tail_end // 4))
        tail_power = np.abs(raw[..., tail_start:tail_end]) ** 2
        noise_floor = np.median(tail_power, axis=-1)

        cropped_power = np.abs(cropped) ** 2
        denoised = np.maximum(cropped_power - noise_floor[..., None], 0.0)
        if float(np.sum(denoised)) <= 0.0:
            raise ValueError("扣除噪声底后没有可用的实测信道功率")

        aggregate = np.sum(denoised, axis=(0, 1))
        aggregate /= np.sum(aggregate)
        selected_indexes, selected_pdp = self._select_dominant_paths(aggregate)

        link_power = np.sum(denoised, axis=-1)
        if float(np.sum(link_power)) <= 0.0:
            link_power = np.sum(cropped_power, axis=-1)
        relative_link_power = link_power * (2.0 / np.sum(link_power))

        raw_window_ratio = float(
            np.sum(cropped_power) / max(np.sum(np.abs(aligned) ** 2), 1e-300)
        )
        # 能量占比用期望噪声功率直接扣除，避免逐点截零后正半边噪声在数百个
        # 尾部抽头上累积，从而低估物理时延窗所包含的主要功率。
        noise_per_delay = float(np.sum(noise_floor))
        inside_excess = max(
            float(np.sum(cropped_power)) - num_taps * noise_per_delay, 0.0
        )
        aligned_tail_end = max(1, tail_end - main_tap)
        full_excess = max(
            float(np.sum(np.abs(aligned[..., :aligned_tail_end]) ** 2))
            - aligned_tail_end * noise_per_delay,
            1e-300,
        )
        denoised_window_ratio = float(
            min(inside_excess / full_excess, 1.0)
        )
        diagonal = float(link_power[0, 0] + link_power[1, 1])
        cross = float(link_power[0, 1] + link_power[1, 0])
        outside_power = np.sum(
            np.abs(aligned[..., num_taps:aligned_tail_end]) ** 2, axis=(0, 1)
        )
        if len(outside_power):
            outside_offset = int(np.argmax(outside_power))
            outside_index = num_taps + outside_offset
            outside_max = float(outside_power[outside_offset])
        else:
            outside_index = -1
            outside_max = 0.0
        aggregate_noise_floor = max(noise_per_delay, 1e-300)
        aggregate_peak = max(float(np.max(aggregate_raw)), 1e-300)
        significant_outside = bool(outside_max >= 10.0 * aggregate_noise_floor)

        self.raw_taps = raw
        self.cropped_taps = cropped
        self.noise_floor_per_link = noise_floor
        self.denoised_power = denoised
        self.aggregate_pdp = aggregate
        self.selected_path_indexes = selected_indexes
        self.selected_pdp = selected_pdp
        self.relative_link_power = relative_link_power
        self.diagnostics = {
            "scenario": self.scenario,
            "mode": self.mode,
            "source_file": self.source_path.name,
            "original_shape": list(raw.shape),
            "processed_shape": list(cropped.shape),
            "sample_rate_Hz": self.SOURCE_SAMPLE_RATE_HZ,
            "tap_spacing_s": 1.0 / self.SOURCE_SAMPLE_RATE_HZ,
            "max_delay_ns": float(self.scenario_info["max_delay_ns"]),
            "recommended_gi_length": int(self.scenario_info["gi_length"]),
            "main_tap_index": main_tap,
            "noise_tail_start_index": tail_start,
            "noise_tail_end_index": tail_end,
            "noise_floor_power_per_link": noise_floor.tolist(),
            "raw_in_window_power_ratio": raw_window_ratio,
            "denoised_in_window_power_ratio": denoised_window_ratio,
            "outside_window_peak_index": outside_index,
            "outside_window_peak_delay_ns": (
                outside_index / self.SOURCE_SAMPLE_RATE_HZ * 1e9
                if outside_index >= 0 else None
            ),
            "outside_window_peak_relative_db": float(
                10.0 * np.log10(max(outside_max, 1e-300) / aggregate_peak)
            ),
            "outside_window_peak_above_noise_db": float(
                10.0 * np.log10(max(outside_max, 1e-300) / aggregate_noise_floor)
            ),
            "significant_power_outside_window": significant_outside,
            "rms_delay_spread_ns": self._rms_delay_spread_ns(
                aggregate, self.SOURCE_SAMPLE_RATE_HZ
            ),
            "retained_power_target": self.retained_power,
            "retained_pdp_power_ratio": float(np.sum(aggregate[selected_indexes])),
            "selected_path_indexes": selected_indexes.tolist(),
            "selected_path_delays_ns": (
                selected_indexes / self.SOURCE_SAMPLE_RATE_HZ * 1e9
            ).tolist(),
            "selected_path_powers": selected_pdp[selected_indexes].tolist(),
            "relative_mimo_link_power": relative_link_power.tolist(),
            "cross_to_diagonal_power_db": float(
                10.0 * np.log10(max(cross, 1e-300) / max(diagonal, 1e-300))
            ),
        }

    def _select_dominant_paths(
        self, normalized_pdp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(normalized_pdp)[::-1]
        count = int(
            np.searchsorted(np.cumsum(normalized_pdp[order]), self.retained_power)
        ) + 1
        selected = np.sort(order[:count])
        profile = np.zeros_like(normalized_pdp, dtype=float)
        profile[selected] = normalized_pdp[selected]
        profile /= np.sum(profile)
        return selected, profile

    @staticmethod
    def _normalize_deterministic(taps: np.ndarray, target_power: float) -> np.ndarray:
        power = float(np.sum(np.abs(taps) ** 2))
        if power <= 0.0:
            raise ValueError("实测信道功率为零，无法归一化")
        return taps * np.sqrt(float(target_power) / power)

    def deterministic_taps(self, full_mimo: bool) -> np.ndarray:
        if full_mimo:
            return self._normalize_deterministic(self.cropped_taps.copy(), 2.0)
        taps = self.cropped_taps[
            self.rx_index : self.rx_index + 1,
            self.tx_index : self.tx_index + 1,
            :,
        ].copy()
        return self._normalize_deterministic(taps, 1.0)

    def random_rayleigh_taps(self, full_mimo: bool) -> np.ndarray:
        profile = self.selected_pdp
        length = len(profile)
        if full_mimo:
            gaussian = (
                self.rng.standard_normal((2, 2, length))
                + 1j * self.rng.standard_normal((2, 2, length))
            ) / np.sqrt(2.0)
            return (
                np.sqrt(self.relative_link_power)[..., None]
                * np.sqrt(profile)[None, None, :]
                * gaussian
            )

        gaussian = (
            self.rng.standard_normal((1, 1, length))
            + 1j * self.rng.standard_normal((1, 1, length))
        ) / np.sqrt(2.0)
        return np.sqrt(profile)[None, None, :] * gaussian

    def load_taps(self, full_mimo: bool | None = None) -> np.ndarray:
        if full_mimo is None:
            full_mimo = bool(self._get("enable_mimo", False))
        if self.mode == self.MODE_DETERMINISTIC:
            taps = self.deterministic_taps(bool(full_mimo))
        else:
            taps = self.random_rayleigh_taps(bool(full_mimo))
        self.channel_impulse_response = taps.copy()
        self.diagnostics["output_shape"] = list(taps.shape)
        self.diagnostics["output_power"] = float(np.sum(np.abs(taps) ** 2))
        self.diagnostics["siso_rx_index"] = self.rx_index
        self.diagnostics["siso_tx_index"] = self.tx_index
        return taps

    def apply(self, signal_dict: dict[str, Any]) -> dict[str, Any]:
        """对 SISO 波形应用实测抽头；噪声和射频损伤仍由 THzChannel 处理。"""
        signal = np.asarray(signal_dict["signal_stream"], dtype=np.complex128)
        if signal.ndim != 1:
            raise ValueError("SISO 实测信道要求一维 signal_stream")
        sample_rate = float(signal_dict["sample_rate_Hz"])
        if not np.isclose(sample_rate, self.SOURCE_SAMPLE_RATE_HZ, rtol=1e-9):
            raise ValueError(
                "实测信道抽头基准采样率为 30 GHz，"
                f"当前信号采样率为 {sample_rate:g} Hz；请使用 OFDM、30 GHz、1x 过采样"
            )
        taps = self.load_taps(full_mimo=False)[0, 0]
        output = np.convolve(signal, taps, mode="full")[: len(signal)]
        result = dict(signal_dict)
        result.update(
            signal_stream=output,
            signal_length=len(output),
            duration_seconds=len(output) / sample_rate,
            signal_power=float(np.mean(np.abs(output) ** 2)),
            measured_channel_impulse_response=taps.copy(),
            measured_channel_diagnostics=dict(self.diagnostics),
        )
        return result

    def preview_data(self) -> dict[str, np.ndarray]:
        """返回前端绘图所需的纯数值数据。"""
        return {
            "raw_pdp": np.mean(np.abs(self.raw_taps) ** 2, axis=(0, 1)),
            "cropped_pdp": self.aggregate_pdp.copy(),
            "selected_pdp": self.selected_pdp.copy(),
            "selected_indexes": self.selected_path_indexes.copy(),
            "delays_ns": (
                np.arange(len(self.aggregate_pdp))
                / self.SOURCE_SAMPLE_RATE_HZ
                * 1e9
            ),
        }
