import os
from pathlib import Path
from typing import Callable, Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.THzReceiver import THzReceiver
from simulation.result_utils import collect_receiver_intermediates


class SimulationManager:
    """仿真管理器：支持参数字典输入、蒙特卡洛多次仿真、种子策略、分阶段Pipeline。"""

    def __init__(
        self,
        base_params: PHYParams = None,
        output_dir: str = "simulation_results",
        save_plots: bool = True,
    ):
        self.base_params = base_params.clone() if (base_params is not None and hasattr(base_params, "clone")) else (base_params or PHYParams())
        self.base_params.validate()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_plots = save_plots

        self.control_params: Dict[str, Any] = {}
        self.seed_counter = 0
        self.run_index = 0

    def set_control_params(self, control_params: Dict[str, Any]):
        """设置仿真控制参数（run_times、seed_strategy、random_seed等）。"""
        self.control_params = control_params.copy() if control_params else {}

    def apply_dict_to_params(self, params: PHYParams, param_dict: Dict[str, Any]):
        """原子更新 PHYParams，避免关联参数逐键校验产生临时非法状态。"""
        valid_updates = {
            key: value for key, value in param_dict.items() if key in params._params
        }
        unknown_keys = [key for key in param_dict if key not in params._params]
        previous = params._params.copy()
        try:
            params._params.update(valid_updates)
            params.validate()
        except Exception:
            params._params = previous
            raise
        return unknown_keys

    def _get_effective_seed(self, params: PHYParams) -> int:
        """根据 seed_strategy 生成本次仿真种子。"""
        strategy = str(params.get("seed_strategy")).strip() or self.control_params.get("seed_strategy")
        user_seed = params.get("random_seed")

        if user_seed is not None:
            return int(user_seed)

        if strategy == "固定种子":
            if "random_seed" in params._params and params.get("random_seed") is not None:
                return int(params.get("random_seed"))
            return 0
        if strategy == "递增种子":
            self.seed_counter += 1
            return self.seed_counter
        if strategy in ["时间种子", ""]:
            import time
            return int((time.time() * 1e6) % (2**32))

        raise ValueError(f"未知随机种子策略: {strategy}")

    def _prepare_params_for_run(self, override_params: Dict[str, Any] = None) -> PHYParams:
        """从基本参数与控制参数构建本次仿真参数。"""
        params = self.base_params.clone() if hasattr(self.base_params, "clone") else PHYParams()
        self.apply_dict_to_params(params, self.control_params)
        if override_params:
            self.apply_dict_to_params(params, override_params)

        # 统一 seed_strategy 字段
        if "seed_strategy" in self.control_params and "seed_strategy" not in params._params:
            params._params["seed_strategy"] = self.control_params["seed_strategy"]

        # 计算 seed
        seed = self._get_effective_seed(params)
        if "random_seed" in params._params:
            params.update(random_seed=seed)
        else:
            params._params["random_seed"] = seed

        return params

    def _run_stage_pipeline(
        self,
        params: PHYParams,
        progress_callback: Callable[[int], None] = None,
    ) -> Dict[str, Any]:
        """完整链路：TX → Channel → RX。"""
        # TX
        if progress_callback:
            progress_callback(5)
        transmitter = THzTransmitter(params)
        tx_signal_dict = transmitter.run()
        if progress_callback:
            progress_callback(35)

        # Channel
        channel = THzChannel(params)
        rx_signal_dict = channel.run(tx_signal_dict)
        if progress_callback:
            progress_callback(55)

        # RX
        receiver = THzReceiver(params, transmitter)
        rx_data = receiver.run(rx_signal_dict)
        if progress_callback:
            progress_callback(90)

        # BER: data bits vs decoded bits
        tx_bits = (transmitter.data_bits_dict.get("signal_stream")
                   if transmitter.data_bits_dict else None)
        rx_bits = rx_data.get("signal_stream") if rx_data else None
        ber = self.compute_ber(tx_bits, rx_bits)

        tx_bit_len = len(tx_bits) if tx_bits is not None else 0
        waveform_duration = float(tx_signal_dict.get("duration_seconds", 0.0))
        raw_throughput_bps = tx_bit_len / waveform_duration if waveform_duration > 0 else np.nan
        effective_throughput_bps = (
            raw_throughput_bps * (1.0 - ber)
            if np.isfinite(raw_throughput_bps) and np.isfinite(ber)
            else np.nan
        )
        mimo_nmse = None
        mimo_condition_number = None
        if getattr(receiver, "rx_equalized", None):
            mimo_nmse = receiver.rx_equalized.get("mimo_channel_nmse")
            diagnostics = receiver.rx_equalized.get("detector_diagnostics", {})
            mimo_condition_number = diagnostics.get("mean_condition_number")
        if progress_callback:
            progress_callback(98)
        return {
            "params": params,
            "seed": params.get("random_seed"),
            "tx_signal": tx_signal_dict,
            "rx_signal": rx_signal_dict,
            "rx_data": rx_data,
            **collect_receiver_intermediates(receiver),
            "noise_var": receiver.noise_var,
            "ber": ber,
            "tx_bit_len": tx_bit_len,
            "raw_throughput_bps": raw_throughput_bps,
            "effective_throughput_bps": effective_throughput_bps,
            "mimo_channel_nmse": mimo_nmse,
            "mimo_mean_condition_number": mimo_condition_number,
            "tx_modulated": transmitter.modulated_data_dict if hasattr(transmitter, 'modulated_data_dict') else None,
        }

    @staticmethod
    def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        if tx_bits is None or rx_bits is None:
            return np.nan
        tx_bits = np.asarray(tx_bits).ravel().astype(np.uint8)
        rx_bits = np.asarray(rx_bits).ravel().astype(np.uint8)
        n = min(len(tx_bits), len(rx_bits))
        if n == 0:
            return np.nan
        return float(np.sum(tx_bits[:n] != rx_bits[:n]) / n)

    def run_once(
        self,
        override_params: Dict[str, Any] = None,
        progress_callback: Callable[[int], None] = None,
    ) -> Dict[str, Any]:
        """执行一次仿真。"""
        self.run_index += 1
        params = self._prepare_params_for_run(override_params)
        result = self._run_stage_pipeline(params, progress_callback=progress_callback)

        return {
            "run_index": self.run_index,
            "params": params,
            "result": result,
            "metrics": {
                "ber": result.get("ber"),
                "raw_throughput_bps": result.get("raw_throughput_bps"),
                "effective_throughput_bps": result.get("effective_throughput_bps"),
                "mimo_channel_nmse": result.get("mimo_channel_nmse"),
                "mimo_mean_condition_number": result.get("mimo_mean_condition_number"),
            },
        }

    def run_monte_carlo(self, override_params: Dict[str, Any] = None, progress_callback: Callable[[int], None] = None) -> List[Dict[str, Any]]:
        """执行多个蒙特卡洛仿真。"""
        self.run_index = 0
        self.seed_counter = 0
        results = []
        params = self._prepare_params_for_run(override_params)
        run_times = params.get("run_times")
        progress_step = max(1, run_times // 20)  # 约每5%发射一次进度
        if progress_callback:
            progress_callback(0)
        for i in range(run_times):
            def report_run_stage(stage_progress, run_index=i):
                if progress_callback:
                    overall = int((run_index + stage_progress / 100) / run_times * 100)
                    progress_callback(min(overall, 99))

            res = self.run_once(override_params, progress_callback=report_run_stage)
            results.append(res)
            if progress_callback and ((i + 1) % progress_step == 0 or i == run_times - 1):
                progress = int((i + 1) / run_times * 100)
                progress_callback(progress)
        return results

    # 以下保留老的可视化扫描接口，兼容测试
    def scan_single_parameter(
        self,
        param_name: str,
        values: List[Any],
        metric_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        save_name: str = None,
    ) -> List[Dict[str, Any]]:
        results = []
        metrics_y = []

        for val in values:
            r = self.run_once({param_name: val})
            results.append(r)
            metrics_y.append(r["metrics"].get("ber", np.nan))

        if self.save_plots:
            x_label = x_label or param_name
            y_label = y_label or "BER"
            title = title or f"单变量扫描：{param_name}"
            save_name = save_name or f"scan_{param_name}.png"
            self._plot(x=values, y=metrics_y, x_label=x_label, y_label=y_label, title=title, filename=save_name)

        return results

    def scan_two_parameters(
        self,
        param1: str,
        values1: List[Any],
        param2: str,
        values2: List[Any],
        metric_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        save_name: str = None,
    ) -> Dict[str, Any]:
        heatmap = np.zeros((len(values1), len(values2)), dtype=float)
        details = []

        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                r = self.run_once({param1: v1, param2: v2})
                ber = r["metrics"].get("ber", np.nan)
                heatmap[i, j] = ber
                details.append({"param1": v1, "param2": v2, "result": r})

        if self.save_plots:
            xlabel = xlabel or param2
            ylabel = ylabel or param1
            title = title or f"双参数扫描：{param1} vs {param2}"
            save_name = save_name or f"scan_{param1}_{param2}.png"
            self._plot_heatmap(heatmap, values2, values1, xlabel, ylabel, title, save_name)

        return {
            "param1": param1,
            "values1": values1,
            "param2": param2,
            "values2": values2,
            "heatmap": heatmap,
            "details": details,
        }

    def scan_multi_parameters(
        self,
        param_grid: Dict[str, List[Any]],
        metric_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        data_sink: Callable[[Dict[str, Any]], None] = None,
    ) -> List[Dict[str, Any]]:
        keys = list(param_grid.keys())
        results = []

        def walk(key_index, partial_override):
            if key_index == len(keys):
                r = self.run_once(partial_override)
                results.append({**partial_override, "metrics": r["metrics"], "result": r["result"]})
                if data_sink is not None:
                    data_sink(r)
                return
            key = keys[key_index]
            for value in param_grid[key]:
                new_override = dict(partial_override)
                new_override[key] = value
                walk(key_index + 1, new_override)

        walk(0, {})
        return results

    def _plot(self, x, y, x_label: str, y_label: str, title: str, filename: str):
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o", linestyle="-")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _plot_heatmap(self, matrix, x_ticks, y_ticks, xlabel, ylabel, title, filename):
        plt.figure(figsize=(9, 6))
        plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="BER")
        plt.xticks(np.arange(len(x_ticks)), x_ticks)
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
    def scan_two_parameters(
        self,
        param1: str,
        values1: List[Any],
        param2: str,
        values2: List[Any],
        metric_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        save_name: str = None,
    ) -> Dict[str, Any]:
        """双参数扫描过程，生成矩阵并保存热图。"""
        heatmap = np.zeros((len(values1), len(values2)), dtype=float)
        details = []

        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                r = self.run_once({param1: v1, param2: v2}, metric_fn=metric_fn, tag=f"{param1}={v1},{param2}={v2}")
                ber = r["metrics"].get("ber", np.nan)
                heatmap[i, j] = ber
                details.append({"param1": v1, "param2": v2, "result": r})

        if self.save_plots:
            xlabel = xlabel or param2
            ylabel = ylabel or param1
            title = title or f"双参数扫描：{param1} vs {param2}"
            save_name = save_name or f"scan_{param1}_{param2}.png"
            self._plot_heatmap(heatmap, values2, values1, xlabel, ylabel, title, save_name)

        return {
            "param1": param1,
            "values1": values1,
            "param2": param2,
            "values2": values2,
            "heatmap": heatmap,
            "details": details,
        }

    def scan_multi_parameters(
        self,
        param_grid: Dict[str, List[Any]],
        metric_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        data_sink: Callable[[Dict[str, Any]], None] = None,
    ) -> List[Dict[str, Any]]:
        """多参数扫描。param_grid: {'param1': [...], 'param2': [...], ...}。"""
        keys = list(param_grid.keys())
        results = []

        def walk(key_index, partial_override):
            if key_index == len(keys):
                r = self.run_once(partial_override, metric_fn=metric_fn, tag="multi")
                results.append({**partial_override, "metrics": r["metrics"], "result": r["result"]})
                if data_sink is not None:
                    data_sink(r)
                return
            key = keys[key_index]
            for value in param_grid[key]:
                new_override = dict(partial_override)
                new_override[key] = value
                walk(key_index + 1, new_override)

        walk(0, {})
        return results

