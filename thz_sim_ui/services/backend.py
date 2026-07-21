from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from PySide6.QtCore import QObject, Signal, QThread

from thz_sim_ui.data.mock_data import RECENT_TASKS, TaskItem
from simulation.SimulationManager import SimulationManager
from params.PHYParams import PHYParams


class SimulationThread(QThread):
    progress = Signal(int)

    def __init__(self, sm):
        super().__init__()
        self.sm = sm
        self.result = None

    def run(self):
        self.result = self.sm.run_monte_carlo(progress_callback=lambda p: self.progress.emit(p))


class BatchCompareThread(QThread):
    progress = Signal(int)
    scheme_progress = Signal(int, int)
    batch_progress = Signal(int)
    compare_chart_saved = Signal(str)  # 新增信号：发送 BER 对比图片路径

    def __init__(self, params, project_folder):
        super().__init__()
        self.params = params
        self.project_folder = project_folder
        self.results = []
        self._is_running = True
        self._lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._completed_tasks = 0
        self._total_tasks = 0
        self._current_task = None
        self._scheme_progress = {}
        self._save_intermediate = params.get('保存中间结果', '是') == '是'
        self._last_progress_update = 0
        self._progress_update_interval = 0.2
        self._plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    def run(self):
        configs = self.params.get('配置方案', [])
        snr_min = self.params.get('SNR最小值', 0)
        snr_max = self.params.get('SNR最大值', 30)
        snr_step = self.params.get('SNR步长', 2)
        run_mode = self.params.get('运行方式', '串行运行')
        max_concurrency = self.params.get('最大并发数', 8)

        snr_points = []
        current_snr = snr_min
        while current_snr <= snr_max:
            snr_points.append(current_snr)
            current_snr += snr_step

        self._snr_points_count = len(snr_points)
        self._total_tasks = len(configs) * len(snr_points)
        self._current_task = None

        self.progress.emit(0)

        for scheme_idx, config in enumerate(configs):
            self._scheme_progress[scheme_idx] = 0

        if run_mode == '并行运行':
            self._run_parallel(configs, snr_points, max_concurrency)
        else:
            self._run_serial(configs, snr_points)

        if self._is_running:
            self.progress.emit(100)
            self._save_compare_results()

    def _run_serial(self, configs, snr_points):
        for scheme_idx, config in enumerate(configs):
            if not self._is_running:
                break

            scheme_name = config.get('结果标签', f'scheme{scheme_idx + 1}')
            scheme_folder = os.path.join(self.project_folder, scheme_name)
            os.makedirs(scheme_folder, exist_ok=True)

            scheme_results = []

            for snr_idx, snr in enumerate(snr_points):
                if not self._is_running:
                    break

                result = self._run_single_simulation(scheme_name, snr, scheme_folder, snr_idx, scheme_idx, len(snr_points))
                scheme_results.append({
                    'snr': snr,
                    'result': result
                })

            with self._lock:
                self.results.append({
                    'scheme': config,
                    'results': scheme_results
                })

            self._save_scheme_results(scheme_name, scheme_results, scheme_folder)

    def _run_parallel(self, configs, snr_points, max_concurrency):
        max_workers = min(max_concurrency, len(configs))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for scheme_idx, config in enumerate(configs):
                scheme_name = config.get('结果标签', f'scheme{scheme_idx + 1}')
                scheme_folder = os.path.join(self.project_folder, scheme_name)
                os.makedirs(scheme_folder, exist_ok=True)

                future = executor.submit(
                    self._run_scheme,
                    scheme_idx, config, snr_points, scheme_folder
                )
                futures[future] = (scheme_idx, scheme_name, scheme_folder)

            for future in as_completed(futures):
                if not self._is_running:
                    break
                scheme_idx, scheme_name, scheme_folder = futures[future]
                try:
                    scheme_results = future.result()
                    with self._lock:
                        self.results.append({
                            'scheme': configs[scheme_idx],
                            'results': scheme_results
                        })
                    self._save_scheme_results(scheme_name, scheme_results, scheme_folder)
                except Exception as e:
                    print(f"方案 {scheme_name} 运行失败: {e}")
                    import traceback
                    traceback.print_exc()

    def _run_scheme(self, scheme_idx, config, snr_points, scheme_folder):
        scheme_results = []
        snr_count = len(snr_points)

        for snr_idx, snr in enumerate(snr_points):
            if not self._is_running:
                break

            result = self._run_single_simulation(
                config.get('结果标签', f'方案{scheme_idx + 1}'),
                snr, scheme_folder, snr_idx, scheme_idx, snr_count
            )
            scheme_results.append({
                'snr': snr,
                'result': result
            })

        return scheme_results

    def _run_single_simulation(self, scheme_name, snr, scheme_folder, snr_idx, scheme_idx, snr_count):
        sm = SimulationManager(base_params=PHYParams())
        sm.set_control_params({'SNRdB': snr})

        snr_folder = os.path.join(scheme_folder, f'SNR_{snr}dB')
        os.makedirs(snr_folder, exist_ok=True)

        try:
            result = sm.run_monte_carlo()
            serializable_result = self._convert_to_serializable(result)
            result_path = os.path.join(snr_folder, 'result.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            self._save_single_simulation_plots(result, snr_folder) if self._save_intermediate else None

            with self._progress_lock:
                self._completed_tasks += 1
                self._scheme_progress[scheme_idx] = self._scheme_progress.get(scheme_idx, 0) + 1
                scheme_prog = int((self._scheme_progress[scheme_idx] / snr_count) * 100)
                
                current_time = time.time()
                if current_time - self._last_progress_update >= self._progress_update_interval:
                    self.scheme_progress.emit(scheme_idx, scheme_prog)
                    if self._total_tasks > 0:
                        overall_progress = int((self._completed_tasks / self._total_tasks) * 100)
                        self.batch_progress.emit(overall_progress)
                    self._last_progress_update = current_time

            return serializable_result
        except Exception as e:
            print(f"仿真失败 {scheme_name} @ {snr}dB: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_single_simulation_plots(self, result, output_folder):
        import numpy as np
        from pathlib import Path

        transmitter_dir = Path(output_folder) / "transmitter"
        channel_dir = Path(output_folder) / "channel"
        receiver_dir = Path(output_folder) / "receiver"
        transmitter_dir.mkdir(exist_ok=True)
        channel_dir.mkdir(exist_ok=True)
        receiver_dir.mkdir(exist_ok=True)

        result_dict = result[0] if isinstance(result, list) and len(result) > 0 else result
        if not isinstance(result_dict, dict):
            result_dict = {}

        actual_result = result_dict.get('result', result_dict)

        self._batch_save_transmitter_plots(actual_result, transmitter_dir)
        self._batch_save_channel_plots(actual_result, channel_dir)
        self._batch_save_matched_filter_plots(actual_result, receiver_dir)

    def _batch_save_transmitter_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        with self._lock:
            plt.clf()
            plt.close('all')
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

            tx_signal_dict = result.get('tx_signal', {})
            tx_signal = tx_signal_dict.get('signal_stream', np.array([]))

            if len(tx_signal) == 0:
                print("发射信号为空，跳过图表生成")
                return

            params = result.get('params')
            symbol_period = 8
            if params is not None:
                symbol_period = params.get("oversampling") if hasattr(params, 'get') else 8
                if symbol_period is None:
                    symbol_period = 8

            n = len(tx_signal)
            time_window = min(5000, n)
            time_start = max(0, n - time_window)
            time_slice = tx_signal[time_start:time_start + time_window]

            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(time_slice)), np.real(time_slice), label='Real', alpha=0.8, linewidth=0.8, color=self._plot_colors[0])
            plt.plot(np.arange(len(time_slice)), np.imag(time_slice), label='Imag', alpha=0.8, linewidth=0.4, color=self._plot_colors[1])
            plt.title(f'TX Signal Time Domain (last {len(time_slice)} samples)')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "transmitter_time.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            if len(tx_signal) > 0 and np.any(tx_signal != 0):
                fft_signal = np.fft.fft(tx_signal)
                sample_rate = tx_signal_dict.get("sample_rate_Hz", 1e9)
                if sample_rate > 0:
                    freq = np.fft.fftfreq(len(fft_signal), 1/sample_rate)
                    magnitude = np.abs(fft_signal)
                    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                    plt.plot(freq/1e9, magnitude_db, color=self._plot_colors[0])
                    plt.title('TX Signal Spectrum')
                    plt.xlabel('Frequency (GHz)')
                    plt.ylabel('Magnitude (dB)')
                    plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
            fig.tight_layout()
            fig.savefig(output_dir / "transmitter_spectrum.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            power_window = min(1000, n)
            power_signal = tx_signal[max(0, n - power_window):]
            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(power_signal)), np.abs(power_signal)**2, linewidth=0.8, color=self._plot_colors[2])
            plt.title(f'TX Signal Power ({len(power_signal)} samples)')
            plt.xlabel('Sample')
            plt.ylabel('Power (W)')
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "transmitter_power.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            constellation_window = min(5000, n)
            constellation_signal = tx_signal[max(0, n - constellation_window):]
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(np.real(constellation_signal), np.imag(constellation_signal), s=5, alpha=0.6, c=self._plot_colors[1])
            plt.title('TX Constellation')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            fig.tight_layout()
            fig.savefig(output_dir / "transmitter_constellation.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            eye_window = min(2000 * symbol_period, n)
            eye_signal = np.real(tx_signal[:eye_window])
            fig = plt.figure(figsize=(10, 5))
            for i in range(0, len(eye_signal) - symbol_period + 1, symbol_period):
                plt.plot(np.arange(symbol_period), eye_signal[i:i + symbol_period], color=self._plot_colors[0], alpha=0.1, linewidth=0.8)
            plt.title('TX Signal Eye Diagram (Real)')
            plt.xlabel('Sample in symbol period')
            plt.ylabel('Signal amplitude (Real)')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, symbol_period)
            fig.tight_layout()
            fig.savefig(output_dir / "transmitter_eye.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"已保存 TX 图表到 {output_dir}")

    def _batch_save_channel_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        with self._lock:
            plt.clf()
            plt.close('all')
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

            rx_signal_dict = result.get('rx_signal', {})
            rx_signal = rx_signal_dict.get('signal_stream', np.array([]))

            if len(rx_signal) == 0:
                print("接收信号为空，跳过图表生成")
                return

            params = result.get('params')
            symbol_period = 8
            if params is not None:
                symbol_period = params.get("oversampling") if hasattr(params, 'get') else 8
                if symbol_period is None:
                    symbol_period = 8

            n = len(rx_signal)
            time_plot_len = min(1000, n)
            time_signal = rx_signal[:time_plot_len]

            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(time_signal)), np.real(time_signal), label='Real', alpha=0.8, linewidth=0.8, color=self._plot_colors[0])
            plt.plot(np.arange(len(time_signal)), np.imag(time_signal), label='Imag', alpha=0.8, linewidth=0.8, color=self._plot_colors[1])
            plt.title(f'RX Signal Time Domain (first {len(time_signal)} samples)')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "channel_time.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            if len(rx_signal) > 0 and np.any(rx_signal != 0):
                fft_signal = np.fft.fft(rx_signal)
                sample_rate = rx_signal_dict.get("sample_rate_Hz", 1e9)
                if sample_rate > 0:
                    freq = np.fft.fftfreq(len(fft_signal), 1/sample_rate)
                    magnitude = np.abs(fft_signal)
                    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                    plt.plot(freq/1e9, magnitude_db, color=self._plot_colors[0])
                    plt.title('RX Signal Spectrum')
                    plt.xlabel('Frequency (GHz)')
                    plt.ylabel('Magnitude (dB)')
                    plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
            fig.tight_layout()
            fig.savefig(output_dir / "channel_spectrum.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            constellation_window = min(1000, n)
            constellation_signal = rx_signal[:constellation_window]
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(np.real(constellation_signal), np.imag(constellation_signal), s=5, alpha=0.6, c=self._plot_colors[1])
            plt.title('RX Constellation')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            fig.tight_layout()
            fig.savefig(output_dir / "channel_constellation.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            power_window = min(1000, n)
            power_signal = rx_signal[:power_window]
            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(power_signal)), np.abs(power_signal)**2, linewidth=0.8, color=self._plot_colors[2])
            plt.title(f'RX Signal Power (first {len(power_signal)} samples)')
            plt.xlabel('Sample')
            plt.ylabel('Power (W)')
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "channel_power.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            eye_window = min(2000 * symbol_period, n)
            eye_signal = np.real(rx_signal[:eye_window])
            fig = plt.figure(figsize=(10, 5))
            for i in range(0, len(eye_signal) - symbol_period + 1, symbol_period):
                plt.plot(np.arange(symbol_period), eye_signal[i:i + symbol_period], color=self._plot_colors[0], alpha=0.1, linewidth=0.8)
            plt.title('RX Signal Eye Diagram (Real)')
            plt.xlabel('Sample in symbol period')
            plt.ylabel('Signal amplitude (Real)')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, symbol_period)
            fig.tight_layout()
            fig.savefig(output_dir / "channel_eye_real.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"已保存 RX 图表到 {output_dir}")

    def _batch_save_matched_filter_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        with self._lock:
            plt.clf()
            plt.close('all')
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

            rx_signal_dict = result.get('rx_signal', {})
            rx_matched_dict = result.get('rx_matched', {})
            tx_signal_dict = result.get('tx_signal', {})
            tx_symbols = result.get('tx_with_gi', {}).get('signal_stream', np.array([]))

            rx_signal = rx_signal_dict.get('signal_stream', np.array([]))
            y_matched = rx_matched_dict.get('signal_stream', np.array([]))
            rx_filtered = y_matched
            tx_signal = tx_signal_dict.get('signal_stream', np.array([]))

            if len(rx_signal) == 0 or len(rx_filtered) == 0 or len(y_matched) == 0:
                print("接收端信号数据不完整，跳过图表生成")
                return

            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.real(tx_signal[:200]), color=self._plot_colors[0])
            plt.title("TX Signal Time Domain (Real)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "matched_tx_time.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.real(rx_signal[:200]), color=self._plot_colors[0])
            plt.title("RX Signal Time Domain (Real)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "matched_rx_time.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            if len(rx_signal) > 0 and np.any(rx_signal != 0):
                RX = np.fft.fftshift(np.fft.fft(rx_signal))
                magnitude = np.abs(RX)
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                plt.plot(magnitude_db, color=self._plot_colors[0])
                plt.title("RX Signal Spectrum")
                plt.xlabel("Frequency Bin")
                plt.ylabel("Magnitude (dB)")
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
            fig.tight_layout()
            fig.savefig(output_dir / "matched_rx_spectrum.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            if len(rx_filtered) > 0 and np.any(rx_filtered != 0):
                MF = np.fft.fftshift(np.fft.fft(rx_filtered))
                magnitude = np.abs(MF)
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                plt.plot(magnitude_db, color=self._plot_colors[1])
                plt.title("Matched Filter Output Spectrum")
                plt.xlabel("Frequency Bin")
                plt.ylabel("Magnitude (dB)")
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
            fig.tight_layout()
            fig.savefig(output_dir / "matched_filtered_spectrum.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.real(tx_symbols[:100]), color=self._plot_colors[2])
            plt.title("TX Symbols Time Domain (Real)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "matched_tx_symbols.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(10, 5))
            plt.plot(np.real(y_matched[:100]), color=self._plot_colors[3])
            plt.title("Recovered Symbols Time Domain (Real)")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "matched_recovered_symbols.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"已保存 RX Matched Filter 图表到 {output_dir}")

    def _convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return self._convert_to_serializable(vars(obj))
        else:
            return str(obj)

    def _save_scheme_results(self, scheme_name, scheme_results, scheme_folder):
        result_path = os.path.join(scheme_folder, 'scheme_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(scheme_results, f, ensure_ascii=False, indent=2)

    def _save_compare_results(self):
        compare_folder = os.path.join(self.project_folder, 'compare_results')
        os.makedirs(compare_folder, exist_ok=True)

        fig = plt.figure(figsize=(10, 6))
        has_data = False

        for scheme_data in self.results:
            scheme_name = scheme_data['scheme'].get('结果标签', '未知方案')
            snrs = []
            bers = []

            for item in scheme_data['results']:
                snr = item.get('snr')
                result = item.get('result')
                if result and isinstance(result, list) and len(result) > 0:
                    try:
                        metrics = result[0].get('metrics', {})
                        ber = metrics.get('ber', 1.0) if isinstance(metrics, dict) else 1.0
                        if ber > 0:
                            snrs.append(snr)
                            bers.append(ber)
                    except Exception:
                        pass

            if snrs:
                plt.semilogy(snrs, bers, marker='o', label=scheme_name, linewidth=2, markersize=6)
                has_data = True

        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        plt.title('BER Comparison')
        if has_data:
            plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        chart_path = os.path.join(compare_folder, 'ber_compare.png')
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # 发送信号通知图片已保存
        self.compare_chart_saved.emit(chart_path)

    def stop(self):
        self._is_running = False


@dataclass
class BackendState:
    project_name: str = ''
    project_version: str = 'v1.0'
    current_task_id: str = 'TASK-2026-03-16-001'
    phase: str = '待机'
    progress: int = 0
    last_saved_at: str = '2026-03-16 16:32:42'
    recovery_points: int = 3
    notices: List[str] = field(default_factory=lambda: ['暂无新的系统通知'])


class BackendService(QObject):
    state_changed = Signal(dict)
    message_emitted = Signal(str)
    tasks_updated = Signal(list)
    scheme_progress_updated = Signal(int, int)
    compare_chart_saved = Signal(str)  # 新增信号：发送 BER 对比图片路径

    def __init__(self) -> None:
        super().__init__()
        self._state = BackendState()
        self._simulation_threads: list[SimulationThread] = []
        self._batch_thread: BatchCompareThread | None = None
        self._plot_lock = threading.Lock()
        self._plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        self.current_project_folder = None

    def snapshot(self) -> Dict[str, Any]:
        return self._state.__dict__.copy()

    def save_project(self) -> None:
        self._state.last_saved_at = '2026-03-16 16:35:12'
        self.message_emitted.emit('工程已保存（占位逻辑）')
        self.state_changed.emit(self.snapshot())

    def save_as_template(self) -> None:
        self.message_emitted.emit('已另存为模板（占位逻辑）')

    def import_config(self) -> None:
        self.message_emitted.emit('导入配置接口已预留，请接入文件选择与配置解析逻辑')

    def export_report(self) -> None:
        self.message_emitted.emit('导出报告接口已预留，请接入报告生成服务')

    def _format_task_tags(self, mapped_params: Dict[str, Any]) -> str:
        code_type = mapped_params.get('code_type', '未知编码')
        ncbps = mapped_params.get('NCBPS')
        mod = '未知调制'
        if ncbps is not None:
            mod_map = {1: 'BPSK', 2: 'QPSK', 3: '8PSK', 4: '16QAM', 6: '64QAM', 8: '256QAM'}
            mod = mod_map.get(int(ncbps), f'NCBPS={ncbps}')
        return f'{mod}, {code_type}'

    def _format_elapsed_str(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def map_ui_params_to_phy_params(ui_params: dict[str, object]) -> dict[str, object]:
        mapped_params = {}

        if '载频' in ui_params:
            mapped_params['fc'] = float(ui_params['载频']) * 1e9
        if '带宽' in ui_params:
            mapped_params['bandwidth'] = float(ui_params['带宽']) * 1e9
        if '采样率' in ui_params:
            mapped_params['sample_rate'] = float(ui_params['采样率']) * 1e6
        if '时长' in ui_params:
            mapped_params['duration'] = float(ui_params['时长']) * 1e-3
        if '数据帧长度' in ui_params:
            mapped_params['subframe_length'] = int(ui_params['数据帧长度'])
        if '单帧数据子帧数量' in ui_params:
            mapped_params['subframe_num'] = int(ui_params['单帧数据子帧数量'])
        if 'GI长度' in ui_params:
            mapped_params['gi_length'] = int(ui_params['GI长度'])

        if '比特源配置' in ui_params:
            mapped_params['bit_source'] = str(ui_params['比特源配置'])
        modulation_map = {
            'BPSK': 1,
            'QPSK': 2,
            '8PSK': 3,
            '16QAM': 4,
            '64QAM': 6,
        }
        if '调制方式' in ui_params and str(ui_params['调制方式']) in modulation_map:
            mapped_params['NCBPS'] = modulation_map[str(ui_params['调制方式'])]

        if '编码方式' in ui_params:
            coding_map = {
                'LDPC': 'LDPC',
                'RS(255,192)': 'RS',
                'RS(255, 192)': 'RS',
            }
            mapped_params['code_type'] = coding_map.get(str(ui_params['编码方式']), str(ui_params['编码方式']))

        if '波形成形' in ui_params:
            mapped_params['filter_type'] = str(ui_params['波形成形'])
        if '滚降因子' in ui_params:
            mapped_params['rolloff'] = float(ui_params['滚降因子'])
        if '滤波器跨度' in ui_params:
            mapped_params['filter_length'] = int(ui_params['滤波器跨度'])
        if '扰码配置' in ui_params:
            mapped_params['scramble'] = (str(ui_params['扰码配置']) == '启用')

        if '前导码配置' in ui_params:
            preamble_map = {
                '短前导': 'short',
                '长前导': 'long',
                '双前导': 'short',
            }
            mapped_params['Preamble_type'] = preamble_map.get(str(ui_params['前导码配置']), 'short')

        if 'SNR' in ui_params:
            mapped_params['SNRdB'] = float(ui_params['SNR'])
        if '路径损耗' in ui_params:
            mapped_params['path_loss'] = float(ui_params['路径损耗'])
        if '载频偏移' in ui_params:
            mapped_params['ppm'] = float(ui_params['载频偏移'])
        if '多径路径数' in ui_params:
            mapped_params['multipath_count'] = int(ui_params['多径路径数'])

        if '译码参数' in ui_params:
            decode_map = {
                '硬译码': 'hard',
                '软译码': 'soft',
            }
            mapped_params['decode_mode'] = decode_map.get(str(ui_params['译码参数']), str(ui_params['译码参数']))

        if '运行次数' in ui_params:
            mapped_params['run_times'] = int(ui_params['运行次数'])
        if '随机种子策略' in ui_params:
            mapped_params['seed_strategy'] = str(ui_params['随机种子策略'])
        if '保存中间结果' in ui_params:
            mapped_params['save_intermediate'] = str(ui_params['保存中间结果'])

        return mapped_params

    def run_simulation(self, params: dict = None, project_folder: str = None) -> None:
        if params and '工程名称' in params:
            project_name = params['工程名称']
            if project_name.strip():
                self._state.project_name = project_name
        project_folder = project_folder or getattr(self, 'current_project_folder', None)
        if project_folder is not None:
            from pathlib import Path
            project_folder = str(project_folder)
            self.current_project_folder = project_folder
            Path(project_folder).mkdir(parents=True, exist_ok=True)
        self._state.phase = '运行中'
        self._state.progress = 0
        if params:
            mapped_params = BackendService.map_ui_params_to_phy_params(params)
            print("映射后的PHY参数：", mapped_params)
            sm = SimulationManager(base_params=PHYParams())
            sm.set_control_params(mapped_params)
            tags = self._format_task_tags(mapped_params)
            start_time = time.time()
            task = TaskItem(
                name=f'THz-Sim-{len(RECENT_TASKS) + 1}',
                project=self._state.project_name,
                mode='蒙特卡洛',
                tags=tags,
                stage='仿真',
                progress=0,
                elapsed='00:00:00',
                eta='未知',
                status='运行中',
                is_current=True,
                start_time=start_time,
                result_path=self.current_project_folder,
            )
            RECENT_TASKS.append(task)
            self.tasks_updated.emit(RECENT_TASKS.copy())
            thread = SimulationThread(sm)
            self._simulation_threads.append(thread)
            thread.progress.connect(lambda p, t=task: self.update_progress(t, p))
            thread.finished.connect(lambda t=task, thr=thread: self.handle_simulation_finished(t, thr.result))
            thread.finished.connect(lambda thr=thread: self._cleanup_thread(thr))
            thread.finished.connect(thread.deleteLater)
            thread.start()
        else:
            self.message_emitted.emit('已触发运行接口')
        self.state_changed.emit(self.snapshot())

    def run_batch_compare_simulation(self, params: dict = None, project_folder: str = None) -> None:
        if project_folder is not None:
            from pathlib import Path
            project_folder = str(project_folder)
            self.current_project_folder = project_folder
            Path(project_folder).mkdir(parents=True, exist_ok=True)

        scheme_count = len(params.get('配置方案', []))
        snr_min = params.get('SNR最小值', 0)
        snr_max = params.get('SNR最大值', 30)
        snr_step = params.get('SNR步长', 2)

        snr_count = int((snr_max - snr_min) / snr_step) + 1
        total_points = scheme_count * snr_count

        project_name = params.get('工程名称', f'BatchCompare_{scheme_count}Schemes')
        self._state.project_name = project_name
        self._state.phase = '批量对比运行中'
        self._state.progress = 0

        start_time = time.time()
        tags = f'{scheme_count}方案, SNR:{snr_min}-{snr_max}dB'
        task = TaskItem(
            name=f'BatchCompare-{len(RECENT_TASKS) + 1}',
            project=self._state.project_name,
            mode='批量对比',
            tags=tags,
            stage='批量仿真',
            progress=0,
            elapsed='00:00:00',
            eta='未知',
            status='运行中',
            is_current=True,
            start_time=start_time,
            result_path=self.current_project_folder,
        )
        RECENT_TASKS.append(task)
        self.tasks_updated.emit(RECENT_TASKS.copy())

        self._batch_thread = BatchCompareThread(params, project_folder)
        self._batch_thread._current_task = task
        self._batch_thread.progress.connect(lambda p, t=task: self.update_progress(t, p))
        self._batch_thread.scheme_progress.connect(lambda idx, prog: self.scheme_progress_updated.emit(idx, prog))
        self._batch_thread.batch_progress.connect(lambda p, t=task: self.update_progress(t, p))
        self._batch_thread.compare_chart_saved.connect(self.compare_chart_saved.emit)  # 连接新信号
        self._batch_thread.finished.connect(lambda t=task, thr=self._batch_thread: self.handle_batch_compare_finished(t, thr.results))
        self._batch_thread.finished.connect(lambda: setattr(self, '_batch_thread', None))
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_thread.start()

        self.state_changed.emit(self.snapshot())
        self.message_emitted.emit(f'批量对比任务已启动，共 {total_points} 个仿真点')

    def update_progress(self, task: TaskItem, progress: int):
        now = time.time()
        task.progress = progress
        task.elapsed = self._format_elapsed_str(now - task.start_time) if task.start_time else '00:00:00'
        if progress > 0 and progress < 100:
            remain = (100 - progress) / progress * (now - task.start_time)
            task.eta = self._format_elapsed_str(remain)
        else:
            task.eta = '-' if progress == 100 else '未知'
        task.stage = '仿真'

        self._state.progress = progress
        
        if not hasattr(self, '_last_state_update'):
            self._last_state_update = 0
        if not hasattr(self, '_last_tasks_update'):
            self._last_tasks_update = 0
        
        update_interval = 0.3
        if now - self._last_state_update >= update_interval:
            self.state_changed.emit(self.snapshot())
            self._last_state_update = now
        if now - self._last_tasks_update >= update_interval:
            self.tasks_updated.emit(RECENT_TASKS.copy())
            self._last_tasks_update = now

    def handle_simulation_finished(self, task: TaskItem, result):
        now = time.time()
        task.status = '已完成'
        task.progress = 100
        task.is_current = False
        task.elapsed = self._format_elapsed_str(now - task.start_time) if task.start_time else '00:00:00'
        task.eta = '-'
        task.stage = '完成'
        task.result = result if isinstance(result, dict) else (result[0] if isinstance(result, list) and result else None)

        project_folder = task.result_path or getattr(self, 'current_project_folder', None)
        if not project_folder and self._state.project_name:
            from pathlib import Path
            project_folder = Path.cwd() / self._state.project_name
            project_folder.mkdir(parents=True, exist_ok=True)
            project_folder = str(project_folder)
            self.current_project_folder = project_folder
        if project_folder:
            task.result_path = str(project_folder)
            save_worker = threading.Thread(target=self._save_simulation_plots_async, args=(result, project_folder), daemon=True)
            save_worker.start()
            self.message_emitted.emit(f'仿真图表保存已触发，保存目录: {project_folder}')

        self._state.phase = '完成'
        self._state.progress = 100
        self.state_changed.emit(self.snapshot())
        self.tasks_updated.emit(RECENT_TASKS.copy())
        
        if result is None:
            self.message_emitted.emit('仿真完成，无结果数据')
        else:
            result_count = len(result) if hasattr(result, '__len__') else 1
            self.message_emitted.emit(f'仿真完成，结果: {result_count} 次运行')
        
        try:
            bers = [r["metrics"]["ber"] for r in result] if result else []
            print('运行结果BER：', bers)
        except Exception:
            pass

    def handle_batch_compare_finished(self, task: TaskItem, results):
        now = time.time()
        task.status = '已完成'
        task.progress = 100
        task.is_current = False
        task.elapsed = self._format_elapsed_str(now - task.start_time) if task.start_time else '00:00:00'
        task.eta = '-'
        task.stage = '批量对比完成'
        task.result = results

        self._state.phase = '完成'
        self._state.progress = 100
        self.state_changed.emit(self.snapshot())
        self.tasks_updated.emit(RECENT_TASKS.copy())

        scheme_count = len(results)
        total_points = sum(len(s['results']) for s in results)
        self.message_emitted.emit(f'批量对比完成，{scheme_count} 个方案，共 {total_points} 个仿真点')

    def _save_simulation_plots_async(self, results, project_folder):
        try:
            with self._plot_lock:
                self.save_simulation_plots(results, project_folder)
            self.tasks_updated.emit(RECENT_TASKS.copy())
        except Exception as exc:
            print(f"图表保存异常: {exc}")
            import traceback
            traceback.print_exc()
            self.message_emitted.emit(f'仿真图表保存失败：{exc}')

    def _cleanup_thread(self, thread: SimulationThread) -> None:
        if thread in self._simulation_threads:
            self._simulation_threads.remove(thread)

    def save_simulation_plots(self, results, project_folder):
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        project_path = Path(project_folder)
        transmitter_dir = project_path / "transmitter"
        channel_dir = project_path / "channel"
        matched_filter_dir = project_path / "receiver"

        transmitter_dir.mkdir(exist_ok=True)
        channel_dir.mkdir(exist_ok=True)
        matched_filter_dir.mkdir(exist_ok=True)

        print(f"开始保存图表到: {project_folder}")

        if results and len(results) > 0:
            result_item = results[0]
            result = result_item.get('result') if isinstance(result_item, dict) else None
            if result is None:
                print("仿真结果格式异常，无法生成图表")
                self.message_emitted.emit('仿真结果格式异常，无法生成图表')
                return
            params = result.get('params', {})

            print(f"结果数据键: {list(result.keys())}")

            self._save_transmitter_plots(result, transmitter_dir)
            self._save_channel_plots(result, channel_dir)
            self._save_matched_filter_plots(result, matched_filter_dir)

            print("图表保存完成")
        else:
            print("无结果数据，无法生成图表")

    def _save_transmitter_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        plt.clf()
        plt.close('all')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        tx_signal_dict = result.get('tx_signal', {})
        tx_signal = tx_signal_dict.get('signal_stream', np.array([]))
        print(f"发射信号长度: {len(tx_signal)}, 类型: {type(tx_signal)}, 非零元素: {np.count_nonzero(tx_signal)}")
        if len(tx_signal) == 0:
            print("发射信号为空，跳过图表生成")
            return

        params = result.get('params')
        symbol_period = 8
        if params is not None:
            symbol_period = params.get("oversampling") if hasattr(params, 'get') else 8
            if symbol_period is None:
                symbol_period = 8

        n = len(tx_signal)
        time_window = min(5000, n)
        time_start = max(0, n - time_window)
        time_slice = tx_signal[time_start:time_start + time_window]

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(time_slice)), np.real(time_slice), label='Real', alpha=0.8, linewidth=0.8, color=self._plot_colors[0])
        plt.plot(np.arange(len(time_slice)), np.imag(time_slice), label='Imag', alpha=0.8, linewidth=0.4, color=self._plot_colors[1])
        plt.title(f'TX Signal Time Domain (last {len(time_slice)} samples)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        if len(tx_signal) > 0 and np.any(tx_signal != 0):
            fft_signal = np.fft.fft(tx_signal)
            sample_rate = tx_signal_dict.get("sample_rate_Hz", 1e9)
            if sample_rate > 0:
                freq = np.fft.fftfreq(len(fft_signal), 1/sample_rate)
                magnitude = np.abs(fft_signal)
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                plt.plot(freq/1e9, magnitude_db, color=self._plot_colors[0])
                plt.title('TX Signal Spectrum')
                plt.xlabel('Frequency (GHz)')
                plt.ylabel('Magnitude (dB)')
                plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        power_window = min(1000, n)
        power_signal = tx_signal[max(0, n - power_window):]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(power_signal)), np.abs(power_signal)**2, linewidth=0.8, color=self._plot_colors[2])
        plt.title(f'TX Signal Power ({len(power_signal)} samples)')
        plt.xlabel('Sample')
        plt.ylabel('Power (W)')
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_power.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        constellation_window = min(5000, n)
        constellation_signal = tx_signal[max(0, n - constellation_window):]
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(np.real(constellation_signal), np.imag(constellation_signal), s=5, alpha=0.6, c=self._plot_colors[1])
        plt.title('TX Constellation')
        plt.xlabel('Real')
        plt.ylabel('Imag')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_constellation.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        eye_window = min(2000 * symbol_period, n)
        eye_signal = np.real(tx_signal[:eye_window])
        fig = plt.figure(figsize=(10, 5))
        for i in range(0, len(eye_signal) - symbol_period + 1, symbol_period):
            plt.plot(np.arange(symbol_period), eye_signal[i:i + symbol_period], color=self._plot_colors[0], alpha=0.1, linewidth=0.8)
        plt.title('TX Signal Eye Diagram (Real)')
        plt.xlabel('Sample in symbol period')
        plt.ylabel('Signal amplitude (Real)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, symbol_period)
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_eye.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _save_channel_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        plt.clf()
        plt.close('all')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        rx_signal_dict = result.get('rx_signal', {})
        rx_signal = rx_signal_dict.get('signal_stream', np.array([]))
        print(f"接收信号长度: {len(rx_signal)}, 类型: {type(rx_signal)}, 非零元素: {np.count_nonzero(rx_signal)}")
        if len(rx_signal) == 0:
            print("接收信号为空，跳过图表生成")
            return

        params = result.get('params')
        symbol_period = 8
        if params is not None:
            symbol_period = params.get("oversampling") if hasattr(params, 'get') else 8
            if symbol_period is None:
                symbol_period = 8

        n = len(rx_signal)
        time_plot_len = min(1000, n)
        time_signal = rx_signal[:time_plot_len]

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(time_signal)), np.real(time_signal), label='Real', alpha=0.8, linewidth=0.8, color=self._plot_colors[0])
        plt.plot(np.arange(len(time_signal)), np.imag(time_signal), label='Imag', alpha=0.8, linewidth=0.8, color=self._plot_colors[1])
        plt.title(f'RX Signal Time Domain (first {len(time_signal)} samples)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "channel_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        if len(rx_signal) > 0 and np.any(rx_signal != 0):
            fft_signal = np.fft.fft(rx_signal)
            sample_rate = rx_signal_dict.get("sample_rate_Hz", 1e9)
            if sample_rate > 0:
                freq = np.fft.fftfreq(len(fft_signal), 1/sample_rate)
                magnitude = np.abs(fft_signal)
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                plt.plot(freq/1e9, magnitude_db, color=self._plot_colors[0])
                plt.title('RX Signal Spectrum')
                plt.xlabel('Frequency (GHz)')
                plt.ylabel('Magnitude (dB)')
                plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
        fig.tight_layout()
        fig.savefig(output_dir / "channel_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        constellation_window = min(1000, n)
        constellation_signal = rx_signal[:constellation_window]
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(np.real(constellation_signal), np.imag(constellation_signal), s=5, alpha=0.6, c=self._plot_colors[1])
        plt.title('RX Constellation')
        plt.xlabel('Real')
        plt.ylabel('Imag')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        fig.tight_layout()
        fig.savefig(output_dir / "channel_constellation.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        power_window = min(1000, n)
        power_signal = rx_signal[:power_window]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(power_signal)), np.abs(power_signal)**2, linewidth=0.8, color=self._plot_colors[2])
        plt.title(f'RX Signal Power (first {len(power_signal)} samples)')
        plt.xlabel('Sample')
        plt.ylabel('Power (W)')
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "channel_power.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        eye_window = min(2000 * symbol_period, n)
        eye_signal = np.real(rx_signal[:eye_window])
        fig = plt.figure(figsize=(10, 5))
        for i in range(0, len(eye_signal) - symbol_period + 1, symbol_period):
            plt.plot(np.arange(symbol_period), eye_signal[i:i + symbol_period], color=self._plot_colors[0], alpha=0.1, linewidth=0.8)
        plt.title('RX Signal Eye Diagram (Real)')
        plt.xlabel('Sample in symbol period')
        plt.ylabel('Signal amplitude (Real)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, symbol_period)
        fig.tight_layout()
        fig.savefig(output_dir / "channel_eye_real.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _save_matched_filter_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        plt.clf()
        plt.close('all')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

        rx_signal_dict = result.get('rx_signal', {})
        rx_matched_dict = result.get('rx_matched', {})
        tx_signal_dict = result.get('tx_signal', {})
        tx_symbols = result.get('tx_with_gi', {}).get('signal_stream', np.array([]))

        rx_signal = rx_signal_dict.get('signal_stream', np.array([]))
        y_matched = rx_matched_dict.get('signal_stream', np.array([]))
        rx_filtered = y_matched
        tx_signal = tx_signal_dict.get('signal_stream', np.array([]))

        print(f"接收信号长度: {len(rx_signal)}, 滤波后长度: {len(rx_filtered)}, 匹配后长度: {len(y_matched)}")
        print(f"发射信号长度: {len(tx_signal)}, 发射符号长度: {len(tx_symbols)}")

        if len(rx_signal) == 0 or len(rx_filtered) == 0 or len(y_matched) == 0:
            print("接收端信号数据不完整，跳过图表生成")
            return

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.real(tx_signal[:200]), color=self._plot_colors[0])
        plt.title("TX Signal Time Domain (Real)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "matched_tx_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.real(rx_signal[:200]), color=self._plot_colors[0])
        plt.title("RX Signal Time Domain (Real)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "matched_rx_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        if len(rx_signal) > 0 and np.any(rx_signal != 0):
            RX = np.fft.fftshift(np.fft.fft(rx_signal))
            magnitude = np.abs(RX)
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
            plt.plot(magnitude_db, color=self._plot_colors[0])
            plt.title("RX Signal Spectrum")
            plt.xlabel("Frequency Bin")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
        fig.tight_layout()
        fig.savefig(output_dir / "matched_rx_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        if len(rx_filtered) > 0 and np.any(rx_filtered != 0):
            MF = np.fft.fftshift(np.fft.fft(rx_filtered))
            magnitude = np.abs(MF)
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
            plt.plot(magnitude_db, color=self._plot_colors[1])
            plt.title("Matched Filter Output Spectrum")
            plt.xlabel("Frequency Bin")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid signal data', ha='center', va='center', transform=plt.gca().transAxes)
        fig.tight_layout()
        fig.savefig(output_dir / "matched_filtered_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.real(tx_symbols[:100]), color=self._plot_colors[2])
        plt.title("TX Symbols Time Domain (Real)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "matched_tx_symbols.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.real(y_matched[:100]), color=self._plot_colors[3])
        plt.title("Recovered Symbols Time Domain (Real)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "matched_recovered_symbols.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def pause_simulation(self) -> None:
        self._state.phase = '已暂停'
        self.message_emitted.emit('已触发暂停接口（占位逻辑）')
        self.state_changed.emit(self.snapshot())

    def stop_simulation(self) -> None:
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.stop()

        for thread in self._simulation_threads:
            if thread.isRunning():
                thread.terminate()

        self._state.phase = '已停止'
        self._state.progress = 0
        self.message_emitted.emit('已触发停止接口')
        self.state_changed.emit(self.snapshot())

    def get_recent_tasks(self) -> List[dict]:
        return [item.__dict__.copy() for item in RECENT_TASKS]

    def get_project_summary(self) -> Dict[str, Any]:
        return {
            '理论峰值速率': '1.18 Tbps',
            '净有效速率': '0.93 Tbps',
            '频谱占用': '38.4 GHz',
            '运算复杂度': 'High',
            '并发任务数': '8',
            '最近恢复点': self._state.recovery_points,
        }
