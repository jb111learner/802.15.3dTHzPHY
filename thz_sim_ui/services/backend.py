from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import threading
import time
import os
import multiprocessing
import tempfile
import traceback

# 子进程会重新导入 galois/numba。显式使用可写缓存目录，避免其反复探测
# Python 安装目录，也兼容只读安装和打包后的应用。
os.environ.setdefault(
    'NUMBA_CACHE_DIR',
    os.path.join(tempfile.gettempdir(), 'thz_sim_numba_cache'),
)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from PySide6.QtCore import QObject, Signal, QThread, QTimer

from thz_sim_ui.data.mock_data import RECENT_TASKS, TaskItem
from simulation.SimulationManager import SimulationManager
from params.PHYParams import PHYParams
from thz_sim_ui.services.result_utils import (
    build_simulation_result_path,
    select_rx_constellation_data,
)


def _plot_stream(signal):
    """绘图统一选取第一根天线，避免二维 MIMO 波形破坏旧图表接口。"""
    array = np.asarray(signal)
    if array.ndim == 2:
        return array[0]
    return array.reshape(-1) if array.ndim > 1 else array


class SimulationThread(QThread):
    progress = Signal(int)

    def __init__(self, sm):
        super().__init__()
        self.sm = sm
        self.result = None

    def run(self):
        self.result = self.sm.run_monte_carlo(progress_callback=lambda p: self.progress.emit(p))


def _load_batch_project_config(project_name: str) -> dict:
    """按关联工程名加载配置，结果标签只用于输出目录和图例。"""
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'projects', 'single',
        project_name, 'config.json'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'关联工程配置不存在: {config_path}')

    import json
    with open(config_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _should_continue_batch_point(
    total_errors: int,
    total_frames: int,
    min_errors: int,
    min_independent_runs: int,
    max_frames: int,
) -> bool:
    return total_frames < max_frames and (
        total_frames < min_independent_runs or total_errors < min_errors
    )


def _execute_batch_point(
    ui_params: dict,
    snr: float,
    max_frames: int,
    min_independent_runs: int,
    min_errors: int,
    progress_callback=None,
) -> dict:
    """在独立进程内执行一个 SNR 点，避免计算线程长期占用 GUI 进程的 GIL。"""
    mapped = BackendService.map_ui_params_to_phy_params(ui_params)
    mapped['SNRdB'] = snr
    mapped['data_source'] = 'PRBS'
    mapped['duration'] = 5e-6
    mapped['sample_length'] = None

    from transmitter.THzTransmitter import THzTransmitter
    from channel.THzChannel import THzChannel
    from receiver.THzReceiver import THzReceiver

    params = PHYParams()
    valid_keys = set(params._params) | PHYParams._LEGACY_MULTIPATH_KEYS
    for key, value in mapped.items():
        if value is not None and key in valid_keys:
            params._params[key] = value

    total_bits, total_errors, total_frames = 0, 0, 0
    tx = THzTransmitter(params)
    tx.run()
    channel = THzChannel(params)
    receiver = THzReceiver(params, tx)
    while _should_continue_batch_point(
        total_errors,
        total_frames,
        min_errors,
        min_independent_runs,
        max_frames,
    ):
        tx.assembler.duration = params.get('duration')
        tx.assembler.sample_length = None
        tx.run()
        rx = channel.run(tx.tx_signal_dict)
        rx_data = receiver.run(rx)
        tx_bits = tx.data_bits_dict['signal_stream']
        rx_bits = rx_data['signal_stream']
        compare_length = min(len(tx_bits), len(rx_bits))
        total_errors += int(np.sum(tx_bits[:compare_length] != rx_bits[:compare_length]))
        total_bits += compare_length
        total_frames += 1
        if progress_callback is not None:
            progress_callback(total_frames, total_errors)

    final_ber = total_errors / total_bits if total_bits > 0 else float('nan')

    # ── 吞吐率与谱效 ──
    bandwidth = float(params.get("bandwidth") or 0.0)
    duration_per_frame = float(tx.tx_signal_dict.get("duration_seconds", 0.0))
    waveform_duration = duration_per_frame * total_frames if total_frames > 0 else 0.0
    raw_throughput_bps = (
        total_bits / waveform_duration if waveform_duration > 0 else float('nan')
    )
    effective_throughput_bps = (
        raw_throughput_bps * (1.0 - final_ber)
        if np.isfinite(raw_throughput_bps) and np.isfinite(final_ber)
        else float('nan')
    )
    spectral_efficiency_bps_per_hz = (
        effective_throughput_bps / bandwidth
        if np.isfinite(effective_throughput_bps) and bandwidth > 0
        else float('nan')
    )
    return {
        'SNRdB': snr,
        'BER': final_ber,
        'total_bits': total_bits,
        'total_errors': total_errors,
        'total_frames': total_frames,
        'raw_throughput_bps': raw_throughput_bps,
        'effective_throughput_bps': effective_throughput_bps,
        'spectral_efficiency_bps_per_hz': spectral_efficiency_bps_per_hz,
    }


def _save_batch_compare_chart(results: list, chart_path: str) -> None:
    """在任务进程内保存对比图，避免 Matplotlib 绘图阻塞 GUI。"""
    plt.rcParams.update({
        'font.family': 'serif', 'axes.unicode_minus': False,
        'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'grid.alpha': 0.25, 'grid.linestyle': '--', 'grid.linewidth': 0.4,
    })
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    colors = ['#2C68B4', '#D95F02', '#3A9D3A', '#9467BD', '#E54B4F', '#8C6B4F']
    has_data = False
    for index, scheme_data in enumerate(results):
        scheme_name = scheme_data['scheme'].get('结果标签', f'方案{index + 1}')
        snrs, bers = [], []
        for item in scheme_data.get('results', []):
            result = item.get('result')
            if isinstance(result, dict) and result.get('BER') is not None and result['BER'] > 0:
                snrs.append(item.get('snr'))
                bers.append(result['BER'])
        if snrs:
            ax.semilogy(snrs, bers, marker='o', ms=6, lw=1.3,
                        color=colors[index % len(colors)], label=scheme_name)
            has_data = True
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('BER Comparison', fontsize=11, pad=6)
    if has_data:
        ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, which='major', alpha=0.25, ls='--', lw=0.4)
    ax.grid(True, which='minor', alpha=0.10, ls='--', lw=0.3)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _atomic_write_json(payload, result_path: str) -> None:
    """先写入同目录临时文件，再原子替换结果文件。"""
    import json

    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix='.scheme_results.', suffix='.tmp', dir=result_dir,
    )
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, result_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _batch_compare_process_entry(connection, params: dict, project_folder: str) -> None:
    """在一个常驻子进程中执行整批任务，避免重复支付重模块/JIT 初始化成本。"""
    try:
        configs = params.get('配置方案', [])
        snr_min = params.get('SNR最小值', 0)
        snr_max = params.get('SNR最大值', 30)
        snr_step = params.get('SNR步长', 2)
        max_frames = int(params.get('每点最大帧数', 300))
        min_independent_runs = int(params.get('每点最少独立运行次数', 20))
        min_errors = int(params.get('每点最少错误比特', 5000))
        snr_points = []
        current_snr = snr_min
        while current_snr <= snr_max:
            snr_points.append(current_snr)
            current_snr += snr_step
        total_points = len(configs) * len(snr_points)
        completed_points = 0
        all_results = []

        for scheme_index, config in enumerate(configs):
            scheme_name = config.get('结果标签', f'方案{scheme_index + 1}')
            project_name = config.get('工程', '')
            ui_params = _load_batch_project_config(project_name)
            scheme_folder = os.path.join(project_folder, scheme_name)
            os.makedirs(scheme_folder, exist_ok=True)
            result_path = os.path.join(scheme_folder, 'scheme_results.json')
            scheme_results = []
            for snr in snr_points:
                try:
                    last_live_update = 0.0

                    def report_point_progress(frame_count, error_count):
                        nonlocal last_live_update
                        now = time.monotonic()
                        if now - last_live_update < 0.5:
                            return
                        max_frame_ratio = frame_count / max_frames if max_frames else 1.0
                        min_run_ratio = (
                            frame_count / min_independent_runs
                            if min_independent_runs else 1.0
                        )
                        error_ratio = error_count / min_errors if min_errors else 1.0
                        point_ratio = min(0.99, max(
                            max_frame_ratio,
                            min(min_run_ratio, error_ratio),
                        ))
                        completed_in_scheme = len(scheme_results) + point_ratio
                        scheme_progress = int(completed_in_scheme / len(snr_points) * 100)
                        overall_progress = int(
                            (completed_points + point_ratio) / total_points * 100
                        ) if total_points else 100
                        connection.send((
                            'progress', scheme_index, scheme_progress, overall_progress,
                        ))
                        last_live_update = now

                    result = _execute_batch_point(
                        ui_params,
                        snr,
                        max_frames,
                        min_independent_runs,
                        min_errors,
                        progress_callback=report_point_progress,
                    )
                except Exception:
                    result = None
                    connection.send(('point_error', scheme_name, snr, traceback.format_exc()))
                scheme_results.append({'snr': snr, 'result': result})
                # 每个仿真点完成后立即持久化，中途终止时仍保留已完成结果。
                _atomic_write_json(scheme_results, result_path)
                completed_points += 1
                scheme_progress = int(len(scheme_results) / len(snr_points) * 100)
                overall_progress = int(completed_points / total_points * 100) if total_points else 100
                connection.send(('progress', scheme_index, scheme_progress, overall_progress))

            all_results.append({'scheme': config, 'results': scheme_results})

        compare_folder = os.path.join(project_folder, 'compare_results')
        os.makedirs(compare_folder, exist_ok=True)
        chart_path = os.path.join(compare_folder, 'ber_compare.png')
        _save_batch_compare_chart(all_results, chart_path)
        connection.send(('finished', all_results, chart_path))
    except BaseException:
        connection.send(('fatal_error', traceback.format_exc()))
    finally:
        connection.close()


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
        self._max_frames = int(params.get('每点最大帧数', 300))
        self._min_independent_runs = int(params.get('每点最少独立运行次数', 20))
        self._min_errors = int(params.get('每点最少错误比特', 5000))
        self._last_progress_update = 0
        self._progress_update_interval = 0.2
        self._plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        self._process_lock = threading.Lock()
        self._active_process = None
        self.error_message = None

    def run(self):
        self.progress.emit(0)
        context = multiprocessing.get_context('spawn')
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_batch_compare_process_entry,
            args=(send_connection, self.params, self.project_folder),
            name='BatchCompareWorker',
        )
        with self._process_lock:
            self._active_process = process
        try:
            process.start()
            send_connection.close()
            finished_payload = None

            def handle_message(message):
                nonlocal finished_payload
                kind = message[0]
                if kind == 'progress':
                    _, scheme_index, scheme_progress, overall_progress = message
                    self.scheme_progress.emit(scheme_index, scheme_progress)
                    self.batch_progress.emit(overall_progress)
                elif kind == 'point_error':
                    _, scheme_name, snr, details = message
                    print(f'仿真失败 {scheme_name} @ {snr}dB:\n{details}')
                elif kind in ('finished', 'fatal_error'):
                    finished_payload = message

            while process.is_alive() and self._is_running:
                try:
                    if receive_connection.poll(0.05):
                        try:
                            message = receive_connection.recv()
                        except EOFError:
                            break
                        handle_message(message)
                except BrokenPipeError:
                    break
            if not self._is_running and process.is_alive():
                process.terminate()
            process.join()
            while True:
                try:
                    if not receive_connection.poll():
                        break
                    handle_message(receive_connection.recv())
                except (EOFError, BrokenPipeError):
                    break
            if self._is_running:
                if finished_payload is None:
                    raise RuntimeError(f'批量任务子进程异常退出，退出码: {process.exitcode}')
                if finished_payload[0] == 'fatal_error':
                    raise RuntimeError(finished_payload[1])
                _, self.results, chart_path = finished_payload
                self.progress.emit(100)
                self.compare_chart_saved.emit(chart_path)
        except Exception as exc:
            if self._is_running:
                self.error_message = str(exc)
                print(f'批量对比任务失败: {exc}')
                traceback.print_exc()
        finally:
            receive_connection.close()
            send_connection.close()
            with self._process_lock:
                self._active_process = None

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

        with self._lock:
            plt.clf(); plt.close('all')

            tx_signal_dict = result.get('tx_signal', {})
            tx_signal = _plot_stream(tx_signal_dict.get('signal_stream', []))
            if len(tx_signal) == 0:
                return

            params = result.get('params')
            sps = 4; preamble_len_sym = 3328
            if params is not None:
                sps = int(params.get("oversampling", 4) or 4)
                preamble_len_sym = 3328 if params.get("Preamble_type", "short") == "short" else 5120

            pskip = preamble_len_sym * sps
            payload = tx_signal[pskip:] if len(tx_signal) > pskip else tx_signal
            if len(payload) == 0:
                payload = tx_signal
            payload = payload / (np.sqrt(np.mean(np.abs(payload)**2)) + 1e-15)
            fs = tx_signal_dict.get("sample_rate_Hz", 1.0)
            color = '#2C68B4'
            mode = str(params.get("link_mode", "sc-fde")).upper() if params is not None else "SC-FDE"

            plt.rcParams.update({
                "font.family": "serif", "axes.unicode_minus": False,
                "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                "xtick.major.size": 4, "ytick.major.size": 4,
                "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
            })

            # time envelope
            fig, ax = plt.subplots(figsize=(6, 3))
            n_show = min(600, len(payload)//sps)
            env = np.abs(payload[:n_show*sps]); t = np.arange(len(env))/sps
            ax.plot(t, env, color=color, lw=0.6)
            ax.axhline(y=1.0, color="gray", ls="--", lw=0.5, alpha=0.5, label="mean")
            ax.set_xlabel("Symbol index"); ax.set_ylabel("|Amplitude|")
            ax.set_title(f"{mode} time envelope", fontsize=10, pad=4)
            ax.legend(fontsize=8); ax.set_xlim([0, n_show])
            fig.tight_layout(); fig.savefig(output_dir / "transmitter_time.png", dpi=300, bbox_inches='tight'); plt.close(fig)

            # spectrum
            fig, ax = plt.subplots(figsize=(6, 3.5))
            nfft = 2048; seg = payload[:nfft]
            spec = np.fft.fftshift(np.fft.fft(seg, nfft))
            freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)) if fs > 0 else np.arange(nfft)
            ax.plot(freq/1e9, 20*np.log10(np.abs(spec)+1e-15), color=color, lw=0.4)
            ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"{mode} spectrum", fontsize=10, pad=4)
            bw = 0.5*fs; ax.set_xlim([-bw/1e9, bw/1e9]); ax.set_ylim([-20, 60])
            fig.tight_layout(); fig.savefig(output_dir / "transmitter_spectrum.png", dpi=300, bbox_inches='tight'); plt.close(fig)

            # PAPR CCDF
            fig, ax = plt.subplots(figsize=(6, 4))
            blk_len = 480 * sps; pv = []
            for b in range(0, len(payload)-blk_len, blk_len):
                blk = payload[b:b+blk_len]
                pv.append(10*np.log10(np.max(np.abs(blk)**2)/(np.mean(np.abs(blk)**2)+1e-15)))
            if pv:
                pv = np.sort(pv); ccdf = 1.0 - np.arange(len(pv))/len(pv)
                ax.semilogy(pv, ccdf, color=color, lw=1.2)
            ax.set_xlabel("PAPR (dB)"); ax.set_ylabel("CCDF")
            ax.set_title(f"{mode} PAPR CCDF", fontsize=10, pad=4); ax.set_ylim([1e-3,1])
            fig.tight_layout(); fig.savefig(output_dir / "transmitter_papr.png", dpi=300, bbox_inches='tight'); plt.close(fig)

            # amplitude histogram
            fig, ax = plt.subplots(figsize=(6, 4))
            amps = np.abs(payload)
            ax.hist(amps, bins=80, density=True, histtype="step", color=color, lw=1.2)
            ax.set_xlabel("|Amplitude|"); ax.set_ylabel("Probability density")
            ax.set_title(f"{mode} amplitude distribution", fontsize=10, pad=4)
            fig.tight_layout(); fig.savefig(output_dir / "transmitter_histogram.png", dpi=300, bbox_inches='tight'); plt.close(fig)

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
            rx_signal = _plot_stream(rx_signal_dict.get('signal_stream', np.array([])))

            if len(rx_signal) == 0:
                print("接收信号为空，跳过图表生成")
                return

            params = result.get('params')
            symbol_period = 8
            if params is not None:
                symbol_period = params.get("oversampling", 8) or 8

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
            tx_symbols = _plot_stream(result.get('tx_with_gi', {}).get('signal_stream', np.array([])))

            rx_signal = _plot_stream(rx_signal_dict.get('signal_stream', np.array([])))
            y_matched = _plot_stream(rx_matched_dict.get('signal_stream', np.array([])))
            rx_filtered = y_matched
            tx_signal = _plot_stream(tx_signal_dict.get('signal_stream', np.array([])))

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

        plt.rcParams.update({
            "font.family": "serif", "axes.unicode_minus": False,
            "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 4, "ytick.major.size": 4,
            "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
        })
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_facecolor("white"); fig.patch.set_facecolor("white")
        colors = ['#2C68B4', '#D95F02', '#3A9D3A', '#9467BD', '#E54B4F', '#8C6B4F']
        has_data = False

        for i, scheme_data in enumerate(self.results):
            scheme_name = scheme_data['scheme'].get('结果标签', f'方案{i+1}')
            snrs, bers = [], []
            for item in scheme_data.get('results', []):
                result = item.get('result')
                if isinstance(result, dict):
                    ber = result.get('BER')
                    if ber is not None and ber > 0:
                        snrs.append(item.get('snr'))
                        bers.append(ber)

            if snrs:
                color = colors[i % len(colors)]
                ax.semilogy(snrs, bers, marker='o', ms=6, lw=1.3,
                            color=color, label=scheme_name)
                has_data = True

        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
        ax.set_title("BER Comparison", fontsize=11, pad=6)
        if has_data:
            ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, which="major", alpha=0.25, ls="--", lw=0.4)
        ax.grid(True, which="minor", alpha=0.10, ls="--", lw=0.3)
        fig.tight_layout()
        chart_path = os.path.join(compare_folder, 'ber_compare.png')
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.compare_chart_saved.emit(chart_path)

    def stop(self):
        self._is_running = False
        with self._process_lock:
            process = self._active_process
        if process is not None and process.is_alive():
            process.terminate()


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
        self._task_clock = QTimer(self)
        self._task_clock.setInterval(1000)
        self._task_clock.timeout.connect(self._refresh_running_task_elapsed)

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

    def _start_task_clock(self) -> None:
        if not self._task_clock.isActive():
            self._task_clock.start()

    def _refresh_running_task_elapsed(self) -> None:
        """时间显示独立于仿真进度，每秒刷新一次且不触发全表重建。"""
        now = time.time()
        has_running_task = False
        changed = False
        for task in RECENT_TASKS:
            if task.status != '运行中' or not task.start_time:
                continue
            has_running_task = True
            elapsed_seconds = max(0.0, now - task.start_time)
            elapsed = self._format_elapsed_str(elapsed_seconds)
            if task.elapsed != elapsed:
                task.elapsed = elapsed
                changed = True
            if 0 < task.progress < 100:
                remaining = (100 - task.progress) / task.progress * elapsed_seconds
                eta = self._format_elapsed_str(remaining)
                if task.eta != eta:
                    task.eta = eta
                    changed = True
        if changed:
            self.tasks_updated.emit(RECENT_TASKS.copy())
        if not has_running_task:
            self._task_clock.stop()

    def map_ui_params_to_phy_params(ui_params: dict[str, object]) -> dict[str, object]:
        mapped_params = {}

        # 空间模式。UI 当前提供固定的 2×2 空间复用入口。
        spatial_mode = str(
            ui_params.get('空间模式分区', ui_params.get('spatial_mode', '非 MIMO'))
        )
        if 'MIMO' in spatial_mode and '非' not in spatial_mode:
            mapped_params.update({
                'enable_mimo': True,
                'num_tx': 2,
                'num_rx': 2,
                'num_spatial_streams': 2,
                'mimo_scheme': 'spatial_multiplexing',
                'mimo_detector': str(ui_params.get('MIMO检测算法', 'mmse')).lower(),
                'mimo_csi_mode': str(ui_params.get('MIMO CSI模式', 'estimated')).lower(),
                'mimo_channel_model': str(ui_params.get('MIMO信道模型', 'iid_rayleigh')).lower(),
                'link_mode': 'ofdm',
            })
        else:
            mapped_params.update({
                'enable_mimo': False,
                'num_tx': 1,
                'num_rx': 1,
                'num_spatial_streams': 1,
            })

        # ── 链路参数 ──
        if '载频' in ui_params:
            mapped_params['fc'] = float(ui_params['载频']) * 1e9
        if '带宽' in ui_params:
            mapped_params['bandwidth'] = float(ui_params['带宽']) * 1e9
        if '采样率' in ui_params:
            mapped_params['sample_rate'] = float(ui_params['采样率']) * 1e6
        if '时长' in ui_params:
            mapped_params['duration'] = float(ui_params['时长']) * 1e-3

        # 链路设计页是链路模式的主入口；参数页中的“波形类型”作为旧工程
        # 和专项模式的兼容入口。避免两个页面值不一致时悄悄运行另一种波形。
        link_mode_partition = str(ui_params.get('链路模式分区', '')).strip()
        partition_mode_map = {
            '单载波模式': 'sc-fde',
            '多载波模式': 'ofdm',
        }
        if not mapped_params.get('enable_mimo') and link_mode_partition in partition_mode_map:
            mapped_params['link_mode'] = partition_mode_map[link_mode_partition]
        elif '波形类型' in ui_params:
            wf = str(ui_params['波形类型'])
            if not mapped_params.get('enable_mimo'):
                mapped_params['link_mode'] = 'ofdm' if 'OFDM' in wf else 'sc-fde'

        # 数据源
        if '数据源配置' in ui_params:
            src = str(ui_params['数据源配置'])
            if src == '文件输入':
                mapped_params['data_source'] = '文件输入'
                if '文件地址' in ui_params and str(ui_params['文件地址']).strip():
                    mapped_params['file_path'] = str(ui_params['文件地址'])
            else:
                mapped_params['data_source'] = 'PRBS'

        # 帧结构
        if '数据子帧长度' in ui_params:
            mapped_params['subframe_length'] = int(ui_params['数据子帧长度'])
        if '单帧数据子帧数量' in ui_params:
            mapped_params['subframe_num'] = int(ui_params['单帧数据子帧数量'])
        if 'CP长度' in ui_params:
            mapped_params['gi_length'] = int(ui_params['CP长度'])

        # 调制
        modulation_map = {'BPSK': 1, 'QPSK': 2, '8PSK': 3, '16QAM': 4, '64QAM': 6}
        if '调制方式' in ui_params and str(ui_params['调制方式']) in modulation_map:
            mapped_params['NCBPS'] = modulation_map[str(ui_params['调制方式'])]

        # 编码
        if '信道编码类型' in ui_params:
            code_str = str(ui_params['信道编码类型'])
            if 'LDPC' in code_str:
                mapped_params['code_type'] = 'LDPC'
                mapped_params['ldpc_standard_rate'] = '14/15'
            elif 'RS' in code_str:
                mapped_params['code_type'] = 'RS'
                if '11,15' in code_str or '11，15' in code_str:
                    mapped_params['rs_nsym'] = 4
                    mapped_params['rs_c_exp'] = 4
                    mapped_params['rs_packet_size'] = 11
                else:
                    mapped_params['rs_nsym'] = 63
                    mapped_params['rs_c_exp'] = 8
                    mapped_params['rs_packet_size'] = 192

        # RS 译码
        if 'RS译码方式' in ui_params:
            mapped_params['decode_mode'] = 'chase' if 'Chase' in str(ui_params['RS译码方式']) else 'hard'
        if 'Chase译码试探数' in ui_params:
            mapped_params['chase_num_per_packet'] = int(ui_params['Chase译码试探数'])

        # 前导码
        if '前导码配置' in ui_params:
            preamble_map = {'短前导': 'short', '长前导': 'long'}
            mapped_params['Preamble_type'] = preamble_map.get(str(ui_params['前导码配置']), 'short')

        # OFDM
        if 'OFDM子载波数' in ui_params:
            mapped_params['subwave_num'] = int(ui_params['OFDM子载波数'])
        if '单帧OFDM符号数' in ui_params:
            mapped_params['subframe_ofdm_num'] = int(ui_params['单帧OFDM符号数'])
        if '块状导频索引' in ui_params:
            import ast
            try:
                mapped_params['pilot_block_indexes'] = ast.literal_eval(str(ui_params['块状导频索引']))
            except (ValueError, SyntaxError):
                mapped_params['pilot_block_indexes'] = [0, 16, 32]

        # 波形成形
        if '波形成形' in ui_params:
            mapped_params['filter_type'] = str(ui_params['波形成形'])
        if '滚降因子' in ui_params:
            mapped_params['rolloff'] = float(ui_params['滚降因子'])
        if '滤波器跨度' in ui_params:
            mapped_params['filter_length'] = int(ui_params['滤波器跨度'])
        if '过采样率' in ui_params:
            ovs = str(ui_params['过采样率']).replace('x', '')
            mapped_params['oversampling'] = int(ovs) if ovs.isdigit() else 4

        # 接收端补偿开关
        if '频偏补偿' in ui_params:
            mapped_params['enable_cfo_compensation'] = (str(ui_params['频偏补偿']) == '是')
        if 'IQ补偿' in ui_params:
            mapped_params['enable_iq_compensation'] = (str(ui_params['IQ补偿']) == '是')
        if '信道估计与均衡' in ui_params:
            mapped_params['enable_channel_equalization'] = (str(ui_params['信道估计与均衡']) == '是')

        # 扰码
        if '扰码配置' in ui_params:
            mapped_params['scramble'] = (str(ui_params['扰码配置']) == '启用')

        # 运行参数（单方案始终 run_times=1）
        mapped_params['run_times'] = 1
        if '随机种子策略' in ui_params:
            mapped_params['seed_strategy'] = str(ui_params['随机种子策略'])

        # ── 信道参数 (来自 channel_integration 页面) ──
        channel_params = ui_params.get('_channel_params', {})
        if isinstance(channel_params, dict):
            for k, v in channel_params.items():
                mapped_params[k] = v
            # CFO 注入时自动启用补偿
            if mapped_params.get("enable_cfo"):
                mapped_params.setdefault("enable_cfo_compensation", True)

        # 实测抽头的原始采样率为 30 GHz。sample_rate 保持为基础波形
        # 采样率；2x/4x 时由发射端和实测 CIR 同步提升至 60/120 GHz。
        # 不得覆盖用户选择的链路模式和过采样率。
        # 场景选择同时确定物理时延窗对应的安全 GI，避免把尾部测量噪声
        # 当成需要覆盖的 512 抽头信道。
        if str(mapped_params.get("multipath_source", "simulated")) == "measured":
            scenario = str(mapped_params.get("measured_channel_scenario", "8cm"))
            gi_by_scenario = {"8cm": 64, "12cm": 64, "50cm": 32}
            if scenario not in gi_by_scenario:
                raise ValueError(f"未知实测信道场景：{scenario}")
            gi_length = gi_by_scenario[scenario]
            mapped_params.update({
                "enable_multipath": True,
                "sample_rate": 30e9,
                "gi_length": gi_length,
                "noise_temperature": None,
                "noise_figure_db": None,
            })

        # 兼容旧参数名 (向后兼容旧的 config.json)
        if '数据帧长度' in ui_params and 'subframe_length' not in mapped_params:
            mapped_params['subframe_length'] = int(ui_params['数据帧长度'])
        if 'GI长度' in ui_params and 'gi_length' not in mapped_params:
            mapped_params['gi_length'] = int(ui_params['GI长度'])
        if '比特源配置' in ui_params and 'data_source' not in mapped_params:
            mapped_params['data_source'] = str(ui_params['比特源配置'])
        if '编码方式' in ui_params:
            old_code = str(ui_params['编码方式'])
            if 'LDPC' in old_code:
                mapped_params.setdefault('code_type', 'LDPC')
            elif 'RS' in old_code:
                mapped_params.setdefault('code_type', 'RS')
        if 'SNR' in ui_params and 'SNRdB' not in mapped_params:
            mapped_params['SNRdB'] = float(ui_params['SNR'])

        return mapped_params

    def run_simulation(self, params: dict = None, project_folder: str = None) -> None:
        if params and '工程名称' in params:
            project_name = params['工程名称']
            if project_name.strip():
                self._state.project_name = project_name
        else:
            project_name = self._state.project_name or 'unnamed'
        # 工程目录保存参数，运行时结果存入临时目录避免污染工程文件夹
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
            task_name = f'THz-Sim-{len(RECENT_TASKS) + 1}'
            # 运行时结果存入临时目录，按工程分组
            task_root = Path(tempfile.gettempdir()) / "thz_sim_runs" / project_name
            result_path = build_simulation_result_path(
                str(task_root),
                task_name,
                time.time_ns(),
            )
            result_path.mkdir(parents=True, exist_ok=True)
            task = TaskItem(
                name=task_name,
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
                result_path=str(result_path) if result_path else None,
            )
            RECENT_TASKS.append(task)
            self._start_task_clock()
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
        self._start_task_clock()
        self.tasks_updated.emit(RECENT_TASKS.copy())

        self._batch_thread = BatchCompareThread(params, project_folder)
        self._batch_thread._current_task = task
        self._batch_thread.progress.connect(lambda p, t=task: self.update_progress(t, p))
        self._batch_thread.scheme_progress.connect(lambda idx, prog: self.scheme_progress_updated.emit(idx, prog))
        self._batch_thread.batch_progress.connect(lambda p, t=task: self.update_progress(t, p))
        self._batch_thread.compare_chart_saved.connect(self.compare_chart_saved.emit)  # 连接新信号
        self._batch_thread.finished.connect(
            lambda t=task, thr=self._batch_thread:
            self.handle_batch_compare_finished(t, thr.results, thr.error_message)
        )
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

    @staticmethod
    def _cleanup_old_runs(run_folder: str) -> None:
        """删除 run_folder 所在 runs/ 目录下的旧运行结果，保留 run_folder 自身。"""
        import shutil
        current = Path(run_folder)
        runs_dir = current.parent
        if not runs_dir.is_dir() or runs_dir.name != "runs":
            return
        dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d != current],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        for old in dirs:
            try:
                shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass

    def handle_batch_compare_finished(self, task: TaskItem, results, error_message: str | None = None):
        now = time.time()
        if error_message:
            task.status = '失败'
            task.is_current = False
            task.elapsed = self._format_elapsed_str(now - task.start_time) if task.start_time else '00:00:00'
            task.eta = '-'
            task.stage = '批量对比失败'
            self._state.phase = '失败'
            self.state_changed.emit(self.snapshot())
            self.tasks_updated.emit(RECENT_TASKS.copy())
            self.message_emitted.emit(f'批量对比失败：{error_message}')
            return
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
            self._save_metrics_json(result, project_path)

            print("图表保存完成")
        else:
            print("无结果数据，无法生成图表")

    @staticmethod
    def _save_metrics_json(result, project_path):
        """将单次仿真关键指标输出到 metrics.json。"""
        import json

        metrics = result.get('metrics') or {}
        params = result.get('params')
        snr = params.get('SNRdB') if params is not None else None

        def _sanitize(value):
            if value is None:
                return None
            try:
                vf = float(value)
                if np.isfinite(vf):
                    return vf
            except (TypeError, ValueError):
                pass
            return None

        payload = {'SNRdB': snr}
        for key in (
            'ber', 'raw_throughput_bps', 'effective_throughput_bps',
            'spectral_efficiency_bps_per_hz',
            'mimo_channel_nmse', 'mimo_mean_condition_number',
        ):
            payload[key] = _sanitize(metrics.get(key))

        # 也尝试从 result 顶层兜底
        if payload['ber'] is None:
            payload['ber'] = _sanitize(result.get('ber'))
            for key in ('raw_throughput_bps', 'effective_throughput_bps',
                        'spectral_efficiency_bps_per_hz'):
                if payload[key] is None:
                    payload[key] = _sanitize(result.get(key))

        with open(project_path / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_transmitter_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.clf(); plt.close('all')

        tx_signal_dict = result.get('tx_signal', {})
        tx_signal = _plot_stream(tx_signal_dict.get('signal_stream', []))
        if len(tx_signal) == 0:
            print("发射信号为空，跳过图表生成")
            return

        params = result.get('params')
        sps = 4
        preamble_len_sym = 3328
        if params is not None:
            sps = int(params.get("oversampling", 4) or 4)
            preamble_len_sym = 3328 if params.get("Preamble_type", "short") == "short" else 5120

        # payload only, normalize
        pskip = preamble_len_sym * sps
        payload = tx_signal[pskip:] if len(tx_signal) > pskip else tx_signal
        if len(payload) == 0:
            payload = tx_signal
        payload = payload / (np.sqrt(np.mean(np.abs(payload)**2)) + 1e-15)
        fs = tx_signal_dict.get("sample_rate_Hz", 1.0)

        color = '#2C68B4'
        mode = str(params.get("link_mode", "sc-fde")).upper() if params is not None else "SC-FDE"

        # —— publication style ——
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
            "axes.unicode_minus": False, "axes.linewidth": 0.8,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 4, "ytick.major.size": 4,
            "axes.labelsize": 11, "legend.fontsize": 9,
            "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
        })

        # —— Fig 1: Time envelope ——
        fig, ax = plt.subplots(figsize=(6, 3))
        n_show = min(600, len(payload) // sps)
        env = np.abs(payload[:n_show * sps])
        t = np.arange(len(env)) / sps
        ax.plot(t, env, color=color, lw=0.6)
        ax.axhline(y=1.0, color="gray", ls="--", lw=0.5, alpha=0.5, label="mean")
        ax.set_xlabel("Symbol index"); ax.set_ylabel("|Amplitude|")
        ax.set_title(f"{mode} time envelope", fontsize=10, pad=4)
        ax.legend(fontsize=8); ax.set_xlim([0, n_show])
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # —— Fig 2: Spectrum ——
        fig, ax = plt.subplots(figsize=(6, 3.5))
        nfft = 2048
        seg = payload[:nfft]
        spec = np.fft.fftshift(np.fft.fft(seg, nfft))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)) if fs > 0 else np.arange(nfft)
        ax.plot(freq/1e9, 20*np.log10(np.abs(spec) + 1e-15), color=color, lw=0.4)
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"{mode} spectrum", fontsize=10, pad=4)
        bw = 0.5 * fs; ax.set_xlim([-bw/1e9, bw/1e9]); ax.set_ylim([-20, 60])
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # —— Fig 3: PAPR CCDF ——
        fig, ax = plt.subplots(figsize=(6, 4))
        blk_len = 480 * sps  # per SC-FDE block
        pv = []
        for b in range(0, len(payload) - blk_len, blk_len):
            blk = payload[b:b+blk_len]
            pv.append(10*np.log10(np.max(np.abs(blk)**2)/(np.mean(np.abs(blk)**2)+1e-15)))
        if pv:
            pv = np.sort(pv)
            ccdf = 1.0 - np.arange(len(pv)) / len(pv)
            ax.semilogy(pv, ccdf, color=color, lw=1.2)
            idx1e3 = np.searchsorted(ccdf, 1e-3)
            if idx1e3 < len(pv):
                p3 = pv[idx1e3]
                ax.axvline(x=p3, color=color, ls=":", lw=0.8, alpha=0.6)
                ax.annotate(f"{p3:.1f} dB @ 1e-3", xy=(p3, 1e-3), xytext=(p3+1.5, 3e-3),
                            fontsize=9, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.6))
        ax.set_xlabel("PAPR (dB)"); ax.set_ylabel("CCDF")
        ax.set_title(f"{mode} PAPR CCDF", fontsize=10, pad=4)
        ax.set_ylim([1e-3, 1])
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_papr.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # —— Fig 4: Amplitude histogram ——
        fig, ax = plt.subplots(figsize=(6, 4))
        amps = np.abs(payload)
        ax.hist(amps, bins=80, density=True, histtype="step", color=color, lw=1.2)
        if "ofdm" in str(mode).lower():
            sigma = np.sqrt(np.mean(amps**2) / 2)
            x_r = np.linspace(0, np.max(amps), 200)
            ax.plot(x_r, x_r/sigma**2*np.exp(-x_r**2/(2*sigma**2)),
                    "--", color="gray", lw=0.8, alpha=0.7, label="Rayleigh ref.")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("|Amplitude|"); ax.set_ylabel("Probability density")
        ax.set_title(f"{mode} amplitude distribution", fontsize=10, pad=4)
        fig.tight_layout()
        fig.savefig(output_dir / "transmitter_histogram.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _save_channel_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.clf(); plt.close('all')
        plt.rcParams.update({
            "font.family": "serif", "axes.unicode_minus": False,
            "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 4, "ytick.major.size": 4,
            "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
        })
        rx_signal_dict = result.get('rx_signal', {})
        rx_signal = _plot_stream(rx_signal_dict.get('signal_stream', []))
        if len(rx_signal) == 0:
            return
        params = result.get('params')
        sps = params.get("oversampling", 4) if params else 4
        fs = rx_signal_dict.get("sample_rate_Hz", 1.0)
        color = '#D95F02'
        n = len(rx_signal)

        # time domain
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.arange(min(200, n)), np.real(rx_signal[:200]), color=color, lw=0.5)
        ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude (Real)")
        ax.set_title("RX signal — time domain", fontsize=10, pad=4)
        fig.tight_layout(); fig.savefig(output_dir / "channel_time.png", dpi=300, bbox_inches='tight'); plt.close(fig)

        # spectrum
        fig, ax = plt.subplots(figsize=(6, 3.5))
        nfft = 2048; seg = rx_signal[:nfft]
        spec = np.fft.fftshift(np.fft.fft(seg, nfft))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs)) if fs > 0 else np.arange(nfft)
        ax.plot(freq/1e9, 20*np.log10(np.abs(spec)+1e-15), color=color, lw=0.4)
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
        ax.set_title("RX signal — spectrum", fontsize=10, pad=4)
        bw = 0.5*fs; ax.set_xlim([-bw/1e9, bw/1e9]); ax.set_ylim([-20, 60])
        fig.tight_layout(); fig.savefig(output_dir / "channel_spectrum.png", dpi=300, bbox_inches='tight'); plt.close(fig)

        # constellation (skip preamble)
        preamble_len_sym = 3328 if params.get("Preamble_type", "short") == "short" else 5120
        pskip = preamble_len_sym * sps
        payload = rx_signal[pskip:] if len(rx_signal) > pskip else rx_signal
        fig, ax = plt.subplots(figsize=(5, 5))
        n_cst = min(3000, len(payload))
        ax.scatter(np.real(payload[:n_cst]), np.imag(payload[:n_cst]), s=3, alpha=0.5, c=color)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_title("RX constellation (payload)", fontsize=10, pad=4)
        ax.axis("equal")
        fig.tight_layout(); fig.savefig(output_dir / "channel_constellation.png", dpi=300, bbox_inches='tight'); plt.close(fig)

        # power distribution
        fig, ax = plt.subplots(figsize=(6, 3.5))
        pwr_data = payload if len(payload) > 0 else rx_signal
        ax.plot(np.arange(min(500, len(pwr_data))), np.abs(pwr_data[:500])**2, color=color, lw=0.5)
        ax.set_xlabel("Sample"); ax.set_ylabel("Power")
        ax.set_title("RX signal power (payload)", fontsize=10, pad=4)
        fig.tight_layout(); fig.savefig(output_dir / "channel_power.png", dpi=300, bbox_inches='tight'); plt.close(fig)

    def _save_matched_filter_plots(self, result, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.clf(); plt.close('all')
        plt.rcParams.update({
            "font.family": "serif", "axes.unicode_minus": False,
            "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 4, "ytick.major.size": 4,
            "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
        })
        params = result.get('params')
        sps = params.get("oversampling", 4) if params else 4
        preamble_len_sym = 3328 if params.get("Preamble_type", "short") == "short" else 5120
        pskip = preamble_len_sym * sps
        color_tx = '#2C68B4'; color_rx = '#D95F02'

        # ---- matched filter spectrum ----
        mf_dict = result.get('rx_matched') or {}
        mf_sig = _plot_stream(mf_dict.get('signal_stream', []))
        if len(mf_sig) > 0:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            nfft = 2048; seg = mf_sig[pskip:pskip+nfft] if len(mf_sig) > pskip else mf_sig[:nfft]
            spec = np.fft.fftshift(np.fft.fft(np.resize(seg, nfft), nfft))
            ax.plot(20*np.log10(np.abs(spec)+1e-15), color=color_rx, lw=0.4)
            ax.set_xlabel("Frequency bin"); ax.set_ylabel("Magnitude (dB)")
            ax.set_title("Matched filter output — spectrum", fontsize=10, pad=4)
            ax.set_ylim([-20, 60])
            fig.tight_layout(); fig.savefig(output_dir / "rx_mf_spectrum.png", dpi=300, bbox_inches='tight'); plt.close(fig)

        # ---- equalized constellation ----
        # 判决导向 IQ 补偿位于均衡之后。开启补偿时应展示补偿后的
        # 星座；未开启补偿（值为 None）时再回退到原始均衡结果。
        eq_dict = select_rx_constellation_data(result)
        eq_sig = _plot_stream(eq_dict.get('signal_stream', []))
        if len(eq_sig) > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            n_cst = min(3000, len(eq_sig))
            ax.scatter(np.real(eq_sig[:n_cst]), np.imag(eq_sig[:n_cst]), s=3, alpha=0.5, c=color_tx)
            ax.set_xlabel("I"); ax.set_ylabel("Q")
            ax.set_title("Equalized constellation", fontsize=10, pad=4)
            ax.axis("equal")
            fig.tight_layout(); fig.savefig(output_dir / "rx_eq_constellation.png", dpi=300, bbox_inches='tight'); plt.close(fig)

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
