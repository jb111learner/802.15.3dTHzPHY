"""功能测试页的四模式全链路 BER 扫描服务。"""
from __future__ import annotations

import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


TARGET_BER = 1e-6
CONFIDENCE = 0.95
VERIFICATION_BITS = math.ceil(-math.log(1.0 - CONFIDENCE) / TARGET_BER)

# 顺序与需求保持一致。界面沿用项目的 (k,n) 标法。
BER_CONFIGS = (
    {"key": "64qam_ofdm", "label": "64QAM OFDM LDPC(1344,1440)", "mode": "ofdm", "ncbps": 6, "code_type": "LDPC",
     "snr_values": tuple(range(20, 27))},
    {"key": "16qam_ofdm", "label": "16QAM OFDM LDPC(1344,1440)", "mode": "ofdm", "ncbps": 4, "code_type": "LDPC",
     "snr_values": tuple(range(14, 21))},
    {"key": "64qam_sc", "label": "64QAM SC LDPC(1344,1440)", "mode": "sc-fde", "ncbps": 6, "code_type": "LDPC",
     "snr_values": tuple(range(17, 24))},
    {"key": "16qam_sc", "label": "16QAM SC LDPC(1344,1440)", "mode": "sc-fde", "ncbps": 4, "code_type": "LDPC",
     "snr_values": tuple(range(11, 18))},
    {"key": "64qam_ofdm_rs", "label": "64QAM OFDM RS(11,15)", "mode": "ofdm", "ncbps": 6, "code_type": "RS",
     "rs_nsym": 4, "rs_c_exp": 4, "rs_packet_size": 11,
     "snr_values": tuple(range(12, 20))},
    {"key": "64qam_sc_rs", "label": "64QAM SC RS(192,255)", "mode": "sc-fde", "ncbps": 6, "code_type": "RS",
     "rs_nsym": 63, "rs_c_exp": 8, "rs_packet_size": 192,
     "snr_values": tuple(range(9, 17))},
)


def required_zero_error_bits(target_ber: float = TARGET_BER,
                             confidence: float = CONFIDENCE) -> int:
    """零误码时使单侧置信上限不超过目标 BER 所需的样本数。"""
    if not 0.0 < float(target_ber) < 1.0:
        raise ValueError("目标 BER 必须位于 (0, 1) 内")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("置信度必须位于 (0, 1) 内")
    return math.ceil(-math.log(1.0 - float(confidence)) / float(target_ber))


def one_sided_ber_upper(errors: int, bits: int,
                        confidence: float = CONFIDENCE) -> float:
    """计算 BER 单侧上限；零误码用精确式，非零误码用 Wilson 上限。"""
    errors, bits = int(errors), int(bits)
    if bits <= 0 or errors < 0 or errors > bits:
        return float("nan")
    if errors == 0:
        return -math.log(1.0 - float(confidence)) / bits
    p = errors / bits
    z = NormalDist().inv_cdf(float(confidence))
    z2 = z * z
    centre = p + z2 / (2.0 * bits)
    radius = z * math.sqrt(p * (1.0 - p) / bits + z2 / (4.0 * bits * bits))
    return min(1.0, (centre + radius) / (1.0 + z2 / bits))


def build_ber_params(config: dict, snr_db: float, seed: int):
    """构造一条无额外损伤的 IEEE 802.15.3d LDPC AWGN 链路。"""
    from params.PHYParams import PHYParams

    params = PHYParams()
    code_type = str(config.get("code_type", "LDPC")).upper()
    coding = ({
        "code_type": "RS", "rs_nsym": int(config["rs_nsym"]),
        "rs_c_exp": int(config["rs_c_exp"]),
        "rs_packet_size": int(config["rs_packet_size"]), "decode_mode": "hard",
    } if code_type == "RS" else {
        "code_type": "LDPC", "ldpc_matrix_type": "ieee802153d_1440",
        "ldpc_standard_rate": "14/15", "ldpc_n": 1440, "ldpc_k": 1344,
    })
    params.update(
        data_source="PRBS", duration=1e-7, sample_length=None,
        random_seed=int(seed), seed_strategy="固定种子",
        link_mode=config["mode"], NCBPS=int(config["ncbps"]), MCS=1,
        SNRdB=float(snr_db),
        enable_awgn=True, enable_multipath=False, enable_phase_noise=False,
        enable_cfo=False, enable_cfo_compensation=False,
        enable_iq_imbalance=False, enable_iq_compensation=False,
        enable_pa=False, enable_channel_equalization=True, **coding,
    )
    return params


def run_ber_frame(config: dict, snr_db: float, seed: int) -> dict:
    """运行一个独立全链路帧并返回可累计统计量。"""
    import contextlib

    from channel.THzChannel import THzChannel
    from receiver.THzReceiver import THzReceiver
    from transmitter.THzTransmitter import THzTransmitter

    params = build_ber_params(config, snr_db, seed)
    # AWGN 模块仍使用 numpy 全局随机源；由配置/SNR/运行号派生固定噪声种子。
    np.random.seed((int(seed) * 1009 + int(round(float(snr_db) * 100))) % (2**32))
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        transmitter = THzTransmitter(params)
        transmitter.run()
        received = THzChannel(params).run(transmitter.tx_signal_dict)
        decoded = THzReceiver(params, transmitter).run(received)

    tx_bits = np.asarray(transmitter.data_bits_dict["signal_stream"]).ravel()
    rx_bits = np.asarray(decoded["signal_stream"]).ravel()
    bits = min(len(tx_bits), len(rx_bits))
    errors = int(np.count_nonzero(tx_bits[:bits] != rx_bits[:bits]))
    waveform_duration = float(transmitter.tx_signal_dict.get("duration_seconds", 0.0))
    raw_rate = bits / waveform_duration if waveform_duration > 0 else float("nan")
    bandwidth = float(params.get("bandwidth") or 0.0)
    spectral_efficiency = raw_rate / bandwidth if bandwidth > 0 else float("nan")
    ebn0 = (float(snr_db) - 10.0 * math.log10(spectral_efficiency)
            if spectral_efficiency > 0 else float("nan"))
    return {
        "bits": bits, "errors": errors, "elapsed_seconds": time.perf_counter() - started,
        "waveform_duration_seconds": waveform_duration, "EbN0_dB": ebn0,
    }


def _point(config: dict, snr_db: float, base_seed: int,
           stop: Callable[[int, int], bool], start: Optional[dict],
           progress_callback: Optional[Callable[[dict], None]],
           should_cancel: Optional[Callable[[], bool]], config_index: int,
           phase: str, confidence: float,
           frame_runner: Callable[[dict, float, int], dict]) -> tuple[dict, bool]:
    point = dict(start or {})
    total_bits = int(point.get("total_bits", 0))
    total_errors = int(point.get("total_errors", 0))
    runs = int(point.get("independent_runs", 0))
    elapsed = float(point.get("elapsed_seconds", 0.0))
    ebn0_values = list(point.get("_ebn0_values", []))
    while stop(total_bits, total_errors):
        if should_cancel is not None and should_cancel():
            break
        trial_seed = int(base_seed) + config_index * 1_000_000 + int(round(snr_db * 1000)) * 100 + runs
        trial = frame_runner(config, snr_db, trial_seed)
        total_bits += int(trial["bits"])
        total_errors += int(trial["errors"])
        elapsed += float(trial.get("elapsed_seconds", 0.0))
        ebn0_values.append(float(trial.get("EbN0_dB", float("nan"))))
        runs += 1
        if progress_callback is not None:
            progress_callback({
                "config_index": config_index, "config_label": config["label"],
                "total_configs": int(config.get("_total_configs", 1)),
                "phase": phase, "snr_db": float(snr_db), "total_bits": total_bits,
                "total_errors": total_errors, "independent_runs": runs,
            })
    cancelled = should_cancel is not None and should_cancel()
    ber = total_errors / total_bits if total_bits else float("nan")
    upper = one_sided_ber_upper(total_errors, total_bits, confidence) if total_bits else float("nan")
    return ({
        "SNRdB": float(snr_db),
        "EbN0_dB": float(np.nanmean(ebn0_values)) if ebn0_values else float("nan"),
        "BER": ber, "ber_upper_95": upper, "total_bits": total_bits,
        "total_errors": total_errors, "independent_runs": runs,
        "elapsed_seconds": elapsed, "phase": phase, "verified": False,
        "passes": False, "_ebn0_values": ebn0_values,
    }, cancelled)


def _serializable_results(results: list[dict]) -> list[dict]:
    clean = []
    for config in results:
        item = {key: value for key, value in config.items() if key != "points"}
        item["points"] = [
            {key: value for key, value in point.items() if not key.startswith("_")}
            for point in config.get("points", [])
        ]
        clean.append(item)
    return clean


def _write_checkpoint(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _plot_results(results: list[dict], target_ber: float) -> tuple[bytes, object]:
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial Unicode MS"],
        "axes.unicode_minus": False, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    colors = ["#2C68B4", "#D95F02", "#3A9D3A", "#9467BD", "#00A6A6", "#8C6B4F"]
    markers = ["o", "s", "^", "D", "P", "X"]
    for result, color, marker in zip(results, colors, markers):
        points = sorted(result.get("points", []), key=lambda value: value["SNRdB"])
        if not points:
            continue
        x = np.asarray([point["SNRdB"] for point in points], dtype=float)
        # 实测非零点与完成置信验证的零误码点组成主曲线。仅做过短预扫描的
        # 零误码点不与主曲线连接，避免其较宽置信上限制造“BER 回升”假象。
        y = np.asarray([
            point["BER"] if point["total_errors"] else
            (point["ber_upper_95"] if point.get("passes") else np.nan)
            for point in points
        ], dtype=float)
        ax.semilogy(x, y, color=color, marker=marker, lw=1.4, ms=5.5,
                    label=result["label"])
        unverified_zero = np.asarray([
            point["total_errors"] == 0 and not point.get("passes") for point in points
        ])
        if np.any(unverified_zero):
            upper = np.asarray([point["ber_upper_95"] for point in points], dtype=float)
            ax.semilogy(x[unverified_zero], upper[unverified_zero],
                        linestyle="none", marker="v",
                        fillstyle="none", color=color, ms=7)
    ax.axhline(target_ber, color="#E5484D", ls="--", lw=1.2,
               label=f"目标 BER = {target_ber:.0e}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER / 零误码时的 95% 上限")
    ax.set_title(f"{len(results)} 模式全链路 BER 曲线（AWGN）")
    ax.grid(True, which="both")
    ax.set_ylim(1e-7, 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
    ax.legend(fontsize=8, ncol=2)
    ax.text(0.01, 0.018, "注：零误码达标点按 95% 单侧 BER 上限绘制",
            transform=ax.transAxes, fontsize=8, color="#596579")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return buffer.getvalue(), fig


def run_ber_test(random_seed: int = 2026, quick_max_bits: int = 300_000,
                 quick_min_errors: int = 100, target_ber: float = TARGET_BER,
                 confidence: float = CONFIDENCE, output_root: Optional[str] = None,
                 save_artifacts: bool = True,
                 progress_callback: Optional[Callable[[dict], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 configs: Optional[tuple[dict, ...]] = None,
                 frame_runner: Callable[[dict, float, int], dict] = run_ber_frame) -> dict:
    """运行四模式 BER 曲线和零误码置信验证。"""
    if int(quick_max_bits) <= 0 or int(quick_min_errors) <= 0:
        raise ValueError("预扫描比特数和最少错误比特必须大于 0")
    verification_bits = required_zero_error_bits(target_ber, confidence)
    selected_configs = tuple(configs or BER_CONFIGS)
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_root) if output_root else Path(__file__).resolve().parents[2] / "simulation_results" / "functional_ber"
    run_dir = root / timestamp
    results: list[dict] = []
    cancelled = False

    def checkpoint() -> None:
        if save_artifacts:
            _write_checkpoint(run_dir / "ber_results.json", {
                "target_BER": target_ber, "confidence": confidence,
                "minimum_verification_bits": verification_bits,
                "cancelled": cancelled, "results": _serializable_results(results),
            })

    for config_index, source in enumerate(selected_configs):
        config = dict(source)
        config["_total_configs"] = len(selected_configs)
        config_result = {"key": config["key"], "label": config["label"],
                         "points": [], "threshold_SNRdB": None, "passes": False}
        results.append(config_result)
        for point_index, snr_db in enumerate(config["snr_values"]):
            stop = lambda bits, errors: bits < int(quick_max_bits) and errors < int(quick_min_errors)
            point, cancelled = _point(
                config, snr_db, int(random_seed), stop, None, progress_callback,
                should_cancel, config_index, "预扫描", confidence, frame_runner)
            config_result["points"].append(point)
            if not cancelled and point["total_errors"] == 0:
                # 零误码预扫描只是候选，立即在同一点继续累计。候选通过后便
                # 结束该模式，避免继续产生统计量较弱、视觉上反而更高的上限点。
                verify_stop = lambda bits, errors: errors == 0 and bits < verification_bits
                verified, cancelled = _point(
                    config, point["SNRdB"], int(random_seed), verify_stop, point,
                    progress_callback, should_cancel, config_index, "达标验证",
                    confidence, frame_runner)
                point.update(verified)
                point["verified"] = point["total_bits"] >= verification_bits
                point["passes"] = bool(
                    point["verified"] and point["total_errors"] == 0
                    and point["ber_upper_95"] <= target_ber)
                if point["passes"]:
                    config_result["passes"] = True
                    config_result["threshold_SNRdB"] = point["SNRdB"]
            if progress_callback is not None:
                progress_callback({
                    "config_index": config_index, "config_label": config["label"],
                    "total_configs": len(selected_configs),
                    "phase": "完成扫描点", "snr_db": float(snr_db),
                    "point_index": point_index + 1, "point_count": len(config["snr_values"]),
                    "total_bits": point["total_bits"], "total_errors": point["total_errors"],
                })
            checkpoint()
            if cancelled or point["passes"]:
                break
        if cancelled:
            break
        checkpoint()

    png, figure = _plot_results(results, target_ber)
    artifact_paths = {}
    if save_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / "ber_curves.png"
        figure.savefig(image_path, dpi=300, bbox_inches="tight", facecolor="white")
        artifact_paths = {"result_json": str(run_dir / "ber_results.json"),
                          "curve_png": str(image_path)}
        checkpoint()
    plt.close(figure)

    rows = []
    for config_result in results:
        for point in sorted(config_result["points"], key=lambda value: value["SNRdB"]):
            status = "通过" if point["passes"] else ("样本不足" if point["total_errors"] == 0 else "未通过")
            rows.append([
                config_result["label"], f"{point['SNRdB']:.1f}", f"{point['EbN0_dB']:.3f}",
                str(point["total_errors"]), f"{point['total_bits']:,}", f"{point['BER']:.3e}",
                f"{point['ber_upper_95']:.3e}", str(point["independent_runs"]), status,
            ])
    completed = len(results) == len(selected_configs) and not cancelled
    passed_count = sum(bool(item["passes"]) for item in results)
    summary = [
        {"label": "目标与置信度", "value": f"BER ≤ {target_ber:.0e}，{confidence:.0%} 单侧置信"},
        {"label": "零误码验证比特数", "value": f"≥ {verification_bits:,} bit"},
        {"label": "达标模式", "value": f"{passed_count}/{len(selected_configs)}"},
    ]
    for item in results:
        value = (f"{item['threshold_SNRdB']:.1f} dB" if item["passes"] else
                 ("已停止" if cancelled else "扫描范围内未达标"))
        summary.append({"label": item["label"], "value": value})
    if artifact_paths:
        summary.append({"label": "结果目录", "value": str(run_dir)})
    return {
        "test_type": "ber", "ok": bool(completed and passed_count == len(selected_configs)),
        "cancelled": cancelled, "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "summary": summary, "checks": [],
        "plots": [{"title": "四模式 BER 曲线", "png": png}],
        "table": {"columns": ["模式", "SNR/dB", "Eb/N0/dB", "误码数", "比特数",
                              "实测 BER", "95% 上限", "运行次数", "判定"], "rows": rows},
        "data": {"target_BER": target_ber, "confidence": confidence,
                 "minimum_verification_bits": verification_bits,
                 "results": _serializable_results(results), "artifact_paths": artifact_paths},
    }
