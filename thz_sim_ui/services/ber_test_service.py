"""功能测试页的全链路 BER 扫描服务。"""
from __future__ import annotations

import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Callable, List, Optional

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

# 误码率测试的四种测试方式：前三种对应 batch 批量对比工程（各含两条链路），
# 最后一种对应 single 单方案工程（单链路）。前三种的默认 SNR 扫描范围取自
# batch 工程设置；0.5Tbps 方式默认 35～45 dB、步进 1 dB。
# keep_channel=True 时保留工程原信道设置（0.5Tbps 为实测 PDP 瑞利信道），
# 其余对比方式统一采用仅 AWGN 口径。
BER_TEST_MODES = (
    {"key": "modulation_compare", "label": "对比调制方式",
     "batch": "(64QAM_vs_16QAM)_OFDM_LDPC"},
    {"key": "codec_compare", "label": "对比编码方式",
     "batch": "64QAM_OFDM_(LDPC_vs_RS)"},
    {"key": "waveform_compare", "label": "对比波形方式",
     "batch": "64QAM_(OFDM_vs_SC)_LDPC"},
    {"key": "tbps05", "label": "0.5Tbps方式",
     "single": "256QAM_MIMO_OFDM_AWGN_LDPC", "keep_channel": True},
)

_PROJECTS_ROOT = Path(__file__).resolve().parents[2] / "projects"

# 0.5Tbps 方式的 MIMO 信道口径：实测确定性回放。逐试次原样回放实测 2×2 CIR，
# 使各 SNR 点只对噪声做统计平均；pdp_rayleigh 会在每次试次间引入额外的
# 信道随机实现，不利于定点 BER 评估。
BER_MEASURED_CHANNEL_MODE = "deterministic"


def _apply_ber_channel_policy(config: dict) -> dict:
    """按 BER 测试口径修正保留工程信道的链路：实测信道固定为确定性回放。"""
    channel = config.get("_channel_params")
    if isinstance(channel, dict) and channel.get("enable_multipath") \
            and str(channel.get("multipath_source", "")).lower() == "measured":
        channel["measured_channel_mode"] = BER_MEASURED_CHANNEL_MODE
    return config


def get_ber_mode(mode_key: str) -> dict:
    """按 key 返回测试方式定义。"""
    for mode in BER_TEST_MODES:
        if mode["key"] == str(mode_key):
            return mode
    raise ValueError(f"未知误码率测试方式：{mode_key}")


def load_ber_project_config(project_name: str) -> dict:
    """加载 single 工程的 config.json。"""
    path = _PROJECTS_ROOT / "single" / str(project_name) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"关联工程配置不存在：{path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_ber_batch_config(batch_name: str) -> dict:
    """加载 batch 工程的 batch_config.json。"""
    path = _PROJECTS_ROOT / "batch" / str(batch_name) / "batch_config.json"
    if not path.exists():
        raise FileNotFoundError(f"批量对比配置不存在：{path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def ber_mode_schemes(mode_key: str) -> List[dict]:
    """返回测试方式下的链路配置列表（工程名、结果标签、config.json 内容）。

    返回的 config 已按 BER 测试口径修正（实测信道固定为确定性回放）。
    """
    mode = get_ber_mode(mode_key)
    if "single" in mode:
        config = _apply_ber_channel_policy(load_ber_project_config(mode["single"]))
        return [{
            "project": mode["single"],
            "label": str(config.get("工程名称") or mode["single"]),
            "config": config,
        }]
    batch = load_ber_batch_config(mode["batch"])
    schemes = []
    for scheme in batch.get("配置方案", []):
        project = str(scheme.get("工程", ""))
        config = _apply_ber_channel_policy(load_ber_project_config(project))
        schemes.append({
            "project": project,
            "label": str(scheme.get("结果标签") or config.get("工程名称") or project),
            "config": config,
        })
    return schemes


def ber_mode_snr_default(mode_key: str) -> tuple:
    """返回测试方式的默认 SNR 扫描范围 (min, max, step)。"""
    mode = get_ber_mode(mode_key)
    if "batch" in mode:
        batch = load_ber_batch_config(mode["batch"])
        return (float(batch["SNR最小值"]), float(batch["SNR最大值"]),
                float(batch["SNR步长"]))
    return (35.0, 45.0, 1.0)


def map_ber_project_config(config: dict, keep_channel: bool = False) -> dict:
    """把工程 config.json 映射为 PHYParams 键值。

    keep_channel=False 时信道段统一替换为仅 AWGN（对比方式口径）；
    keep_channel=True 时保留工程原信道设置（如 0.5Tbps 的实测 PDP 瑞利信道）。
    """
    from thz_sim_ui.services.backend import BackendService

    normalized = dict(config)
    if not keep_channel:
        normalized["_channel_params"] = {
            "enable_awgn": True, "SNRdB": 42.0,
            "noise_temperature": None, "noise_figure_db": None,
            "enable_multipath": False, "multipath_source": "simulated",
            "enable_cfo": False, "ppm": 1.0,
            "enable_phase_noise": False, "enable_iq_imbalance": False,
            "enable_pa": False,
        }
    else:
        normalized["_channel_params"] = dict(normalized.get("_channel_params") or {})
        _apply_ber_channel_policy(normalized)
    return BackendService.map_ui_params_to_phy_params(normalized)


def build_mode_configs(mode_key: str, snr_min: float, snr_max: float,
                       snr_step: float) -> List[dict]:
    """按测试方式与 SNR 扫描范围构造扫描配置列表。"""
    mode = get_ber_mode(mode_key)
    keep_channel = bool(mode.get("keep_channel", False))
    snr_min, snr_max, snr_step = float(snr_min), float(snr_max), float(snr_step)
    if snr_step <= 0:
        raise ValueError("SNR 步长必须大于 0")
    if snr_max < snr_min:
        raise ValueError("SNR 最大值不能小于最小值")
    values = [round(v, 4) for v in
              np.arange(snr_min, snr_max + snr_step / 2.0, snr_step)]
    if not values:
        raise ValueError("SNR 扫描范围内至少需要一个扫描点")
    configs = []
    for index, scheme in enumerate(ber_mode_schemes(mode_key)):
        configs.append({
            "key": f"{mode_key}:{index}",
            "label": scheme["label"],
            "mapped": map_ber_project_config(scheme["config"], keep_channel),
            "keep_channel": keep_channel,
            "snr_values": tuple(values),
        })
    return configs


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
    """构造一条无额外损伤的 AWGN 链路。

    工程方式（含 mapped 字段）：以 batch/single 工程的 config.json 映射结果
    为基础，仅叠加 AWGN 与扫描 SNR；旧六模式（mode/ncbps/code_type 字段）
    保持原有构造逻辑。
    """
    from params.PHYParams import PHYParams

    params = PHYParams()
    mapped = config.get("mapped")
    if mapped:
        valid_keys = set(params._params) | PHYParams._LEGACY_MULTIPATH_KEYS
        params._params.update({
            key: value for key, value in mapped.items()
            if value is not None and key in valid_keys
        })
        keep_channel = bool(config.get("keep_channel", False))
        overrides = dict(
            data_source="PRBS", sample_length=None,
            random_seed=int(seed), seed_strategy="固定种子",
            SNRdB=float(snr_db),
            enable_awgn=True, enable_phase_noise=False,
            enable_cfo=False, enable_cfo_compensation=False,
            enable_iq_imbalance=False, enable_iq_compensation=False,
            enable_pa=False, enable_channel_equalization=True,
        )
        if not keep_channel:
            # 仅 AWGN 口径：关闭多径；MIMO 使用恒等信道；固定短时长保证
            # 每次试次仅组装一帧，进度粒度细且可随时停止（与旧六模式的
            # 扫描口径一致）。
            overrides["enable_multipath"] = False
            overrides["duration"] = 1e-7
        # keep_channel（0.5Tbps 方式）：duration 沿用工程配置的时长
        # （256QAM_MIMO_OFDM_AWGN_LDPC 为 0.04 ms），每次试次的数据量
        # 与单方案运行一致。
        params.update(**overrides)
        if params.get("enable_mimo") and not keep_channel:
            params.update(mimo_channel_model="identity")
        return params

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


def _plot_results(results: list[dict], target_ber: float,
                  x_key: str = "SNRdB",
                  xlabel: str = "SNR (dB)",
                  title: Optional[str] = None) -> tuple[bytes, object]:
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
        points = sorted(result.get("points", []),
                        key=lambda value: float(value.get(x_key, float("nan"))))
        if not points:
            continue
        x = np.asarray([point[x_key] for point in points], dtype=float)
        # 实测非零点与完成置信验证的零误码点组成主曲线。仅做过短预扫描的
        # 零误码点不与主曲线连接，避免其较宽置信上限制造“BER 回升”假象。
        y = np.asarray([
            point["BER"] if point["total_errors"] else
            (point["ber_upper_95"] if point.get("passes") else np.nan)
            for point in points
        ], dtype=float)
        ax.semilogy(x, y, color=color, marker=marker, lw=1.4, ms=5.5,
                    label=comparison_title([result["label"]]))
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    if title is None:
        title = comparison_title([item.get("label") for item in results])
    ax.set_title(title)
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


def comparison_title(labels) -> str:
    """从结果标签推导「对比的量」标题。

    单链路直接使用链路标签；多链路先去掉标签中的编码维度括号（如
    (1056,1440)、(11,15)），再按下划线分段取公共前后缀，把各标签独有的
    中间差异段按「(A vs B)」拼接，例如 64QAM_LDPC_(OFDM vs SC)、
    (64QAM vs 16QAM)_LDPC_OFDM、64QAM_(LDPC vs RS)_OFDM。
    标签中的 AWGN 字样一律剔除：0.5Tbps 方式实际使用实测确定性回放信道，
    工程名残留的 AWGN 会误导图题（如 256QAM_MIMO_OFDM_LDPC）。
    """
    import re

    labels = [str(label).strip() for label in labels if str(label).strip()]
    if not labels:
        return "全链路 BER 曲线"
    parts = [
        [part for part in re.sub(r"\([^()]*\)", "", label).split("_")
         if part.upper() != "AWGN"]
        for label in labels
    ]
    parts = [part for part in parts if part]
    if not parts:
        return "全链路 BER 曲线"
    if len(parts) == 1:
        return "_".join(parts[0])

    prefix: List[str] = []
    for group in zip(*parts):
        if len(set(group)) == 1:
            prefix.append(group[0])
        else:
            break
    suffix: List[str] = []
    for group in zip(*(part[len(prefix):][::-1] for part in parts)):
        if len(set(group)) == 1:
            suffix.append(group[0])
        else:
            break
    suffix.reverse()

    diffs = []
    for part in parts:
        middle = part[len(prefix):]
        middle = middle[:-len(suffix)] if suffix else middle
        diffs.append("_".join(middle))
    if not any(diffs):
        return "多链路 BER 曲线对比"
    pieces = []
    if prefix:
        pieces.append("_".join(prefix))
    pieces.append("(" + " vs ".join(diffs) + ")")
    if suffix:
        pieces.append("_".join(suffix))
    return "_".join(pieces)


def generate_saved_curves(result_json_path) -> List[str]:
    """根据已保存的 ber_results.json 重画两幅 BER 曲线并保存 PNG。

    输出到 JSON 同目录下的 ber_curves.png（SNR 横轴）与
    ber_curves_ebn0.png（Eb/N0 横轴）；两图标题统一为对比的量
    （由各链路标签推导），纵轴统一为 BER。返回输出文件路径列表。
    """
    path = Path(result_json_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    results = document.get("results") or []
    target_ber = float(document.get("target_BER", TARGET_BER))
    outputs = []
    for name, x_key, xlabel in (
        ("ber_curves.png", "SNRdB", "SNR (dB)"),
        ("ber_curves_ebn0.png", "EbN0_dB", "Eb/N0 (dB)"),
    ):
        _, figure = _plot_results(results, target_ber,
                                  x_key=x_key, xlabel=xlabel)
        output = path.with_name(name)
        figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.append(str(output))
    return outputs


def run_ber_test(random_seed: int = 2026, quick_max_bits: int = 300_000,
                 quick_min_errors: int = 100, target_ber: float = TARGET_BER,
                 confidence: float = CONFIDENCE, output_root: Optional[str] = None,
                 save_artifacts: bool = True,
                 progress_callback: Optional[Callable[[dict], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 configs: Optional[tuple[dict, ...]] = None,
                 mode_key: Optional[str] = None,
                 snr_min: float = 14.0, snr_max: float = 23.0, snr_step: float = 1.0,
                 frame_runner: Callable[[dict, float, int], dict] = run_ber_frame) -> dict:
    """运行全链路 BER 曲线和零误码置信验证。

    未显式传入 configs 时按测试方式 mode_key（默认「对比调制方式」）读取
    batch/single 工程配置，并按 SNR 扫描范围生成配置列表；显式传入
    configs 时保持旧行为（默认六模式）。
    """
    if int(quick_max_bits) <= 0 or int(quick_min_errors) <= 0:
        raise ValueError("预扫描比特数和最少错误比特必须大于 0")
    verification_bits = required_zero_error_bits(target_ber, confidence)
    mode_label = ""
    if configs:
        selected_configs = tuple(configs)
    elif mode_key is not None:
        selected_configs = tuple(build_mode_configs(
            str(mode_key), float(snr_min), float(snr_max), float(snr_step)))
        mode_label = get_ber_mode(mode_key)["label"]
    else:
        selected_configs = tuple(BER_CONFIGS)
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
    _, ebn0_figure = _plot_results(results, target_ber,
                                   x_key="EbN0_dB", xlabel="Eb/N0 (dB)")
    artifact_paths = {}
    if save_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / "ber_curves.png"
        figure.savefig(image_path, dpi=300, bbox_inches="tight", facecolor="white")
        ebn0_path = run_dir / "ber_curves_ebn0.png"
        ebn0_figure.savefig(ebn0_path, dpi=300, bbox_inches="tight", facecolor="white")
        artifact_paths = {"result_json": str(run_dir / "ber_results.json"),
                          "curve_png": str(image_path),
                          "ebn0_curve_png": str(ebn0_path)}
        checkpoint()
    plt.close(figure)
    plt.close(ebn0_figure)

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
        {"label": "达标链路", "value": f"{passed_count}/{len(selected_configs)}"},
    ]
    if mode_label:
        summary.insert(0, {"label": "测试方式", "value": mode_label})
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
        "plots": [{"title": "全链路 BER 曲线", "png": png}],
        "table": {"columns": ["模式", "SNR/dB", "Eb/N0/dB", "误码数", "比特数",
                              "实测 BER", "95% 上限", "运行次数", "判定"], "rows": rows},
        "data": {"target_BER": target_ber, "confidence": confidence,
                 "minimum_verification_bits": verification_bits,
                 "results": _serializable_results(results), "artifact_paths": artifact_paths},
    }
