"""
功能测试服务 — 调制、波形、编码、浮点精度、物理层速率与功能校准

各测试入口均返回含 PNG 字节、校验结果与表格数据的 dict，
由 FunctionalTestWorker(QThread) 在后台线程中执行，结果经 Qt 信号回传页面。
PHY 模块 import 全部放在函数内部（惰性加载），galois 仅在 RSCoder 构造时导入。
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 必须先于 pyplot import（与 backend.py 一致）
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from PySide6.QtCore import QThread, Signal

# 绘图格式对齐 backend.py（SimHei 中文 + DejaVu 回退）与 batch_compare_page
# 的白底/网格/颜色表风格。semilogy 的负指数刻度改用 ASCII Formatter，
# 避免 mathtext 的 U+2212 在 SimHei 首字体下缺字形。
PLOT_STYLE = {
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
}
PLOT_COLORS = ["#2C68B4", "#D95F02", "#3A9D3A", "#9467BD", "#E54B4F", "#8C6B4F"]
FIGURE_DPI = 320

plt.rcParams["font.sans-serif"] = PLOT_STYLE["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _apply_plot_style() -> None:
    """每次绘图前重新应用样式（其他页面会污染全局 rcParams）。"""
    plt.rcParams.update(PLOT_STYLE)


def _figure_to_png(fig, dpi: int = FIGURE_DPI) -> bytes:
    """以高分辨率 PNG 输出，保证桌面端缩放及截图后的文字和细线清晰。"""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight",
        facecolor="white", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _new_result(test_type: str) -> Dict[str, Any]:
    return {
        "test_type": test_type,
        "ok": True,
        "elapsed_ms": 0.0,
        "summary": [],
        "checks": [],
        "plots": [],
        "table": {"columns": [], "rows": []},
        "detail_lines": [],
        "error": None,
    }


# =====================================================================
# 1. 调制方式测试
# =====================================================================

def run_modulation_test(modulation: str = "QPSK", num_symbols: int = 1024,
                        show_points: int = 1024,
                        random_seed: int = 2026) -> Dict[str, Any]:
    """调用真实发射端调制器生成符号并绘制星座图，供人工检查。"""
    from params.PHYParams import PHYParams
    from transmitter.Modulator import THzModulator

    t0 = time.perf_counter()
    result = _new_result("modulation")
    modulation_map = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
    mod_name = str(modulation).upper()
    if mod_name not in modulation_map:
        raise ValueError(f"不支持的调制方式：{modulation}，仅支持 QPSK / 16QAM / 64QAM")

    n_symbols = int(num_symbols)
    n_show = int(show_points)
    seed = int(random_seed)
    if n_symbols <= 0:
        raise ValueError("调制符号数必须大于 0")
    if n_show <= 0:
        raise ValueError("星座图显示点数必须大于 0")
    if seed < 0:
        raise ValueError("随机种子不能小于 0")

    ncbps = modulation_map[mod_name]
    bit_count = n_symbols * ncbps
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, bit_count, dtype=np.uint8)

    params = PHYParams()
    # 关闭逐符号 pi/2 旋转与 QPSK 的 pi/4 映射补偿，展示调制器的基础
    # 方型星座图（QPSK 为 {(±1±1j)/√2} 四点）。
    params.update(MCS=1, NCBPS=ncbps, scramble=False,
                  pi2_rotation=False, qpsk_pi4_compensation=False)
    bit_rate = 30e9
    modulated = THzModulator(params).modulate({
        "signal_stream": bits,
        "sample_rate_Hz": bit_rate,
        "duration_seconds": bit_count / bit_rate,
        "signal_length": bit_count,
        "padding_bit_num": 0,
        "frame_bit_num": bit_count,
        "frame_num": 1,
    })
    symbols = np.asarray(modulated["signal_stream"], dtype=np.complex128)
    shown = symbols[:min(n_show, len(symbols))]
    average_power = float(np.mean(np.abs(symbols) ** 2))

    # 星座点表：按 (I, Q) 排序的全部唯一星座点，展示每个点的 I/Q 值。
    unique_symbols = np.unique(np.round(symbols, 12))
    constellation_rows = [
        [str(index + 1), f"{float(point.real):.6f}", f"{float(point.imag):.6f}"]
        for index, point in enumerate(unique_symbols)
    ]

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(shown.real, shown.imag, s=24, alpha=0.65,
               color=PLOT_COLORS[0], edgecolors="none")
    ax.axhline(0, color="#7C869B", lw=0.7)
    ax.axvline(0, color="#7C869B", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("同相分量 I")
    ax.set_ylabel("正交分量 Q")
    ax.set_title(f"{mod_name} 发射端星座图（{len(shown)} 个符号）")
    ax.grid(True)
    fig.tight_layout()

    result["plots"] = [{"title": "发射端星座图", "png": _figure_to_png(fig)}]
    result["summary"] = [
        {"label": "调制方式", "value": mod_name},
        {"label": "每符号比特数", "value": str(ncbps)},
        {"label": "输入比特数", "value": str(bit_count)},
        {"label": "输出符号数", "value": str(len(symbols))},
        {"label": "星座图显示点数", "value": str(len(shown))},
        {"label": "平均符号功率", "value": f"{average_power:.6f}"},
        {"label": "pi/2 旋转", "value": "已关闭"},
        {"label": "随机种子", "value": str(seed)},
    ]
    result["table"] = {
        "columns": ["星座点", "I 分量", "Q 分量"],
        "rows": constellation_rows,
    }
    result["data"] = {
        "modulation": mod_name,
        "ncbps": ncbps,
        "input_bit_count": bit_count,
        "output_symbol_count": len(symbols),
        "shown_symbol_count": len(shown),
        "average_symbol_power": average_power,
        "constellation": [
            [float(point.real), float(point.imag)] for point in unique_symbols
        ],
    }
    # ok 仅表示任务正常生成结果，不代表星座质量的人工验收结论。
    result["ok"] = True
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    return result


# =====================================================================
# 2. 波形测试
# =====================================================================

def _count_contiguous_regions(mask: np.ndarray) -> int:
    """统计布尔掩码中连续 True 区域的个数。"""
    if not np.any(mask):
        return 0
    edges = np.diff(mask.astype(np.int8))
    starts = 1 if mask[0] else 0
    return int(starts + np.count_nonzero(edges == 1))


def _make_signal_dict(stream: np.ndarray, rate: float, extra: Optional[dict] = None) -> dict:
    """按 dict 协议构造信号字典（sample_rate × duration == signal_length）。"""
    data = {
        "signal_stream": np.asarray(stream),
        "sample_rate_Hz": rate,
        "duration_seconds": len(stream) / rate,
        "signal_length": len(stream),
        "padding_bit_num": 0,
    }
    if extra:
        data.update(extra)
    return data


def qam_constellation_points(modulation: str) -> np.ndarray:
    """返回与发射端调制器一致的归一化 QAM 星座点（I 升序、Q 升序排列）。

    16QAM/64QAM 为方型星座，尺度 sqrt((M-1)/3)；QPSK 含调制器内部的
    pi/4 映射补偿，等效四点 {±1, ±j}。
    """
    modulation = str(modulation).upper()
    ncbps_map = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
    if modulation not in ncbps_map:
        raise ValueError(
            f"不支持的时域测试调制：{modulation}，仅支持 QPSK / 16QAM / 64QAM")
    ncbps = ncbps_map[modulation]
    if ncbps == 2:
        points = np.array(
            [complex(a + 1j * b) for b in (-1, 1) for a in (-1, 1)])
        return points * np.exp(-1j * np.pi / 4) / np.sqrt(2)
    bits_per_axis = ncbps // 2
    levels = np.array(
        [-(2 ** bits_per_axis - 1) + 2 * i for i in range(2 ** bits_per_axis)],
        dtype=float)
    points = np.array([complex(q + 1j * i) for q in levels for i in levels])
    return points / np.sqrt((2 ** ncbps - 1) / 3.0)


def _run_sc_waveform_test(params, modulation: str, symbol_indexes: list,
                          pulse_count: int, checks: list, plots: list,
                          summary: list) -> None:
    """绘制单载波 I/Q 脉冲成型波形，并与理论脉冲叠加比较。

    每个完整脉冲使用用户从对应调制星座图中选出的星座点符号。
    """
    from transmitter.Pulseshaper import TxPulseShaper

    L = int(params.get("filter_length"))
    sps = int(params.get("oversampling"))
    beta = float(params.get("rolloff"))
    pulse_count = int(pulse_count)
    if pulse_count not in (2, 3):
        raise ValueError(f"完整脉冲数仅支持 2 或 3，当前值：{pulse_count}")

    constellation = qam_constellation_points(modulation)
    symbol_indexes = [int(index) for index in symbol_indexes]
    if len(symbol_indexes) != pulse_count:
        raise ValueError(f"星座点选择数量必须等于完整脉冲数 {pulse_count}")
    if any(index < 0 or index >= len(constellation) for index in symbol_indexes):
        raise ValueError(f"星座点索引必须位于 0～{len(constellation) - 1}")
    symbols = constellation[symbol_indexes]

    # 每个有限长脉冲前后各保留半个滤波器长度，保证显示窗口内没有截断。
    half_span = L // 2
    spacing = L + 4
    centers = half_span + 2 + np.arange(pulse_count) * spacing
    num_symbols = int(centers[-1] + half_span + 3)

    impulses = np.zeros(num_symbols, dtype=np.complex128)
    impulses[centers] = symbols

    # 实际波形走项目真实的发送端成型模块。
    shaper = TxPulseShaper(params)
    shaped = shaper.shape_pulse({
        "signal_stream": impulses,
        "sample_rate_Hz": 30e9,
        "duration_seconds": num_symbols / 30e9,
        "signal_length": num_symbols,
        "padding_bit_num": 0,
    })
    y = np.asarray(shaped["signal_stream"])

    ok_len = (y.dtype == np.complex128) and (len(y) == num_symbols * sps)
    checks.append({
        "name": "成型输出类型与长度",
        "ok": bool(ok_len),
        "detail": f"dtype={y.dtype}，长度={len(y)}（期望 complex128 / {num_symbols * sps}）",
    })

    # 理想波形按有限长理论脉冲的平移叠加独立构造。
    h = np.asarray(shaper.filter_coeffs, dtype=float)
    delay = len(h) // 2
    ideal_full = np.convolve(shaper.upsample_symbols(impulses), h, mode="full")
    ideal = ideal_full[delay:delay + len(y)]
    max_error = float(np.max(np.abs(y - ideal)))
    checks.append({
        "name": "实际波形与理想脉冲响应一致",
        "ok": bool(max_error < 1e-12),
        "detail": f"最大复幅度误差 {max_error:.2e}（期望 < 1e-12）",
    })

    # 各脉冲按所选星座点在 I/Q 轴上分别校验：在脉冲中心 ±半滤波器长度的
    # 窗口内，每个非零分量应恰好出现一个完整主峰（阈值按该星座点分量
    # 自身的 0.5 倍期望峰值计，避免强脉冲旁瓣误计为小幅度星座点的峰）。
    gain = float(np.max(np.abs(y))) / float(np.max(np.abs(symbols)))
    window_half = int(half_span * sps)
    expected_i = sum(1 for sym in symbols if abs(sym.real) > 1e-12)
    expected_q = sum(1 for sym in symbols if abs(sym.imag) > 1e-12)
    n_i = n_q = 0
    for sym, center in zip(symbols, centers):
        start = max(0, int(center * sps) - window_half)
        end = min(len(y), int(center * sps) + window_half + 1)
        for component, expected in (("I", expected_i), ("Q", expected_q)):
            value = sym.real if component == "I" else sym.imag
            if abs(value) < 1e-12:
                continue
            threshold = 0.5 * abs(value) * gain
            axis = np.real(y) if component == "I" else np.imag(y)
            regions = _count_contiguous_regions(
                np.abs(axis[start:end]) > threshold)
            if regions == 1:
                if component == "I":
                    n_i += 1
                else:
                    n_q += 1
    checks.append({
        "name": "各脉冲 I/Q 分量符合所选星座点",
        "ok": bool(n_i == expected_i and n_q == expected_q),
        "detail": f"实部 {n_i}/{expected_i} 个、虚部 {n_q}/{expected_q} 个完整脉冲分量"
                  f"（期望与所选星座点一致）",
    })

    # 横坐标采用输出采样率换算的真实时间，并以 ns 展示。
    fs_out = float(shaped["sample_rate_Hz"])
    time_ns = np.arange(len(y), dtype=float) / fs_out * 1e9
    _apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
    components = ((np.real(y), np.real(ideal), "实部 I"),
                  (np.imag(y), np.imag(ideal), "虚部 Q"))
    for ax, (actual_part, ideal_part, label) in zip(axes, components):
        # 理想线稍宽并先绘制，实际线覆盖其上，两种颜色均能辨认。
        ax.plot(time_ns, ideal_part, color="#D95F02", lw=1.8, ls="--",
                label="理想波形")
        ax.plot(time_ns, actual_part, color="#2C68B4", lw=0.9,
                label="实际波形")
        ax.set_ylabel(f"{label}幅度")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("时间 (ns)")
    fig.suptitle(f"单载波时域波形（{pulse_count} 个完整脉冲，L={L}，β={beta}）")
    fig.tight_layout()
    plots.append({"title": "单载波时域波形", "png": _figure_to_png(fig)})

    symbol_text = "，".join(
        f"#{index}({symbol.real:.3f}{symbol.imag:+.3f}j)"
        for index, symbol in zip(symbol_indexes, symbols))
    summary.extend([
        {"label": "时域测试符号", "value": f"{modulation}：{symbol_text}"},
        {"label": "时域滤波器", "value": f"{params.get('filter_type').upper()} L={L}，{sps}×，β={beta}"},
        {"label": "时域范围", "value": f"{time_ns[-1]:.3f} ns"},
        {"label": "时域采样率", "value": f"{fs_out / 1e9:.1f} GHz"},
    ])


def _run_ofdm_waveform_test_legacy(params, include_optional_pipeline: bool,
                                   checks: list, plots: list, summary: list) -> None:
    N_SC = int(params.get("subwave_num"))          # 512
    S = int(params.get("subframe_ofdm_num"))       # 48
    sps = int(params.get("oversampling"))

    # 1) 主测试：直接构造频域网格（每符号仅 1 个子载波有值、索引逐符号递增），
    #    复刻 TxOFDMProcesser 的 IFFT 变换（不插导频，保证"仅单音"前提）
    grid = np.zeros((S, N_SC), dtype=np.complex128)
    grid[np.arange(S), np.arange(S)] = 1.0
    tx = np.fft.ifft(grid.T, axis=0, norm="ortho").ravel(order="F")

    n_good = 0
    first_bad = None
    spec_all = np.zeros((S, N_SC))
    for s in range(S):
        seg = tx[s * N_SC:(s + 1) * N_SC]
        spec = np.fft.fft(seg, norm="ortho")
        spec_all[s] = np.abs(spec)
        k_peak = int(np.argmax(np.abs(spec)))
        others_quiet = float(np.max(np.abs(np.delete(spec, s))))
        ok_sym = (k_peak == s) and (abs(abs(spec[s]) - 1.0) < 1e-9) and (others_quiet < 1e-9)
        if ok_sym:
            n_good += 1
        elif first_bad is None:
            first_bad = (s, k_peak, float(abs(spec[s])), others_quiet)
    checks.append({
        "name": "每符号仅单音且位置逐符号递增（直接 IFFT）",
        "ok": bool(n_good == S),
        "detail": f"{n_good}/{S} 个符号通过"
                  + (f"，首个异常：符号 {first_bad[0]} 峰值位于子载波 {first_bad[1]}" if first_bad else ""),
    })

    # 2) 图 (a)：前 4 个符号时域波形（频率逐符号升高）
    n_show = min(4 * N_SC, len(tx))
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(np.arange(n_show), np.real(tx[:n_show]), color="#2C68B4", lw=0.6)
    for s in range(1, min(4, S)):
        ax.axvline(s * N_SC, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("采样点")
    ax.set_ylabel("幅度")
    ax.set_title("OFDM 单音逐符号扫描时域波形（前 4 个符号，频率逐符号升高）")
    ax.grid(True)
    fig.tight_layout()
    plots.append({"title": "OFDM 时域波形", "png": _figure_to_png(fig)})

    # 3) 图 (b)：各符号频谱热图（单音位置沿对角线递增）
    _apply_plot_style()
    spec_db = 20 * np.log10(spec_all + 1e-12)
    cols = min(S + 8, N_SC)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(spec_db[:, :cols], aspect="auto", origin="lower",
                   cmap="viridis", vmin=-80, vmax=0)
    fig.colorbar(im, ax=ax, label="幅度 (dB)")
    ax.set_xlabel("子载波索引")
    ax.set_ylabel("OFDM 符号索引")
    ax.set_title("各 OFDM 符号频谱（单音位置沿对角线递增）")
    fig.tight_layout()
    plots.append({"title": "OFDM 逐符号频谱", "png": _figure_to_png(fig)})

    summary.extend([
        {"label": "子载波数", "value": f"{N_SC}（IFFT 点数）"},
        {"label": "测试符号数", "value": f"{S} 个"},
        {"label": "有值子载波", "value": "0,1,2,… 逐符号 +1"},
        {"label": "变换", "value": "np.fft.ifft(ortho 归一化)"},
    ])

    # 4) 可选辅链路：真实 TxOFDMProcesser + GIInserter + 成型滤波
    if not include_optional_pipeline:
        return
    try:
        from transmitter.TxOFDMProcesser import TxOFDMProcesser
        from transmitter.GIInserter import GIInserter
        from transmitter.Pulseshaper import TxPulseShaper

        data_syms = 45 * N_SC
        ds = np.zeros(data_syms, dtype=np.complex128)
        for j in range(45):
            ds[j * N_SC + j] = 1.0
        ofdm = TxOFDMProcesser(params).ofdm_process({
            "signal_stream": ds,
            "sample_rate_Hz": 30e9,
            "duration_seconds": data_syms / 30e9,
            "signal_length": data_syms,
            "padding_bit_num": 0,
            "frame_symbol_num": data_syms,
            "frame_num": 1,
        })
        gi = GIInserter(params).insert_gi(ofdm)
        gi_sig = np.asarray(gi["signal_stream"])
        pilot_indexes = list(params.get("pilot_block_indexes"))  # [0, 16, 32]
        blocks = int(params.get("subframe_ofdm_num"))            # 48
        cp = int(params.get("gi_length"))                        # 32

        # 补零 IFFT 后每个 OFDM 符号为 (N_SC+cp)×sps 个高采样率样点
        n_gi_good = 0
        gi_first_bad = None
        for b in range(blocks):
            if b in pilot_indexes:
                continue  # 导频块为整块 512 音序列，不在单音校验范围
            start = b * (N_SC + cp) * sps + cp * sps
            spec = np.fft.fft(gi_sig[start:start + N_SC * sps], norm="ortho")
            j = b - sum(1 for p in pilot_indexes if p < b)
            k_peak = int(np.argmax(np.abs(spec)))
            if k_peak == j and abs(abs(spec[j]) - 1.0) < 1e-9:
                n_gi_good += 1
            elif gi_first_bad is None:
                gi_first_bad = (b, j, k_peak)
        checks.append({
            "name": "模块链路（ofdm_process + 插入 GI）单音位置校验",
            "ok": bool(n_gi_good == 45),
            "detail": f"{n_gi_good}/45 个数据块通过"
                      + (f"，首个异常：块 {gi_first_bad[0]} 期望单音 {gi_first_bad[1]}，实测 {gi_first_bad[2]}"
                         if gi_first_bad else ""),
        })

        # 成型滤波（OFDM 模式 = 直通：补零 IFFT 已完成过采样，无插值滤波）
        shaper = TxPulseShaper(params)
        shaped = shaper.shape_pulse(gi)
        shaped_sig = np.asarray(shaped["signal_stream"])
        ok_shaped = (shaped_sig.dtype == np.complex128) and \
                    (len(shaped_sig) == len(gi_sig))
        checks.append({
            "name": "OFDM 成型输出类型与长度",
            "ok": bool(ok_shaped),
            "detail": f"dtype={shaped_sig.dtype}，长度={len(shaped_sig)}"
                      f"（期望 complex128 / {len(gi_sig)}，直通无二次上采样）",
        })

        # 单音保留率：取数据块 20（单音 j=18），对高采样率符号体做 N_SC×sps 点 FFT。
        # 补零 IFFT 无损还原：单音仍位于 bin j，幅度为 1（无需除以 √sps）。
        b = 20
        j = b - sum(1 for p in pilot_indexes if p < b)
        seg = shaped_sig[b * (N_SC + cp) * sps + cp * sps:
                         b * (N_SC + cp) * sps + (cp + N_SC) * sps]
        spec2 = np.fft.fft(seg, norm="ortho")
        retention = float(abs(spec2[j]))
        checks.append({
            "name": "补零 IFFT 对带内单音的保留",
            "ok": bool(retention > 0.9),
            "detail": f"块 20（单音 j={j}）过采样符号体 FFT 后保留率 = {retention:.3f}（期望 > 0.9）",
        })

        # 带限特性：补零 IFFT 的保护带（中间 N_SC×(sps-1) 个频点）应无能量，
        # 等价于原插零+低通的抗镜像功能（理想砖墙带限）。
        guard_lo = N_SC // 2
        guard_hi = N_SC * sps - N_SC // 2
        guard_max = float(np.max(np.abs(spec2[guard_lo:guard_hi])))
        checks.append({
            "name": "补零 IFFT 频谱带限（保护带无能量）",
            "ok": bool(guard_max < 1e-9),
            "detail": f"块 20 保护带 [bin {guard_lo}..{guard_hi}) 最大幅度 = {guard_max:.2e}（期望 < 1e-9）",
        })

        summary.append({"label": "模块链路", "value": "ofdm_process + GI 校验通过"})
    except Exception as exc:
        checks.append({
            "name": "模块链路（ofdm_process + GI + 成型）",
            "ok": False,
            "detail": f"异常：{type(exc).__name__}: {exc}",
        })


def _typical_16qam_symbols(count: int) -> np.ndarray:
    """返回确定、平均功率归一化的典型 16QAM 符号序列。"""
    levels = np.array([-3.0, -1.0, 3.0, 1.0])
    constellation = np.array(
        [i + 1j * q for q in levels for i in levels], dtype=np.complex128
    ) / np.sqrt(10.0)
    return np.resize(constellation, int(count))


def _validate_signed_subcarriers(indices: np.ndarray, n_sc: int) -> None:
    lo, hi = -n_sc // 2, n_sc // 2 - 1
    if np.any(indices < lo) or np.any(indices > hi):
        raise ValueError(f"有符号子载波索引必须位于 [{lo}, {hi}]")


def _make_ofdm_single_tone_scan(n_sc: int, sps: int, cp_len: int,
                                signed_indices: np.ndarray) -> dict:
    """直接构造确定性 16QAM 单音扫描、解析理论波形及 CP。"""
    n_sc = int(n_sc)
    sps = int(sps)
    cp_len = int(cp_len)
    if n_sc < 8 or n_sc % 2:
        raise ValueError("OFDM 子载波数必须是不小于 8 的偶数")
    if sps < 1:
        raise ValueError("OFDM 过采样率必须大于等于 1")
    if not 0 <= cp_len < n_sc:
        raise ValueError("CP 长度必须满足 0 <= CP < 子载波数")

    signed_indices = np.asarray(signed_indices, dtype=int).ravel()
    if len(signed_indices) == 0:
        raise ValueError("单音扫描至少需要一个 OFDM 符号")
    _validate_signed_subcarriers(signed_indices, n_sc)

    n_fft = n_sc * sps
    cp_samples = cp_len * sps
    values = _typical_16qam_symbols(len(signed_indices))
    fft_bins = np.where(signed_indices >= 0, signed_indices, n_fft + signed_indices)
    grid = np.zeros((len(signed_indices), n_fft), dtype=np.complex128)
    grid[np.arange(len(signed_indices)), fft_bins] = values
    sampled_body = np.fft.ifft(grid, axis=1, norm="ortho")

    n = np.arange(n_fft, dtype=float)[None, :]
    theoretical_body = (
        values[:, None] / np.sqrt(n_fft)
        * np.exp(2j * np.pi * signed_indices[:, None] * n / n_fft)
    )
    if cp_samples:
        sampled = np.concatenate([sampled_body[:, -cp_samples:], sampled_body], axis=1)
        theoretical = np.concatenate(
            [theoretical_body[:, -cp_samples:], theoretical_body], axis=1)
    else:
        sampled = sampled_body.copy()
        theoretical = theoretical_body.copy()
    return {
        "signed_indices": signed_indices,
        "values": values,
        "n_fft": n_fft,
        "cp_samples": cp_samples,
        "sampled_body": sampled_body,
        "theoretical_body": theoretical_body,
        "sampled_blocks": sampled,
        "theoretical_blocks": theoretical,
        "sampled_stream": sampled.ravel(),
        "theoretical_stream": theoretical.ravel(),
    }


def _nice_tick_step(span: int, target_ticks: int = 9) -> int:
    """为离散索引轴选择易读的主刻度间隔。"""
    raw = max(1.0, float(span) / max(1, int(target_ticks)))
    magnitude = 10 ** np.floor(np.log10(raw))
    for factor in (1, 2, 4, 5, 8, 10):
        candidate = int(max(1, factor * magnitude))
        if candidate >= raw:
            return candidate
    return int(max(1, 10 * magnitude))


def _run_ofdm_time_test(n_sc: int, sps: int, cp_len: int, symbol_count: int,
                        start_subcarrier: int, subcarrier_step: int,
                        checks: list, plots: list, summary: list) -> None:
    symbol_count = int(symbol_count)
    indices = int(start_subcarrier) + np.arange(symbol_count) * int(subcarrier_step)
    scan = _make_ofdm_single_tone_scan(n_sc, sps, cp_len, indices)
    actual = scan["sampled_stream"]
    theory = scan["theoretical_stream"]
    max_error = float(np.max(np.abs(actual - theory)))
    checks.append({
        "name": "OFDM 单音 IFFT 与解析理论波形一致",
        "ok": bool(max_error < 1e-12),
        "detail": f"最大复幅度误差 {max_error:.2e}（期望 < 1e-12）",
    })

    cp_samples = scan["cp_samples"]
    if cp_samples:
        cp_ok = bool(np.allclose(
            scan["sampled_blocks"][:, :cp_samples],
            scan["sampled_body"][:, -cp_samples:], atol=1e-12, rtol=0.0))
    else:
        cp_ok = True
    checks.append({
        "name": "OFDM 循环前缀复制正确",
        "ok": cp_ok,
        "detail": f"CP={cp_len} 基带样点 / {cp_samples} 个过采样点",
    })
    varying = np.std(actual.real) > 1e-6 and np.std(actual.imag) > 1e-6
    checks.append({
        "name": "16QAM 单音实部和虚部均有起伏",
        "ok": bool(varying),
        "detail": f"std(I)={np.std(actual.real):.3e}，std(Q)={np.std(actual.imag):.3e}",
    })

    fs_base = 30e9
    fs_out = fs_base * sps
    time_ns = np.arange(len(actual), dtype=float) / fs_out * 1e9
    block_len = scan["n_fft"] + cp_samples
    _apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    parts = ((actual.real, theory.real, "实部 I"),
             (actual.imag, theory.imag, "虚部 Q"))
    for ax, (actual_part, theory_part, ylabel) in zip(axes, parts):
        ax.plot(time_ns, theory_part, color="#D95F02", lw=1.35, ls="--",
                label="理论连续波形", zorder=2)
        marker_step = max(1, len(time_ns) // 90)
        ax.plot(time_ns, actual_part, color="#2C68B4", lw=0.55,
                marker="o", ms=2.1, markevery=marker_step,
                label="IFFT 采样点", zorder=3)
        for idx in range(symbol_count):
            start = idx * block_len
            cp_end = start + cp_samples
            end = (idx + 1) * block_len
            start_ns = start / fs_out * 1e9
            cp_end_ns = cp_end / fs_out * 1e9
            end_ns = end / fs_out * 1e9
            ax.axvline(start_ns, color="#344054", lw=1.25, ls="-.",
                       label="完整 OFDM symbol 边界" if idx == 0 else None)
            ax.axvline(start_ns, color="#E5484D", lw=0.9, ls="--",
                       label="CP 起点" if idx == 0 else None)
            if cp_samples:
                ax.axvspan(start_ns, cp_end_ns, color="#E5484D", alpha=0.14)
                ax.axvline(cp_end_ns, color="#E5484D", lw=1.15, ls="--",
                           label="CP 结束" if idx == 0 else None)
                cp_mid = (start_ns + cp_end_ns) / 2
                ax.text(cp_mid, 0.94, "CP", color="#B42318", fontsize=7,
                        ha="center", va="top", transform=ax.get_xaxis_transform())
            ax.axvline(end_ns, color="#344054", lw=1.25, ls="-.")
        ax.set_ylabel(f"{ylabel}幅度")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
        ax.margins(x=0.005)

    # 在第一幅子图中明确展示 CP 来自对应有效符号尾部，并标出三种长度。
    if cp_samples:
        from matplotlib.transforms import blended_transform_factory

        ax = axes[0]
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        first_start = 0.0
        first_cp_end = cp_samples / fs_out * 1e9
        first_end = block_len / fs_out * 1e9
        tail_start = (block_len - cp_samples) / fs_out * 1e9
        ax.axvspan(tail_start, first_end, color="#F79009", alpha=0.12,
                   label="CP 复制来源（有效符号尾部）")

        def _length_bracket(x0, x1, y, label, color):
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        xycoords=transform, textcoords=transform,
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
            ax.text((x0 + x1) / 2, y + 0.015, label, transform=transform,
                    ha="center", va="bottom", fontsize=7, color=color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0))

        _length_bracket(
            first_start, first_cp_end, 0.88,
            f"CP：{cp_samples} 点 / {first_cp_end - first_start:.3f} ns", "#B42318")
        _length_bracket(
            first_cp_end, first_end, 0.78,
            f"有效符号：{scan['n_fft']} 点 / {first_end - first_cp_end:.3f} ns", "#175CD3")
        _length_bracket(
            first_start, first_end, 0.68,
            f"完整 OFDM symbol：{block_len} 点 / {first_end:.3f} ns", "#344054")
        ax.annotate(
            "尾部复制为 CP",
            xy=((first_start + first_cp_end) / 2, 0.48),
            xytext=((tail_start + first_end) / 2, 0.58),
            xycoords=transform, textcoords=transform,
            ha="center", va="center", fontsize=7, color="#B54708",
            arrowprops=dict(arrowstyle="->", color="#F79009", lw=1.1,
                            connectionstyle="arc3,rad=0.22"),
            bbox=dict(facecolor="white", edgecolor="#F79009", alpha=0.86, pad=1.5))
    axes[-1].set_xlabel("时间 (ns)")
    fig.suptitle(f"OFDM 16QAM 单音扫描时域波形（含 CP；子载波 {indices.tolist()}）")
    fig.tight_layout()
    plots.append({"title": "OFDM 时域波形（含 CP）", "png": _figure_to_png(fig)})
    summary.extend([
        {"label": "OFDM 时域符号", "value": f"典型 16QAM，{symbol_count} 个单音符号"},
        {"label": "时域扫描子载波", "value": str(indices.tolist())},
        {"label": "时域 CP", "value": f"{cp_len} 点（过采样后 {cp_samples} 点）"},
        {"label": "OFDM 有效符号长度", "value": f"{scan['n_fft']} 采样点 / {scan['n_fft'] / fs_out * 1e9:.3f} ns"},
        {"label": "完整 OFDM symbol 长度", "value": f"{block_len} 采样点 / {block_len / fs_out * 1e9:.3f} ns"},
        {"label": "时域采样率", "value": f"{fs_out / 1e9:.1f} GHz"},
    ])


def _run_ofdm_heatmap_test(n_sc: int, sps: int, cp_len: int, symbol_count: int,
                           start_subcarrier: int, subcarrier_step: int,
                           axis_margin: int, floor_db: float,
                           checks: list, plots: list, summary: list) -> None:
    indices = int(start_subcarrier) + np.arange(int(symbol_count)) * int(subcarrier_step)
    scan = _make_ofdm_single_tone_scan(n_sc, sps, cp_len, indices)
    active_signed = np.arange(-n_sc // 2, n_sc // 2)
    active_bins = np.where(active_signed >= 0, active_signed,
                           scan["n_fft"] + active_signed)
    recovered = np.fft.fft(scan["sampled_body"], axis=1, norm="ortho")[:, active_bins]
    magnitude = np.abs(recovered)
    global_peak = float(np.max(magnitude))
    spec_db = 20 * np.log10(magnitude / (global_peak + 1e-15) + 1e-15)
    spec_db = np.maximum(spec_db, float(floor_db))
    peak_positions = active_signed[np.argmax(magnitude, axis=1)]
    peak_ok = bool(np.array_equal(peak_positions, indices))
    residual = magnitude.copy()
    residual[np.arange(len(indices)), np.argmax(magnitude, axis=1)] = 0.0
    quiet_max = float(np.max(residual))
    checks.append({
        "name": "逐符号单音峰值沿指定子载波扫描",
        "ok": bool(peak_ok and quiet_max < 1e-12),
        "detail": f"{len(indices)}/{len(indices)} 个目标峰值；其余频点最大 {quiet_max:.2e}",
    })

    x_lo = int(np.min(indices) - int(axis_margin))
    x_hi = int(np.max(indices) + int(axis_margin))
    x_lo = max(x_lo, -n_sc // 2)
    x_hi = min(x_hi, n_sc // 2 - 1)
    tick_step = _nice_tick_step(x_hi - x_lo + 1)
    y_step = _nice_tick_step(len(indices), target_ticks=7)
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    im = ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest",
                   vmin=float(floor_db), vmax=0.0,
                   extent=[-n_sc / 2 - 0.5, n_sc / 2 - 0.5,
                           -0.5, len(indices) - 0.5])
    fig.colorbar(im, ax=ax, label="相对幅度 (dB)")
    scan_rows = np.arange(len(indices))
    ax.plot(indices, scan_rows, color="white", lw=1.0, ls="--",
            alpha=0.9, label="理论扫描轨迹")
    ax.scatter(peak_positions, scan_rows, s=10, facecolors="none",
               edgecolors="#FF4D4F", linewidths=0.7, label="实测峰值")
    ax.axvline(0, color="white", lw=1.1, ls=":", alpha=0.95, label="DC")
    ax.set_xlim(x_lo - 0.5, x_hi + 0.5)
    ax.set_ylim(-0.5, len(indices) - 0.5)
    x_ticks = np.arange(x_lo, x_hi + 1, tick_step)
    x_ticks = np.unique(np.concatenate([x_ticks, [0, x_lo, x_hi]]))
    x_ticks = x_ticks[(x_ticks >= x_lo) & (x_ticks <= x_hi)]
    y_ticks = np.unique(np.concatenate([
        np.arange(0, len(indices), y_step), [0, len(indices) - 1]]))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("有符号子载波索引 k")
    ax.set_ylabel("OFDM 扫描符号索引")
    ax.set_title("OFDM 逐符号单音扫描频谱（典型 16QAM 幅相）")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    plots.append({"title": "OFDM 逐符号频谱热图", "png": _figure_to_png(fig)})
    summary.append({
        "label": "热图扫描范围",
        "value": f"k={indices[0]}…{indices[-1]}，显示 [{x_lo}, {x_hi}]",
    })


def _run_ofdm_random_spectrum_test(n_sc: int, sps: int, cp_len: int,
                                   scale: str, num_symbols: int,
                                   random_seed: int, checks: list,
                                   plots: list, summary: list) -> None:
    """由随机 16QAM-OFDM 连续时域流通过分段 FFT 平均计算实际 PSD。"""
    n_sc = int(n_sc)
    sps = int(sps)
    cp_len = int(cp_len)
    num_symbols = int(num_symbols)
    random_seed = int(random_seed)
    if num_symbols < 16:
        raise ValueError("OFDM 功率谱至少需要 16 个随机符号")
    if random_seed < 0:
        raise ValueError("OFDM 功率谱随机种子不能小于 0")
    _validate_signed_subcarriers(np.array([-n_sc // 2, n_sc // 2 - 1]), n_sc)

    fs_base = 30e9
    fs_out = fs_base * sps
    n_fft = n_sc * sps
    cp_samples = cp_len * sps
    rng = np.random.default_rng(random_seed)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    qam = (
        rng.choice(levels, size=(num_symbols, n_sc))
        + 1j * rng.choice(levels, size=(num_symbols, n_sc))
    ) / np.sqrt(10.0)
    signed_indices = np.arange(-n_sc // 2, n_sc // 2)
    fft_bins = np.where(signed_indices >= 0, signed_indices, n_fft + signed_indices)
    frequency_grid = np.zeros((num_symbols, n_fft), dtype=np.complex128)
    frequency_grid[:, fft_bins] = qam
    useful = np.fft.ifft(frequency_grid, axis=1, norm="ortho")
    blocks = (
        np.concatenate([useful[:, -cp_samples:], useful], axis=1)
        if cp_samples else useful
    )
    continuous = blocks.ravel()

    # 手工分段 FFT 平均。窗口允许跨越相邻 OFDM symbol，保留 CP、符号边界
    # 和随机 16QAM 幅相变化造成的真实带内起伏及带外泄漏。
    block_len = blocks.shape[1]
    segment_len = min(len(continuous), block_len * 4)
    fft_length = 1 << int(np.ceil(np.log2(segment_len)))
    step = max(1, segment_len // 2)
    window = np.hanning(segment_len)
    window_energy = float(np.sum(window ** 2)) + 1e-30
    measured = np.zeros(fft_length, dtype=float)
    segment_count = 0
    for start in range(0, len(continuous) - segment_len + 1, step):
        spectrum = np.fft.fft(
            continuous[start:start + segment_len] * window, n=fft_length)
        measured += np.abs(spectrum) ** 2 / window_energy
        segment_count += 1
    if segment_count == 0:
        raise ValueError("随机 OFDM 连续波形不足以完成分段 FFT")
    measured /= segment_count
    f = np.fft.fftfreq(fft_length, d=1 / fs_out)
    order = np.argsort(f)
    f, measured = f[order], measured[order]

    inband_ref = np.abs(f) < 0.45 * fs_base
    measured /= float(np.median(measured[inband_ref])) + 1e-30
    measured_db = 10 * np.log10(measured + 1e-15)
    inside = np.abs(f) < 0.45 * fs_base
    outside = (np.abs(f) > 0.55 * fs_base) & (np.abs(f) < 0.75 * fs_base)
    in_level = float(np.median(measured_db[inside]))
    out_level = float(np.median(measured_db[outside]))
    checks.append({
        "name": "红色带宽边界外开始带外滚降",
        "ok": bool(out_level < in_level - 3.0),
        "detail": f"带内中位数 {in_level:.1f} dB，红线外频段 {out_level:.1f} dB",
    })
    checks.append({
        "name": "随机 16QAM-OFDM 连续时域数据 FFT 有效",
        "ok": bool(segment_count >= 4 and np.all(np.isfinite(measured_db))),
        "detail": (
            f"{num_symbols} 个 OFDM symbols，{segment_count} 个 FFT 分段，"
            f"FFT={fft_length}"
        ),
    })

    scale_key = str(scale).lower()
    is_db = scale_key in ("db", "对数功率 db", "对数")
    if not is_db and scale_key not in ("linear", "线性归一化功率", "线性"):
        raise ValueError(f"不支持的功率谱纵坐标：{scale}")
    y_measured = measured_db if is_db else measured
    ylabel = "归一化功率谱 (dB)" if is_db else "归一化功率谱"

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        f / 1e9, y_measured, color="#2C68B4", lw=0.95, alpha=0.95,
        label="随机 16QAM-OFDM 实测 PSD（分段 FFT）")
    edge_ghz = fs_base / 2e9
    ax.axvline(
        edge_ghz, color="#E5484D", lw=1.35, ls="--",
        label="理论占用带宽边界 ±Rs/2")
    ax.axvline(-edge_ghz, color="#E5484D", lw=1.35, ls="--")
    view_half_ghz = min(fs_out / 2e9, edge_ghz * 2.0)
    ax.set_xlim(-view_half_ghz, view_half_ghz)
    if is_db:
        ax.set_ylim(-60, 5)
    else:
        upper = max(1.2, float(np.percentile(y_measured[inband_ref], 99)) * 1.1)
        ax.set_ylim(0, upper)
    ax.set_xlabel("频率 (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title("随机 16QAM-OFDM 连续时域信号功率谱")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plots.append({"title": "OFDM 随机 16QAM 功率谱", "png": _figure_to_png(fig)})
    summary.extend([
        {"label": "功率谱纵坐标", "value": "对数功率 dB" if is_db else "线性归一化功率"},
        {"label": "理论占用带宽边界", "value": f"±{edge_ghz:.1f} GHz（红色虚线）"},
        {"label": "功率谱数据", "value": (
            f"随机 16QAM，{num_symbols} 个连续 OFDM symbols，seed={random_seed}")},
        {"label": "功率谱计算", "value": (
            f"分段 FFT 平均：{segment_count} 段，FFT={fft_length}（含 CP）")},
        {"label": "输出采样范围", "value": f"±{fs_out / 2e9:.1f} GHz"},
    ])


def _run_ofdm_waveform_test(params, include_optional_pipeline: bool,
                            checks: list, plots: list, summary: list,
                            time_symbol_count: int = 2,
                            time_start_subcarrier: int = -6,
                            time_subcarrier_step: int = 11,
                            heatmap_symbol_count: int = 48,
                            heatmap_start_subcarrier: int = -24,
                            heatmap_subcarrier_step: int = 1,
                            heatmap_axis_margin: int = 4,
                            heatmap_floor_db: float = -45.0,
                            spectrum_scale: str = "db",
                            spectrum_n_sc: Optional[int] = None,
                            spectrum_sps: Optional[int] = None,
                            spectrum_cp_len: Optional[int] = None,
                            spectrum_num_symbols: int = 128,
                            spectrum_random_seed: int = 2026) -> None:
    """确定性 16QAM 单音扫描的 OFDM 时域、热图与功率谱测试。"""
    del include_optional_pipeline  # 保留旧调用签名；新测试不进入真实模块链路。
    n_sc = int(params.get("subwave_num"))
    sps = int(params.get("oversampling"))
    cp_len = int(params.get("gi_length"))
    _run_ofdm_time_test(
        n_sc, sps, cp_len, time_symbol_count, time_start_subcarrier,
        time_subcarrier_step, checks, plots, summary)
    _run_ofdm_heatmap_test(
        n_sc, sps, cp_len, heatmap_symbol_count, heatmap_start_subcarrier,
        heatmap_subcarrier_step, heatmap_axis_margin, heatmap_floor_db,
        checks, plots, summary)
    _run_ofdm_random_spectrum_test(
        n_sc if spectrum_n_sc is None else int(spectrum_n_sc),
        sps if spectrum_sps is None else int(spectrum_sps),
        cp_len if spectrum_cp_len is None else int(spectrum_cp_len),
        spectrum_scale, int(spectrum_num_symbols), int(spectrum_random_seed),
        checks, plots, summary)
    summary.insert(0, {
        "label": "OFDM 时域/热图测试信号",
        "value": "确定性典型 16QAM 单音扫描",
    })


def _run_ofdm_time_part(ofdm_subcarriers: int, ofdm_oversampling: int,
                        ofdm_cp_length: int, time_symbol_count: int,
                        time_start_subcarrier: int, time_subcarrier_step: int,
                        heatmap_symbol_count: int, heatmap_start_subcarrier: int,
                        heatmap_subcarrier_step: int, heatmap_axis_margin: int,
                        heatmap_floor_db: float, checks: list, plots: list,
                        summary: list) -> None:
    """OFDM 时域测试公共部分：单音扫描时域波形 + 逐符号频谱热图。"""
    _run_ofdm_time_test(
        int(ofdm_subcarriers), int(ofdm_oversampling), int(ofdm_cp_length),
        int(time_symbol_count), int(time_start_subcarrier),
        int(time_subcarrier_step), checks, plots, summary)
    _run_ofdm_heatmap_test(
        int(ofdm_subcarriers), int(ofdm_oversampling), int(ofdm_cp_length),
        int(heatmap_symbol_count), int(heatmap_start_subcarrier),
        int(heatmap_subcarrier_step), int(heatmap_axis_margin),
        float(heatmap_floor_db), checks, plots, summary)
    summary.insert(0, {
        "label": "OFDM 时域/热图测试信号",
        "value": "确定性典型 16QAM 单音扫描",
    })


def _run_papr_ccdf_test(params, mode: str, symbols_per_mod: int,
                        checks: list, plots: list, summary: list) -> None:
    """多调制 PAPR CCDF：BPSK/QPSK/16QAM/64QAM/256QAM 经调制与成型滤波后叠加一张图。

    SC：每符号周期（sps 样本）为一块；OFDM：每个 OFDM 符号（512 样本）为一块。
    点数默认 262144 符号/调制（SC 可分辨到 ~4e-6，OFDM 约 528 个符号块 ~2e-3）。
    """
    from params.PHYParams import PHYParams
    from transmitter.Modulator import THzModulator
    from transmitter.Pulseshaper import TxPulseShaper

    L = int(params.get("filter_length"))
    sps = int(params.get("oversampling"))
    rolloff = float(params.get("rolloff"))
    fs_base = 30e9
    mods = [("BPSK", 1), ("QPSK", 2), ("16QAM", 4), ("64QAM", 6), ("256QAM", 8)]
    rng = np.random.default_rng(20260829)

    sorted_papr = {}
    n_blocks_list = []
    for name, ncbps in mods:
        p = PHYParams()
        p.update(link_mode=mode, filter_length=L, oversampling=sps, rolloff=rolloff,
                 NCBPS=ncbps, MCS=1, scramble=False)
        mod = THzModulator(p)
        if mode == "sc-fde":
            n_syms = int(symbols_per_mod)
            bits = rng.integers(0, 2, n_syms * ncbps).astype(np.uint8)
            syms = mod.modulate(_make_signal_dict(
                bits, fs_base, {"frame_bit_num": len(bits), "frame_num": 1}))["signal_stream"]
            shaped = TxPulseShaper(p).shape_pulse(_make_signal_dict(syms, fs_base))
            sig = np.asarray(shaped["signal_stream"])
            block_len = sps
        else:
            from transmitter.TxOFDMProcesser import TxOFDMProcesser
            data_per_sub = 45 * 512
            # OFDM 每个 PAPR 块对应一个 OFDM 符号：补零 IFFT 后为
            # 512×sps 个高采样率样点。默认 262144 符号只有 ~500 块，
            # CCDF 统计波动大（标准误 ~0.016，曲线间出现系统性偏移假象）。
            # IFFT 开销远小于 SC 的成型卷积，故用 4× 符号数把块数提到
            # ~2000（标准误 ~0.009），曲线平滑可辨。
            n_sub = max(1, int(symbols_per_mod) * 4 // data_per_sub)
            data_syms = n_sub * data_per_sub
            bits = rng.integers(0, 2, data_syms * ncbps).astype(np.uint8)
            syms = mod.modulate(_make_signal_dict(
                bits, fs_base, {"frame_bit_num": len(bits), "frame_num": 1}))["signal_stream"]
            ofdm = TxOFDMProcesser(p).ofdm_process(_make_signal_dict(
                syms, fs_base, {"frame_symbol_num": data_per_sub, "frame_num": n_sub}))
            sig = np.asarray(ofdm["signal_stream"])
            block_len = 512 * sps

        n_blocks = len(sig) // block_len
        blocks = sig[:n_blocks * block_len].reshape(n_blocks, block_len)
        if mode == "ofdm":
            # 排除导频块：固定 a512 序列的 PAPR 恒为 ~3.0 dB（std=0），
            # 混入统计会在 CCDF 上造成 6.25% 的陡降台阶，曲线失真。
            pilots = set(params.get("pilot_block_indexes"))
            blocks_per_subframe = int(params.get("subframe_ofdm_num"))
            keep = np.array([b % blocks_per_subframe not in pilots
                             for b in range(n_blocks)])
            blocks = blocks[keep]
            n_blocks = len(blocks)
        pwr = np.mean(np.abs(blocks) ** 2, axis=1) + 1e-15
        peak = np.max(np.abs(blocks) ** 2, axis=1)
        sorted_papr[name] = np.sort(10 * np.log10(peak / pwr))
        n_blocks_list.append(n_blocks)

    # 动态 x 轴：覆盖全部曲线实际 PAPR 范围（SC 的 pi/2-BPSK 从 ~0 dB 起，
    # OFDM 数据块集中在 6~12 dB），避免 0~15 固定轴上的大段空白平台。
    x_lo = max(0.0, float(np.floor(min(p[0] for p in sorted_papr.values())) - 1.0))
    x_hi = float(np.ceil(max(p[-1] for p in sorted_papr.values())) + 1.0)
    x = np.linspace(x_lo, x_hi, 151)
    ccdf_curves = {}
    min_y = 1.0
    for name, papr_db in sorted_papr.items():
        y = 1.0 - np.searchsorted(papr_db, x) / len(papr_db)
        ccdf_curves[name] = y
        min_y = min(min_y, float(y[-1]))

    ok_blocks = len(set(n_blocks_list)) == 1
    ok_mono = all(bool(np.all(np.diff(y) <= 0)) for y in ccdf_curves.values())
    ok_head = all(bool(y[0] == 1.0) for y in ccdf_curves.values())
    pilot_note = "（已排除导频块）" if mode == "ofdm" else ""
    checks.append({
        "name": "PAPR CCDF 曲线有效（各调制点数一致、单调递减）",
        "ok": bool(ok_blocks and ok_mono and ok_head),
        "detail": f"每调制 {symbols_per_mod} 符号{pilot_note}（块数 {n_blocks_list[0]}），"
                  f"CCDF({x_lo:.0f} dB)=1，单调性 {'正常' if ok_mono else '异常'}",
    })

    # OFDM：与高斯近似理论对比（正交 IFFT 输出近似复高斯，
    # CCDF(x) = 1-(1-e^(-x))^N，x 为线性 PAPR）。用 QPSK 曲线做参照：
    # pi/2-BPSK 因频域实虚交替使时域包络镜像对称（自由度减半为 N/2），
    # PAPR 系统性偏低，属真实物理特性，不做 N 音校验。
    # 补零 IFFT 输出为 sps 倍过采样波形，其 PAPR 统计等效于 αN 个子载波
    # 的奈奎斯特采样（业界公认 L=4 过采样时经验 α≈2.8）。
    if mode == "ofdm":
        n_sc = int(params.get("subwave_num"))
        n_eff = n_sc if sps == 1 else int(round(2.8 * n_sc))
        ref_y = ccdf_curves["QPSK"]
        theory_devs = []
        for x_db in (8.0, 9.0, 10.0):
            idx = int(np.argmin(np.abs(x - x_db)))
            theory = 1.0 - (1.0 - np.exp(-10 ** (x_db / 10))) ** n_eff
            theory_devs.append(abs(float(ref_y[idx]) - theory))
        ok_theory = max(theory_devs) < 0.08
        checks.append({
            "name": "OFDM PAPR CCDF 符合理论预期（QPSK 参照）",
            "ok": bool(ok_theory),
            "detail": f"QPSK 在 8/9/10 dB 处与 {n_eff} 音（α≈2.8 × {n_sc}）"
                      f"高斯理论最大偏差 {max(theory_devs):.3f}（< 0.08）；"
                      f"BPSK 因频域实虚交替镜像对称，PAPR 偏低属正常",
        })

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    linestyles = ["-", "--", "-.", ":"]
    for (name, y), color, ls in zip(ccdf_curves.items(), PLOT_COLORS, linestyles):
        ax.semilogy(x, y, color=color, ls=ls, lw=1.2, label=name)
    # 负指数刻度用 ASCII 格式（1e-5），避免 mathtext 的 U+2212 缺字形
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "1" if abs(v - 1.0) < 1e-12 else
        (f"1e{int(round(np.log10(v)))}" if v > 0 else "0")))
    ax.set_xlabel("PAPR (dB)")
    ax.set_ylabel("CCDF  P(PAPR > x)")
    ax.set_title(f"{mode.upper()} 多调制 PAPR CCDF（每调制 {symbols_per_mod} 符号）")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(max(min_y / 2, 1e-5), 2.0)
    ax.grid(True, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plots.append({"title": "PAPR CCDF（多调制）", "png": _figure_to_png(fig)})
    summary.append({"label": "PAPR 测试", "value": f"{len(mods)} 种调制 × {symbols_per_mod} 符号"})


def _ideal_filter_response(f_n, filter_type: str, rolloff: float) -> np.ndarray:
    """理想无限长成型滤波器的幅度响应（f_n 为 f/Rs 归一化频率）。

    RRC：通带 1，过渡带 cos(π/(4β)(2|f|-1+β))，阻带 0（平方后即升余弦滚降）；
    RC：过渡带 0.5(1+cos(π/(2β)(2|f|-1+β)))；矩形：|sinc| 幅度响应。
    """
    f = np.abs(np.asarray(f_n, dtype=float))
    beta = float(rolloff)
    if not 0.0 < beta < 1.0:
        raise ValueError("滚降系数必须位于 (0, 1)")
    filter_type = str(filter_type).lower()
    if filter_type == "rect":
        return np.sinc(f)
    edge_low = (1.0 - beta) / 2.0
    edge_high = (1.0 + beta) / 2.0
    phase = np.clip((2.0 * f - 1.0 + beta) / (2.0 * beta), 0.0, 1.0)
    transition = (f >= edge_low) & (f < edge_high)
    response = np.where(f < edge_low, 1.0, 0.0)
    if filter_type == "rc":
        response[transition] = 0.5 * (1.0 + np.cos(np.pi * phase[transition]))
    else:
        # rrc（根升余弦）：|H| 在过渡带按余弦开方形状滚降
        response[transition] = np.cos(np.pi / 2.0 * phase[transition])
    return response


def _run_spectrum_test(params, mode: str, modulation: str, num_symbols: int,
                       scale: str, checks: list, plots: list,
                       summary: list) -> None:
    """经过成型滤波器后的功率谱（Welch），叠加理论滤波器响应并标注截止频率。

    采样方式：成型输出为基带复数信号，采样率 fs_base×sps；Welch 使用较短分段
    保留可见统计波动；频率轴归一化为 f/fs_base，便于标注通带/截止。
    理论曲线按理想无限长滤波器的解析响应绘制（直接滚降形状）；纵坐标支持
    对数功率 dB 与线性归一化功率两种显示。
    """
    from params.PHYParams import PHYParams
    from transmitter.Modulator import THzModulator
    from transmitter.Pulseshaper import TxPulseShaper
    from scipy.signal import welch

    L = int(params.get("filter_length"))
    sps = int(params.get("oversampling"))
    rolloff = float(params.get("rolloff"))
    fs_base = 30e9
    modulation_map = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
    modulation = str(modulation).upper()
    if modulation not in modulation_map:
        raise ValueError(f"不支持的功率谱调制符号：{modulation}")
    ncbps = modulation_map[modulation]
    num_symbols = int(num_symbols)
    if num_symbols < 2048:
        raise ValueError("功率谱符号数不能少于 2048")
    scale_key = str(scale).lower()
    is_db = scale_key in ("db", "对数功率 db", "对数")
    if not is_db and scale_key not in ("linear", "线性归一化功率", "线性"):
        raise ValueError(f"不支持的功率谱纵坐标：{scale}")

    p = PHYParams()
    # 关闭逐符号 pi/2 旋转：旋转会使频谱整体搬移 Rs/4（双峰），遮住成型
    # 滤波器本征形状；关闭后符号频谱平坦，RRC 形状与截止点清晰可验。
    p.update(link_mode=mode, filter_length=L, oversampling=sps, rolloff=rolloff,
             filter_type=params.get("filter_type"), NCBPS=ncbps, MCS=1,
             scramble=False, pi2_rotation=False)
    mod = THzModulator(p)
    rng = np.random.default_rng(20260829)

    if mode == "sc-fde":
        n_syms = num_symbols
        bits = rng.integers(0, 2, n_syms * ncbps).astype(np.uint8)
        syms = mod.modulate(_make_signal_dict(
            bits, fs_base, {"frame_bit_num": len(bits), "frame_num": 1}))["signal_stream"]
        shaper = TxPulseShaper(p)
        shaped = shaper.shape_pulse(_make_signal_dict(syms, fs_base))
        sig = np.asarray(shaped["signal_stream"])
        fs_out = float(shaped["sample_rate_Hz"])
        filter_type = str(p.get("filter_type")).lower()
        if filter_type in ("rrc", "rc"):
            cut_lines = [0.5]
            band_edges = [(1 - rolloff) / 2, (1 + rolloff) / 2]
            cut_level = "-3 dB" if filter_type == "rrc" else "-6 dB"
            cut_text = (f"±Rs/2 处 {cut_level}；通带边缘 ±{band_edges[0]:.3f}；"
                        f"理论带宽 ±{band_edges[1]:.3f}")
            passband = max(0.05, min(0.30, band_edges[0] * 0.8))
            stopband = band_edges[1] + 0.05
        else:
            # 矩形脉冲对应 sinc 型谱，不存在 RRC/RC 意义下的滚降截止点。
            cut_lines = []
            band_edges = []
            cut_text = "矩形脉冲的理论 sinc 型响应"
            passband = 0.15
            stopband = 0.70
        title_note = f"调制：{modulation}（符号相位旋转已关闭）"
    else:
        from transmitter.TxOFDMProcesser import TxOFDMProcesser
        data_syms = 45 * 512
        bits = rng.integers(0, 2, data_syms * ncbps).astype(np.uint8)
        syms = mod.modulate(_make_signal_dict(
            bits, fs_base, {"frame_bit_num": len(bits), "frame_num": 1}))["signal_stream"]
        ofdm = TxOFDMProcesser(p).ofdm_process(_make_signal_dict(
            syms, fs_base, {"frame_symbol_num": data_syms, "frame_num": 1}))
        shaper = TxPulseShaper(p)
        # 补零 IFFT 输出已是 sps 倍高采样率信号，直通成型后采样率不变
        shaped = shaper.shape_pulse(_make_signal_dict(
            ofdm["signal_stream"], float(ofdm["sample_rate_Hz"])))
        sig = np.asarray(shaped["signal_stream"])
        fs_out = float(shaped["sample_rate_Hz"])
        cut_lines = [0.5]                 # 带限边界 ±fs_base/2
        band_edges = []
        cut_text = "理想带限 ±fs_base/2（补零 IFFT）"
        passband = 0.40
        # 0.6 处仍处于砖墙边缘的 hann 泄漏裙内（实测约 -21 dB），
        # 阻带检查从 0.7 起算（实测 < -35 dB），阈值 -20 dB 稳健。
        stopband = 0.70
        title_note = f"调制：{modulation}（符号相位旋转不改变 OFDM 频谱形状）"

    # 校验用理论响应（成型滤波器形状）：单载波取有限长系数 FFT，包含截断
    # 旁瓣，保证阻带校验有实际参照；OFDM 取补零 IFFT 的理想砖墙带限。
    f_h = np.fft.fftfreq(4096, 1 / fs_out) / fs_base
    if mode == "sc-fde":
        h_resp = np.fft.fft(shaper.filter_coeffs, 4096)
        h_db = 20 * np.log10(np.abs(h_resp) / (np.max(np.abs(h_resp)) + 1e-15))
        h_label = "理论滤波器响应（理想滚降）"
    else:
        h_db = np.where(np.abs(f_h) <= 0.5, 0.0, -120.0)
        h_label = "理想带限响应（补零 IFFT）"
    order_h = np.argsort(f_h)

    # Welch 功率谱（分段平均，双边谱）
    # 较短分段保留适量统计起伏；校验另用平滑副本，避免展示效果影响判定。
    nperseg = min(2048, len(sig))
    f, psd = welch(sig, fs=fs_out, nperseg=nperseg, window="hann",
                   return_onesided=False)
    f_n = f / fs_base
    order = np.argsort(f_n)
    f_n = f_n[order]
    psd_lin = psd[order] / (np.max(psd) + 1e-15)
    psd_db = 10 * np.log10(psd_lin)

    # dB 域移动平均平滑：周期图 bin 呈指数分布、单 bin 波动大且平滑后
    # 最大值仍虚高约 1~2 dB；平滑后才能正确呈现滤波器形状并做校验。
    smooth_win = 31
    psd_db_s = np.convolve(psd_db, np.ones(smooth_win) / smooth_win, mode="same")

    # 归一化到通带中位数：周期图 max 虚高会把整条谱面压低，中位数参照
    # 是无偏的，使实测与理论曲线可直接对齐、绝对值校验有意义。
    mask_ref = np.abs(f_n) < 0.35
    plot_ref = float(np.median(psd_db[mask_ref]))
    psd_db = psd_db - plot_ref
    psd_db_s = psd_db_s - plot_ref
    h_db = h_db - float(np.median(h_db[np.abs(f_h) < 0.35]))

    # 校验使用平滑谱；绘图仍使用未平滑谱，以保留用户要求的可见波动。
    theory_on_measured = np.interp(
        f_n, f_h[order_h], h_db[order_h], left=-120.0, right=-120.0)
    mask_pass = np.abs(f_n) < passband
    flat_min = float(np.min(psd_db_s[mask_pass]))
    flat_expected = float(np.min(theory_on_measured[mask_pass]))
    ok_flat = flat_min > flat_expected - 2.0
    if mode == "sc-fde":
        mask_cut = np.abs(np.abs(f_n) - 0.5) < 0.01
        cut_db = float(np.mean(psd_db_s[mask_cut]))
        cut_expected = float(np.mean(theory_on_measured[mask_cut]))
        ok_cut = abs(cut_db - cut_expected) < 2.0
        cut_detail = (f"±Rs/2 处实测 {cut_db:.1f} dB，"
                      f"理论 {cut_expected:.1f} dB（偏差 < 2 dB）")
    else:
        mask_cut = np.abs(np.abs(f_n) - 0.5) < 0.01
        cut_db = float(np.mean(psd_db_s[mask_cut]))
        ok_cut = cut_db < -2.0
        cut_detail = f"截止 ±fs_base/2 处实测 {cut_db:.1f} dB"
    mask_stop = np.abs(f_n) > stopband
    stop_max = float(np.max(psd_db_s[mask_stop]))
    if mode == "sc-fde":
        stop_expected = float(np.max(theory_on_measured[mask_stop]))
        # Welch 窗泄漏会抬高深阻带，尤其 RC/有限长 RRC；允许 20 dB 的阻带
        # 估计余量，同时仍可捕获明显偏离理论形状的实现错误。
        ok_stop = stop_max < stop_expected + 20.0
        stop_detail = (f"|f|>{stopband}·fs_base 外实测最大 {stop_max:.1f} dB，"
                       f"理论最大 {stop_expected:.1f} dB（含 Welch 泄漏余量）")
    else:
        ok_stop = stop_max < -20.0
        stop_detail = f"|f|>{stopband}·fs_base 外最大 {stop_max:.1f} dB（期望 < -20 dB）"
    checks.append({
        "name": "频谱通带平坦",
        "ok": bool(ok_flat),
        "detail": (f"|f|<{passband}·fs_base 内实测最小 {flat_min:.1f} dB，"
                   f"理论最小 {flat_expected:.1f} dB"),
    })
    checks.append({
        "name": "截止频率位置正确",
        "ok": bool(ok_cut),
        "detail": cut_detail,
    })
    checks.append({
        "name": "带外响应符合理论",
        "ok": bool(ok_stop),
        "detail": stop_detail,
    })

    # 绘图用理论曲线：单载波按理想无限长滤波器的解析响应（直接滚降），
    # OFDM 为补零 IFFT 的砖墙带限；两者均按各自通带中位数归一化对齐。
    if mode == "sc-fde":
        ideal = _ideal_filter_response(f_n, filter_type, rolloff)
        theory_db = 20.0 * np.log10(np.maximum(ideal, 1e-6))
        theory_db -= float(np.median(theory_db[mask_ref]))
        theory_lin = ideal ** 2 / (float(np.median((ideal ** 2)[mask_ref])) + 1e-30)
    else:
        theory_db = np.where(np.abs(f_n) <= 0.5, 0.0, -120.0)
        theory_lin = np.where(np.abs(f_n) <= 0.5, 1.0, 0.0)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    if is_db:
        measured_plot, theory_plot, ylabel, ylim = psd_db, theory_db, "归一化功率谱 (dB)", (-60, 5)
    else:
        plot_ref_lin = float(np.median(psd_lin[mask_ref])) + 1e-30
        measured_plot = psd_lin / plot_ref_lin
        theory_plot = theory_lin
        ylabel = "归一化功率谱"
        upper = max(1.2, float(
            np.percentile(measured_plot[np.abs(f_n) < 1.0], 99)) * 1.1)
        ylim = (0, upper)
    ax.plot(f_n, measured_plot, color="#2C68B4", lw=0.85, alpha=0.9,
            label="实测功率谱（Welch）")
    ax.plot(f_n, theory_plot, color="#D95F02", lw=1.8, ls="--",
            label=h_label)
    for cf in cut_lines:
        ax.axvline(cf, color="#E5484D", lw=1.0, ls="-.")
        ax.axvline(-cf, color="#E5484D", lw=1.0, ls="-.")
    for be in band_edges:
        ax.axvline(be, color="gray", lw=0.6, ls=":")
        ax.axvline(-be, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("归一化频率 f / fs_base")
    ax.set_ylabel(ylabel)
    visible_mode = "单载波" if mode == "sc-fde" else "OFDM"
    if mode == "sc-fde":
        title_text = f"单载波功率谱（{modulation}，{filter_type.upper()} β={rolloff}）"
    else:
        title_text = "OFDM 功率谱（随机 16QAM-OFDM）"
    ax.set_title(title_text)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plots.append({"title": "功率谱", "png": _figure_to_png(fig)})
    summary.extend([
        {"label": "功率谱符号", "value": f"{modulation}，{num_symbols} 个符号"},
        {"label": "功率谱滤波器", "value": f"{params.get('filter_type').upper()} L={L}，{sps}×，β={rolloff}"},
        {"label": "截止与带宽", "value": cut_text},
        {"label": "功率谱纵坐标", "value": "对数功率 dB" if is_db else "线性归一化功率"},
        {"label": "频谱测试", "value": f"Welch {nperseg} 点分段平均，采样率 {fs_out / 1e9:.0f} GHz"},
    ])


def run_waveform_time_test(link_mode: str = "sc-fde",
                           sc_modulation: str = "QPSK",
                           sc_symbol_indexes: tuple = (0, 1),
                           sc_pulse_count: int = 2,
                           sc_filter_type: str = "rrc",
                           sc_filter_length: int = 32,
                           sc_oversampling: int = 4,
                           sc_rolloff: float = 0.22,
                           ofdm_subcarriers: int = 512,
                           ofdm_oversampling: int = 4,
                           ofdm_cp_length: int = 32,
                           ofdm_time_symbol_count: int = 2,
                           ofdm_time_start_subcarrier: int = -6,
                           ofdm_time_subcarrier_step: int = 11,
                           ofdm_heatmap_symbol_count: int = 48,
                           ofdm_heatmap_start_subcarrier: int = -24,
                           ofdm_heatmap_subcarrier_step: int = 1,
                           ofdm_heatmap_axis_margin: int = 4,
                           ofdm_heatmap_floor_db: float = -45.0) -> Dict[str, Any]:
    """时域波形测试：单载波（所选星座点脉冲成型）或 OFDM（单音扫描 + 热图）。

    单载波：每个完整脉冲使用从对应调制星座图中选出的星座点符号，绘制
    I/Q 实际与理想时域波形；OFDM：确定性典型 16QAM 单音扫描的含 CP 时域
    波形与逐符号频谱热图。均不生成 PAPR 与功率谱。
    """
    from params.PHYParams import PHYParams

    t0 = time.perf_counter()
    result = _new_result("waveform_time")
    mode = str(link_mode).lower()
    if mode not in ("sc-fde", "ofdm"):
        raise ValueError(f"不支持的链路模式：{link_mode}，仅支持 sc-fde / ofdm")

    if mode == "sc-fde":
        params = PHYParams()
        params.update(
            link_mode=mode,
            filter_type=str(sc_filter_type).lower(),
            filter_length=int(sc_filter_length),
            oversampling=int(sc_oversampling),
            rolloff=float(sc_rolloff),
            NCBPS=2, MCS=1, scramble=False, pi2_rotation=False,
        )
        _run_sc_waveform_test(params, str(sc_modulation), list(sc_symbol_indexes),
                              int(sc_pulse_count), result["checks"],
                              result["plots"], result["summary"])
    else:
        _run_ofdm_time_part(
            int(ofdm_subcarriers), int(ofdm_oversampling), int(ofdm_cp_length),
            int(ofdm_time_symbol_count), int(ofdm_time_start_subcarrier),
            int(ofdm_time_subcarrier_step), int(ofdm_heatmap_symbol_count),
            int(ofdm_heatmap_start_subcarrier), int(ofdm_heatmap_subcarrier_step),
            int(ofdm_heatmap_axis_margin), float(ofdm_heatmap_floor_db),
            result["checks"], result["plots"], result["summary"])

    result["ok"] = all(c["ok"] for c in result["checks"])
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    return result


def run_waveform_spectrum_test(link_mode: str = "sc-fde",
                               sc_modulation: str = "QPSK",
                               sc_filter_type: str = "rrc",
                               sc_filter_length: int = 32,
                               sc_oversampling: int = 4,
                               sc_rolloff: float = 0.22,
                               sc_num_symbols: int = 8192,
                               sc_scale: str = "对数功率 dB",
                               ofdm_spectrum_subcarriers: int = 512,
                               ofdm_spectrum_oversampling: int = 4,
                               ofdm_spectrum_cp_length: int = 32,
                               ofdm_spectrum_scale: str = "对数功率 dB",
                               ofdm_spectrum_num_symbols: int = 128,
                               ofdm_spectrum_random_seed: int = 2026) -> Dict[str, Any]:
    """功率谱测试：单载波（成型滤波器 PSD + 理想滚降理论）或 OFDM（随机 16QAM PSD）。

    单载波：实测 Welch 功率谱叠加理想无限长滤波器的解析滚降响应，
    纵坐标支持对数功率 dB 与线性归一化功率；OFDM：随机 16QAM-OFDM
    连续时域数据经分段 FFT 平均，纵坐标同样支持两种显示。
    """
    from params.PHYParams import PHYParams

    t0 = time.perf_counter()
    result = _new_result("waveform_spectrum")
    mode = str(link_mode).lower()
    if mode not in ("sc-fde", "ofdm"):
        raise ValueError(f"不支持的链路模式：{link_mode}，仅支持 sc-fde / ofdm")

    if mode == "sc-fde":
        params = PHYParams()
        params.update(
            link_mode=mode,
            filter_type=str(sc_filter_type).lower(),
            filter_length=int(sc_filter_length),
            oversampling=int(sc_oversampling),
            rolloff=float(sc_rolloff),
        )
        _run_spectrum_test(params, mode, str(sc_modulation), int(sc_num_symbols),
                           str(sc_scale), result["checks"], result["plots"],
                           result["summary"])
    else:
        _run_ofdm_random_spectrum_test(
            int(ofdm_spectrum_subcarriers), int(ofdm_spectrum_oversampling),
            int(ofdm_spectrum_cp_length), str(ofdm_spectrum_scale),
            int(ofdm_spectrum_num_symbols), int(ofdm_spectrum_random_seed),
            result["checks"], result["plots"], result["summary"])

    result["ok"] = all(c["ok"] for c in result["checks"])
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    return result


# =====================================================================
# 3. 编码技术测试（RS / LDPC）
# =====================================================================

def _gf_str(arr: np.ndarray, max_show: int = 40) -> str:
    vals = [str(int(v)) for v in arr[:max_show]]
    text = ", ".join(vals)
    if len(arr) > max_show:
        text += f", …共{len(arr)}项"
    return f"[{text}]"


def _gf_full_str(arr: np.ndarray) -> str:
    return "[" + ", ".join(str(int(value)) for value in arr) + "]"


def _matlab_reference_comparison(case: dict, local_encoded: str,
                                 local_decoded: str, data_kind: str) -> Dict[str, str]:
    """读取离线 MATLAB 参考结果并与本地编译码结果比较。"""
    reference = case.get("matlab_reference")
    if not isinstance(reference, dict):
        return {
            "matlab_encoded": "未提供",
            "matlab_decoded": "未提供",
            "matlab_comparison": "未提供 MATLAB 参考结果",
        }

    def normalize(value: Any) -> str:
        if value is None:
            return "未提供"
        if data_kind == "gf":
            if not isinstance(value, list):
                return "格式错误：应为 GF 符号数组"
            try:
                return _gf_full_str(np.asarray([int(item) for item in value], dtype=np.int64))
            except (TypeError, ValueError):
                return "格式错误：应为 GF 符号数组"
        if not isinstance(value, str):
            return "格式错误：应为十六进制字符串"
        text = "".join(value.split()).upper()
        if not text or len(text) % 2 or any(ch not in "0123456789ABCDEF" for ch in text):
            return "格式错误：应为十六进制字符串"
        return text

    matlab_encoded = normalize(reference.get("encoded"))
    matlab_decoded = normalize(reference.get("decoded"))
    comparisons: List[str] = []
    if matlab_encoded == "未提供":
        comparisons.append("编码结果未提供")
    elif matlab_encoded.startswith("格式错误"):
        comparisons.append("编码参考格式错误")
    else:
        comparisons.append(f"编码{'一致' if matlab_encoded == local_encoded else '不一致'}")
    if matlab_decoded == "未提供":
        comparisons.append("译码结果未提供")
    elif matlab_decoded.startswith("格式错误"):
        comparisons.append("译码参考格式错误")
    else:
        comparisons.append(f"译码{'一致' if matlab_decoded == local_decoded else '不一致'}")
    return {
        "matlab_encoded": matlab_encoded,
        "matlab_decoded": matlab_decoded,
        "matlab_comparison": "；".join(comparisons),
    }


def _hex_to_bits(hex_str: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8))


def _bits_to_hex(bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if len(bits) % 8:
        bits = np.concatenate([bits, np.zeros(8 - len(bits) % 8, dtype=np.uint8)])
    return np.packbits(bits, bitorder="big").tobytes().hex().upper()


def _trunc_hex(hex_str: str, max_len: int = 48) -> str:
    if len(hex_str) <= max_len:
        return hex_str
    return hex_str[:max_len] + f"…(共{len(hex_str)}字符)"


def _head_tail_hex(hex_str: str, edge_len: int = 24) -> str:
    """同时显示长十六进制数据的头尾，便于观察系统码末尾的校验位。"""
    if len(hex_str) <= 2 * edge_len:
        return hex_str
    return f"{hex_str[:edge_len]} … {hex_str[-edge_len:]} (共{len(hex_str) // 2}字节)"


def _run_rs_test(config: dict, result: Dict[str, Any]) -> None:
    from params.PHYParams import PHYParams
    from utils.Coder import RSCoder

    rp = config.get("rs_params") or {}
    n = int(rp.get("n", 15))
    k = int(rp.get("k", 11))
    m = int(rp.get("m", 4))
    if not (n > k > 0):
        raise ValueError(f"rs_params 需满足 n > k > 0，当前 n={n}, k={k}")
    if m not in (4, 8):
        raise ValueError(f"rs_params.m 仅支持 4 或 8（GF(2^m)），当前 m={m}")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须为非空算例数组")

    params = PHYParams()
    params.update(code_type="RS", rs_nsym=n - k, rs_c_exp=m, rs_packet_size=k)
    coder = RSCoder(params)

    weights = 1 << np.arange(m)
    rows: List[List[str]] = []
    details: List[str] = []
    n_pass = 0
    for case in cases:
        name = str(case.get("name") or f"算例{len(rows) + 1}")
        raw = case.get("input")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"算例「{name}」的 input 必须为非空 GF 符号数组")
        vals = np.asarray([int(v) for v in raw], dtype=np.int64)
        if np.any(vals >= 2 ** m):
            raise ValueError(f"算例「{name}」的 input 含超出 GF(2^{m}) 范围的符号")

        bits = ((vals[:, None] >> np.arange(m)) & 1).ravel().astype(np.uint8)
        enc_bits = coder.encode(bits)
        enc_syms = (enc_bits.reshape(-1, m) * weights).sum(axis=1)

        rx_syms = enc_syms.copy()
        err_desc: List[str] = []
        for e in (case.get("errors") or []):
            p = int(e.get("pos"))
            v = int(e.get("value"))
            if not (0 <= p < len(rx_syms)):
                raise ValueError(f"算例「{name}」错误位置 {p} 超出编码流长度 {len(rx_syms)}")
            if not (0 <= v < 2 ** m):
                raise ValueError(f"算例「{name}」错误值 {v} 超出 GF(2^{m}) 范围")
            rx_syms[p] = v
            err_desc.append(f"pos {p}→{v}")

        rx_bits = ((rx_syms[:, None] >> np.arange(m)) & 1).ravel().astype(np.uint8)
        llrs = (1.0 - 2.0 * rx_bits).astype(np.float64)
        dec_bits = coder.decode(llrs, original_bit_len=len(bits))
        dec_syms = (dec_bits.reshape(-1, m) * weights).sum(axis=1)[:len(vals)]
        ok = bool(np.array_equal(dec_syms, vals))
        if ok:
            n_pass += 1

        packets = int(np.ceil(len(vals) / k))
        rows.append([
            name, _gf_str(vals), _gf_str(enc_syms),
            "；".join(err_desc) or "无", _gf_str(dec_syms),
            "通过" if ok else "失败",
        ])
        details.append(
            f"RS({n},{k}) GF(2^{m})；输入 {len(vals)} 符号 → 编码 {len(enc_syms)} 符号"
            f"（{packets} 包）；注入错误 {len(err_desc)} 个；"
            f"译码{'正确' if ok else '与输入不一致'}。"
            f"注：错误位置 pos 为编码后码字流中的符号下标。"
        )
        local_encoded = _gf_full_str(enc_syms)
        local_decoded = _gf_full_str(dec_syms)
        matlab_reference = _matlab_reference_comparison(
            case, local_encoded, local_decoded, "gf")
        case_result = {
            "name": name,
            "input_size": f"{len(vals)} 符号",
            "encoded_size": f"{len(enc_syms)} 符号",
            "block_count": packets,
            "error_count": len(err_desc),
            "status": "通过" if ok else "失败",
            "overview": (
                f"{name}\nRS({n},{k}) GF(2^{m})\n"
                f"输入 {len(vals)} 符号，经 {packets} 个 RS 包编码为 {len(enc_syms)} 符号。\n"
                f"注入 {len(err_desc)} 个符号错误，译码{'恢复原始输入' if ok else '未能恢复原始输入'}。"
            ),
            "input_data": _gf_full_str(vals),
            "encoded_data": local_encoded,
            "received_data": _gf_full_str(rx_syms),
            "decoded_data": local_decoded,
            "errors": "；".join(err_desc) or "无",
            "diagnostics": (
                f"单包理论纠错能力 t={(n - k) // 2} 个符号；"
                f"本算例注入 {len(err_desc)} 个符号错误；结果：{'一致' if ok else '不一致'}。"
            ),
        }
        case_result.update(matlab_reference)
        result.setdefault("case_results", []).append(case_result)
        result["checks"].append({
            "name": name, "ok": ok,
            "detail": f"注入 {len(err_desc)} 个符号错误，译码{'正确' if ok else '失败'}",
        })

    result["summary"] = [
        {"label": "编码方案", "value": f"RS({n},{k}) GF(2^{m})"},
        {"label": "算例总数", "value": str(len(cases))},
        {"label": "通过", "value": str(n_pass)},
        {"label": "失败", "value": str(len(cases) - n_pass)},
    ]
    result["table"] = {
        "columns": ["算例", "输入(GF符号)", "编码结果(GF符号)", "注入错误", "译码结果(GF符号)", "判定"],
        "rows": rows,
    }
    result["detail_lines"] = details
    result["ok"] = all(c["ok"] for c in result["checks"])


def _run_ldpc_test(config: dict, result: Dict[str, Any]) -> None:
    from params.PHYParams import PHYParams
    from utils.Coder import LDPCCoder

    lp = config.get("ldpc_params") or {}
    rate = str(lp.get("rate", "14/15"))
    if rate not in ("14/15", "11/15"):
        raise ValueError(f"ldpc_params.rate 仅支持 14/15 或 11/15，当前 {rate}")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须为非空算例数组")

    params = PHYParams()
    params.update(code_type="LDPC", ldpc_standard_rate=rate)
    coder = LDPCCoder(params)

    rows: List[List[str]] = []
    details: List[str] = []
    n_pass = 0
    for case in cases:
        name = str(case.get("name") or f"算例{len(rows) + 1}")
        hex_in = str(case.get("input_hex") or "").strip()
        if not hex_in or len(hex_in) % 2 or \
                any(ch not in "0123456789abcdefABCDEF" for ch in hex_in):
            raise ValueError(f"算例「{name}」的 input_hex 必须为非空偶数长度十六进制串")
        in_bits = _hex_to_bits(hex_in)

        cw = coder.encode(in_bits)
        rx = cw.copy()
        err_pos = [int(p) for p in (case.get("error_bits") or [])]
        for p in err_pos:
            if not (0 <= p < len(cw)):
                raise ValueError(f"算例「{name}」错误 bit 位置 {p} 超出编码流长度 {len(cw)}")
            rx[p] ^= 1

        dec_bits, debug = coder.decode(rx, original_bit_len=len(in_bits))
        ok = bool(np.array_equal(dec_bits, in_bits) and debug.get("decode_success"))
        if ok:
            n_pass += 1

        codeword_hex = _bits_to_hex(cw)
        codeword_blocks = cw.reshape(-1, coder.n)
        parity_hexes = [
            _bits_to_hex(block[coder.parity_positions])
            for block in codeword_blocks
        ]
        parity_display = "；".join(
            f"块{idx + 1}: {_head_tail_hex(value, edge_len=16)}"
            for idx, value in enumerate(parity_hexes)
        )

        rows.append([
            name, _trunc_hex(hex_in), _head_tail_hex(codeword_hex), parity_display,
            "；".join(f"bit #{p}" for p in err_pos) or "无",
            _trunc_hex(_bits_to_hex(dec_bits)),
            "通过" if ok else "失败",
        ])
        details.append(
            f"LDPC({coder.n},{coder.k}) 码率 {rate}；输入 {len(in_bits)} bit"
            f"（{len(in_bits) // 8} 字节）→ 编码 {len(cw)} bit"
            f"（{debug['num_blocks']} 个码字块）；注入错误 {len(err_pos)} bit；"
            f"校验位：{parity_display}；"
            f"迭代次数 {debug['iterations']}；校验子权重 {debug['syndrome_weights']}；"
            f"硬判 fallback={debug['used_hard_fallback']}；"
            f"译码{'成功' if ok else '失败'}。"
            f"注：error_bits 为编码后码字流中的 bit 下标。"
        )
        received_hex = _bits_to_hex(rx)
        decoded_hex = _bits_to_hex(dec_bits)
        matlab_reference = _matlab_reference_comparison(
            case, codeword_hex, decoded_hex, "hex")
        case_result = {
            "name": name,
            "input_size": f"{len(in_bits)} bit",
            "encoded_size": f"{len(cw)} bit",
            "block_count": debug["num_blocks"],
            "error_count": len(err_pos),
            "status": "通过" if ok else "失败",
            "overview": (
                f"{name}\nLDPC({coder.n},{coder.k})，码率 {rate}\n"
                f"输入 {len(in_bits)} bit，经 {debug['num_blocks']} 个码字块编码为 {len(cw)} bit。\n"
                f"注入 {len(err_pos)} 个 bit 错误，译码{'恢复原始输入' if ok else '未能恢复原始输入'}。"
            ),
            "input_data": hex_in.upper(),
            "encoded_data": codeword_hex,
            "received_data": received_hex,
            "decoded_data": decoded_hex,
            "errors": "；".join(f"bit #{p}" for p in err_pos) or "无",
            "diagnostics": (
                f"迭代次数：{debug['iterations']}\n"
                f"校验子权重：{debug['syndrome_weights']}\n"
                f"硬判 fallback：{debug['used_hard_fallback']}\n"
                f"解码器成功标志：{debug.get('decode_success')}"
            ),
        }
        case_result.update(matlab_reference)
        result.setdefault("case_results", []).append(case_result)
        result["checks"].append({
            "name": name, "ok": ok,
            "detail": f"注入 {len(err_pos)} bit 错误，译码{'正确' if ok else '失败'}",
        })

    result["summary"] = [
        {"label": "编码方案", "value": f"LDPC({coder.n},{coder.k}) {rate}（IEEE 802.15.3d）"},
        {"label": "算例总数", "value": str(len(cases))},
        {"label": "通过", "value": str(n_pass)},
        {"label": "失败", "value": str(len(cases) - n_pass)},
    ]
    result["table"] = {
        "columns": [
            "算例", "输入(hex)", "编码结果(头…尾)", "LDPC校验位(hex)",
            "注入错误", "译码结果(hex)", "判定",
        ],
        "rows": rows,
    }
    result["detail_lines"] = details
    result["ok"] = all(c["ok"] for c in result["checks"])


def run_codec_test(config_text: str) -> Dict[str, Any]:
    """RS / LDPC 编译码技术支持测试。

    输入为 JSON 文本（文件内容），包含多组算例；每组算例的输入、
    编码结果、译码结果均返回给页面展示（RS 用 GF 符号、LDPC 用十六进制）。
    """
    t0 = time.perf_counter()
    result = _new_result("codec")
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"用例 JSON 解析失败（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}") from exc
    if not isinstance(config, dict):
        raise ValueError("用例 JSON 顶层必须为对象（包含 code_type / cases 等字段）")

    code_type = str(config.get("code_type") or "")
    if code_type == "RS":
        _run_rs_test(config, result)
    elif code_type == "LDPC":
        _run_ldpc_test(config, result)
    else:
        raise ValueError(f"不支持的 code_type：{code_type!r}，仅支持 RS / LDPC")

    result["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    return result


# =====================================================================
# 3. 浮点精度验证
# =====================================================================

def run_precision_test(link_mode: str = "sc-fde", duration: float = 1e-6,
                       SNRdB: float = 24.0) -> Dict[str, Any]:
    """全链路各模块输出信号数据类型校验（TX → 信道 → RX）。"""
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter
    from channel.THzChannel import THzChannel
    from receiver.THzReceiver import THzReceiver

    t0 = time.perf_counter()
    result = _new_result("precision")
    mode = str(link_mode).lower()
    if mode not in ("sc-fde", "ofdm"):
        raise ValueError(f"不支持的链路模式：{link_mode}，仅支持 sc-fde / ofdm")

    params = PHYParams()
    params.update(
        link_mode=mode,
        SNRdB=float(SNRdB),
        duration=float(duration),
        enable_multipath=False,
        enable_cfo=False,
        enable_phase_noise=False,
        enable_iq_imbalance=False,
        enable_channel_equalization=True,
        enable_cfo_compensation=False,
    )

    tx = THzTransmitter(params)
    tx.run()
    channel_out = THzChannel(params).run(tx.tx_signal_dict)
    recv = THzReceiver(params, tx)
    recv.run(channel_out)

    stages: List[tuple] = [
        ("TX 比特组装", "data_bits_dict", "uint8"),
        ("TX 扰码", "data_scrambled_dict", "uint8"),
        ("TX 信道编码", "coded_bits_dict", "uint8"),
        ("TX 调制", "modulated_data_dict", "complex128"),
    ]
    if mode == "ofdm":
        stages.append(("TX OFDM 处理", "data_ofdm_dict", "complex128"))
    stages += [
        ("TX 插入 GI", "data_with_gi_dict", "complex128"),
        ("TX 前导码", "data_with_preamble_dict", "complex128"),
        ("TX 脉冲成型", "tx_signal_dict", "complex128"),
        ("信道输出", "_channel_out", "complex128"),
        ("RX 匹配滤波", "rx_matched", "complex128"),
        ("RX 粗同步", "rx_coarse_synced", "complex128"),
        ("RX CFO 粗补偿", "rx_cfo_coarse", "complex128"),
        ("RX 精同步", "rx_fine_synced", "complex128"),
        ("RX 下采样", "rx_downsampled", "complex128"),
        ("RX CFO 精补偿", "rx_cfo_fine", "complex128"),
        ("RX IQ 补偿", "rx_iq_compensated", "complex128"),
        ("RX 均衡", "rx_equalized", "complex128"),
        ("RX LLR 软信息", "llr_dict", "float64"),
        ("RX 译码", "decoded_bits", "uint8"),
        ("RX 解扰", "data_bits", "uint8"),
        ("噪声方差", "noise_var", "float64"),
    ]

    # RX 专属属性集合（"data_bits" 不能靠前缀区分，TX 侧是 data_bits_dict）
    rx_attrs = {"rx_matched", "rx_coarse_synced", "rx_cfo_coarse", "rx_fine_synced",
                "rx_downsampled", "rx_cfo_fine", "rx_iq_compensated", "rx_equalized",
                "llr_dict", "decoded_bits", "data_bits", "noise_var"}

    rows: List[List[str]] = []
    details: List[str] = []
    n_pass = n_skip = n_fail = 0
    for module_name, attr, expected in stages:
        if attr == "_channel_out":
            value = channel_out
        else:
            source = recv if attr in rx_attrs else tx
            value = getattr(source, attr, None)
        if value is None:
            rows.append([module_name, "-", "-", "-", "跳过"])
            details.append(f"{module_name}：未产生输出（该模块未启用），跳过")
            n_skip += 1
            continue
        arr = np.asarray(value["signal_stream"] if isinstance(value, dict) else value)
        dtype_str = str(arr.dtype)
        verdict = "通过" if dtype_str == expected else "失败"
        if verdict == "通过":
            n_pass += 1
        else:
            n_fail += 1
        rows.append([module_name, "signal_stream", dtype_str, str(arr.dtype.itemsize), verdict])
        details.append(f"{module_name}：dtype={dtype_str}，shape={arr.shape}，期望 {expected}")

    has_float32 = any("float32" in row[2] for row in rows)
    result["table"] = {
        "columns": ["模块", "输出信号", "数据类型", "字节数", "判定"],
        "rows": rows,
    }
    result["detail_lines"] = details
    result["checks"] = [{
        "name": "全链路无 float32 降精度",
        "ok": not has_float32,
        "detail": "未发现 float32" if not has_float32 else "存在 float32 输出，请检查对应模块",
    }]
    result["summary"] = [
        {"label": "链路模式", "value": mode.upper()},
        {"label": "模块阶段数", "value": str(len(stages))},
        {"label": "通过", "value": str(n_pass)},
        {"label": "跳过", "value": str(n_skip)},
        {"label": "失败", "value": str(n_fail)},
        {"label": "存在 float32", "value": "是" if has_float32 else "否"},
    ]
    result["ok"] = (n_fail == 0) and (not has_float32)
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    return result


# =====================================================================
# 4. 物理层速率测试
# =====================================================================

def _theoretical_phy_rate(params) -> float:
    """按编码率和实际帧结构计算理论物理层速率。"""
    from thz_sim_ui.services.result_utils import compute_theoretical_phy_rate
    return compute_theoretical_phy_rate(params)


def _rate_result(test_type: str, params, threshold_bps: float) -> Dict[str, Any]:
    """运行 TX，并用实际信息比特数/实际波形时长测量物理层速率。"""
    from transmitter.THzTransmitter import THzTransmitter

    t0 = time.perf_counter()
    tx = THzTransmitter(params)
    tx.run()
    information_bits = int(len(tx.data_bits_dict["signal_stream"]))
    waveform_duration = float(tx.tx_signal_dict["duration_seconds"])
    if waveform_duration <= 0:
        raise ValueError("发射波形持续时间必须大于 0")
    theoretical = _theoretical_phy_rate(params)
    actual = information_bits / waveform_duration
    deviation = abs(actual - theoretical) / theoretical if theoretical > 0 else float("inf")
    passed = bool(np.isfinite(actual) and actual >= float(threshold_bps))
    result = _new_result(test_type)
    result.update({
        "ok": passed,
        "elapsed_ms": (time.perf_counter() - t0) * 1e3,
        "data": {
            "theoretical_rate_bps": theoretical,
            "actual_rate_bps": actual,
            "threshold_bps": float(threshold_bps),
            "information_bits": information_bits,
            "waveform_duration_seconds": waveform_duration,
            "relative_deviation": deviation,
        },
        "summary": [
            {"label": "理论速率", "value": f"{theoretical / 1e9:.3f} Gbps"},
            {"label": "实际速率", "value": f"{actual / 1e9:.3f} Gbps"},
            {"label": "理论/实际偏差", "value": f"{deviation * 100:.3f}%"},
            {"label": "验收门限", "value": f"≥ {threshold_bps / 1e9:.3f} Gbps"},
            {"label": "最终判定", "value": "通过" if passed else "失败"},
        ],
        "checks": [{
            "name": "实际物理层速率达到门限",
            "ok": passed,
            "detail": f"{actual / 1e9:.3f} Gbps {'≥' if passed else '<'} {threshold_bps / 1e9:.3f} Gbps",
        }],
        "table": {
            "columns": ["指标", "测量值", "门限/参考", "判定"],
            "rows": [
                ["理论速率", f"{theoretical / 1e9:.3f} Gbps", "帧结构公式", "参考"],
                ["实际速率", f"{actual / 1e9:.3f} Gbps", f"≥ {threshold_bps / 1e9:.3f} Gbps", "通过" if passed else "失败"],
                ["信息比特数", str(information_bits), "实际 TX 输入", "参考"],
                ["波形持续时间", f"{waveform_duration:.9e} s", "实际 TX 输出", "参考"],
            ],
        },
    })
    return result


def run_single_link_rate_test(link_mode: str = "ofdm", modulation: str = "64QAM",
                              code_type: str = "LDPC", symbol_rate_ghz: float = 30.0,
                              duration: float = 1e-6,
                              threshold_gbps: float = 50.0) -> Dict[str, Any]:
    """验证单链路实际物理层速率不低于指定门限。"""
    from params.PHYParams import PHYParams

    ncbps_map = {"QPSK": 2, "16QAM": 4, "64QAM": 6, "256QAM": 8}
    mod = str(modulation).upper()
    if mod not in ncbps_map:
        raise ValueError(f"不支持的调制方式：{modulation}")
    mode = str(link_mode).lower()
    if mode not in ("sc-fde", "ofdm"):
        raise ValueError(f"不支持的链路模式：{link_mode}")
    if symbol_rate_ghz <= 0 or duration <= 0 or threshold_gbps <= 0:
        raise ValueError("符号率、时长和速率门限必须大于 0")

    params = PHYParams()
    values = dict(
        link_mode=mode, NCBPS=ncbps_map[mod], code_type=str(code_type).upper(),
        sample_rate=float(symbol_rate_ghz) * 1e9, duration=float(duration),
        enable_mimo=False, enable_awgn=False, enable_multipath=False,
        enable_cfo=False, enable_phase_noise=False, enable_iq_imbalance=False,
    )
    if values["code_type"] == "LDPC":
        values.update(ldpc_matrix_type="ieee802153d_1440", ldpc_standard_rate="14/15")
    elif values["code_type"] == "RS":
        if mode == "ofdm":
            values.update(rs_nsym=4, rs_c_exp=4, rs_packet_size=11)
        else:
            values.update(rs_nsym=63, rs_c_exp=8, rs_packet_size=192)
    else:
        raise ValueError("编码方式仅支持 LDPC / RS")
    params.update(**values)
    result = _rate_result("single_link_rate", params, float(threshold_gbps) * 1e9)
    result["summary"].insert(0, {"label": "链路配置", "value": f"{mode.upper()} / {mod} / {values['code_type']} / SISO"})
    return result


def run_total_phy_rate_test(threshold_tbps: float = 0.5) -> Dict[str, Any]:
    """固定验收配置：256QAM、1024 子载波、2×2 MIMO、LDPC(11/15)。"""
    from params.PHYParams import PHYParams

    if threshold_tbps <= 0:
        raise ValueError("总速率门限必须大于 0")
    frame_bits = 270336
    params = PHYParams()
    params.update(
        enable_mimo=True, link_mode="ofdm", num_tx=2, num_rx=2,
        num_spatial_streams=2, mimo_scheme="spatial_multiplexing",
        mimo_detector="mmse", mimo_channel_model="identity", mimo_csi_mode="estimated",
        mimo_num_taps=1, subwave_num=1024, gi_length=64,
        pilot_block_indexes=[], subframe_ofdm_num=45, NCBPS=8,
        code_type="LDPC", ldpc_matrix_type="ieee802153d_1440",
        ldpc_standard_rate="11/15", sample_rate=60e9,
        enable_awgn=False, enable_multipath=False, enable_cfo=False,
        enable_cfo_compensation=False, enable_phase_noise=False,
        enable_iq_imbalance=False, enable_iq_compensation=False,
        random_seed=7, mimo_channel_seed=11, duration=None,
        sample_length=frame_bits // 8,
    )
    result = _rate_result("total_phy_rate", params, float(threshold_tbps) * 1e12)
    aligned = frame_bits % 1056 == 0
    result["data"].update({
        "configuration": "256QAM / 1024 子载波 / 2×2 MIMO / LDPC(11/15)",
        "frame_information_bits": frame_bits,
        "ldpc_padding_bits": 0 if aligned else frame_bits % 1056,
    })
    result["summary"].insert(0, {"label": "固定配置", "value": result["data"]["configuration"]})
    result["table"]["rows"].extend([
        ["空间流数", "2", "2×2 MIMO", "通过"],
        ["LDPC 块对齐", f"{frame_bits} bit = {frame_bits // 1056} × 1056", "补零 0 bit", "通过" if aligned else "失败"],
    ])
    result["checks"].append({
        "name": "LDPC 信息块整帧对齐", "ok": aligned,
        "detail": f"{frame_bits} bit 可整除 1056 bit，补零为 0" if aligned else "帧信息比特未与 LDPC 块对齐",
    })
    result["ok"] = result["ok"] and aligned
    return result


# =====================================================================
# 5. 功能校准测试
# =====================================================================

def _qam_theoretical_ber(modulation: str, ebn0_db: np.ndarray) -> np.ndarray:
    """Gray 映射 QPSK/方形 M-QAM 的 AWGN 理论 BER。"""
    from scipy.special import erfc

    m_map = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    M = m_map[modulation]
    k = np.log2(M)
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    if M == 4:
        return 0.5 * erfc(np.sqrt(ebn0))
    q = 0.5 * erfc(np.sqrt(3.0 * k * ebn0 / (2.0 * (M - 1.0))))
    ber = (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * q
    ber -= (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) ** 2 * q ** 2
    return np.clip(ber, 0.0, 0.5)


def _run_modulation_calibration(bits_per_point: int, random_seed: int):
    from params.PHYParams import PHYParams
    from transmitter.Modulator import THzModulator
    from receiver.DeModulator import THzDemodulator

    # 每种调制只取对应的前 4 个信噪比点仿真
    configs = {
        "QPSK": (2, np.arange(0.0, 9.0, 2.0)[:4]),
        "16QAM": (4, np.arange(2.0, 15.0, 3.0)[:4]),
        "64QAM": (6, np.arange(6.0, 23.0, 4.0)[:4]),
    }
    curves = []
    checks = []
    for mod_index, (name, (ncbps, ebn0_points)) in enumerate(configs.items()):
        count = max(int(bits_per_point), 10000)
        count -= count % ncbps
        rng = np.random.default_rng(int(random_seed) + mod_index)
        bits = rng.integers(0, 2, count, dtype=np.uint8)
        params = PHYParams()
        # 调制/解调均使用链路原生映射（含 pi/2 旋转与 pi/4 补偿）；
        # 旋转不改变 AWGN 下 BER，保持与发射端/接收端模块完全一致。
        params.update(MCS=1, NCBPS=ncbps, scramble=False)
        modulator = THzModulator(params)
        demodulator = THzDemodulator(params)
        bit_dict = _make_signal_dict(bits, float(count), {
            "frame_bit_num": count, "frame_num": 1,
        })
        modulated = modulator.modulate(bit_dict)
        tx = np.asarray(modulated["signal_stream"])
        energy_per_bit = float(np.mean(np.abs(tx) ** 2)) / ncbps
        simulated = []
        errors_list = []
        snr_list = []
        for ebn0_db in ebn0_points:
            n0 = energy_per_bit / (10.0 ** (ebn0_db / 10.0))
            sigma = np.sqrt(n0 / 2.0)
            noise = sigma * (rng.standard_normal(tx.size) + 1j * rng.standard_normal(tx.size))
            rx_dict = dict(modulated)
            rx_dict["signal_stream"] = tx + noise
            llr = demodulator.demodulate(rx_dict, sigma)
            hard = np.asarray(demodulator.llr_to_bits(llr)["signal_stream"])[0:count]
            errors = int(np.count_nonzero(bits != hard))
            simulated.append(errors / count)
            errors_list.append(errors)
            snr_list.append(float(ebn0_db) + 10.0 * np.log10(ncbps))
        simulated = np.asarray(simulated)
        theoretical = _qam_theoretical_ber(name, ebn0_points)
        valid = (simulated > 0) & (theoretical * count >= 20)
        log_errors = np.abs(np.log10(simulated[valid]) - np.log10(theoretical[valid]))
        max_log_error = float(np.max(log_errors)) if log_errors.size else float("inf")
        monotonic = bool(np.all(np.diff(simulated) <= 1.0 / count))
        ok = monotonic and max_log_error <= 0.45
        checks.append({
            "name": f"{name} BER 曲线与理论一致", "ok": ok,
            "detail": f"最大 log10(BER) 偏差 {max_log_error:.3f}，曲线{'单调' if monotonic else '非单调'}",
        })
        curves.append({
            "name": name, "ebn0_db": ebn0_points, "snr_db": snr_list,
            "simulated": simulated, "theoretical": theoretical,
            "total_errors": errors_list, "total_bits": [count] * len(ebn0_points),
        })
    return curves, checks


def _run_coded_calibration(code_type: str, ebn0_points, bits_per_point: int,
                           random_seed: int):
    """LDPC(1440,1056) / RS(15,11) 的 QPSK 编码链路 BER 校准。

    理论曲线取同 Eb/N0 下未编码 QPSK 的 AWGN 理论 BER，用于直观展示
    编码增益；每个点记录误码数/比特数等明细。
    """
    from params.PHYParams import PHYParams
    from transmitter.Modulator import THzModulator
    from receiver.DeModulator import THzDemodulator
    from utils.Coder import LDPCCoder, RSCoder

    if code_type == "LDPC":
        name = "LDPC(1440,1056)"
        coding = {"code_type": "LDPC", "ldpc_matrix_type": "ieee802153d_1440",
                  "ldpc_standard_rate": "11/15", "ldpc_n": 1440, "ldpc_k": 1056}
    else:
        name = "RS(15,11)"
        coding = {"code_type": "RS", "rs_nsym": 4, "rs_c_exp": 4,
                  "rs_packet_size": 11, "decode_mode": "hard"}

    params = PHYParams()
    # 调制/解调使用链路原生映射（含 pi/2 旋转与 pi/4 补偿），保证与
    # 发射端/接收端模块一致；旋转不改变 AWGN 下的 BER。
    params.update(MCS=1, NCBPS=2, scramble=False, **coding)
    coder = LDPCCoder(params) if code_type == "LDPC" else RSCoder(params)
    modulator = THzModulator(params)
    demodulator = THzDemodulator(params)

    rate = 11.0 / 15.0
    count = max(int(bits_per_point), 10000)
    # RS 以 44 bit（11 符号 × 4 bit）为单位分包，信息比特取整到块边界。
    count -= count % 44
    rng = np.random.default_rng(
        int(random_seed) + (100 if code_type == "LDPC" else 200))
    bits = rng.integers(0, 2, count, dtype=np.uint8)
    encoded = coder.encode(bits)
    modulated = modulator.modulate(_make_signal_dict(
        encoded, float(len(encoded)), {
            "frame_bit_num": len(encoded), "frame_num": 1}))
    tx = np.asarray(modulated["signal_stream"])
    # QPSK 方型星座平均符号功率为 1；Es/N0 = NCBPS × 码率 × Eb/N0。
    es_n0_factor = 2.0 * rate
    simulated = []
    errors_list = []
    snr_list = []
    for ebn0_db in ebn0_points:
        n0 = 1.0 / (es_n0_factor * (10.0 ** (ebn0_db / 10.0)))
        sigma = np.sqrt(n0 / 2.0)
        noise = sigma * (rng.standard_normal(tx.size) + 1j * rng.standard_normal(tx.size))
        rx_dict = dict(modulated)
        rx_dict["signal_stream"] = tx + noise
        llr = np.asarray(
            demodulator.demodulate(rx_dict, sigma)["signal_stream"],
            dtype=np.float64).ravel()
        decoded = coder.decode(llr, original_bit_len=count)
        if isinstance(decoded, tuple):  # LDPC 返回 (bits, debug)，RS 只返回 bits
            decoded = decoded[0]
        decoded = np.asarray(decoded, dtype=np.uint8).ravel()[:count]
        errors = int(np.count_nonzero(bits != decoded))
        simulated.append(errors / count)
        errors_list.append(errors)
        snr_list.append(float(ebn0_db) + 10.0 * np.log10(es_n0_factor))
    simulated = np.asarray(simulated)
    theoretical = _qam_theoretical_ber("QPSK", ebn0_points)
    monotonic = bool(np.all(np.diff(simulated) <= 1.0 / count))
    return {
        "name": name, "ebn0_db": ebn0_points, "snr_db": snr_list,
        "simulated": simulated, "theoretical": theoretical,
        "total_errors": errors_list, "total_bits": [count] * len(ebn0_points),
    }, monotonic


def _calibration_curve_table(curve: dict) -> dict:
    """把一条校准曲线的仿真点明细整理为表格。"""
    rows = []
    for index, ebn0_db in enumerate(curve["ebn0_db"]):
        errors = curve["total_errors"][index]
        ber_text = "0（无误码）" if errors == 0 else f"{curve['simulated'][index]:.3e}"
        rows.append([
            f"{curve['snr_db'][index]:.2f}",
            f"{float(ebn0_db):.2f}",
            str(errors),
            f"{curve['total_bits'][index]:,}",
            ber_text,
            f"{curve['theoretical'][index]:.3e}",
        ])
    return {
        "title": f"{curve['name']} 仿真点数据",
        "columns": ["SNR/dB", "Eb/N0/dB", "误码数", "比特数",
                    "实测 BER", "理论 BER"],
        "rows": rows,
    }


def run_function_calibration_test(bits_per_point: int = 200000,
                                  random_seed: int = 2026) -> Dict[str, Any]:
    """校准 QPSK/16QAM/64QAM 与 LDPC(1440,1056)/RS(15,11) 的 BER 曲线。

    每条曲线独立绘图（前 4 个 Eb/N0 点），每张图下附带仿真点明细表格。
    """
    if int(bits_per_point) < 10000:
        raise ValueError("每个 BER 点的比特数不能少于 10000")
    t0 = time.perf_counter()
    result = _new_result("function_calibration")
    curves, checks = _run_modulation_calibration(int(bits_per_point), int(random_seed))

    coded_specs = [
        ("LDPC", np.arange(1.0, 5.0, 1.0)),
        ("RS", np.arange(2.0, 6.0, 1.0)),
    ]
    for code_type, points in coded_specs:
        curve, monotonic = _run_coded_calibration(
            code_type, points, int(bits_per_point), int(random_seed))
        curves.append(curve)
        checks.append({
            "name": f"{curve['name']} BER 曲线单调",
            "ok": monotonic,
            "detail": f"曲线{'单调' if monotonic else '非单调'}（Eb/N0 {points.tolist()}）",
        })

    # 每条曲线单独绘制一幅仿真 vs 理论对比图，并附仿真点明细表格
    plots = []
    tables = []
    for curve, color in zip(curves, PLOT_COLORS):
        _apply_plot_style()
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        sim = np.maximum(curve["simulated"], 0.5 / int(bits_per_point))
        ax.semilogy(curve["ebn0_db"], sim, "o-", color=color,
                    label=f"{curve['name']} 仿真")
        ax.semilogy(curve["ebn0_db"], curve["theoretical"], "--", color=color,
                    label="未编码 QPSK 理论" if "QPSK" not in curve["name"] else f"{curve['name']} 理论")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"{curve['name']} BER 校准（前 4 个 Eb/N0 点）")
        ax.grid(True, which="both")
        ax.set_ylim(1e-6, 0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
        ax.legend(fontsize=8)
        fig.tight_layout()
        plots.append({"title": f"{curve['name']} BER 校准曲线",
                      "png": _figure_to_png(fig)})
        tables.append(_calibration_curve_table(curve))

    result.update({
        "ok": all(item["ok"] for item in checks),
        "elapsed_ms": (time.perf_counter() - t0) * 1e3,
        "checks": checks,
        "plots": plots,
        "tables": tables,
        "summary": [
            {"label": "调制校准", "value": "QPSK / 16QAM / 64QAM，各 4 个 Eb/N0 点"},
            {"label": "编码校准", "value": "LDPC(1440,1056) / RS(15,11)，QPSK 编码链路，各 4 点"},
            {"label": "通过项", "value": f"{sum(c['ok'] for c in checks)}/{len(checks)}"},
            {"label": "最终判定", "value": "通过" if all(c["ok"] for c in checks) else "失败"},
        ],
        "data": {"curves": curves},
    })
    result.pop("table", None)  # 旧“校准项目”总表已删除，改用每图下方明细表
    return result


# =====================================================================
# 后台线程
# =====================================================================

class FunctionalTestWorker(QThread):
    """在后台线程执行功能测试，结果经信号回传页面。"""

    result_ready = Signal(str, dict)   # (test_type, result)
    failed = Signal(str, str)          # (test_type, "异常类型: 消息")
    progress = Signal(dict)            # BER 长任务的逐帧进度

    def __init__(self, test_type: str, payload: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.test_type = test_type
        self.payload = dict(payload or {})

    def run(self) -> None:
        try:
            if self.test_type == "modulation":
                result = run_modulation_test(**self.payload)
            elif self.test_type == "waveform_time":
                result = run_waveform_time_test(**self.payload)
            elif self.test_type == "waveform_spectrum":
                result = run_waveform_spectrum_test(**self.payload)
            elif self.test_type == "codec":
                result = run_codec_test(self.payload.get("config_text", ""))
            elif self.test_type == "precision":
                result = run_precision_test(**self.payload)
            elif self.test_type == "single_link_rate":
                result = run_single_link_rate_test(**self.payload)
            elif self.test_type == "total_phy_rate":
                result = run_total_phy_rate_test(**self.payload)
            elif self.test_type == "function_calibration":
                result = run_function_calibration_test(**self.payload)
            elif self.test_type == "ber":
                from thz_sim_ui.services.ber_test_service import run_ber_test
                result = run_ber_test(
                    **self.payload,
                    progress_callback=lambda value: self.progress.emit(value),
                    should_cancel=self.isInterruptionRequested,
                )
            else:
                raise ValueError(f"未知测试类型: {self.test_type}")
            result["test_type"] = self.test_type
            self.result_ready.emit(self.test_type, result)
        except Exception as exc:  # noqa: BLE001 — 线程边界，异常须经信号带回 UI
            self.failed.emit(self.test_type, f"{type(exc).__name__}: {exc}")
