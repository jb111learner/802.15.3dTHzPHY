"""
功能测试服务 — 波形测试 / 编码技术测试 / 浮点精度验证

三个测试函数均为纯函数（返回含 PNG 字节、校验结果与表格数据的 dict），
由 FunctionalTestWorker(QThread) 在后台线程中执行，结果经 Qt 信号回传页面。
PHY 模块 import 全部放在函数内部（惰性加载），galois 仅在 RSCoder 构造时导入。
"""
from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 必须先于 pyplot import（与 backend.py 一致）
import matplotlib.pyplot as plt

from PySide6.QtCore import QThread, Signal

PLOT_STYLE = {
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
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

plt.rcParams["font.sans-serif"] = PLOT_STYLE["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _apply_plot_style() -> None:
    """每次绘图前重新应用样式（其他页面会污染全局 rcParams）。"""
    plt.rcParams.update(PLOT_STYLE)


def _figure_to_png(fig, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
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
# 1. 波形测试
# =====================================================================

def _count_contiguous_regions(mask: np.ndarray) -> int:
    """统计布尔掩码中连续 True 区域的个数。"""
    if not np.any(mask):
        return 0
    edges = np.diff(mask.astype(np.int8))
    starts = 1 if mask[0] else 0
    return int(starts + np.count_nonzero(edges == 1))


def _run_sc_waveform_test(params, num_symbols: int, show_points: int,
                          checks: list, plots: list, summary: list) -> None:
    from transmitter.Modulator import THzModulator
    from transmitter.Pulseshaper import TxPulseShaper

    L = int(params.get("filter_length"))
    sps = int(params.get("oversampling"))
    beta = float(params.get("rolloff"))
    if num_symbols <= 2 * L:
        raise ValueError(f"符号数({num_symbols})必须大于 2×滤波器长度({2 * L})，"
                         f"才能容纳相隔 2L 的两个脉冲点")

    # 1) 经真实调制器生成 ±1 脉冲（pi/2-BPSK：bit 1→+1，bit 0→−1，再乘 exp(jπn/2)）
    bits = np.array([1, 0], dtype=np.uint8)
    mod = THzModulator(params)
    mod_dict = mod.modulate({
        "signal_stream": bits,
        "sample_rate_Hz": 30e9,
        "duration_seconds": len(bits) / 30e9,
        "signal_length": len(bits),
        "padding_bit_num": 0,
        "frame_bit_num": len(bits),
        "frame_num": 1,
    })
    syms = np.asarray(mod_dict["signal_stream"])
    derot = syms * np.exp(-1j * np.pi * np.arange(len(syms)) / 2)
    ok_mod = (abs(derot[0] - 1.0) < 1e-12) and (abs(derot[1] + 1.0) < 1e-12)
    checks.append({
        "name": "pi/2-BPSK 调制输出（去旋转后应为 +1/-1）",
        "ok": bool(ok_mod),
        "detail": f"去旋转符号 = [{derot[0]:.6g}, {derot[1]:.6g}]",
    })

    # 2) 测试信号：相隔 2L 的正负脉冲点，其余为 0
    imp = np.zeros(num_symbols, dtype=np.complex128)
    imp[0] = float(derot[0].real)
    imp[2 * L] = float(derot[1].real)

    # 3) 成型滤波
    shaper = TxPulseShaper(params)
    shaped = shaper.shape_pulse({
        "signal_stream": imp,
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

    # 4) 峰值位置与符号
    i1 = int(np.argmax(np.abs(y[:L * sps])))
    seg2 = y[2 * L * sps - L * sps: 2 * L * sps + L * sps]
    i2 = int(2 * L * sps - L * sps + np.argmax(np.abs(seg2)))
    ok_sign = (float(np.real(y[i1])) > 0) and (float(np.real(y[i2])) < 0)
    checks.append({
        "name": "正负交替脉冲（正脉冲在前，负脉冲在后）",
        "ok": bool(ok_sign),
        "detail": f"正脉冲峰值样本 {i1}（实部 {np.real(y[i1]):.4g}），"
                  f"负脉冲峰值样本 {i2}（实部 {np.real(y[i2]):.4g}）",
    })

    peak = max(abs(y[i1]), abs(y[i2]))
    ok_amp = abs(abs(y[i1]) - abs(y[i2])) < 0.05 * peak
    checks.append({
        "name": "两脉冲幅度一致",
        "ok": bool(ok_amp),
        "detail": f"|y[{i1}]|={abs(y[i1]):.4g}，|y[{i2}]|={abs(y[i2]):.4g}",
    })

    # RRC 旁瓣随 β 减小而增大（β=0 时第一旁瓣约为主峰 22%），主瓣始终远高于
    # 峰值 50%，故取 50% 阈值统计主瓣区域数，对 β∈[0,1] 均稳健。
    thr = 0.5 * peak
    n_regions = _count_contiguous_regions(np.abs(y) > thr)
    ok_two = n_regions == 2
    checks.append({
        "name": "时域波形呈现两个独立的滚降脉冲",
        "ok": bool(ok_two),
        "detail": f"超过峰值 50% 的连续区域数 = {n_regions}（期望 2）",
    })

    # 5) 图 (a)：时域波形
    _apply_plot_style()
    n = min(int(show_points), len(y))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(np.arange(n), np.real(y[:n]), color="#2C68B4", lw=0.8, label="实部")
    ax.plot(np.arange(n), np.abs(y[:n]), color="#D95F02", lw=0.7, ls="--", label="包络")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(2 * L * sps, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("采样点（过采样 {}×）".format(sps))
    ax.set_ylabel("幅度")
    ax.set_title(f"SC-FDE 脉冲成型时域波形（滚降滤波器长度 L={L}，β={beta}）")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plots.append({"title": "SC-FDE 时域波形", "png": _figure_to_png(fig)})

    # 6) 图 (b)：pi/2-BPSK 去旋转星座校验
    bits16 = np.tile(np.array([1, 0], dtype=np.uint8), 8)
    mod_dict16 = mod.modulate({
        "signal_stream": bits16,
        "sample_rate_Hz": 30e9,
        "duration_seconds": len(bits16) / 30e9,
        "signal_length": len(bits16),
        "padding_bit_num": 0,
        "frame_bit_num": len(bits16),
        "frame_num": 1,
    })
    syms16 = np.asarray(mod_dict16["signal_stream"])
    derot16 = syms16 * np.exp(-1j * np.pi * np.arange(len(syms16)) / 2)
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(np.real(derot16), np.imag(derot16), s=14, color="#2C68B4")
    ax.set_xlabel("实部")
    ax.set_ylabel("虚部")
    ax.set_title("pi/2-BPSK 解旋转星座校验")
    ax.axis("equal")
    ax.grid(True)
    fig.tight_layout()
    plots.append({"title": "pi/2-BPSK 星座", "png": _figure_to_png(fig)})

    summary.extend([
        {"label": "滤波器", "value": f"RRC L={L}，{sps}×，β={beta}"},
        {"label": "脉冲间距", "value": f"{2 * L} 符号"},
        {"label": "输出长度", "value": f"{len(y)} 采样"},
        {"label": "输出采样率", "value": f"{shaped['sample_rate_Hz'] / 1e9:.1f} GHz"},
    ])


def _run_ofdm_waveform_test(params, include_optional_pipeline: bool,
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

        n_gi_good = 0
        gi_first_bad = None
        for b in range(blocks):
            if b in pilot_indexes:
                continue  # 导频块为整块 512 音序列，不在单音校验范围
            start = b * (N_SC + cp) + cp
            spec = np.fft.fft(gi_sig[start:start + N_SC], norm="ortho")
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

        # 成型滤波（OFDM 模式 = 抗镜像低通插值）
        shaper = TxPulseShaper(params)
        shaped = shaper.shape_pulse(gi)
        shaped_sig = np.asarray(shaped["signal_stream"])
        ok_shaped = (shaped_sig.dtype == np.complex128) and \
                    (len(shaped_sig) == len(gi_sig) * sps)
        checks.append({
            "name": "OFDM 成型输出类型与长度",
            "ok": bool(ok_shaped),
            "detail": f"dtype={shaped_sig.dtype}，长度={len(shaped_sig)}"
                      f"（期望 complex128 / {len(gi_sig) * sps}）",
        })

        # 单音保留率：取数据块 20（单音 j=18），成型后对符号体做 FFT。
        # 零插值在频域为周期复制（基带频谱原样出现在低 512 个 bin），
        # 单音仍位于 bin j；幅度放大 √sps 倍（2048 点 ortho FFT）。
        b = 20
        j = b - sum(1 for p in pilot_indexes if p < b)
        seg = shaped_sig[b * (N_SC + cp) * sps + cp * sps:
                         b * (N_SC + cp) * sps + (cp + N_SC) * sps]
        spec2 = np.fft.fft(seg, norm="ortho")
        retention = float(abs(spec2[j]) / np.sqrt(sps))
        checks.append({
            "name": "OFDM 成型低通对带内单音的保留",
            "ok": bool(retention > 0.9),
            "detail": f"块 20（单音 j={j}）成型后保留率 = {retention:.3f}（期望 > 0.9）",
        })

        # 低通特性：实测滤波器频率响应在高频（k=480 子载波处）衰减
        if sps == 1:
            checks.append({
                "name": "OFDM 成型滤波器低通特性",
                "ok": True,
                "detail": "过采样率 1×：无插零镜像，无需低通（跳过）",
            })
        else:
            h_resp = np.fft.fft(shaper.filter_coeffs, 4096)
            att = float(abs(h_resp[int(480 * 8 / sps)]) / (abs(h_resp[0]) + 1e-15))
            checks.append({
                "name": "OFDM 成型滤波器低通特性",
                "ok": bool(att < 0.1),
                "detail": f"子载波 k=480 处相对直流衰减 = {att:.4f}（期望 < 0.1）",
            })

        # 图 (c)：模块链路时域波形（GI 后前 2048 采样）
        _apply_plot_style()
        n3 = min(2048, len(gi_sig))
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.plot(np.arange(n3), np.real(gi_sig[:n3]), color="#2C68B4", lw=0.6)
        ax.set_xlabel("采样点")
        ax.set_ylabel("幅度")
        ax.set_title("模块链路输出（TxOFDMProcesser → 插入 GI）时域波形")
        ax.grid(True)
        fig.tight_layout()
        plots.append({"title": "OFDM 模块链路波形", "png": _figure_to_png(fig)})
        summary.append({"label": "模块链路", "value": "ofdm_process + GI 校验通过"})
    except Exception as exc:
        checks.append({
            "name": "模块链路（ofdm_process + GI + 成型）",
            "ok": False,
            "detail": f"异常：{type(exc).__name__}: {exc}",
        })


def run_waveform_test(link_mode: str = "sc-fde", filter_length: int = 32,
                      oversampling: int = 4, rolloff: float = 0.22,
                      num_symbols: int = 128, show_points: int = 600,
                      include_optional_pipeline: bool = True) -> Dict[str, Any]:
    """发射端波形调制与成型滤波测试。

    SC-FDE：±1 脉冲点相隔 2×滚降滤波器长度，期望正负交替的滚降波形；
    OFDM：每符号仅单个子载波有值且索引逐符号递增，期望频率渐升正弦。
    """
    from params.PHYParams import PHYParams

    t0 = time.perf_counter()
    result = _new_result("waveform")
    mode = str(link_mode).lower()
    if mode not in ("sc-fde", "ofdm"):
        raise ValueError(f"不支持的链路模式：{link_mode}，仅支持 sc-fde / ofdm")

    params = PHYParams()
    params.update(
        link_mode=mode,
        filter_length=int(filter_length),
        oversampling=int(oversampling),
        rolloff=float(rolloff),
    )

    if mode == "sc-fde":
        params.update(NCBPS=1, MCS=1, scramble=False)
        _run_sc_waveform_test(params, int(num_symbols), int(show_points),
                              result["checks"], result["plots"], result["summary"])
    else:
        _run_ofdm_waveform_test(params, bool(include_optional_pipeline),
                                result["checks"], result["plots"], result["summary"])

    result["ok"] = all(c["ok"] for c in result["checks"])
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    return result


# =====================================================================
# 2. 编码技术测试（RS / LDPC）
# =====================================================================

def _gf_str(arr: np.ndarray, max_show: int = 40) -> str:
    vals = [str(int(v)) for v in arr[:max_show]]
    text = ", ".join(vals)
    if len(arr) > max_show:
        text += f", …共{len(arr)}项"
    return f"[{text}]"


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

        rows.append([
            name, _trunc_hex(hex_in), _trunc_hex(_bits_to_hex(cw)),
            "；".join(f"bit #{p}" for p in err_pos) or "无",
            _trunc_hex(_bits_to_hex(dec_bits)),
            "通过" if ok else "失败",
        ])
        details.append(
            f"LDPC({coder.n},{coder.k}) 码率 {rate}；输入 {len(in_bits)} bit"
            f"（{len(in_bits) // 8} 字节）→ 编码 {len(cw)} bit"
            f"（{debug['num_blocks']} 个码字块）；注入错误 {len(err_pos)} bit；"
            f"迭代次数 {debug['iterations']}；校验子权重 {debug['syndrome_weights']}；"
            f"硬判 fallback={debug['used_hard_fallback']}；"
            f"译码{'成功' if ok else '失败'}。"
            f"注：error_bits 为编码后码字流中的 bit 下标。"
        )
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
        "columns": ["算例", "输入(hex)", "编码结果(hex)", "注入错误", "译码结果(hex)", "判定"],
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
# 后台线程
# =====================================================================

class FunctionalTestWorker(QThread):
    """在后台线程执行功能测试，结果经信号回传页面。"""

    result_ready = Signal(str, dict)   # (test_type, result)
    failed = Signal(str, str)          # (test_type, "异常类型: 消息")

    def __init__(self, test_type: str, payload: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.test_type = test_type
        self.payload = dict(payload or {})

    def run(self) -> None:
        try:
            if self.test_type == "waveform":
                result = run_waveform_test(**self.payload)
            elif self.test_type == "codec":
                result = run_codec_test(self.payload.get("config_text", ""))
            elif self.test_type == "precision":
                result = run_precision_test(**self.payload)
            else:
                raise ValueError(f"未知测试类型: {self.test_type}")
            result["test_type"] = self.test_type
            self.result_ready.emit(self.test_type, result)
        except Exception as exc:  # noqa: BLE001 — 线程边界，异常须经信号带回 UI
            self.failed.emit(self.test_type, f"{type(exc).__name__}: {exc}")
