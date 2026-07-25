"""
全链路 BER vs SNR 仿真 — AWGN 信道

遍历:
  - 调制:        QPSK / 16QAM / 64QAM
  - 编码:        LDPC / RS
  - 链路模式:    SC-FDE / OFDM

每调制独立 SNR 范围:
  - QPSK:   2 – 12 dB
  - 16QAM:  8 – 18 dB
  - 64QAM: 14 – 24 dB

RS 参数按模式区分:
  - SC-FDE: RS(255,192), GF(2^8), c_exp=8
  - OFDM:   RS(15,11),  GF(2^4), c_exp=4
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.THzReceiver import THzReceiver

# ==================== 全局绘图风格 ====================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.major.size": 4,
    "ytick.minor.size": 2.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#cccccc",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})

# ==================== 仿真参数 ====================
# 每调制独立 SNR 范围 (start, stop, step)
# 实测瀑布区（AWGN, SC-FDE, LDPC/RS）：
#   QPSK:  BER=6e-2@-2dB → 0@4dB   → 范围 -2:1:8
#   16QAM: BER=3e-2@6dB  → 0@10dB  → 范围  4:1:14
#   64QAM: BER=3e-2@12dB → 9e-5@18dB → 范围 10:1:20
SNR_RANGES = {
    "QPSK":  (-2, 2.5,  0.5),    # -2:1:8
    "16QAM": (4,  8.5, 0.5),    #  4:1:14
    "64QAM": (10, 15, 0.5),    # 10:1:19
}

DURATION    = 5e-6             # 单帧数据时长 (秒)
MIN_ERRORS  = 10000              # 每 SNR 点最少错误比特
MAX_FRAMES  = 300              # 每 SNR 点最大帧数

MODULATIONS = {
    "QPSK":  2,
    "16QAM": 4,
    "64QAM": 6,
}

CODINGS = ["LDPC", "RS"]
MODES   = ["sc-fde", "ofdm"]

# 颜色与标记
COLOR_MAP   = {"QPSK": "#2C68B4", "16QAM": "#D95F02", "64QAM": "#3A9D3A"}
MARKER_MAP  = {"QPSK": "o",       "16QAM": "s",        "64QAM": "^"}

OUT_DIR = "simulation_results"


# ==================== RS 参数 ====================
def get_rs_params(mode):
    """返回 (rs_nsym, rs_c_exp, rs_packet_size)"""
    if mode == "sc-fde":
        return 63, 8, 192      # RS(255, 192)
    else:
        return 4, 4, 11        # RS(15, 11)


# ==================== 单次仿真运行 ====================
def run_one_snr(mode, snr_db, ncbps, code_type, duration):
    """运行单 SNR 点仿真，返回 (total_bits, total_errors, n_frames)"""
    params = PHYParams()
    params.update(
        link_mode=mode,
        SNRdB=snr_db,
        duration=duration,
        NCBPS=ncbps,
        code_type=code_type,
        enable_multipath=False,
        enable_phase_noise=False,
        enable_cfo=False,
        enable_iq_imbalance=False,
        enable_channel_equalization=True,
    )

    if code_type == "RS":
        nsym, c_exp, pkt = get_rs_params(mode)
        params.update(rs_nsym=nsym, rs_c_exp=c_exp, rs_packet_size=pkt)

    tx = THzTransmitter(params)
    tx.run()
    n_frames = (tx.ofdm_processer.frame_num if mode == "ofdm"
                else tx.assembler.frame_num)

    ch = THzChannel(params)
    rx = ch.run(tx.tx_signal_dict)

    recv = THzReceiver(params, tx)
    rx_data = recv.run(rx)

    tx_bits = tx.data_bits_dict["signal_stream"]
    rx_bits = rx_data["signal_stream"]
    cmp = min(len(tx_bits), len(rx_bits))
    errors = int(np.sum(tx_bits[:cmp] != rx_bits[:cmp]))

    return cmp, errors, n_frames


# ==================== BER vs SNR 主循环 ====================
def run_ber_sweep(mode, snr_values, ncbps, code_type, duration,
                  min_bit_errors, max_frames):
    """对一组 SNR 点执行 BER 仿真，返回 ber 数组"""
    ber_list = []
    mod_name = [k for k, v in MODULATIONS.items() if v == ncbps][0]
    label = f"{mode.upper()}+{code_type}+{mod_name}"
    print(f"\n{'='*60}")
    print(f"  {label}  |  SNR {snr_values[0]}–{snr_values[-1]} dB")
    print(f"{'='*60}")

    for snr_db in snr_values:
        total_bits, total_errors, total_frames = 0, 0, 0

        while total_errors < min_bit_errors and total_frames < max_frames:
            bits, errors, nf = run_one_snr(mode, snr_db, ncbps, code_type, duration)
            total_bits += bits
            total_errors += errors
            total_frames += nf

        ber = total_errors / total_bits if total_bits > 0 else float("nan")
        ber_list.append(ber)
        print(f"  SNR={snr_db:5.1f} dB  BER={ber:.2e}  "
              f"({total_errors}/{total_bits} err, {total_frames} fr)")

    return np.array(ber_list)


# ==================== 绘图函数 ====================
def make_figure(xlabel="SNR (dB)", ylabel="BER", figsize=(7, 5)):
    """创建统一风格画布"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25, ls="--", lw=0.4)
    ax.grid(True, which="minor", alpha=0.10, ls="--", lw=0.3)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "0"))
    return fig, ax


def plot_single_curve(ax, snr_vals, ber, color, marker, label):
    """绘制单条 BER 曲线（含 BER=0 的空心标记处理）"""
    mask = ber > 0
    if np.any(mask):
        ax.semilogy(snr_vals[mask], ber[mask],
                    color=color, marker=marker, ms=7, lw=1.5,
                    label=label, markevery=1)
    if np.any(~mask):
        y_min = max(np.min(ber[mask]) / 10 if np.any(mask) else 1e-6, 1e-8)
        ax.semilogy(snr_vals[~mask], np.full(np.sum(~mask), y_min),
                    color=color, marker=marker, ms=7, lw=0,
                    fillstyle="none", markevery=1)


def plot_curves(ax, curves, xlim=None):
    """
    在 ax 上绘制多条 BER 曲线，每条曲线可有独立的 SNR 向量。

    :param curves: list of dicts with keys:
        snr_values, ber, color, marker, ls, label, [fillstyle]
    """
    for c in curves:
        snr = c["snr_values"]
        ber = c["ber"]
        mask = ber > 0
        if np.any(mask):
            ax.semilogy(snr[mask], ber[mask],
                        color=c["color"], marker=c["marker"],
                        ls=c.get("ls", "-"), ms=5.5, lw=1.3,
                        label=c["label"], markevery=1)
        # BER=0 的点画在底部（空心标记）
        if np.any(~mask):
            y_min = max(np.min(ber[mask]) / 10 if np.any(mask) else 1e-6, 1e-8)
            ax.semilogy(snr[~mask], np.full(np.sum(~mask), y_min),
                        color=c["color"], marker=c["marker"], ms=5.5, lw=0,
                        fillstyle="none", markevery=1)

    if xlim:
        ax.set_xlim(xlim)
    ax.legend(loc="lower left", ncol=1)


def save_figure(fig, name):
    """保存为 pdf + png"""
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=600 if ext == "pdf" else 300)
    print(f"  → {OUT_DIR}/{name}.pdf|png")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"Modulations: {list(MODULATIONS.keys())}")
    print(f"SNR ranges:  {SNR_RANGES}")
    print(f"Codings:     {CODINGS}")
    print(f"Modes:       {MODES}")
    print(f"Min errors/SNR point: {MIN_ERRORS}, Max frames: {MAX_FRAMES}")
    print(f"Duration/frame: {DURATION}s")

    # ———— 逐组合仿真 + 立即出单图 ————
    # all_results[key] = (snr_values, ber_array)
    all_results = {}
    MODE_LABEL = {"sc-fde": "SC", "ofdm": "OFDM"}
    combo_idx = 0

    for mode in MODES:
        for code_type in CODINGS:
            for mod_name, ncbps in MODULATIONS.items():
                combo_idx += 1
                key = f"{mode}+{code_type}+{mod_name}"
                snr_vals = np.arange(*SNR_RANGES[mod_name])

                # —— 仿真 ——
                ber = run_ber_sweep(
                    mode, snr_vals, ncbps, code_type, DURATION,
                    MIN_ERRORS, MAX_FRAMES,
                )
                all_results[key] = (snr_vals, ber)

                # —— 立即出单图 ——
                fig, ax = make_figure(figsize=(6, 4.5))
                plot_single_curve(ax, snr_vals, ber,
                                  color=COLOR_MAP[mod_name],
                                  marker=MARKER_MAP[mod_name],
                                  label=f"{MODE_LABEL[mode]} + {code_type} + {mod_name}")
                ax.set_xlim(snr_vals[0] - 0.3, snr_vals[-1] + 0.3)
                ax.set_title(f"{MODE_LABEL[mode]}  |  {code_type}  |  {mod_name}", pad=6)
                ax.legend(loc="lower left", fontsize=8)
                fig.tight_layout()
                name = f"ber_{mode}_{code_type.lower()}_{mod_name.lower()}"
                save_figure(fig, name)
                plt.close(fig)

    # ———— 汇总打印（按调制分组） ————
    for mod_name in MODULATIONS:
        snr_vals = np.arange(*SNR_RANGES[mod_name])
        print(f"\n{'='*80}")
        print(f"  {mod_name}  BER Summary  (SNR {snr_vals[0]}–{snr_vals[-1]} dB)")
        print(f"{'='*80}")
        header = f"  {'SNR':>4s}"
        combo_keys = [f"{m}+{c}+{mod_name}" for m in MODES for c in CODINGS]
        for k in combo_keys:
            header += f"  {k:>22s}"
        print(header)
        for i, s in enumerate(snr_vals):
            row = f"  {s:5.1f} dB"
            for k in combo_keys:
                ber = all_results[k][1][i]
                row += f"  {ber:.2e}" if ber > 0 else f"  {'  —':>8s}"
            print(row)

    # ================================================================
    #  图 13–16：按模式+编码分组（单图已出，此处出分组汇总）
    for mode in MODES:
        for code in CODINGS:
            fig, ax = make_figure()
            curves = []
            x_min, x_max = 99, -99
            for mod_name in MODULATIONS:
                key = f"{mode}+{code}+{mod_name}"
                snr_vals, ber = all_results[key]
                curves.append({
                    "snr_values": snr_vals, "ber": ber,
                    "color": COLOR_MAP[mod_name],
                    "marker": MARKER_MAP[mod_name],
                    "ls": "-",
                    "label": mod_name,
                })
                x_min = min(x_min, snr_vals[0])
                x_max = max(x_max, snr_vals[-1])
            mode_label = "SC" if mode == "sc-fde" else "OFDM"
            plot_curves(ax, curves, xlim=(x_min - 0.3, x_max + 0.3))
            ax.set_title(f"{mode_label}  —  {code}", pad=6)
            fig.tight_layout()
            save_figure(fig, f"ber_{mode}_{code.lower()}")

    # ================================================================
    #  图 17：综合对比 — 4 子图
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("white")
    for idx, (mode, code) in enumerate([("sc-fde", "LDPC"), ("sc-fde", "RS"),
                                         ("ofdm", "LDPC"),   ("ofdm", "RS")]):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("white")
        curves = []
        x_min, x_max = 99, -99
        for mod_name in MODULATIONS:
            key = f"{mode}+{code}+{mod_name}"
            snr_vals, ber = all_results[key]
            curves.append({
                "snr_values": snr_vals, "ber": ber,
                "color": COLOR_MAP[mod_name],
                "marker": MARKER_MAP[mod_name],
                "ls": "-",
                "label": mod_name,
            })
            x_min = min(x_min, snr_vals[0])
            x_max = max(x_max, snr_vals[-1])
        plot_curves(ax, curves, xlim=(x_min - 0.3, x_max + 0.3))
        mode_label = "SC" if mode == "sc-fde" else "OFDM"
        ax.set_title(f"{mode_label}  —  {code}", pad=6, fontsize=11)

    for ax in axes.flat:
        ax.set_xlabel("SNR (dB)", fontsize=10)
        ax.set_ylabel("BER", fontsize=10)
        ax.grid(True, which="major", alpha=0.25, ls="--", lw=0.4)
        ax.grid(True, which="minor", alpha=0.10, ls="--", lw=0.3)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$" if y > 0 else "0"))
    fig.tight_layout()
    save_figure(fig, "ber_summary_4panel")

    # ================================================================
    #  图 18–20：按调制分组，对比 2 模式 × 2 编码（4 条曲线）
    # ================================================================
    LINESTYLES  = {"sc-fde+LDPC": "-",  "sc-fde+RS": "--",
                   "ofdm+LDPC":   "-.", "ofdm+RS":   ":"}
    COLORS_C    = {"sc-fde+LDPC": "#2C68B4", "sc-fde+RS": "#4A90D9",
                   "ofdm+LDPC":   "#D95F02", "ofdm+RS":   "#E8A040"}
    MARKERS_C   = {"sc-fde+LDPC": "o",  "sc-fde+RS": "s",
                   "ofdm+LDPC":   "^",  "ofdm+RS":   "D"}

    for mod_name in MODULATIONS:
        fig, ax = make_figure(figsize=(7, 5))
        snr_vals = np.arange(*SNR_RANGES[mod_name])
        curves = []
        for mode in MODES:
            for code in CODINGS:
                key = f"{mode}+{code}+{mod_name}"
                combo = f"{mode}+{code}"
                _, ber = all_results[key]
                mode_label = "SC" if mode == "sc-fde" else "OFDM"
                curves.append({
                    "snr_values": snr_vals, "ber": ber,
                    "color": COLORS_C[combo],
                    "marker": MARKERS_C[combo],
                    "ls": LINESTYLES[combo],
                    "label": f"{mode_label}+{code}",
                })
        plot_curves(ax, curves, xlim=(snr_vals[0] - 0.3, snr_vals[-1] + 0.3))
        ax.set_title(f"{mod_name}", pad=6)
        ax.legend(loc="lower left", ncol=1, fontsize=9)
        fig.tight_layout()
        save_figure(fig, f"ber_{mod_name.lower()}_compare")

    # ================================================================
    #  图 21–22：按模式分组，对比全部（调制+编码），6 条曲线
    # ================================================================
    for mode in MODES:
        fig, ax = make_figure(figsize=(7, 5))
        mode_label = "SC" if mode == "sc-fde" else "OFDM"
        curves = []
        x_min, x_max = 99, -99
        for mod_name in MODULATIONS:
            for code in CODINGS:
                key = f"{mode}+{code}+{mod_name}"
                snr_vals, ber = all_results[key]
                curves.append({
                    "snr_values": snr_vals, "ber": ber,
                    "color": COLOR_MAP[mod_name],
                    "marker": MARKER_MAP[mod_name],
                    "ls": "-" if code == "LDPC" else "--",
                    "label": f"{mod_name}+{code}",
                })
                x_min = min(x_min, snr_vals[0])
                x_max = max(x_max, snr_vals[-1])
        plot_curves(ax, curves, xlim=(x_min - 0.3, x_max + 0.3))
        ax.set_title(f"{mode_label}", pad=6)
        ax.legend(loc="lower left", ncol=2, fontsize=7.5)
        fig.tight_layout()
        save_figure(fig, f"ber_{mode}_all")

    print(f"\nAll figures saved to {OUT_DIR}/")
    plt.show()
