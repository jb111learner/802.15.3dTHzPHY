"""
MIMO 2×2 双流 + 256QAM + LDPC(1440,1056) 11/15 + AWGN 全链路 BER 测试

方案配置（带宽翻倍）:
  - 子载波数:    1024（512 翻倍）
  - 采样率:      60 GHz（30e9 翻倍，子载波间隔保持 58.6 MHz，占用带宽真翻倍）
  - 空间流:      2×2 空间复用，MMSE 检测，理想恒等信道 + 估计 CSI
  - 调制:        256QAM (NCBPS=8)
  - 编码:        LDPC ieee802153d_1440, 码率 11/15 (k=1056, n=1440)
  - 信道:        仅 AWGN（多径/CFO/IQ 不平衡全部关闭）
  - 帧结构:      subframe_ofdm_num=45, 无块导频（MIMO 自带训练符号）
                 → frame_bit_num = 8×1024×45×11/15 = 270336 = 256×1056
                   与 LDPC 码块精确对齐，无补零

用法:
  python simulation/test_mimo_256qam_ber.py        # 全 SNR 扫描 + 出图
  pytest simulation/test_mimo_256qam_ber.py -q     # 单测模式（快速检查）
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.THzReceiver import THzReceiver

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
})

OUT_DIR = "simulation_results"

# 每帧信息比特数 = NCBPS × subwave_num × n_data_syms × k/n（见模块 docstring）
FRAME_BIT_NUM = 270336


def _mimo_params(**overrides):
    """构造 MIMO 2×2 / 1024 子载波 / 256QAM / LDPC 11/15 的测试参数。"""
    params = PHYParams()
    values = dict(
        enable_mimo=True,
        link_mode="ofdm",
        num_tx=2,
        num_rx=2,
        num_spatial_streams=2,
        mimo_scheme="spatial_multiplexing",
        mimo_detector="mmse",
        mimo_channel_model="identity",
        mimo_csi_mode="estimated",
        mimo_num_taps=1,
        subwave_num=1024,          # 子载波数翻倍
        gi_length=64,
        pilot_block_indexes=[],    # MIMO 自带训练符号，无块导频
        subframe_ofdm_num=45,      # → frame_bit_num=270336=256×1056 精确对齐
        NCBPS=8,                   # 256QAM
        code_type="LDPC",
        ldpc_matrix_type="ieee802153d_1440",
        ldpc_standard_rate="11/15",
        sample_rate=60e9,          # 带宽翻倍：子载波间隔保持 58.6 MHz
        enable_multipath=False,
        enable_awgn=False,         # 各用例按需开启
        enable_cfo=False,
        enable_cfo_compensation=False,
        enable_iq_imbalance=False,
        enable_iq_compensation=False,
        scramble=True,
        random_seed=7,
        mimo_channel_seed=11,
        duration=None,
        sample_length=FRAME_BIT_NUM // 8,  # 默认 1 帧
    )
    values.update(overrides)
    params.update(**values)
    return params


def _run_link(params):
    """TX → 信道 → RX 全链路，返回 (transmitter, channel, receiver, decoded)。"""
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    receiver = THzReceiver(params, transmitter)
    decoded = receiver.run(rx_signal)
    return transmitter, channel, receiver, decoded


def run_one_snr(snr_db, frames=1, seed_offset=0):
    """运行单个 SNR 点（多帧），返回 (cmp_bits, errors, noise_var, nmse)。"""
    params = _mimo_params(
        enable_awgn=True,
        SNRdB=float(snr_db),
        sample_length=FRAME_BIT_NUM // 8 * frames,
    )
    np.random.seed(1000 + int(round(snr_db * 10)) + seed_offset)
    transmitter, _, receiver, decoded = _run_link(params)

    tx_bits = transmitter.data_bits_dict["signal_stream"]
    rx_bits = decoded["signal_stream"]
    cmp = min(len(tx_bits), len(rx_bits))
    errors = int(np.sum(tx_bits[:cmp] != rx_bits[:cmp]))
    noise_var = float(receiver.noise_var)
    nmse = float(
        np.mean(receiver.rx_equalized.get("mimo_channel_nmse", np.nan))
    ) if receiver.rx_equalized is not None else float("nan")
    assert len(rx_bits) == len(tx_bits), (
        f"解码长度不匹配: rx={len(rx_bits)}, tx={len(tx_bits)}"
    )
    return cmp, errors, noise_var, nmse


def run_ber_sweep(snr_values, frames=1):
    """对一组 SNR 点执行 BER 仿真，返回 (snr_values, ber_list)。"""
    ber_list = []
    for snr_db in snr_values:
        cmp, errors, noise_var, nmse = run_one_snr(snr_db, frames=frames)
        ber = errors / cmp if cmp else 0.0
        ber_list.append(ber)
        print(f"  SNR={snr_db:5.1f} dB  BER={ber:.3e} "
              f"({errors}/{cmp})  noise_var={noise_var:.2e}  NMSE={nmse:.2e}")
    return np.asarray(snr_values), np.asarray(ber_list)


# ==================== pytest 用例（快速检查） ====================

def test_zero_ber_very_high_snr():
    """40 dB 高信噪比下应零误码（256QAM + LDPC 11/15 + MMSE 双流）。"""
    cmp, errors, _, _ = run_one_snr(40.0, frames=1)
    assert cmp > 0 and errors == 0


def test_ber_monotone_decrease():
    """24→40 dB 范围内 BER 应单调不增。"""
    snr_values, ber_list = run_ber_sweep(np.arange(24.0, 40.01, 2.0), frames=1)
    assert np.all(np.diff(ber_list) <= 0), f"BER 非单调: {ber_list}"


def test_frame_bit_alignment():
    """帧比特数与 LDPC(1440,1056) 码块精确对齐（无补零）。"""
    k, n = 1056, 1440
    assert FRAME_BIT_NUM % k == 0
    assert (FRAME_BIT_NUM * n // k) % (1024 * 45 * 8) == 0  # 每符号数整数


# ==================== 主程序：SNR 扫描 + 出图 ====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    snr_values = np.arange(20.0, 40.01, 2.0)
    print("=" * 64)
    print("  MIMO 2×2 | 1024 子载波 | 256QAM | LDPC(1440,1056) 11/15 | AWGN")
    print(f"  带宽 60 GHz, 每帧 {FRAME_BIT_NUM} bit = 256×1056, 帧数/点: 2")
    print("=" * 64)

    _, ber_list = run_ber_sweep(snr_values, frames=2)

    # —— BER vs SNR 图 ——
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogy(snr_values, ber_list, "o-", color="#2C68B4",
                label="MIMO 2×2 256QAM LDPC 11/15")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("MIMO 2×2 · 1024 subcarriers · 256QAM · LDPC 11/15 · AWGN",
                 fontsize=10)
    ax.grid(True, which="both")
    ax.set_ylim([1e-7, 1])
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/ber_mimo_256qam_ldpc.{ext}",
                    dpi=600 if ext == "pdf" else 300)
    plt.close(fig)
    print(f"\nFigure saved to {OUT_DIR}/ber_mimo_256qam_ldpc.png|pdf")


if __name__ == "__main__":
    main()
