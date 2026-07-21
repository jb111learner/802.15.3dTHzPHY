"""
OFDM / SC-FDE 双模 BER vs SNR 仿真
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.THzReceiver import THzReceiver

# ==================== 仿真开关 ====================
SIM_MODES = ["ofdm"]       # ["ofdm"] / ["sc-fde"] / ["ofdm","sc-fde"]

# ==================== 仿真参数 ====================
SNR_RANGE    = (10, 25, 2)   # 10:2:28 dB
DURATION     = 5e-6           # 每帧数据时长 (秒)
MIN_ERRORS   = 500            # 每 SNR 点最少错误比特
MAX_FRAMES   = 100            # 每 SNR 点最大帧数
COLORS       = {"ofdm": "C0", "sc-fde": "C1"}


def run_ber_for_mode(mode, snr_values, duration, min_bit_errors, max_frames):
    """对指定模式仿真 BER vs SNR"""
    ber_list = []
    print(f"\n{'='*55}")
    print(f"  {mode.upper()} BER vs SNR  {snr_values[0]}-{snr_values[-1]} dB")
    print(f"{'='*55}")

    for snr_db in snr_values:
        total_bits, total_errors, n_frames = 0, 0, 0
        print(f"\n--- SNR = {snr_db} dB ---")

        while total_errors < min_bit_errors and n_frames < max_frames:
            params = PHYParams()
            params.update(link_mode=mode, SNRdB=snr_db, duration=duration)
            tx = THzTransmitter(params)
            tx.run()
            N_SUB = (tx.ofdm_processer.frame_num if mode == "ofdm"
                     else tx.assembler.frame_num)

            ch = THzChannel(params)
            rx = ch.run(tx.tx_signal_dict)

            recv = THzReceiver(params, tx)
            rx_data = recv.run(rx)

            tx_bits = tx.data_bits_dict["signal_stream"]
            rx_bits = rx_data["signal_stream"]
            cmp = min(len(tx_bits), len(rx_bits))
            errors = np.sum(tx_bits[:cmp] != rx_bits[:cmp])
            total_errors += errors
            total_bits += cmp
            n_frames += N_SUB

        ber = total_errors / total_bits if total_bits > 0 else 0
        ber_list.append(ber)
        print(f"  => SNR={snr_db:2d}dB  BER={ber:.2e}  "
              f"({total_errors}/{total_bits} errors, {n_frames} frames)")

    return np.array(ber_list)


def plot_ber(snr_values, results, save_path=None):
    """绘制双模 BER 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, ber_list in results.items():
        ax.semilogy(snr_values, ber_list, 'o-', ms=6, lw=1.5,
                    color=COLORS.get(mode), label=f'{mode.upper()}')

    # 1e-6 参考线
    all_bers = np.concatenate([b for b in results.values()])
    if np.any(all_bers < 1e-6) and np.any(all_bers > 1e-6):
        ax.axhline(y=1e-6, color='red', ls='--', lw=1, alpha=0.7)
        ax.text(snr_values[0] + 0.5, 1.2e-6, '1e-6', color='red', fontsize=8, va='bottom')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('BER vs SNR')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0e}'))

    y_min = max(min(np.min(b[b > 0]) for b in results.values()) / 5, 1e-7)
    ax.set_ylim([y_min, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    snr_values = np.arange(*SNR_RANGE)
    results = {}

    for mode in SIM_MODES:
        ber = run_ber_for_mode(mode, snr_values, DURATION, MIN_ERRORS, MAX_FRAMES)
        results[mode] = ber

    print(f"\n{'='*55}")
    print("  Results Summary")
    print(f"{'='*55}")
    header = f"  {'SNR':>5s}"
    for m in SIM_MODES:
        header += f"  {m.upper():>12s}"
    print(header)
    for i, s in enumerate(snr_values):
        row = f"  {s:3d} dB"
        for m in SIM_MODES:
            row += f"  {results[m][i]:.2e}"
        print(row)

    if results:
        plot_ber(snr_values, results)
