import numpy as np
import matplotlib.pyplot as plt
from core.BaseTransmitter import BaseTransmitter
from transmitter.HeaderGenerator import HeaderGenerator
from transmitter.PreambleInsertor import PreambleInsertor
from transmitter.Modulator import THzModulator as Modulator
from transmitter.GIInserter import GIInserter
from transmitter.DataProcesser import BitStreamProcessor
# from utils.Rotator import Rotator 
from transmitter.Encoder import Encoder 
from transmitter.Pulseshaper import TxPulseShaper
from params.PHYParams import PHYParams
from transmitter.Scrambler import Scrambler 
from transmitter.TxOFDMProcesser import TxOFDMProcesser

class THzTransmitter(BaseTransmitter):
    """
    THz发射机主类
    """
    def __init__(self, params):
        super().__init__(params)
        # 初始化子模块
        self.preamble_gen = PreambleInsertor(params)
        self.modulator = Modulator(params)
        self.coder = Encoder(params)
        self.gi_inserter = GIInserter(params)
        self.pulse_shaper = TxPulseShaper(params)
        self.scrambler = Scrambler(params)  # 初始化扰码器
        self.assembler = BitStreamProcessor(params)  # 初始化比特流处理器
        self.ofdm_processer = TxOFDMProcesser(params)
        self.is_scramble = self.params.get("scramble")  # 是否使用扰码
        self.oversampling = self.params.get("oversampling")  # 上采样率\
        self.link_mode = self.params.get("link_mode") 
        # self.rotator = Rotator()  
    
    def assemble_frame(self):
        """组装帧"""
        self.data_bits_dict = self.assembler.run()
        return self.data_bits_dict

    def scramble_data(self, data_bits_dict):
        """扰码数据"""
        self.data_scrambled_dict = self.scrambler.scramble(data_bits_dict)
        return self.data_scrambled_dict

    def channel_encode(self, data_bits_dict):
        """信道编码（调用RSCoder）"""
        self.coded_bits_dict = self.coder.encode(data_bits_dict)
        return self.coded_bits_dict

    def modulate(self, data_bits_dict):
        """调制数据"""
        self.modulated_data_dict = self.modulator.modulate(data_bits_dict)
        return self.modulated_data_dict

    def insert_gi(self, data_bits_dict):
        """插入GI"""
        self.data_with_gi_dict = self.gi_inserter.insert_gi(data_bits_dict)
        return self.data_with_gi_dict
    
    def insert_preamble(self, data_bits_dict):
        """插入前导码"""
        self.data_with_preamble_dict = self.preamble_gen.insert_preamble(data_bits_dict)
        self.preamble = self.preamble_gen.preamble
        self.sync = self.preamble_gen.sync
        self.sfd = self.preamble_gen.sfd
        self.ces = self.preamble_gen.ces
        return self.data_with_preamble_dict
    
    def ofdm_process(self, data_bits_dict):
        """OFDM处理流程"""
        self.data_ofdm_dict = self.ofdm_processer.ofdm_process(data_bits_dict)
        return self.data_ofdm_dict

    def pulse_shaping(self, data_bits_dict):
        """脉冲成型"""
        self.tx_signal_dict = self.pulse_shaper.shape_pulse(data_bits_dict)
        self.sync_upsampled = self.tx_signal_dict["signal_stream"][:len(self.sync) * self.oversampling]  # 提取上采样后的SYNC序列
        self.sfd_upsampled = self.tx_signal_dict["signal_stream"][len(self.sync) * self.oversampling : (len(self.sync) + len(self.sfd)) * self.oversampling]  # 提取上采样后的SFD序列
        self.ces_upsampled = self.tx_signal_dict["signal_stream"][(len(self.sync) + len(self.sfd)) * self.oversampling : (len(self.sync) + len(self.sfd) + len(self.ces)) * self.oversampling]  # 提取上采样后的CES序列
        return self.tx_signal_dict

    def run(self):
        """执行完整发射流程"""
        if self.link_mode == "sc-fde":
            data_bits_dict = self.assemble_frame()
            if self.is_scramble:
                data_bits_dict = self.scramble_data(data_bits_dict)
            data_bits_dict = self.channel_encode(data_bits_dict)
            data_bits_dict = self.modulate(data_bits_dict)
            data_bits_dict = self.insert_gi(data_bits_dict)
            # self.generate_preamble()
            data_bits_dict = self.insert_preamble(data_bits_dict)
            tx_signal_dict = self.pulse_shaping(data_bits_dict)
        elif self.link_mode == "ofdm":
            data_bits_dict = self.assemble_frame()
            if self.is_scramble:
                data_bits_dict = self.scramble_data(data_bits_dict)
            data_bits_dict = self.channel_encode(data_bits_dict)
            data_bits_dict = self.modulate(data_bits_dict)
            data_bits_dict = self.ofdm_process(data_bits_dict)
            data_bits_dict = self.insert_gi(data_bits_dict)
            self.generate_preamble()
            data_bits_dict = self.insert_preamble(data_bits_dict)
            tx_signal_dict = self.pulse_shaping(data_bits_dict)
        else:
            raise ValueError("链路模式选择错误，当前仅支持sc-fde与ofdm")  

        return tx_signal_dict




def plot_eye_diagram(signal, symbol_period, num_symbols=1000, oversampling=1, ax=None):
    """
    绘制眼图
    参数：
        signal: 输入的复信号（时域波形）
        symbol_period: 符号周期（采样点数）
        num_symbols: 用于绘制眼图的符号数量
        oversampling: 上采样率
        ax: 绘图的坐标轴对象
    """
    if ax is None:
        ax = plt.gca()
    
    # 提取实部和虚部分别绘制眼图
    signal_real = np.real(signal)
    signal_imag = np.imag(signal)
    
    # 计算每个符号的采样点数
    samples_per_symbol = symbol_period * oversampling
    
    # 截取有效信号段（跳过前导码和延迟，取数据段）
    start_idx = int(len(signal) * 0.1)  # 跳过前10%的信号（前导码+延迟）
    end_idx = start_idx + num_symbols * samples_per_symbol
    if end_idx > len(signal):
        end_idx = len(signal)
    signal_segment = signal_real[start_idx:end_idx]
    
    # 重排数据为二维数组：每行一个符号周期的采样点
    num_rows = len(signal_segment) // samples_per_symbol
    eye_data = signal_segment[:num_rows * samples_per_symbol].reshape(-1, samples_per_symbol)
    
    # 绘制实部眼图
    for i in range(num_rows):
        ax.plot(eye_data[i], alpha=0.1, color='blue')
    
    # 美化眼图
    ax.set_title('信号眼图（实部）', fontweight='bold')
    ax.set_xlabel('符号周期内采样点')
    ax.set_ylabel('幅值（实部）')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, samples_per_symbol-1)

# if __name__ == "__main__":    
#     params = PHYParams()
#     tx = THzTransmitter(params)
#     tx.run()

# test
if __name__ == "__main__":
    import os

    # ---- publication style ----
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
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.framealpha": 0.8,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
    })

    out_dir = "simulation_results"
    os.makedirs(out_dir, exist_ok=True)

    colors = {"sc-fde": "#2C68B4", "ofdm": "#D95F02"}  # blue / orange
    labels = {"sc-fde": "SC", "ofdm": "OFDM"}

    # ---- generate waveforms ----
    signals = {}
    paprs = {}
    for mode in ["sc-fde", "ofdm"]:
        params = PHYParams()
        params.update(link_mode=mode)
        tx = THzTransmitter(params)
        tx.run()
        sig = tx.tx_signal_dict["signal_stream"]
        sps = params.get("oversampling")
        # payload only (skip preamble), normalize to unit avg power
        pskip = len(tx.preamble) * sps
        payload = sig[pskip:]
        payload = payload / np.sqrt(np.mean(np.abs(payload)**2))
        signals[mode] = {"sig": payload, "sps": sps,
                         "fs": tx.tx_signal_dict.get("sample_rate_Hz"),
                         "n_sc": params.get("subwave_num")}
        # PAPR per OFDM-symbol-equivalent block (N_SC × sps = 2048 samples)
        blk_len = params.get("subframe_length") * sps  # 512 × 4 = 2048
        pv = []
        for b in range(0, len(payload) - blk_len, blk_len):
            blk = payload[b:b+blk_len]
            pv.append(10*np.log10(np.max(np.abs(blk)**2)/(np.mean(np.abs(blk)**2)+1e-15)))
        paprs[mode] = np.array(pv)

    # ===== Fig 1a/1b: Time-domain envelope (separate) =====
    for mode in ["sc-fde", "ofdm"]:
        fig, ax = plt.subplots(figsize=(6, 3))
        s = signals[mode]
        n_show = 600
        env = np.abs(s["sig"][:n_show * s["sps"]])
        t = np.arange(len(env)) / s["sps"]
        ax.plot(t, env, color=colors[mode], lw=0.6)
        ax.axhline(y=1.0, color="gray", ls="--", lw=0.5, alpha=0.5, label="mean")
        ax.set_xlabel("Symbol index"); ax.set_ylabel("|Amplitude|")
        ax.set_title(f"{labels[mode]} time envelope", fontsize=10, pad=4)
        ax.legend(fontsize=8); ax.set_xlim([0, n_show])
        fig.tight_layout()
        tag = "sc" if mode == "sc-fde" else "ofdm"
        fig.savefig(f"{out_dir}/fig1_time_{tag}.pdf", dpi=600)
        fig.savefig(f"{out_dir}/fig1_time_{tag}.png", dpi=300)
        plt.close(fig)
    print("Fig1a/1b saved")

    # ===== Fig 2a/2b: Spectrum (separate) =====
    for mode in ["ofdm", "sc-fde"]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        s = signals[mode]
        nfft = 2048
        seg = s["sig"][:nfft]
        spec = np.fft.fftshift(np.fft.fft(seg, nfft))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1/s["fs"]))
        ax.plot(freq / 1e9, 20*np.log10(np.abs(spec) + 1e-15),
                color=colors[mode], lw=0.4)
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"{labels[mode]} spectrum", fontsize=10, pad=4)
        bw = 0.5 * s["fs"]; ax.set_xlim([-bw/1e9, bw/1e9]); ax.set_ylim([-20, 60])
        if mode == "ofdm":
            inset = ax.inset_axes([0.15, 0.5, 0.35, 0.4])
            mask = (freq > -3e9) & (freq < 3e9)
            inset.plot(freq[mask]/1e9, 20*np.log10(np.abs(spec[mask])+1e-15),
                       color=colors[mode], lw=0.2)
            inset.set_xlim([-2, 2])
            inset.set_xticklabels([]); inset.set_yticklabels([])
            inset.set_title("subcarriers", fontsize=7, pad=2)
        fig.tight_layout()
        tag = "ofdm" if mode == "ofdm" else "sc"
        fig.savefig(f"{out_dir}/fig2_spectrum_{tag}.pdf", dpi=600)
        fig.savefig(f"{out_dir}/fig2_spectrum_{tag}.png", dpi=300)
        plt.close(fig)
    print("Fig2a/2b saved")

    # ===== Fig 3a/3b: PAPR CCDF (separate) =====
    for mode in ["sc-fde", "ofdm"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        pv = np.sort(paprs[mode])
        ccdf = 1.0 - np.arange(len(pv)) / len(pv)
        ax.semilogy(pv, ccdf, color=colors[mode], lw=1.2)
        idx1e3 = np.searchsorted(ccdf, 1e-3)
        if idx1e3 < len(pv):
            p3 = pv[idx1e3]
            ax.axvline(x=p3, color=colors[mode], ls=":", lw=0.8, alpha=0.6)
            ax.annotate(f"{p3:.1f} dB @ 1e-3", xy=(p3, 1e-3), xytext=(p3+1.5, 3e-3),
                         fontsize=9, color=colors[mode],
                         arrowprops=dict(arrowstyle="->", color=colors[mode], lw=0.6))
        ax.set_xlabel("PAPR (dB)"); ax.set_ylabel("CCDF")
        ax.set_title(f"{labels[mode]} PAPR CCDF", fontsize=10, pad=4)
        ax.set_ylim([1e-3, 1])
        fig.tight_layout()
        tag = "sc" if mode == "sc-fde" else "ofdm"
        fig.savefig(f"{out_dir}/fig3_papr_{tag}.pdf", dpi=600)
        fig.savefig(f"{out_dir}/fig3_papr_{tag}.png", dpi=300)
        plt.close(fig)
    print("Fig3a/3b saved")

    # ===== Fig 4a/4b: Amplitude histogram (separate) =====
    for mode in ["sc-fde", "ofdm"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        amps = np.abs(signals[mode]["sig"])
        ax.hist(amps, bins=80, density=True, histtype="step",
                color=colors[mode], lw=1.2)
        if mode == "ofdm":
            sigma = np.sqrt(np.mean(amps**2) / 2)
            x_r = np.linspace(0, np.max(amps), 200)
            ax.plot(x_r, x_r/sigma**2*np.exp(-x_r**2/(2*sigma**2)),
                    "--", color="gray", lw=0.8, alpha=0.7, label="Rayleigh ref.")
            ax.annotate("block pilots", xy=(1.0, 1.2), xytext=(1.6, 1.5),
                        fontsize=8, color=colors[mode],
                        arrowprops=dict(arrowstyle="->", color=colors[mode], lw=0.6))
        ax.set_xlabel("|Amplitude|"); ax.set_ylabel("Probability density")
        ax.set_title(f"{labels[mode]} amplitude distribution", fontsize=10, pad=4)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        tag = "sc" if mode == "sc-fde" else "ofdm"
        fig.savefig(f"{out_dir}/fig4_hist_{tag}.pdf", dpi=600)
        fig.savefig(f"{out_dir}/fig4_hist_{tag}.png", dpi=300)
        plt.close(fig)
    print("Fig4a/4b saved")

    print(f"All 8 figures saved to {out_dir}/")
