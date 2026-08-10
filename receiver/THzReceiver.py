import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from core.BaseReceiver import BaseReceiver
from receiver.sync.CoarseSync import CoarseSync
from receiver.sync.FineSync import FineSync
from receiver.CFOEstimator import CFOEstimator
from receiver.ChannelEstimator import ChannelEstimator
from receiver.NoiseEstimator import NoiseEstimator
from receiver.DeModulator import THzDemodulator
from receiver.MatchedFilter import RxMatchedFilter
from receiver.Equalizer import FreqDomainEqualizer
from receiver.RxOFDMProcesser import RxOFDMProcesser
from receiver.Decoder import Decoder
from receiver.DeScrambler import DeScrambler
from receiver.Downsampler import Downsampler
from receiver.IQCompensator import IQCompensator, DecisionDirectedIQCompensator
from receiver.IQCompensatorOFDM import (
    BlindRXIQWhiteningCompensator,
    ConfiguredRXIQCompensator,
    resolve_iq_compensation_method,
)
from receiver.MIMOReceiverProcessor import MIMOReceiverProcessor


class THzReceiver(BaseReceiver):
    """
    THz 接收机主类 — 支持 SC-FDE / OFDM 双模式

    流程:
      MF → CoarseSync → CFO coarse → FineSync → Downsample → CFO fine
      → NoiseEst → channel equalization → decision-directed IQ compensation
      → [SC-FDE: ChannelEst + FreqDomainEqualizer]
      → [OFDM:   RxOFDMProcesser (内置LS+相位跟踪)]
      → Demodulate → Decode → DeScramble
    """

    def __init__(self, params, transmitter):
        super().__init__(params, transmitter)
        self.link_mode = params.get("link_mode").lower()
        self.enable_mimo = bool(params.get("enable_mimo", False))

        # 中间结果缓存（SISO/MIMO 共用）
        self.rx_matched = None
        self.rx_coarse_synced = None
        self.rx_cfo_coarse = None
        self.rx_fine_synced = None
        self.rx_downsampled = None
        self.rx_cfo_fine = None
        self.rx_iq_compensated = None
        self.rx_iq_frontend_compensated = None
        self.rx_equalized = None
        self.noise_var = None
        self.llr_dict = None
        self.decoded_bits = None
        self.data_bits = None

        if self.enable_mimo:
            self.mimo_processor = MIMOReceiverProcessor(params, transmitter)
            self.demodulator = THzDemodulator(params)
            self.decoder = Decoder(params)
            self.descrambler = DeScrambler(params)
            self.enable_channel_est = True
            self.enable_cfo = bool(params.get("enable_cfo_compensation", False))
            return

        self.rx_matched_filter = RxMatchedFilter(transmitter)
        self.coarse_sync = CoarseSync(transmitter)
        self.fine_sync = FineSync(transmitter)
        self.cfo_estimator = CFOEstimator(transmitter)
        self.downsampler = Downsampler(transmitter)
        self.iq_dd_compensator = None
        self.iq_compensator = None
        self.iq_configured_compensator = None
        self.iq_blind_frontend_compensator = None
        self.iq_compensation_method = resolve_iq_compensation_method(
            self.params, self.link_mode
        )
        if self.iq_compensation_method == "ces":
            self.iq_compensator = IQCompensator(
                transmitter,
                filter_len=self.params.get("iq_comp_filter_len"),
                ridge_lambda=self.params.get("iq_comp_ridge_lambda"),
                compensation_mode=self.params.get("iq_compensation_mode"),
            )
        elif self.iq_compensation_method == "decision_directed":
            self.iq_dd_compensator = DecisionDirectedIQCompensator(
                params,
                filter_len=self.params.get("iq_comp_filter_len"),
                ridge_lambda=self.params.get("iq_comp_ridge_lambda"),
                iterations=self.params.get("iq_comp_dd_iterations"),
            )
        elif self.iq_compensation_method == "configured_inverse":
            self.iq_configured_compensator = ConfiguredRXIQCompensator(params)
        elif (
            self.iq_compensation_method == "ofdm_widely_linear"
            and self.params.get("enable_cfo", False)
            and self.params.get("enable_cfo_compensation", False)
        ):
            self.iq_blind_frontend_compensator = BlindRXIQWhiteningCompensator(
                params
            )
        self.channel_estimator = ChannelEstimator(transmitter)
        self.equalizer = FreqDomainEqualizer(transmitter)
        self.enable_channel_est = params.get("enable_channel_equalization")
        if self.enable_channel_est is None:
            self.enable_channel_est = True
        self.enable_cfo = params.get("enable_cfo_compensation")
        if self.enable_cfo is None:
            self.enable_cfo = True
        self.noise_estimator = NoiseEstimator(transmitter)
        self.demodulator = THzDemodulator(params)
        self.decoder = Decoder(params)
        self.descrambler = DeScrambler(params)
        if self.link_mode == "ofdm":
            self.rx_ofdm = RxOFDMProcesser(transmitter)

    # ==================== 子模块调用 ====================

    def matched_filter(self, signal_dict):
        self.rx_matched = self.rx_matched_filter.matched_filter(signal_dict)
        return self.rx_matched

    def coarse_sync_detect(self, signal_dict):
        self.rx_coarse_synced = self.coarse_sync.detect_sync(signal_dict)
        return self.rx_coarse_synced

    def compensate_cfo_coarse(self, signal_dict):
        # Single-frame CFO — avoids per-frame phase discontinuity
        fs = signal_dict["sample_rate_Hz"]
        cfo_est = self.cfo_estimator.estimate_cfo_coarse(
            signal_dict["signal_stream"], fs=fs)
        sig = self.cfo_estimator.compensate_cfo(
            signal_dict["signal_stream"], fs=fs, cfo_est=cfo_est)
        self.rx_cfo_coarse = dict(signal_dict)
        self.rx_cfo_coarse["signal_stream"] = sig
        self.rx_cfo_coarse["signal_length"] = len(sig)
        return self.rx_cfo_coarse

    def fine_sync_frame(self, signal_dict):
        self.rx_fine_synced = self.fine_sync.fine_sync(signal_dict)
        return self.rx_fine_synced

    def downsample(self, signal_dict):
        self.rx_downsampled = self.downsampler.recover_symbol(signal_dict)
        return self.rx_downsampled

    def compensate_cfo_fine(self, signal_dict):
        # Single-frame CFO — avoids per-frame phase discontinuity
        fs = signal_dict["sample_rate_Hz"]
        cfo_est = self.cfo_estimator.estimate_cfo_fine(
            signal_dict["signal_stream"], fs=fs)
        sig = self.cfo_estimator.compensate_cfo(
            signal_dict["signal_stream"], fs=fs, cfo_est=cfo_est)
        self.rx_cfo_fine = dict(signal_dict)
        self.rx_cfo_fine["signal_stream"] = sig
        self.rx_cfo_fine["signal_length"] = len(sig)
        return self.rx_cfo_fine

    def compensate_iq_decision_directed(self, signal_dict):
        """对均衡后的数据符号执行判决导向残余 IQ 补偿。"""
        if self.iq_dd_compensator is None:
            return signal_dict
        self.rx_iq_compensated = self.iq_dd_compensator.compensate(signal_dict)
        return self.rx_iq_compensated

    def compensate_iq_configured(self, signal_dict):
        """在同步与 CFO 处理前执行已知或盲 RX IQ 前端校正。"""
        compensator = getattr(self, "iq_configured_compensator", None)
        if compensator is not None and compensator.is_applicable():
            self.rx_iq_frontend_compensated = compensator.compensate(signal_dict)
            return self.rx_iq_frontend_compensated
        compensator = getattr(self, "iq_blind_frontend_compensator", None)
        if compensator is None or not compensator.is_applicable():
            return signal_dict
        self.rx_iq_frontend_compensated = compensator.compensate(signal_dict)
        return self.rx_iq_frontend_compensated

    def compensate_iq_imbalance(self, signal_dict):
        """在噪声/信道估计前执行基于 CES 的 IQ 补偿。"""
        if getattr(self, "iq_compensator", None) is None:
            return signal_dict
        self.rx_iq_compensated = self.iq_compensator.compensate(
            signal_dict, tail_mode="error"
        )
        return self.rx_iq_compensated

    def estimate_channel(self, signal_dict):
        self.rx_estimated = self.channel_estimator.channel_estimate(signal_dict)
        return self.rx_estimated

    def equalize(self, signal_dict):
        self.rx_equalized = self.equalizer.equalize(signal_dict)
        return self.rx_equalized

    def ofdm_demodulate(self, signal_dict):
        self.rx_equalized = self.rx_ofdm.ofdm_demodulate(
            signal_dict, H_init_list=None)
        frontend = self.rx_iq_frontend_compensated
        if frontend is not None:
            stages = [
                frontend.get(
                    "iq_frontend_compensation_method",
                    frontend.get("iq_compensation_method"),
                )
            ]
            if (
                self.rx_equalized.get("iq_compensation_method")
                == "ofdm_widely_linear"
            ):
                stages.append("ofdm_widely_linear")
            self.rx_equalized["iq_compensation_stages"] = [
                stage for stage in stages if stage
            ]
            self.rx_equalized["iq_frontend_diagnostics"] = (
                frontend.get("iq_blind_diagnostics")
                or frontend.get("iq_configured_diagnostics")
            )
            if self.rx_equalized.get("iq_compensation_method") == "none":
                self.rx_equalized["iq_compensation_method"] = stages[0]
        if (
            self.rx_equalized.get("iq_compensation_method")
            == "ofdm_widely_linear"
            or frontend is not None
        ):
            self.rx_iq_compensated = self.rx_equalized
        return self.rx_equalized

    def estimate_noise(self, signal_dict):
        result = self.noise_estimator.noise_estimate(signal_dict)
        var_list = result.get("noise_var", [0.01])
        if isinstance(var_list, list):
            self.noise_var = np.mean(var_list)
        else:
            self.noise_var = var_list
        return self.noise_var

    def demodulate(self, signal_dict):
        # 使用噪声方差估计值计算 sigma
        noise_var = signal_dict.get(
            "post_equalization_noise_var",
            signal_dict.get("noise_var", self.noise_var),
        )
        if isinstance(noise_var, (list, tuple, np.ndarray)):
            noise_var = float(np.mean(noise_var))
        noise_var = float(noise_var) if noise_var is not None else 0.01
        if not np.isfinite(noise_var) or noise_var <= 0:
            noise_var = 0.01
        sigma = np.sqrt(noise_var / 2)
        # 重建兼容的 dict（确保 sample_rate * duration == signal_length）
        sym = signal_dict["signal_stream"]
        fs = signal_dict.get("sample_rate_Hz", 1.0)
        compat_dict = {
            "signal_stream": sym,
            "sample_rate_Hz": fs,
            "signal_length": len(sym),
            "duration_seconds": len(sym) / fs if fs > 0 else 0,
            "padding_bit_num": signal_dict.get("padding_bit_num", 0),
        }
        self.llr_dict = self.demodulator.demodulate(compat_dict, sigma)
        return self.llr_dict

    def decode(self, signal_dict):
        # Pad/truncate LLR to expected coded bit length if needed
        llr = signal_dict["signal_stream"]
        expected_len = round(signal_dict["sample_rate_Hz"] * signal_dict["duration_seconds"])
        if len(llr) > expected_len:
            signal_dict = dict(signal_dict)
            signal_dict["signal_stream"] = llr[:expected_len]
            signal_dict["signal_length"] = expected_len
        elif len(llr) < expected_len:
            signal_dict = dict(signal_dict)
            pad = np.zeros(expected_len - len(llr), dtype=llr.dtype)
            signal_dict["signal_stream"] = np.concatenate([llr, pad])
            signal_dict["signal_length"] = expected_len
        self.decoded_bits = self.decoder.decode(signal_dict)
        if self.params.get("scramble"):
            self.data_bits = self.descrambler.descramble(self.decoded_bits)
        else:
            self.data_bits = self.decoded_bits
        return self.data_bits

    # ==================== 主流程 ====================

    def run(self, rx_signal_dict):
        """
        完整接收流程 — 分支 SC-FDE / OFDM
        """
        if getattr(self, "enable_mimo", False):
            sig = self.mimo_processor.process(rx_signal_dict)
            self.rx_matched = self.mimo_processor.compensated_signal
            self.rx_coarse_synced = self.mimo_processor.compensated_signal
            self.rx_cfo_coarse = self.mimo_processor.compensated_signal
            self.rx_fine_synced = self.mimo_processor.compensated_signal
            self.rx_downsampled = self.mimo_processor.compensated_signal
            self.rx_cfo_fine = self.mimo_processor.compensated_signal
            self.rx_equalized = sig
            self.noise_var = max(float(sig.get("noise_var", 0.0)), 1e-12)
            sig = self.demodulate(sig)
            return self.decode(sig)

        # ⓪ 已知 RX FID IQ 参数：损伤注入的逆序处理，先于 CFO/滤波/抽取。
        sig = self.compensate_iq_configured(rx_signal_dict)

        # ① 匹配滤波
        sig = self.matched_filter(sig)

        # ② 粗同步（SYNC 互相关定位）
        sig = self.coarse_sync_detect(sig)

        # ③ 细同步（在匹配滤波原始坐标系中合并 coarse+fine 偏移）
        sig = self.fine_sync_frame(sig)

        # ④ 粗 CFO（帧起点已对齐，SYNC 重复段从索引 0 开始）
        if getattr(self, "enable_cfo", True):
            sig = self.compensate_cfo_coarse(sig)

        # ⑤ 下采样（过采样率 → 符号率）
        sig = self.downsample(sig)

        # ⑥ 细 CFO（逐帧估计+补偿，符号率）
        if getattr(self, "enable_cfo", True):
            sig = self.compensate_cfo_fine(sig)

        # ⑥b 基于 CES 的宽线性 IQ 补偿（如启用）
        sig = self.compensate_iq_imbalance(sig)

        # ⑦ 噪声方差估计（基于 SYNC，需在 OFDM 解调前）
        self.estimate_noise(sig)
        # 将噪声估计显式送入后续信道估计/均衡链路；此前该值只保存在
        # receiver 成员中，MMSE 和均衡后噪声传播都无法读取。
        sig = dict(sig)
        sig["noise_var"] = self.noise_var

        if self.enable_channel_est:
            if self.link_mode == "ofdm":
                # OFDM: 导频LS信道估计 + 相位跟踪 + 均衡
                sig = self.ofdm_demodulate(sig)
            else:
                # SC-FDE: CES信道估计 + 频域均衡
                sig = self.estimate_channel(sig)
                sig = self.equalize(sig)
        else:
            # 关闭信道估计/均衡——仍执行去前导+去CP+FFT/IFFT
            if self.link_mode == "ofdm":
                sig = self.ofdm_demodulate(sig)
            else:
                params = getattr(self, "params", None)
                n_sc = params.get("subframe_length") if params is not None else 1
                sig["channel_freq_response"] = np.ones(n_sc, dtype=np.complex128)
                sig = self.equalize(sig)

        # ⑩ SC-FDE 专用判决导向IQ补偿；OFDM 在频域网格内完成补偿。
        if getattr(self, "iq_dd_compensator", None) is not None:
            sig = self.compensate_iq_decision_directed(sig)

        # ⑪ 解调（LLR，使用噪声方差）
        sig = self.demodulate(sig)

        # ⑫ 解码 + 解扰
        data = self.decode(sig)
        return data

# ==================== 测试 ====================
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from receiver.DeModulator import THzDemodulator

    out_dir = "simulation_results"
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
        "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
    })

    test_cases = [
        ("no_iq",       False, False),   # 无IQ不平衡
        ("iq_no_comp",  True,  False),   # 有IQ不平衡，不补偿
        ("iq_comp",     True,  True),    # 有IQ不平衡，补偿
    ]

    for mode in ["sc-fde", "ofdm"]:
        print(f"\n{'='*60}")
        print(f"  {mode.upper()}  —  IQ 不平衡补偿测试")
        print(f"{'='*60}")

        for tag, iq_on, iq_comp in test_cases:
            print(f"\n--- {mode} / {tag} ---")

            params = PHYParams()
            params.update(
                link_mode=mode,
                enable_awgn=True, SNRdB=24,
                enable_cfo=False, enable_phase_noise=False,
                enable_multipath=False,
                code_type="LDPC",
                enable_iq_imbalance=iq_on,
                enable_iq_compensation=iq_comp,
                random_seed=20260722, seed_strategy="固定种子",
            )
            if iq_on:
                params.update(
                    iq_imbalance_position="rx",
                    iq_imbalance_model="fd",
                    rx_iq_gain_imbalance_db=2.0,
                    rx_iq_phase_imbalance_deg=5.0,
                    rx_iq_gI_taps=[1.0, 0.08, -0.03],
                    rx_iq_gQ_taps=[1.0, -0.12, 0.04],
                )
            np.random.seed(20260722)
            tx = THzTransmitter(params); tx.run()
            ch = THzChannel(params); rx = ch.run(tx.tx_signal_dict)
            recv = THzReceiver(params, tx); rx_data = recv.run(rx)

            # ———— BER ————
            tx_bits = tx.data_bits_dict["signal_stream"]
            rx_bits = rx_data["signal_stream"]
            cmp = min(len(tx_bits), len(rx_bits))
            ber = np.sum(tx_bits[:cmp] != rx_bits[:cmp]) / cmp

            # ———— EVM & IRR（补偿后符号）————
            # DD补偿后信号在 rx_iq_compensated 中，未补偿时退回 rx_equalized
            eq_dict = recv.rx_iq_compensated if recv.rx_iq_compensated is not None \
                      else recv.rx_equalized
            rx_syms = eq_dict["signal_stream"]
            tx_syms = tx.modulated_data_dict["signal_stream"]
            cmp_sym = min(len(tx_syms), len(rx_syms))
            tx_cmp = tx_syms[:cmp_sym]
            rx_cmp = rx_syms[:cmp_sym]
            evm = np.sqrt(np.mean(np.abs(rx_cmp - tx_cmp)**2)
                          / np.mean(np.abs(tx_cmp)**2))
            image_design = np.column_stack((tx_cmp, np.conj(tx_cmp)))
            direct_gain, image_gain = np.linalg.lstsq(
                image_design, rx_cmp, rcond=None)[0]
            irr_db = 10 * np.log10(
                (np.abs(direct_gain)**2 + 1e-15)
                / (np.abs(image_gain)**2 + 1e-15))

            print(f"  noise_var={recv.noise_var:.2e}  BER={ber:.2e}"
                  f"  EVM={100*evm:.2f}%  IRR={irr_db:.2f} dB")

            # ———— 星座图 ————
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.plot(tx_cmp[::10].real, tx_cmp[::10].imag, ".", ms=2,
                    alpha=0.3, label="TX")
            ax.plot(rx_cmp[::10].real, rx_cmp[::10].imag, ".", ms=2,
                    alpha=0.7, label="RX")
            ax.set_xlabel("I"); ax.set_ylabel("Q")
            ax.set_title(f"{mode.upper()} — {tag}  |  EVM={100*evm:.1f}%  "
                         f"IRR={irr_db:.1f} dB", fontsize=10, pad=6)
            ax.axis("equal"); ax.legend(fontsize=8, markerscale=3)
            fig.tight_layout()
            for ext in ["pdf", "png"]:
                fig.savefig(f"{out_dir}/iq_{mode}_{tag}.{ext}",
                            dpi=600 if ext == "pdf" else 300)
            plt.close(fig)

    print(f"\nFigures saved to {out_dir}/")
