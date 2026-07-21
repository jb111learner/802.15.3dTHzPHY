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


class THzReceiver(BaseReceiver):
    """
    THz 接收机主类 — 支持 SC-FDE / OFDM 双模式

    流程:
      MF → CoarseSync → CFO coarse → FineSync → Downsample → CFO fine
      → [SC-FDE: ChannelEst + FreqDomainEqualizer]
      → [OFDM:   RxOFDMProcesser (内置LS+相位跟踪)]
      → NoiseEst → Demodulate → Decode → DeScramble
    """

    def __init__(self, params, transmitter):
        super().__init__(params, transmitter)
        self.link_mode = params.get("link_mode").lower()

        self.rx_matched_filter = RxMatchedFilter(transmitter)
        self.coarse_sync = CoarseSync(transmitter)
        self.fine_sync = FineSync(transmitter)
        self.cfo_estimator = CFOEstimator(transmitter)
        self.downsampler = Downsampler(transmitter)
        self.channel_estimator = ChannelEstimator(transmitter)
        self.equalizer = FreqDomainEqualizer(transmitter)
        self.noise_estimator = NoiseEstimator(transmitter)
        self.demodulator = THzDemodulator(params)
        self.decoder = Decoder(params)
        self.descrambler = DeScrambler(params)
        if self.link_mode == "ofdm":
            self.rx_ofdm = RxOFDMProcesser(transmitter)

        # 中间结果缓存
        self.rx_matched = None
        self.rx_coarse_synced = None
        self.rx_cfo_coarse = None
        self.rx_fine_synced = None
        self.rx_downsampled = None
        self.rx_cfo_fine = None
        self.rx_equalized = None
        self.noise_var = None
        self.llr_dict = None
        self.decoded_bits = None
        self.data_bits = None

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

    def estimate_channel(self, signal_dict):
        self.rx_estimated = self.channel_estimator.channel_estimate(signal_dict)
        return self.rx_estimated

    def equalize(self, signal_dict):
        self.rx_equalized = self.equalizer.equalize(signal_dict)
        return self.rx_equalized

    def ofdm_demodulate(self, signal_dict):
        self.rx_equalized = self.rx_ofdm.ofdm_demodulate(
            signal_dict, H_init_list=None)
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
        noise_var = self.noise_var if self.noise_var and self.noise_var > 0 else 0.01
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
        self.data_bits = self.descrambler.descramble(self.decoded_bits)
        return self.data_bits

    # ==================== 主流程 ====================

    def run(self, rx_signal_dict):
        """
        完整接收流程 — 分支 SC-FDE / OFDM
        """
        # ① 匹配滤波
        sig = self.matched_filter(rx_signal_dict)

        # ② 粗同步（SYNC 互相关定位）
        sig = self.coarse_sync_detect(sig)

        # ③ 粗 CFO（逐帧估计+补偿）
        sig = self.compensate_cfo_coarse(sig)

        # ④ 细同步（SFD 帧定界）
        sig = self.fine_sync_frame(sig)

        # ⑤ 下采样
        sig = self.downsample(sig)

        # ⑥ 细 CFO（逐帧估计+补偿）
        sig = self.compensate_cfo_fine(sig)

        # ⑦ 噪声方差估计（基于 SYNC，需在 OFDM 解调前）
        self.estimate_noise(sig)

        if self.link_mode == "ofdm":
            # ⑧ OFDM 解调（内置导频LS + 相位跟踪 + 均衡）
            sig = self.ofdm_demodulate(sig)
        else:
            # ⑧ SC-FDE: 信道估计 + 频域均衡
            sig = self.estimate_channel(sig)
            sig = self.equalize(sig)

        # ⑨ 解调（LLR，使用噪声方差）
        sig = self.demodulate(sig)

        # ⑩ 解码 + 解扰
        data = self.decode(sig)
        return data

# ==================== 测试 ====================
if __name__ == "__main__":

    params = PHYParams()
    tx = THzTransmitter(params)
    tx.run()
    ch = THzChannel(params)
    rx = ch.run(tx.tx_signal_dict)

    receiver = THzReceiver(params, tx)
    rx_data = receiver.run(rx)

    tx_bits = tx.data_bits_dict["signal_stream"]
    rx_bits = rx_data["signal_stream"]
    cmp = min(len(tx_bits), len(rx_bits))
    ber = np.sum(tx_bits[:cmp] != rx_bits[:cmp]) / cmp
    print(f"  noise_var = {receiver.noise_var:.2e}")
    print(f"  BER = {ber:.2e}  ({np.sum(tx_bits[:cmp] != rx_bits[:cmp])} / {cmp})")    


# # ==================== 测试 ====================
# if __name__ == "__main__":
#     for mode in ["sc-fde", "ofdm"]:
#         print(f"\n{'='*55}")
#         print(f"     {mode.upper()} 模式")
#         print(f"{'='*55}")

#         params = PHYParams()
#         params.update(link_mode=mode)
#         tx = THzTransmitter(params)
#         tx.run()
#         ch = THzChannel(params)
#         rx = ch.run(tx.tx_signal_dict)

#         receiver = THzReceiver(params, tx)
#         rx_data = receiver.run(rx)

#         tx_bits = tx.data_bits_dict["signal_stream"]
#         rx_bits = rx_data["signal_stream"]
#         cmp = min(len(tx_bits), len(rx_bits))
#         ber = np.sum(tx_bits[:cmp] != rx_bits[:cmp]) / cmp
#         print(f"  noise_var = {receiver.noise_var:.2e}")
#         print(f"  BER = {ber:.2e}  ({np.sum(tx_bits[:cmp] != rx_bits[:cmp])} / {cmp})")

#     print("\nDone")
