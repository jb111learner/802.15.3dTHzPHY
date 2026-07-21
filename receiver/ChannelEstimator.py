import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.CFOEstimator import CFOEstimator

class ChannelEstimator:
    """
    信道估计器（符号率输入），支持 SC-FDE 和 OFDM 两种模式：

      SC-FDE — CES互相关信道估计（时域相关 → CIR → FFT → CSI）
      OFDM   — CES LS信道估计（CES段 FFT → 频域LS → CSI）

    支持两种帧策略：
      - 'per_frame' : 每帧独立估计
      - 'global'     : 仅用第一帧估计，所有帧共享
    """
    def __init__(self, transmitter, estimation_mode='per_frame'):
        self.params = transmitter.params
        self.link_mode = self.params.get("link_mode").lower()

        # —————— CES 公共参数 ——————
        self.N_ces = len(transmitter.preamble_gen.a512)    # 512 (a512/b512长度)
        self.Lh = self.params.get("gi_length")             # CP长度，也是CIR保留长度
        self.nfft = self.params.get("subframe_length")     # FFT点数 = 512
        self.len_b128 = len(transmitter.preamble_gen.b128) # 128
        self.sync_len = len(transmitter.sync)              # SYNC符号数
        self.sfd_len = len(transmitter.sfd)                # SFD符号数
        self.ces_seq = transmitter.ces                     # 完整CES序列（符号级）
        self.ces_len = len(self.ces_seq)                   # CES总长度 = 1408

        # —————— OFDM LS估计专用：本地CES频域参考 ——————
        self.tx_a512 = transmitter.preamble_gen.a512       # 512点 BPSK
        self.tx_b512 = transmitter.preamble_gen.b512       # 512点 BPSK

        # —————— 帧长度（符号级）——————
        preamble_len = len(transmitter.preamble)           # 前导码总长
        subframe_len = self.params.get("subframe_length")  # 512
        gi_len = self.params.get("gi_length")              # 32

        if self.link_mode == "ofdm":
            # OFDM: 每帧 = 前导码 + 48个OFDM符号×(子载波+GI)
            subframe_ofdm_num = self.params.get("subframe_ofdm_num")  # 48
            self.frame_symbol_num = preamble_len + \
                (subframe_len + gi_len) * subframe_ofdm_num
        else:
            # SC-FDE: 每帧 = 前导码 + subframe_num个块×(块长+GI)
            self.frame_symbol_num = preamble_len + \
                (subframe_len + gi_len) * self.params.get("subframe_num")

        self.estimation_mode = estimation_mode

        # 输出缓存
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0
        self.channel_freq_response = None
        self.channel_time_response = None

    def _verification_data(self, data_dict):
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("采样率与时长不匹配")
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.signal_length = data_dict["signal_length"]
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]

    def _split_into_frames(self, rx_signal, frame_len, tail_mode='discard'):
        """
        将一维/二维信号分割为帧列表
        :param rx_signal: 输入信号 (1D 或 2D)
        :param tail_mode: 'discard' 丢弃不足一帧的尾部，'zero_pad' 补零至整帧
        :return: (frames, num_frames, processed_len)
        """
        total_len = len(rx_signal)
        if frame_len <= 0:
            raise ValueError("帧长度必须大于 0")
        num_frames = total_len // frame_len
        remainder = total_len % frame_len

        if remainder == 0:
            processed_signal = rx_signal
            num_frames = total_len // frame_len
        else:
            if tail_mode == 'discard':
                processed_signal = rx_signal[:num_frames * frame_len]
                print(f"丢弃尾部 {remainder} 个采样点（不足一帧）")
            elif tail_mode == 'zero_pad':
                pad_len = frame_len - remainder
                if rx_signal.ndim == 1:
                    pad = np.zeros(pad_len, dtype=rx_signal.dtype)
                else:
                    pad = np.zeros((pad_len, rx_signal.shape[1]), dtype=rx_signal.dtype)
                processed_signal = np.concatenate([rx_signal, pad], axis=0)
                num_frames = num_frames + 1
                print(f"尾部补零 {pad_len} 个采样点")
            else:
                raise ValueError("tail_mode 必须为 'discard' 或 'zero_pad'")

        frames = np.split(processed_signal, num_frames, axis=0)
        return frames, num_frames, len(processed_signal)

    def ces_corr_estimate(self, ces_field, ces_seq):
        """
        核心CES互相关信道估计（与原逻辑一致，未修改）
        """
        # 提取a512和b512
        start_a512 = self.len_b128
        a512 = ces_seq[start_a512 : start_a512 + self.N_ces]
        start_b512 = start_a512 + self.N_ces
        b512 = ces_seq[start_b512 : start_b512 + self.N_ces]

        ra512 = ces_field[start_a512 : start_a512 + self.N_ces]
        rb512 = ces_field[start_b512 : start_b512 + self.N_ces]

        ra = np.convolve(ra512, np.flip(np.conj(a512)), mode='full')
        rb = np.convolve(rb512, np.flip(np.conj(b512)), mode='full')
        c = (ra + rb) / (2 * self.N_ces)

        abs_c = np.abs(c)
        center = np.argmax(abs_c)

        start_region = int(center - self.Lh / 2)
        end_region = int(center + self.Lh / 2)
        start_region = max(0, start_region)
        end_region = min(len(c)-1, end_region)
        c_region = c[start_region:end_region+1]
        abs_c_region = np.abs(c_region)

        # peaks, _ = find_peaks(abs_c_region, height=0.1 * abs_c.max())
        # if len(peaks) == 0:
        #     peaks = [np.argmax(abs_c_region)]
        peaks, _ = find_peaks(abs_c_region, height=0.1 * abs_c.max())
        if len(peaks) == 0:
            peaks = np.array([np.argmax(abs_c_region)])   # 转为numpy数组
        else:
            peaks = np.array(peaks)                       # 确保为numpy数组        

        idx0 = peaks + start_region
        fine_offset = idx0[0] - self.N_ces
        idx_h = idx0 - idx0[0]

        h_est = np.zeros((self.Lh,), dtype=np.complex128)
        valid_idx = [i for i in idx_h if 0 <= i < self.Lh]
        valid_c = [c[idx0[k]] for k in range(len(idx0)) if 0 <= idx_h[k] < self.Lh]
        h_est[valid_idx] = valid_c

        H_est = np.fft.fftshift(np.fft.fft(h_est, self.nfft))
        return H_est, fine_offset, h_est

    # ==================== OFDM LS 信道估计 ====================
    def _ls_estimate_ofdm(self, ces_field):
        """
        OFDM模式：基于CES序列的LS（最小二乘）信道估计

        关键修正：线性卷积 vs 循环卷积
          - 多径信道下接收CES是 tx * h 的线性卷积，长度 = 512 + Lh - 1
          - 必须用扩展FFT窗口捕获完整卷积响应，否则LS幅值失真
          - 做法：提取 512+Lh-1 点 → FFT → LS → IFFT → 截断至 Lh

        :param ces_field: 接收CES时域信号（符号率，长度1408）
        :return: H_est (512点频域CSI), fine_offset (0), h_est (Lh点时域CIR)
        """
        full_len = self.N_ces + self.Lh - 1   # 512 + 31 = 543（完整线性卷积长度）

        # 1. 提取 a512 / b512 的完整线性卷积响应
        start_a512 = self.len_b128
        rx_a512_full = ces_field[start_a512 : start_a512 + full_len]   # 543
        start_b512 = start_a512 + self.N_ces
        rx_b512_full = ces_field[start_b512 : start_b512 + full_len]   # 543

        # 2. TX参考序列零填充至 full_len
        tx_a512_padded = np.pad(self.tx_a512, (0, self.Lh - 1))  # 543
        tx_b512_padded = np.pad(self.tx_b512, (0, self.Lh - 1))  # 543

        # 3. FFT（full_len点，捕获完整线性卷积）
        Y_a = np.fft.fft(rx_a512_full)
        Y_b = np.fft.fft(rx_b512_full)
        X_a = np.fft.fft(tx_a512_padded)
        X_b = np.fft.fft(tx_b512_padded)

        # 4. LS估计: H = Y / X（阈值保护低功率子载波）
        threshold = np.max([np.max(np.abs(X_a)), np.max(np.abs(X_b))]) * 1e-3
        mask_a = np.abs(X_a) > threshold
        mask_b = np.abs(X_b) > threshold

        H_a = np.zeros(full_len, dtype=np.complex128)
        H_b = np.zeros(full_len, dtype=np.complex128)
        H_a[mask_a] = Y_a[mask_a] / X_a[mask_a]
        H_b[mask_b] = Y_b[mask_b] / X_b[mask_b]

        # 5. 两组平均
        mask_both = mask_a & mask_b
        H_est_full = np.zeros(full_len, dtype=np.complex128)
        H_est_full[mask_both] = (H_a[mask_both] + H_b[mask_both]) / 2.0
        H_est_full[mask_a & ~mask_b] = H_a[mask_a & ~mask_b]
        H_est_full[mask_b & ~mask_a] = H_b[mask_b & ~mask_a]

        # 6. IFFT → 时域CIR（full_len点），截断至 Lh
        h_full = np.fft.ifft(H_est_full)
        h_est = h_full[:self.Lh].copy()

        # 7. 输出512点频域CSI（零填充CIR → FFT，与SC-FDE格式一致）
        h_padded = np.zeros(self.N_ces, dtype=np.complex128)
        h_padded[:self.Lh] = h_est
        H_est = np.fft.fft(h_padded)

        fine_offset = 0

        return H_est, fine_offset, h_est

    def channel_estimate(self, signal_dict):
        """
        多帧信道估计入口

        SC-FDE模式: CES互相关 → 时域CIR → FFT → 频域CSI
        OFDM模式:   CES LS估计 → 频域CSI → IFFT → 时域CIR

        :param signal_dict: 符号率接收信号字典
        :return: result_dict (含 channel_freq_response, channel_time_response, fine_offset)
        """
        self._verification_data(signal_dict)
        rx_symbols = signal_dict["signal_stream"]

        # 分割帧
        frames, num_frames, _ = self._split_into_frames(rx_symbols, self.frame_symbol_num)
        if num_frames == 0:
            raise ValueError("信号不包含完整帧，无法进行信道估计")

        # CES偏移（相对于帧起始）
        ces_offset = self.sync_len + self.sfd_len

        H_list = []
        h_list = []
        fine_offset_list = []

        # global模式：先估计第一帧作为共享值
        first_H = first_fine = first_h = None
        if self.estimation_mode == 'global' and num_frames > 0:
            first_frame = frames[0]
            ces_field = first_frame[ces_offset : ces_offset + self.ces_len]
            if self.link_mode == "ofdm":
                first_H, first_fine, first_h = self._ls_estimate_ofdm(ces_field)
            else:
                first_H, first_fine, first_h = self.ces_corr_estimate(ces_field, self.ces_seq)

        for i, frame in enumerate(frames):
            if self.estimation_mode == 'per_frame':
                ces_field = frame[ces_offset : ces_offset + self.ces_len]
                if self.link_mode == "ofdm":
                    H_est, fine_offset, h_est = self._ls_estimate_ofdm(ces_field)
                else:
                    H_est, fine_offset, h_est = self.ces_corr_estimate(ces_field, self.ces_seq)
                H_list.append(H_est)
                h_list.append(h_est)
                fine_offset_list.append(fine_offset)
            else:  # 'global'
                H_list.append(first_H.copy())
                h_list.append(first_h.copy())
                fine_offset_list.append(first_fine)

        # 缓存
        if num_frames == 1:
            self.channel_freq_response = H_list[0]
            self.channel_time_response = h_list[0]
        else:
            self.channel_freq_response = H_list
            self.channel_time_response = h_list

        result_dict = {
            "signal_stream": rx_symbols,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
            "channel_freq_response": H_list,
            "channel_time_response": h_list,
            "fine_offset": fine_offset_list,
        }
        return result_dict



# ==================== 测试 ====================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    for mode in ["sc-fde", "ofdm"]:
        print(f"\n{'='*50}")
        print(f"     {mode.upper()} 模式")
        print(f"{'='*50}")

        # ---- TX ----
        params = PHYParams()
        params.update(link_mode=mode)
        tx = THzTransmitter(params)
        tx.run()
        ch = THzChannel(params)
        rx = ch.run(tx.tx_signal_dict)

        sps = params.get("oversampling")
        Lh = params.get("gi_length")
        nfft = params.get("subframe_length")

        # ---- RX: MF -> coarse CFO -> fine CFO -> ChEst ----
        mf = RxMatchedFilter(tx)
        mf_out = mf.matched_filter(rx)
        mf_sig = mf_out["signal_stream"]
        fd = mf.filter_delay
        mf_fs = mf_out["sample_rate_Hz"]
        sym_fs = mf_fs / sps
        tx_len = len(tx.data_with_preamble_dict["signal_stream"])

        ce = CFOEstimator(tx)
        cfo_c = ce.estimate_cfo_coarse(mf_sig, fs=mf_fs)
        mf_c = ce.compensate_cfo(mf_sig, fs=mf_fs, cfo_est=cfo_c)
        rs = mf_c[fd:fd+tx_len*sps:sps]
        cfo_f = ce.estimate_cfo_fine(rs, fs=sym_fs)
        rs = ce.compensate_cfo(rs, fs=sym_fs, cfo_est=cfo_f)
        rs = np.pad(rs[:tx_len], (0, max(0, tx_len-len(rs))))

        nd = dict(tx.data_with_preamble_dict)
        nd["signal_stream"] = rs
        r = ChannelEstimator(tx).channel_estimate(nd)
        h_est = r["channel_time_response"][0]
        H_est = r["channel_freq_response"][0]

        # ---- true equivalent symbol-rate channel ----
        h_phys = np.zeros(Lh*sps, dtype=np.complex128)
        h_phys[:len(ch.chan_true)] = ch.chan_true
        pt, pr = tx.pulse_shaper.filter_coeffs, mf.h_rx
        heq = np.convolve(np.convolve(h_phys, pt), pr)
        gd = (len(pt)-1)//2 + (len(pr)-1)//2
        ht = heq[gd::sps][:Lh]
        Ht = np.fft.fft(ht, nfft)

        nmse_t = np.linalg.norm(ht-h_est)**2/(np.linalg.norm(ht)**2+np.finfo(float).eps)
        nmse_f = np.linalg.norm(Ht-H_est)**2/(np.linalg.norm(Ht)**2+np.finfo(float).eps)
        cfo_err = abs(cfo_c+cfo_f-ch.cfo.freq_offset) if hasattr(ch, 'cfo') else 0

        print(f"  CFO coarse={cfo_c:.1f}Hz fine={cfo_f:.1f}Hz residual={cfo_err:.1f}Hz")
        print(f"  CIR NMSE={10*np.log10(nmse_t):.1f}dB  CSI NMSE={10*np.log10(nmse_f):.1f}dB")

    # ---- visualization ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for idx, mode in enumerate(["sc-fde", "ofdm"]):
        params = PHYParams()
        params.update(link_mode=mode)
        sps, Lh, nfft = params.get("oversampling"), params.get("gi_length"), params.get("subframe_length")
        tx = THzTransmitter(params); tx.run()
        ch = THzChannel(params); rx = ch.run(tx.tx_signal_dict)
        mf = RxMatchedFilter(tx); mf_out = mf.matched_filter(rx)
        mf_sig = mf_out["signal_stream"]; mf_fs = mf_out["sample_rate_Hz"]
        fd = mf.filter_delay; sym_fs = mf_fs/sps
        tx_len = len(tx.data_with_preamble_dict["signal_stream"])
        ce = CFOEstimator(tx)
        cfo_c = ce.estimate_cfo_coarse(mf_sig, fs=mf_fs)
        mf_c = ce.compensate_cfo(mf_sig, fs=mf_fs, cfo_est=cfo_c)
        rs = mf_c[fd:fd+tx_len*sps:sps]
        cfo_f = ce.estimate_cfo_fine(rs, fs=sym_fs)
        rs = ce.compensate_cfo(rs, fs=sym_fs, cfo_est=cfo_f)
        rs = np.pad(rs[:tx_len], (0, max(0, tx_len-len(rs))))
        h_phys = np.zeros(Lh*sps, dtype=np.complex128)
        h_phys[:len(ch.chan_true)] = ch.chan_true
        pt, pr = tx.pulse_shaper.filter_coeffs, mf.h_rx
        heq = np.convolve(np.convolve(h_phys, pt), pr)
        gd = (len(pt)-1)//2 + (len(pr)-1)//2
        ht = heq[gd::sps][:Lh]; Ht = np.fft.fft(ht, nfft)
        nd = dict(tx.data_with_preamble_dict); nd["signal_stream"] = rs
        r = ChannelEstimator(tx).channel_estimate(nd)
        he = r["channel_time_response"][0]; He = r["channel_freq_response"][0]
        nmse = np.linalg.norm(Ht-He)**2/(np.linalg.norm(Ht)**2+np.finfo(float).eps)
        cfo_res = abs(cfo_c+cfo_f-ch.cfo.freq_offset) if hasattr(ch, 'cfo') else 0
        ax0 = axes[0, idx]
        ax0.stem(np.abs(ht), linefmt='b-', markerfmt='bo', basefmt=' ', label='|h_true|')
        ax0.stem(np.abs(he), linefmt='r--', markerfmt='rx', basefmt=' ', label='|h_est|')
        ax0.set_title(f'{mode.upper()} CIR (CFO res={cfo_res:.0f}Hz NMSE={10*np.log10(nmse):.1f}dB)')
        ax0.set_xlabel('tap'); ax0.set_ylabel('|h|'); ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)
        ax1 = axes[1, idx]
        ax1.plot(np.abs(Ht), 'b-', lw=1.2, alpha=0.8, label='|H_true|')
        ax1.plot(np.abs(He), 'r--', lw=0.8, alpha=0.8, label='|H_est|')
        ax1.set_title(f'{mode.upper()} CSI (NMSE={10*np.log10(nmse):.1f}dB)')
        ax1.set_xlabel('subcarrier'); ax1.set_ylabel('|H|'); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
    fig.suptitle('Channel Estimation with Sync+CFO (SC-FDE corr vs OFDM LS)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()
    print("\nDone")
