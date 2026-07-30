import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


class RxOFDMProcesser:
    """
    OFDM 接收端处理器：完成完整的 OFDM 解调流程

    流程：
      1. 去前导码 → 去GI(CP) → 时域OFDM符号
      2. 串并转换 → FFT → 频域OFDM网格 (N_SUB, 48, 512)
      3. 块状导频LMS信道估计
      4. 信道补偿（迫零均衡）
      5. 提取数据OFDM符号 → 输出QAM符号流
    """

    def __init__(self, transmitter):
        self.params = transmitter.params
        self.preamble_len = len(transmitter.preamble)

        # —————— 信道估计开关 ——————
        try:
            self.enable_ch_est = self.params.get("enable_channel_equalization")
        except KeyError:
            self.enable_ch_est = True

        # —————— OFDM 参数 ——————
        self.N_SC = self.params.get("subwave_num")             # 512 子载波
        self.N_SYM = self.params.get("subframe_ofdm_num")      # 48 OFDM符号/子帧
        self.pilot_indexes = self.params.get("pilot_block_indexes")  # [0, 16, 32]
        self.gi_len = self.params.get("gi_length")             # 32
        self.N_DATA = self.N_SYM - len(self.pilot_indexes)     # 45 数据符号/子帧

        # —————— 导频参考 ——————
        self.pilot_seq = transmitter.ofdm_processer.pilot_seq  # a512 (BPSK, |·|=1)

        # —————— LMS 参数 ——————
        self.lms_mu = 0.1          # LMS 步长
        self.lms_n_iter = 3        # 迭代次数（多次遍历导频）

        # —————— 输出缓存 ——————
        self.sample_rate = None
        self.duration = None
        self.symbol_length = None
        self.padding_bit_num = 0
        self.frame_num = None

        # —————— 中间结果（调试用） ——————
        self.ofdm_freq_grid = None       # 频域OFDM网格
        self.H_est_per_subframe = None   # 每子帧的信道估计
        self.data_symbols = None         # 解调后的数据符号

    def _verification_data(self, data_dict):
        """校验输入数据字典，处理多帧信号"""
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        total_len = int(data_dict["signal_length"])
        base_rate = data_dict["sample_rate_Hz"]
        base_dur = data_dict["duration_seconds"]

        if round(base_rate * base_dur) != total_len:
            raise ValueError("采样率与时长不匹配")

        # 每帧 = 前导码 + N_SYM个OFDM符号块（每块=GI+子载波）
        frame_symbol_num = self.preamble_len + self.N_SYM * (self.gi_len + self.N_SC)
        if total_len % frame_symbol_num != 0:
            raise ValueError(
                f"信号总长度({total_len})不是帧长({frame_symbol_num})的整数倍"
            )

        self.frame_num = total_len // frame_symbol_num
        self.total_ofdm_blocks = self.frame_num * self.N_SYM  # 总OFDM符号数

        self.symbol_length = self.total_ofdm_blocks * self.N_SC
        self.sample_rate = base_rate
        self.duration = self.symbol_length / base_rate
        self.padding_bit_num = data_dict["padding_bit_num"]

    # ==================== 逐帧去前导码 + 去GI ====================
    def _strip_preamble_and_gi(self, rx_signal):
        """
        多帧信号：逐帧去除前导码和GI，返回时域OFDM符号矩阵

        :param rx_signal: 1D 符号率接收信号（含多帧，每帧=前导码+GI数据）
        :return: (N_SC, total_ofdm_blocks) 时域OFDM符号矩阵
        """
        frame_symbol_num = self.preamble_len + self.N_SYM * (self.gi_len + self.N_SC)
        block_len = self.gi_len + self.N_SC

        # 按帧切分
        frames = rx_signal.reshape(self.frame_num, frame_symbol_num)

        all_blocks = []
        for s in range(self.frame_num):
            frame = frames[s]
            # 去除前导码
            data_with_gi = frame[self.preamble_len:]  # length = N_SYM * block_len
            # 重塑为 (N_SYM, block_len)，每行一个OFDM符号块
            blocks_with_gi = data_with_gi.reshape(self.N_SYM, block_len)
            # 去除每块前 gi_len 个CP样本
            blocks_no_gi = blocks_with_gi[:, self.gi_len:]  # (N_SYM, N_SC)
            all_blocks.append(blocks_no_gi)

        # 堆叠所有帧的OFDM符号 → (total_blocks, N_SC)
        all_blocks = np.concatenate(all_blocks, axis=0)  # (total_ofdm_blocks, N_SC)
        # 转置为 (N_SC, total_ofdm_blocks)，每列一个OFDM符号
        return all_blocks.T

    # ==================== 频域转换 ====================
    def _fft_demodulate(self, time_grid):
        """
        OFDM 解调：时域 → 频域
        :param time_grid: (N_SC, num_blocks) 时域OFDM符号矩阵
        :return: (N_SC, num_blocks) 频域OFDM符号矩阵
        """
        # ortho模式：与发射端 ifft(norm='ortho') 匹配，完美还原
        return np.fft.fft(time_grid, axis=0, norm='ortho')

    # ==================== 导频LS + 全局线性相位 + 残余CFO补偿 ====================
    def _ls_track_and_equalize(self, freq_grid, H_init_list=None):
        """
        Pilot LS + global linear-phase + residual CFO self-compensation.

        Strategy (CFO-robust like SC-FDE):
          1. H = pilot-LS average (3 pilots, no windowing)
          2. Measure common phase at all pilots, unwrap
          3. Fit global linear trend from pilot 0 -> pilot 32:
               Δφ_per_sym = (φ32 - φ0) / 32
             This uses max time baseline for best frequency resolution.
          4. Apply per-symbol phase correction: φ(sym) = φ0 + sym·Δφ
          5. Feed-forward: de-rotate pilots with estimated phase, re-estimate
             residual common phase at middle pilot for verification

        Unlike piecewise interpolation, a single linear trend across the
        full 32-symbol span gives optimal residual-CFO frequency estimate.
        """
        eps = np.finfo(np.complex128).eps
        has_init = H_init_list is not None and len(H_init_list) >= self.frame_num
        pilot_set = set(self.pilot_indexes)
        pilot_syms = sorted(self.pilot_indexes)  # [0, 16, 32]
        n_sym = self.N_SYM

        eq_grid = np.zeros_like(freq_grid)
        H_est_list = []
        h_est_list = []

        for s in range(self.frame_num):
            # ---- H: pilot-LS average ----
            if has_init:
                H = H_init_list[s].copy().astype(np.complex128)
            else:
                H_raw = np.zeros(self.N_SC, dtype=np.complex128)
                for p_idx in pilot_syms:
                    col = p_idx * self.frame_num + s
                    Y = freq_grid[:, col]
                    H_raw += Y / (self.pilot_seq + eps)
                H_raw /= len(pilot_syms)
                H = H_raw

            weak = np.abs(H) < max(1e-10, np.max(np.abs(H)) * 1e-3)
            if np.any(weak):
                H[weak] = np.mean(H[~weak])

            # ---- Phase measurement at pilots (weighted) ----
            w = np.abs(H) ** 2
            w_sum = np.sum(w) + eps

            pilot_phases = {}
            for sym in range(n_sym):
                col = sym * self.frame_num + s
                Y = freq_grid[:, col]
                X_eq = Y / (H + eps)
                eq_grid[:, col] = X_eq

                if sym in pilot_set:
                    corr = X_eq * np.conj(self.pilot_seq)
                    pilot_phases[sym] = np.angle(np.sum(w * corr) / w_sum)

            # ---- Global linear trend: φ(sym) = φ₀ + sym × Δφ ----
            pv_raw = [pilot_phases[ps] for ps in pilot_syms]
            pv = list(np.unwrap(pv_raw))  # [φ₀, φ₁₆, φ₃₂]

            # Per-symbol phase increment from full 32-symbol baseline
            phi_0 = pv[0]
            phi_last = pv[-1]
            dphi_per_sym = (phi_last - phi_0) / (pilot_syms[-1] - pilot_syms[0])

            # # ---- Gate: 低SNR下跳过伪相位跟踪 ----
            # dphi_threshold = 0.01  # rad/sym, 真实CFO(100Hz@30GHz)≈0.1 rad/sym
            # if abs(dphi_per_sym) < dphi_threshold:
            #     if abs(dphi_per_sym) > 1e-6:
            #         print(f"  [OFDM] frame {s}: |dphi|={abs(dphi_per_sym):.1e} < "
            #               f"threshold, 跳过相位跟踪")
            #     dphi_per_sym = 0.0
            #     phi_0 = 0.0

            # ---- Apply linear phase ramp to all data symbols ----
            for sym in range(n_sym):
                if sym in pilot_set:
                    continue
                col = sym * self.frame_num + s
                phase = phi_0 + sym * dphi_per_sym
                eq_grid[:, col] *= np.exp(-1j * phase)

            H_est_list.append(H.copy())
            h_est_list.append(np.fft.ifft(H)[:self.gi_len])

        return eq_grid, H_est_list, h_est_list
    def _extract_data_symbols(self, eq_grid):
        """
        从均衡后的OFDM网格中提取数据符号（跳过导频位置）

        列布局（transpose(2,1,0)决定的）: col = sym * N_SUB + s
        即 OFDM符号索引变化慢，子帧索引变化快

        :param eq_grid: (N_SC, total_blocks) 均衡后的频域OFDM网格
        :return: 1D QAM符号流（与发射端调制输出顺序一致）
        """
        pilot_set = set(self.pilot_indexes)
        data_cols = []

        # 按 sym→s 顺序收集数据列（与TX端data_3d[s, d, :]对齐）
        for sym in range(self.N_SYM):
            if sym not in pilot_set:
                for s in range(self.frame_num):
                    data_cols.append(sym * self.frame_num + s)

        # 提取数据列 → (N_SC, N_DATA * frame_num)
        data_grid = eq_grid[:, data_cols]  # (N_SC, N_DATA * N_SUB)

        # 列 k = d*N_SUB + s → data_3d[s, d, :]
        # 重塑为 (frame_num, N_DATA, N_SC) 与TX的data_3d对齐
        data_3d = data_grid.T.reshape(self.N_DATA, self.frame_num, self.N_SC) \
                            .transpose(1, 0, 2)  # (frame_num, N_DATA, N_SC)

        return data_3d.ravel()  # 与 tx_symbols 顺序一致

    # ==================== 主处理流程 ====================
    def ofdm_demodulate(self, signal_dict, H_init_list=None):
        """
        OFDM 接收主流程

        每帧独立处理：LS初值 → 逐符号LMS跟踪（导频+判决引导）→ 均衡

        :param signal_dict: 符号率接收信号字典
        :param H_init_list: 可选，每子帧的LS信道估计初值
        :return: result_dict
        """
        self._verification_data(signal_dict)
        rx_signal = signal_dict["signal_stream"]

        # ———— 1. 逐帧去前导码 + 去GI → 时域OFDM符号矩阵 ————
        time_grid = self._strip_preamble_and_gi(rx_signal)

        # ———— 2. FFT → 频域 ————
        freq_grid = self._fft_demodulate(time_grid)
        self.ofdm_freq_grid = freq_grid

        # ———— 3. LS跟踪+均衡（可开关） ————
        if self.enable_ch_est:
            eq_grid, H_est_list, h_est_list = self._ls_track_and_equalize(
                freq_grid, H_init_list)
            self.H_est_per_subframe = H_est_list
        else:
            # 旁路：仅 pilot-LS + ZF（补偿滤波器响应），跳过相位跟踪
            eps = np.finfo(np.complex128).eps
            eq_grid = np.zeros_like(freq_grid)
            H_est_list = []
            h_est_list = []
            for s in range(self.frame_num):
                H_raw = np.zeros(self.N_SC, dtype=np.complex128)
                for p_idx in self.pilot_indexes:
                    col = p_idx * self.frame_num + s
                    H_raw += freq_grid[:, col] / (self.pilot_seq + eps)
                H_raw /= len(self.pilot_indexes)
                for sym in range(self.N_SYM):
                    col = sym * self.frame_num + s
                    eq_grid[:, col] = freq_grid[:, col] / (H_raw + eps)
                H_est_list.append(H_raw.copy())
                h_est_list.append(np.fft.ifft(H_raw)[:self.gi_len])

        # ———— 4. 提取数据符号 ————
        data_symbols = self._extract_data_symbols(eq_grid)
        self.data_symbols = data_symbols

        # ———— 5. 组装输出 ————
        result_dict = {
            "signal_stream": data_symbols,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": len(data_symbols),
            "padding_bit_num": self.padding_bit_num,
            "frame_symbol_num": self.N_DATA * self.N_SC,
            "frame_num": self.frame_num,
            "channel_freq_response": H_est_list,
            "channel_time_response": h_est_list,
            "ofdm_freq_grid": freq_grid,
        }
        return result_dict


# ==================== 测试 ====================
if __name__ == "__main__":
    from receiver.MatchedFilter import RxMatchedFilter
    from receiver.sync.CoarseSync import CoarseSync
    from receiver.sync.FineSync import FineSync
    from receiver.CFOEstimator import CFOEstimator
    from receiver.Downsampler import Downsampler
    from channel.THzChannel import THzChannel
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("     OFDM Full RX Chain Test")
    print("=" * 55)

    # ---- TX (default params) ----
    params = PHYParams()
    params.update(link_mode="ofdm")
    tx = THzTransmitter(params)
    tx.run()

    sps = params.get("oversampling")
    Lh = params.get("gi_length")
    nfft = params.get("subframe_length")

    # ---- Channel ----
    ch = THzChannel(params)
    rx_dict = ch.run(tx.tx_signal_dict)
    cfo_true = ch.cfo.freq_offset if hasattr(ch, "cfo") else 0.0
    print(f"  CFO true = {cfo_true:.1f} Hz")

    # ---- 1. RxMatchedFilter ----
    mf = RxMatchedFilter(tx)
    mf_out = mf.matched_filter(rx_dict)
    print(f"[1] MF: len={mf_out['signal_length']}")

    # ---- 2. CoarseSync (SYNC correlation -> frame start) ----
    cs = CoarseSync(tx)
    sync_out = cs.detect_sync(mf_out)
    print(f"[2] CoarseSync: SYNC offset={cs.sync_offset}")

    # ---- 3. CFO coarse: per-frame (zero-pad, exclude padded frame from avg) ----
    cfo = CFOEstimator(tx)
    N_SUB = tx.ofdm_processer.frame_num  # valid frame count
    coarse_out = cfo.estimate_and_compensate_cfo_coarse(sync_out)
    # Exclude padded frame(s) from average (their SYNC is zeros -> bad estimate)
    cfo_coarse = np.mean(coarse_out["cfo_estimates_Hz"][:N_SUB])
    print(f"[3] CFO coarse (per-frame): avg={cfo_coarse:.1f} Hz "
          f"(valid frames only, total_frames={len(coarse_out['cfo_estimates_Hz'])})")

    # ---- 4. FineSync (SFD frame boundary) ----
    fsync = FineSync(tx)
    fine_out = fsync.fine_sync(coarse_out)
    print(f"[4] FineSync: SFD offset={fine_out['sync_offset']}")

    # ---- 5. Manual downsample from index 0 (CoarseSync already aligned) ----
    sym_raw = fine_out["signal_stream"][::sps]
    sym_fs = fine_out["sample_rate_Hz"] / sps
    fl_sym = len(tx.preamble) + 48 * (512 + 32)
    N_SUB = tx.ofdm_processer.frame_num
    sym = sym_raw[:N_SUB * fl_sym]
    sym_dict = {
        "signal_stream": sym,
        "sample_rate_Hz": sym_fs,
        "signal_length": len(sym),
        "duration_seconds": len(sym) / sym_fs,
        "padding_bit_num": fine_out["padding_bit_num"],
    }
    print(f"[5] Downsample (manual): sym_len={len(sym)} (={N_SUB}x{fl_sym})")

    # ---- 6. CFO fine: per-frame (exclude padded frames from avg) ----
    fine_cfo_out = cfo.estimate_and_compensate_cfo_fine(sym_dict)
    cfo_fine = np.mean(fine_cfo_out["cfo_estimates_Hz"][:N_SUB])
    cfo_total = cfo_coarse + cfo_fine
    rxd = fine_cfo_out
    print(f"[6] CFO fine (per-frame): avg={cfo_fine:.1f} Hz, total={cfo_total:.1f} Hz")

    # ---- 8. RxOFDMProcesser (内置导频LS+相位插值) ----
    rx_ofdm = RxOFDMProcesser(tx)
    ofdm_result = rx_ofdm.ofdm_demodulate(rxd, H_init_list=None)
    rx_sym = ofdm_result["signal_stream"]
    tx_sym = tx.modulated_data_dict["signal_stream"]
    cmp = min(len(tx_sym), len(rx_sym))
    evm = np.sqrt(np.mean(np.abs(rx_sym[:cmp] - tx_sym[:cmp])**2)) * 100
    print(f"[8] OFDM demod EVM={evm:.1f}%")

    H_est = ofdm_result["channel_freq_response"][0]
    h_est = ofdm_result["channel_time_response"][0]

    # ---- 9. True equivalent symbol-rate channel ----
    h_phys = np.zeros(Lh * sps, dtype=np.complex128)
    h_phys[:len(ch.chan_true)] = ch.chan_true
    pt, pr = tx.pulse_shaper.filter_coeffs, mf.h_rx
    heq = np.convolve(np.convolve(h_phys, pt), pr)
    gd = (len(pt)-1)//2 + (len(pr)-1)//2
    h_true = heq[gd::sps][:Lh]
    H_true = np.fft.fft(h_true, nfft)
    nmse = (np.linalg.norm(H_true-H_est)**2
            / (np.linalg.norm(H_true)**2 + np.finfo(float).eps))
    cfo_residual = abs(cfo_total - cfo_true)
    print(f"[9] NMSE={10*np.log10(nmse):.1f}dB  CFO_residual={cfo_residual:.0f}Hz")

    # ==================== Visualization ====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax0 = axes[0, 0]
    ax0.stem(np.abs(h_true), linefmt='b-', markerfmt='bo', basefmt=' ',
             label='|h_true|')
    ax0.stem(np.abs(h_est), linefmt='r--', markerfmt='rx', basefmt=' ',
             label=f'Pilot-LS ({10*np.log10(nmse):.1f}dB)')
    ax0.set_title('CIR'); ax0.set_xlabel('tap'); ax0.set_ylabel('|h|')
    ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)

    ax1 = axes[0, 1]
    ax1.plot(np.abs(H_true), 'b-', lw=1.2, alpha=0.8, label='|H_true|')
    ax1.plot(np.abs(H_est), 'r--', lw=0.8, alpha=0.8, label='|H_est|')
    ax1.set_title('CSI'); ax1.set_xlabel('subcarrier'); ax1.set_ylabel('|H|')
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 2]
    n = min(300, cmp)
    n = 30000
    ax2.plot(tx_sym[:n].real, tx_sym[:n].imag, 'b.', ms=4, alpha=0.5, label='TX')
    ax2.plot(rx_sym[:n].real, rx_sym[:n].imag, 'r.', ms=4, alpha=0.6,
             label=f'RX EVM={evm:.1f}%')
    ax2.set_title('Constellation'); ax2.axis('equal')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(np.abs(rx_sym[:cmp] - tx_sym[:cmp])[:300], 'r-', lw=0.5)
    ax3.set_title('|error| first 300 symbols'); ax3.set_xlabel('symbol')
    ax3.set_ylabel('|error|'); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if rx_ofdm.ofdm_freq_grid is not None:
        fg = rx_ofdm.ofdm_freq_grid
        n_show = min(48, fg.shape[1])
        dsp = 20*np.log10(np.abs(np.fft.fftshift(fg[:,:n_show], axes=0))+1e-15)
        im = ax4.imshow(dsp, aspect='auto', origin='lower', cmap='viridis',
                        extent=[0, n_show, -nfft//2, nfft//2-1])
        for pi in rx_ofdm.pilot_indexes:
            if pi < n_show:
                ax4.axvline(x=pi, color='red', linestyle='--', alpha=0.7, lw=1)
        ax4.set_title('RX OFDM spectrogram'); ax4.set_xlabel('OFDM symbol')
        ax4.set_ylabel('subcarrier'); plt.colorbar(im, ax=ax4, label='dB')

    ax5 = axes[1, 2]; ax5.axis('off')
    summary = (
        f"CFO true = {cfo_true:.0f} Hz\n"
        f"CFO est  = {cfo_total:.0f} Hz\n"
        f"residual = {cfo_residual:.0f} Hz\n\n"
        f"EVM = {evm:.1f}%\n\n"
        f"NMSE = {10*np.log10(nmse):.1f} dB"
    )
    ax5.text(0.15, 0.5, summary, transform=ax5.transAxes, fontsize=11,
             va='center', family='monospace')

    fig.suptitle('OFDM RX: MF->CoarseSync->CFOc->FineSync->DS->CFOf->ChEst->Demod',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("\nDone")
