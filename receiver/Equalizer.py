import numpy as np
import math
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.CFOEstimator import CFOEstimator
from receiver.sync.FineSync import FineSync
from receiver.NoiseEstimator import NoiseEstimator
from receiver.ChannelEstimator import ChannelEstimator
from receiver.MatchedFilter import RxMatchedFilter
from receiver.Downsampler import Downsampler
class FreqDomainEqualizer:
    """
    单载波频域均衡器（支持多帧，ZF/MMSE）
    """
    def __init__(self, transmitter):
        self.params = transmitter.params
        self.subframe_length = self.params.get("subframe_length")
        self.gi_length = self.params.get("gi_length")
        self.gi_type = self.params.get("gi_type")
        self.method = self.params.get("equalizer_method")
        if self.method not in ('zf', 'mmse'):
            raise ValueError("method 必须为 'zf' 或 'mmse'")
        self.preamble_len = len(transmitter.preamble)  # 前导码长度（SYNC+SFD+CES）

        # 计算帧长度（符号级，与前保持一致）
        subframe_len = self.params.get("subframe_length")
        gi_len = self.params.get("gi_length")
        preamble_len = len(transmitter.preamble)
        self.frame_symbol_num = (subframe_len + gi_len) * self.params.get("subframe_num") + preamble_len

        # 输出参数
        self.sample_rate = None
        self.duration = None
        self.symbol_length = None
        self.padding_bit_num = 0

    def _verification_data(self, data_dict):
        """简化校验，仅检查必要键值"""
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num", "channel_freq_response"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
        # MMSE需要噪声方差
        if self.method == 'mmse' and "noise_var" not in data_dict:
            raise KeyError("MMSE 均衡需要 noise_var")
        # 暂不检查信号长度一致性，交给后续处理
        self.sample_rate = data_dict["sample_rate_Hz"]
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

    def _equalize_frame(self, frame_data, H, N0=None):
        """
        对单帧数据（已跳过前导码的数据部分）进行GI移除和均衡
        :param frame_data: 1D数组，包含GI+数据
        :param H: 频域信道响应 (fftshift后)
        :param N0: 噪声方差 (MMSE)
        :return: 均衡后的数据 (1D)
        """
        gi_length = self.gi_length
        subframe_length = self.subframe_length

        # 1. 移除GI
        if self.gi_type == "cp":
            # CP模式：每个块 = GI + data
            block_len = subframe_length + gi_length
            # 校验长度
            if len(frame_data) % block_len != 0:
                # 截断或补零？此处截断（与之前一致）
                num_blocks = len(frame_data) // block_len
                frame_data = frame_data[:num_blocks * block_len]
            # 重塑为 行=block_len, 列=num_blocks
            data_blocks = frame_data.reshape(-1, block_len).T  # 转置后行=block_len，列=num_blocks
            data_blocks = data_blocks[gi_length:, :]  # 移除CP
        elif self.gi_type == "golay":
            # Golay模式：每个块前有Golay序列，末尾有Golay序列
            # 这里按标准做法：先移除末尾的GI，再移除块前的GI
            frame_data = frame_data[:-gi_length]  # 移除末尾GI
            block_len = subframe_length + gi_length
            if len(frame_data) % block_len != 0:
                num_blocks = len(frame_data) // block_len
                frame_data = frame_data[:num_blocks * block_len]
            data_blocks = frame_data.reshape(-1, block_len).T
            data_blocks = data_blocks[gi_length:, :]
        else:
            raise ValueError(f"不支持的GI类型: {self.gi_type}")

        # 2. FFT + fftshift
        Y = np.fft.fftshift(np.fft.fft(data_blocks, axis=0), axes=0)

        # 3. 计算均衡权重
        if self.method == 'zf':
            eps = 1e-12
            W = 1.0 / (H + eps)
        else:  # mmse
            if N0 is None:
                raise ValueError("MMSE需要提供噪声方差")
            den = np.abs(H) ** 2 + N0
            W = np.conj(H) / den

        # 4. 均衡
        EQ = Y * W[:, np.newaxis]

        # 5. IFFT + ifftshift
        yt_blocks = np.fft.ifft(np.fft.ifftshift(EQ, axes=0), axis=0)

        # 6. 按列拉平（先列后行）
        y = yt_blocks.ravel(order='F')
        return y

    def equalize(self, data_dict):
        """
        多帧频域均衡入口
        """
        self._verification_data(data_dict)
        rx_signal = data_dict["signal_stream"]
        fs = self.sample_rate

        # 获取信道响应和噪声方差（支持列表或单值）
        H_in = data_dict["channel_freq_response"]
        N0_in = data_dict.get("noise_var", None)

        # 分割帧
        frames, num_frames, _ = self._split_into_frames(rx_signal, self.frame_symbol_num)
        if num_frames == 0:
            raise ValueError("无有效帧")

        # 检查H和N0是否与帧数匹配
        if isinstance(H_in, list):
            if len(H_in) != num_frames:
                raise ValueError(f"信道响应列表长度 ({len(H_in)}) 与帧数 ({num_frames}) 不匹配")
            H_list = H_in
        else:
            H_list = [H_in] * num_frames

        if N0_in is None:
            N0_list = [None] * num_frames
        elif isinstance(N0_in, list):
            if len(N0_in) != num_frames:
                raise ValueError(f"噪声方差列表长度 ({len(N0_in)}) 与帧数 ({num_frames}) 不匹配")
            N0_list = N0_in
        else:
            N0_list = [N0_in] * num_frames

        # 逐帧均衡
        equalized_frames = []
        total_symbol_len = 0
        for i, frame in enumerate(frames):
            # 提取数据部分（跳过前导码）
            data_part = frame[self.preamble_len:]
            if len(data_part) == 0:
                print(f"警告：第 {i+1} 帧无数据部分（前导码可能超出帧长度），跳过")
                continue
            H_i = H_list[i]
            N0_i = N0_list[i]
            eq_data = self._equalize_frame(data_part, H_i, N0_i)
            equalized_frames.append(eq_data)
            total_symbol_len += len(eq_data)

        if len(equalized_frames) == 0:
            raise ValueError("没有成功均衡任何帧")

        # 拼接所有均衡数据
        y = np.concatenate(equalized_frames)

        # 更新时长和长度
        self.symbol_length = total_symbol_len
        self.duration = total_symbol_len / fs

        result_dict = {
            "signal_stream": y,
            "sample_rate_Hz": fs,
            "duration_seconds": self.duration,
            "signal_length": total_symbol_len,
            "padding_bit_num": self.padding_bit_num,
        }
        return result_dict
    
if __name__ == "__main__":
    from receiver.sync.CoarseSync import CoarseSync
    from receiver.IQCompensator import DecisionDirectedIQCompensator
    from receiver.DeModulator import THzDemodulator
    import os
    import time

    out_dir = "simulation_results"
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "STIX"],
        "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
    })

    # ---- RX chain following THzReceiver SC-FDE flow ----
    for tag, iq_imbalance, iq_comp in [("no_iq", False, False),
                                         ("iq_no_comp", True, False),
                                         ("iq_comp", True, True)]:
        scenario_start = time.perf_counter()
        print(f"\n--- {tag} ---", flush=True)
        params = PHYParams()
        test_subframe_num = 15
        one_frame_sample_length = int(
            params.get("subframe_length") * test_subframe_num
            * params.get("rs_packet_size")
            / (params.get("rs_packet_size") + params.get("rs_nsym"))
        )
        params.update(
            enable_awgn=True,
            SNRdB=24,
            enable_cfo=False,
            enable_phase_noise=False,
            enable_multipath=False,
            link_mode="sc-fde",
            code_type="RS",
            Preamble_type="short",
            subframe_num=test_subframe_num,
            duration=None,
            sample_length=one_frame_sample_length,
            random_seed=20260722,
            seed_strategy="固定种子",
            enable_iq_imbalance=iq_imbalance,
            enable_iq_compensation=iq_comp,
            iq_compensation_method="decision_directed",
        )
        if iq_imbalance:
            params.update(
                iq_imbalance_position="rx",
                iq_imbalance_model="fd",
                rx_iq_gain_imbalance_db=2.0,
                rx_iq_phase_imbalance_deg=5.0,
                rx_iq_gI_taps=[1.0, 0.08, -0.03],
                rx_iq_gQ_taps=[1.0, -0.12, 0.04],
            )
        tx = THzTransmitter(params); tx.run()
        print(f"  TX complete: {time.perf_counter() - scenario_start:.2f}s", flush=True)
        np.random.seed(20260722)
        ch = THzChannel(params); rx = ch.run(tx.tx_signal_dict)
        print(f"  Channel complete: {time.perf_counter() - scenario_start:.2f}s", flush=True)

        # ① MF → ② CoarseSync
        mf = RxMatchedFilter(tx); sig = mf.matched_filter(rx)
        sig = CoarseSync(tx).detect_sync(sig)
        print(f"  Sync complete: {time.perf_counter() - scenario_start:.2f}s", flush=True)
        cfo = CFOEstimator(tx)

        # ③ CFO coarse → ④ FineSync
        mf_fs = sig["sample_rate_Hz"]
        cfo_c = cfo.estimate_cfo_coarse(sig["signal_stream"], fs=mf_fs)
        sig_c = cfo.compensate_cfo(sig["signal_stream"], fs=mf_fs, cfo_est=cfo_c)
        sig = dict(sig); sig["signal_stream"] = sig_c; sig["signal_length"] = len(sig_c)
        sig = FineSync(tx).fine_sync(sig)

        # ⑤ Downsample
        ds = Downsampler(tx); sig = ds.recover_symbol(sig)

        # ⑥ CFO fine
        sym_fs = sig["sample_rate_Hz"]
        cfo_f = cfo.estimate_cfo_fine(sig["signal_stream"], fs=sym_fs)
        sig_f = cfo.compensate_cfo(sig["signal_stream"], fs=sym_fs, cfo_est=cfo_f)
        sig = dict(sig); sig["signal_stream"] = sig_f; sig["signal_length"] = len(sig_f)

        # ⑦ Noise est → ⑧ Channel est → ⑨ Equalize
        sig = NoiseEstimator(tx).noise_estimate(sig)
        noise_var = sig["noise_var"]
        sig = ChannelEstimator(tx).channel_estimate(sig)
        sig = FreqDomainEqualizer(tx).equalize(sig)
        print(f"  Equalization complete: {time.perf_counter() - scenario_start:.2f}s", flush=True)

        # ⑩ Decision-directed IQ compensation (only when enabled)
        if iq_comp:
            iq_c = DecisionDirectedIQCompensator(
                params, filter_len=5, ridge_lambda=0.0, iterations=2
            )
            sig = iq_c.compensate(sig)

        # plot constellation (skip preamble)
        rx_syms = sig["signal_stream"]
        tx_syms = tx.modulated_data_dict["signal_stream"]
        cmp = min(len(tx_syms), len(rx_syms))

        # Metrics use the same aligned TX/RX symbol and coded-bit ranges.
        tx_cmp = tx_syms[:cmp]
        rx_cmp = rx_syms[:cmp]
        evm = np.sqrt(
            np.mean(np.abs(rx_cmp - tx_cmp) ** 2)
            / np.mean(np.abs(tx_cmp) ** 2)
        )
        image_design = np.column_stack((tx_cmp, np.conj(tx_cmp)))
        direct_gain, image_gain = np.linalg.lstsq(
            image_design, rx_cmp, rcond=None
        )[0]
        irr_db = 10 * np.log10(
            (np.abs(direct_gain) ** 2 + 1e-15)
            / (np.abs(image_gain) ** 2 + 1e-15)
        )
        noise_scalar = float(np.mean(noise_var)) if isinstance(noise_var, list) \
            else float(noise_var)
        sigma = np.sqrt(max(noise_scalar, 1e-15) / 2)
        hard_bits = THzDemodulator(params).llr_to_bits(
            THzDemodulator(params).demodulate(sig, sigma)
        )["signal_stream"]
        tx_bits = tx.coded_bits_dict["signal_stream"]
        bit_cmp = min(len(tx_bits), len(hard_bits))
        coded_ber = np.mean(tx_bits[:bit_cmp] != hard_bits[:bit_cmp])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(tx_syms[:cmp:10].real, tx_syms[:cmp:10].imag, ".", ms=2, alpha=0.3, label="TX")
        ax.plot(rx_syms[:cmp:10].real, rx_syms[:cmp:10].imag, ".", ms=2, alpha=0.7, label="RX")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_title(f"SC-FDE — {tag}", fontsize=11, pad=6)
        ax.axis("equal"); ax.legend(fontsize=8, markerscale=3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/eq_{tag}.pdf", dpi=600)
        fig.savefig(f"{out_dir}/eq_{tag}.png", dpi=300)
        plt.close(fig)
        diagnostics = sig.get("iq_dd_diagnostics", [])
        condition_text = ""
        if diagnostics:
            condition_text = (
                f", cond(design)={diagnostics[-1]['design_condition_number']:.3e}, "
                f"cond(post)={diagnostics[-1]['post_condition_number']:.3e}"
            )
        print(
            f"{tag}: EVM={100 * evm:.2f}%, IRR={irr_db:.2f} dB, "
            f"coded BER={coded_ber:.3e}{condition_text}; saved"
        )

    print(f"\nFigures saved to {out_dir}/")
