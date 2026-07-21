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
    # # 初始化参数和发射机
    params = PHYParams()
    # params.apply_dict(
    # {
    #     "enable_iq_imbalance": True,
    #     "iq_imbalance_position": "rx",
    #     "iq_imbalance_model": "fd",
    #     "iq_gain_imbalance_db": 2.0,
    #     "iq_phase_imbalance_deg": 5.0,
    #     "iq_gI_taps": [1.0, 0.08, -0.03],
    #     "iq_gQ_taps": [1.0, -0.12, 0.04],
    #     "iq_power_normalize": False,
    # }
    # )    
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)

    num_points = 30000
    # 假设您有原始发射数据符号（未加GI和前导码）
    tx_data_symbols = transmitter.modulated_data_dict['signal_stream']   # 需要从发射机中获取
    tx_data_symbols_with_gi = transmitter.data_with_gi_dict['signal_stream']  # 包含GI的符号流
    tx_data_symbols_with_gi_samples = tx_data_symbols_with_gi[:num_points]  # 截取前2048个符号以匹配接收端截取的点数
    tx_data_symbols_samples = tx_data_symbols[:num_points]  # 截取前2048个符号以匹配接收端截取的点数
    

    # 接收端匹配滤波
    rx_matched_filter = RxMatchedFilter(transmitter)
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)
    # 细同步恢复符号
    fine_sync = FineSync(transmitter)
    rx_sampled_signal_dict = fine_sync.fine_sync(rx_matched_signal_dict)

    rx_symbol_with_gi = rx_sampled_signal_dict['signal_stream'][len(transmitter.preamble):]  # 去除Preamble
    rx_symbol_with_gi_samples = rx_symbol_with_gi[:num_points]  # 截取前2048个点以避免过密
    

    # 粗频偏估计与补偿
    cfo_estimator = CFOEstimator(transmitter)
    rx_signal_compensate_dict = cfo_estimator.estimate_and_compensate_cfo_coarse(rx_sampled_signal_dict) 
    print(f"估计频偏：{rx_signal_compensate_dict['cfo_estimates_Hz'][0]:.2f} Hz")
    rx_signal_compensate = rx_signal_compensate_dict['signal_stream']
    cfo_est_after = cfo_estimator.estimate_cfo_coarse(rx_signal_compensate, fs=rx_signal_dict['sample_rate_Hz'])
    print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # 下采样
    downsampler = Downsampler(transmitter)
    rx_downsampled_signal_dict = downsampler.recover_symbol(rx_signal_compensate_dict)

    # 细频偏估计与补偿
    rx_signal_compensate_dict = cfo_estimator.estimate_and_compensate_cfo_fine(rx_downsampled_signal_dict)
    print(f"细频偏估计：{rx_signal_compensate_dict['cfo_estimates_Hz'][0]:.2f} Hz")
    rx_signal_compensate = rx_signal_compensate_dict['signal_stream']
    cfo_est_after = cfo_estimator.estimate_cfo_fine(rx_signal_compensate, fs=rx_downsampled_signal_dict['sample_rate_Hz'])
    print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # 初始化信道估计器
    channel_estimator = ChannelEstimator(transmitter)
    # ========== 执行CES互相关信道估计（对齐MATLAB逻辑） ==========
    rx_estimated_dict = channel_estimator.channel_estimate(rx_signal_compensate_dict)
    H_est_list = rx_estimated_dict["channel_freq_response"]
    h_est_list = rx_estimated_dict["channel_time_response"]
    fine_offset_list = rx_estimated_dict["fine_offset"]
    H_est = H_est_list[0]
    h_est = h_est_list[0]
    fine_offset = fine_offset_list[0]
    # print(f"估计的细同步修正量（采样点数）：{fine_offset}")
    oversampling = params.get("oversampling")  # 过采样率
    # 获取真实信道参数（用于对比）
    nfft = params.get("subframe_length")  # 与ChannelEstimator默认nfft一致
    h_true = np.zeros((params.get("gi_length") * oversampling,), dtype=np.complex128)
    h_true[:len(channel.chan_true)] = channel.chan_true
    #  等效基带处理：h_cont 与成型滤波器及匹配滤波器卷积
    #  假设发射端成型滤波器为 p，接收端匹配滤波为 p_rev = p[::-1]
    p = transmitter.pulse_shaper.filter_coeffs   # 采样率下的脉冲响应
    p_mf = np.conj(p[::-1])                      # 匹配滤波器
    h_eq_sampled = np.convolve(h_true, p, mode='full')
    h_eq_sampled = np.convolve(h_eq_sampled, p_mf, mode='full')
    group_delay = (len(p) - 1) // 2   # 假设线性相位
    start_idx = group_delay * 2       # 卷积后总延迟补偿
    h_true_symbol = h_eq_sampled[start_idx :: oversampling]   # 每隔 L 个采样点取一个
    Lh = params.get("gi_length")
    h_true_symbol = h_true_symbol[:Lh]
    H_true_symbol = np.fft.fftshift(np.fft.fft(h_true_symbol, nfft))

    rx_symbol_with_gi = rx_signal_compensate[len(transmitter.preamble) + fine_offset:]  # 去除Preamble
    rx_symbol_with_gi_samples = rx_symbol_with_gi[:num_points]  # 截取前2048个点以避免过密

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(rx_symbol_with_gi_samples.real, rx_symbol_with_gi_samples.imag, '.')
    axes[0].set_title("匹配滤波+下采样+频偏补偿后星座图")
    axes[0].set_xlabel("实部")
    axes[0].set_ylabel("虚部")
    axes[1].plot(tx_data_symbols_with_gi_samples.real, tx_data_symbols_with_gi_samples.imag, 'x')
    axes[1].set_title(f'原始数据符号星座图')
    axes[1].axis('equal')
    plt.grid()
    plt.show()    

    # 绘制频偏补偿后的时域波形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(tx_data_symbols_with_gi_samples[:100])
    axes[0].set_title("发射端原始符号（前100点）")
    axes[0].set_xlabel("符号索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(rx_symbol_with_gi_samples[:100], color='orange')
    axes[1].set_title("频偏补偿后的符号（前100点）")
    axes[1].set_xlabel("符号索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    # 估计噪声方差
    noise_estimator = NoiseEstimator(transmitter)
    result_dict = noise_estimator.noise_estimate(rx_estimated_dict)
    noise_var_est_list = result_dict['noise_var']
    noise_var_est = noise_var_est_list[0]
    print(f"估计噪声方差：{noise_var_est:.6f}")


    # 初始化均衡器
    equalizer = FreqDomainEqualizer(transmitter)
    # 进行均衡
    equalized_dict = equalizer.equalize(result_dict)

    equalized_signal = equalized_dict["signal_stream"]

    fig, axes = plt.subplots(figsize=(12, 5))
    axes.stem(np.abs(h_true_symbol), linefmt='b-', basefmt='b-', label='真实信道 |h_true|', markerfmt='bo')
    axes.stem(np.abs(h_est), linefmt='r-', basefmt='r-', label='信道估计 |h_est|')
    axes.set_title("信道估计")
    axes.set_xlabel("子载波索引")
    axes.set_ylabel("幅度")
    axes.legend()
    plt.tight_layout()
    plt.show()

    # 绘制星座图
    fig, axes = plt.subplots(1, 2 ,figsize=(12, 5))
    axes[0].plot(tx_data_symbols_samples.real, tx_data_symbols_samples.imag, 'x')
    axes[0].set_title("原始数据符号星座图")

    equalized_signal_samples = equalized_signal[:num_points]  # 截取前2048个点以避免过密
    axes[1].plot(equalized_signal_samples.real, equalized_signal_samples.imag, '.')
    axes[1].set_title(f'均衡后星座图')
    axes[1].axis('equal')
    plt.grid()
    plt.show()

    # 绘制均衡后信号的时域波形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(tx_data_symbols_samples)
    axes[0].set_title("发射端原始符号")
    axes[0].set_xlabel("符号索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(equalized_signal, color='orange')
    axes[1].set_title("均衡后的符号")
    axes[1].set_xlabel("符号索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    

