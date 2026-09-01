
import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.CFOEstimator import CFOEstimator
from receiver.sync.FineSync import FineSync
from receiver.ChannelEstimator import ChannelEstimator
from receiver.MatchedFilter import RxMatchedFilter
import matplotlib.pyplot as plt
from receiver.Downsampler import Downsampler



# class NoiseEstimator:
#     """
#     噪声方差估计器：利用SYNC序列的重复块结构，通过块间差分消去信号，
#     仅保留噪声，实现无偏噪声方差估计（适用于复/实高斯白噪声）。
#     """
#     def __init__(self, transmitter):
#         self.oversampling = transmitter.params.get("oversampling")
#         self.base_sequences_length = len(transmitter.preamble_gen.a128)  # 每个块的长度（过采样后）
#         self.sync_len = len(transmitter.sync)  # SYNC序列总长度
    
#         # 输出参数    
#         self.noise_var = None  # 噪声方差


#     def _verification_data(self, data_dict):
#         """
#         校验输入数据字典合法性
#         :param data_dict: 输入数据字典，包含以下键值：
#             "signal_stream": 信号流,
#             "sample_rate_Hz": 采样率,
#             "duration_seconds": 时长,
#             "signal_length": 信号长度,
#             "padding_bit_num": 补零比特数,
#         """
#         # 校验输入字典完整性
#         required_keys = [
#             "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num"
#         ]
#         for key in required_keys:
#             if key not in data_dict:
#                 raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
#         # 校验分帧信息一致性
#         if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
#             raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
#         self.sample_rate = data_dict["sample_rate_Hz"]
#         self.signal_length = data_dict["signal_length"]
#         self.duration = data_dict["duration_seconds"]
#         self.padding_bit_num = data_dict["padding_bit_num"]       


#     def estimate_noise_var(self, rx_sync_seq):
#         """
#         估计噪声方差（仅输入接收的SYNC序列）
#         :param rx_sync_seq: 同步后的接收SYNC序列（一维数组）
#         :return: 噪声方差估计值（无偏），若块数不足则返回NaN
#         """
        
#         # 计算SYNC序列中的完整块数
#         Nblks = np.floor(len(rx_sync_seq) / self.base_sequences_length).astype(int)
#         if Nblks < 2:
#             # 至少需要2个块才能做差分，无法估计噪声方差
#             print("Warning: Not enough blocks (Nblks < 2) for noise estimation, returning NaN.")
#             return np.nan

#         # 截取完整块的部分，重塑为 [块长度 × 块数] 矩阵
#         rx_sync_valid = rx_sync_seq[:self.base_sequences_length * Nblks]
#         sync_blk = np.reshape(rx_sync_valid, (self.base_sequences_length, Nblks), order='F')

#         # 按列差分：相邻块相减，消去信号
#         diff_blk = np.diff(sync_blk, axis=1)

#         # 差分后的总功率
#         total_power = np.sum(np.abs(diff_blk) ** 2)

#         # 无偏归一化：差分方差 = 2*σ²（无论实/复信号）
#         norm_factor = self.base_sequences_length * (Nblks - 1) * 2

#         # 计算噪声方差
#         noise_var = total_power / norm_factor
#         return noise_var
    
#     def noise_estimate(self, signal_dict):
#         """直接输入信号字典，提取SYNC段进行噪声估计"""
#         self._verification_data(signal_dict)
#         rx_signal = signal_dict['signal_stream']
#         rx_sync_seq = rx_signal[:self.sync_len]  # 提取SYNC段
#         self.noise_var = self.estimate_noise_var(rx_sync_seq)
#         result_dict = signal_dict.copy()
#         result_dict['noise_var'] = self.noise_var
#         return result_dict

class NoiseEstimator:
    """
    噪声方差估计器（支持多帧）
    利用每帧开头的 SYNC 序列（由多个重复块组成）进行无偏噪声方差估计。
    模式：
      - 'global'    : 所有帧的 SYNC 拼接后估计一个方差（默认）
      - 'per_frame' : 每帧独立估计，返回列表
    """
    def __init__(self, transmitter, estimation_mode='per_frame'):
        self.params = transmitter.params
        self.link_mode = self.params.get("link_mode").lower()
        self.oversampling = transmitter.params.get("oversampling")
        # OFDM 补零 IFFT 后信号全程处于高采样率域，SYNC 块长与帧长 ×sps；
        # 估计公式不变（块长与总功率同步 ×sps，归一化因子自动正确）。
        sps = (
            int(self.oversampling)
            if self.link_mode == "ofdm" and self.oversampling
            else 1
        )
        self.base_sequences_length = len(transmitter.preamble_gen.a128) * sps  # 128×sps
        self.sync_len = len(transmitter.sync) * sps  # SYNC 总样点数
        self.estimation_mode = estimation_mode  # 'global' 或 'per_frame'

        # 计算帧长度 — SC-FDE / OFDM 自适应
        gi_len = self.params.get("gi_length")
        preamble_len = len(transmitter.preamble)
        if self.link_mode == "ofdm":
            block_len = self.params.get("subwave_num")
            subframe_count = self.params.get("subframe_ofdm_num")
        else:
            block_len = self.params.get("subframe_length")
            subframe_count = self.params.get("subframe_num")
        self.frame_symbol_num = (
            (block_len + gi_len) * subframe_count + preamble_len
        ) * sps

        # 输出缓存
        self.noise_var = None   # 若 global，则为标量；若 per_frame，则为列表
        self.sample_rate = None
        self.signal_length = None
        self.duration = None
        self.padding_bit_num = 0

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

    def estimate_noise_var(self, rx_sync_seq):
        """
        核心估计函数：利用重复块差分估计噪声方差（单段 SYNC）
        :param rx_sync_seq: 接收的 SYNC 序列（一维数组，符号级）
        :return: 噪声方差估计值，若块数不足则返回 NaN
        """
        Nblks = len(rx_sync_seq) // self.base_sequences_length
        if Nblks < 2:
            return np.nan
        # 截取完整块并重塑为列矩阵（每列一个块）
        rx_sync_valid = rx_sync_seq[:self.base_sequences_length * Nblks]
        sync_blk = np.reshape(rx_sync_valid, (self.base_sequences_length, Nblks), order='F')
        # 相邻块差分
        diff_blk = np.diff(sync_blk, axis=1)
        if self.link_mode == "ofdm" and diff_blk.shape[1] >= 3:
            # OFDM 补零 IFFT 后前导码经 resample_poly 插值：首块含滤波器
            # 启动瞬态、末块被后续 SFD（-a128）的滤波器拖尾污染，破坏
            # “相邻块完全相同”的模型假设，丢弃首尾两个差分列。
            diff_blk = diff_blk[:, 1:-1]
        total_power = np.sum(np.abs(diff_blk) ** 2)
        # 无偏归一化：差分方差 = 2*σ²
        norm_factor = self.base_sequences_length * diff_blk.shape[1] * 2
        noise_var = total_power / norm_factor
        return noise_var

    def noise_estimate(self, signal_dict):
        """
        多帧噪声估计入口
        :param signal_dict: 输入信号字典（符号级，包含完整帧）
        :return: 更新后的字典，添加 'noise_var' 字段
        """
        self._verification_data(signal_dict)
        rx_signal = signal_dict["signal_stream"]

        # 分割帧（信号不足一帧时直接返回默认值）
        if len(rx_signal) < self.frame_symbol_num:
            self.noise_var = 0.01  # fallback
            return {"noise_var": self.noise_var}
        frames, num_frames, _ = self._split_into_frames(rx_signal, self.frame_symbol_num)
        if num_frames == 0:
            self.noise_var = 0.01
            return {"noise_var": self.noise_var}

        if self.estimation_mode == 'global':
            # 拼接所有帧的 SYNC
            all_sync = np.concatenate([frame[:self.sync_len] for frame in frames])
            noise_var = self.estimate_noise_var(all_sync)
            self.noise_var = noise_var
            result_dict = signal_dict.copy()
            result_dict['noise_var'] = noise_var
        else:  # per_frame
            noise_list = []
            for frame in frames:
                sync_seq = frame[:self.sync_len]
                nv = self.estimate_noise_var(sync_seq)
                noise_list.append(nv)
            self.noise_var = noise_list
            result_dict = signal_dict.copy()
            result_dict['noise_var'] = noise_list

        # 保留其他字段
        result_dict.update({
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
        })
        return result_dict

    # def estimate_noise_var_with_gi(self, rx_sync_seq, skip_first=True, skip_last=True):
    #     """
    #     进阶版：支持跳过第一个块（GI）和最后一个块
    #     :param rx_sync_seq: 接收的SYNC序列
    #     :param skip_first: 是否跳过第一个块（作为GI）
    #     :param skip_last: 是否跳过最后一个块
    #     :return: 噪声方差估计值
    #     """
    #     Nblks = np.floor(len(rx_sync_seq) / self.base_sequences_length).astype(int)
    #     if Nblks < 3:  # 跳过GI后至少需要2个有效块
    #         print("Warning: Not enough valid blocks after skipping GI, using all blocks.")
    #         return self.estimate_noise_var(rx_sync_seq)

    #     # 截取有效块（跳过第一个/最后一个）
    #     start_idx = 1 if skip_first else 0
    #     end_idx = Nblks - 1 if skip_last else Nblks
    #     rx_sync_valid = rx_sync_seq[self.base_sequences_length * start_idx : self.base_sequences_length * end_idx]

    #     # 复用基础估计逻辑
    #     return self.estimate_noise_var(rx_sync_valid)

def calculate_snr(rx_sync_seq, noise_var):
    """
    计算信噪比（SNR）：基于SYNC序列的总功率和估计的噪声方差
    :param rx_sync_seq: 同步后的接收SYNC序列
    :param noise_var: 估计的噪声方差
    :return: snr_linear（线性域SNR）, snr_dB（分贝域SNR）
    """
    # 步骤1：计算SYNC序列的总功率（信号功率 + 噪声功率）
    total_power = np.mean(np.abs(rx_sync_seq) ** 2)
    
    # 步骤2：计算纯信号功率（总功率 - 噪声功率）
    # 注：由于噪声方差=噪声功率（平稳噪声），因此直接相减即可
    signal_power = total_power - noise_var
    
    # 防止信号功率为负（极端低信噪比场景）
    signal_power = max(signal_power, 1e-10)
    
    # 步骤3：计算线性域和分贝域SNR
    snr_linear = signal_power / noise_var
    snr_dB = 10 * np.log10(snr_linear)
    
    return snr_linear, snr_dB

# 测试
if __name__ == "__main__":
    # # # 初始化参数和发射机
    # params = PHYParams()
    # transmitter = THzTransmitter(params)    
    # # 执行完整发射流程  
    # tx_signal_dict = transmitter.run() 
    # # 初始化信道并生成接收信号
    # channel = THzChannel(params)
    # rx_signal_dict = channel.run(tx_signal_dict)

    # # 接收端匹配滤波
    # rx_matched_filter = RxMatchedFilter(transmitter)
    # rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)
    # # 细同步恢复符号
    # fine_sync = FineSync(transmitter)
    # rx_sampled_signal_dict = fine_sync.recover_symbol(rx_matched_signal_dict)
    # # 频偏估计与补偿
    # cfo_estimator = CFOEstimator(transmitter)
    # rx_signal_compensate_dict = cfo_estimator.estimate_and_compensate_cfo(rx_sampled_signal_dict) 
    # print(f"未匹配滤波估计频偏：{rx_signal_compensate_dict['estimated_cfo_Hz']:.2f} Hz，真实频偏：{channel.cfo.freq_offset} Hz，估计误差：{abs(rx_signal_compensate_dict['estimated_cfo_Hz'] - channel.cfo.freq_offset):.2f} Hz")

    # rx_signal_compensate = rx_signal_compensate_dict['signal_stream']
    # # 估计补偿后频偏
    # cfo_est_after = cfo_estimator.estimate_cfo(rx_signal_compensate, fs=rx_signal_dict['sample_rate_Hz'])
    # print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # # 初始化信道估计器
    # channel_estimator = ChannelEstimator(transmitter)
    # # ========== 执行CES互相关信道估计（对齐MATLAB逻辑） ==========
    # rx_estimated_dict = channel_estimator.channel_estimate(rx_signal_compensate_dict)
    # H_est = rx_estimated_dict["channel_freq_response"]
    # h_est = rx_estimated_dict["channel_time_response"]
    # fine_offset = rx_estimated_dict["fine_offset"]
    # print(f"估计的细同步修正量（采样点数）：{fine_offset}")
    # oversampling = params.get("oversampling")  # 过采样率
    # # 获取真实信道参数（用于对比）
    # nfft = params.get("subframe_length")  # 与ChannelEstimator默认nfft一致
    # h_true = np.zeros((params.get("gi_length") * oversampling,), dtype=np.complex128)
    # h_true[:len(channel.chan_true)] = channel.chan_true
    # #  等效基带处理：h_cont 与成型滤波器及匹配滤波器卷积
    # #  假设发射端成型滤波器为 p，接收端匹配滤波为 p_rev = p[::-1]
    # p = transmitter.pulse_shaper.filter_coeffs   # 采样率下的脉冲响应
    # p_mf = np.conj(p[::-1])                      # 匹配滤波器
    # h_eq_sampled = np.convolve(h_true, p, mode='full')
    # h_eq_sampled = np.convolve(h_eq_sampled, p_mf, mode='full')
    # group_delay = (len(p) - 1) // 2   # 假设线性相位
    # start_idx = group_delay * 2       # 卷积后总延迟补偿
    # h_true_symbol = h_eq_sampled[start_idx :: oversampling]   # 每隔 L 个采样点取一个
    # Lh = params.get("gi_length")
    # h_true_symbol = h_true_symbol[:Lh]
    # H_true_symbol = np.fft.fftshift(np.fft.fft(h_true_symbol, nfft))

    # num_points = min(2048, len(rx_signal_compensate))
    # rx_symbol_with_gi = rx_signal_compensate[len(transmitter.preamble) + fine_offset:]  # 去除Preamble
    # rx_symbol_with_gi_samples = rx_symbol_with_gi[:num_points]  # 截取前2048个点以避免过密
    # # 假设您有原始发射数据符号（未加GI和前导码）
    # tx_data_symbols = transmitter.modulated_data_dict['signal_stream']   # 需要从发射机中获取
    # tx_data_symbols_samples = tx_data_symbols[:num_points]  # 截取前2048个符号以匹配接收端截取的点数

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # axes[0].plot(rx_symbol_with_gi_samples.real, rx_symbol_with_gi_samples.imag, '.')
    # axes[0].set_title("匹配滤波+下采样+频偏补偿后星座图")
    # axes[0].set_xlabel("实部")
    # axes[0].set_ylabel("虚部")
    # axes[1].plot(tx_data_symbols_samples.real, tx_data_symbols_samples.imag, 'x')
    # axes[1].set_title(f'原始数据符号星座图')
    # axes[1].axis('equal')
    # plt.grid()
    # plt.show()    
    # # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    print(f"发射信号长度：{len(tx_signal_dict['signal_stream'])}, 接收信号长度：{len(rx_signal_dict['signal_stream'])}")
    print(f"采样率：{rx_signal_dict['sample_rate_Hz']} Hz，时长：{rx_signal_dict['duration_seconds']} 秒")

    # 接收端匹配滤波
    rx_matched_filter = RxMatchedFilter(transmitter)
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)
    # 细同步恢复符号
    fine_sync = FineSync(transmitter)
    rx_sampled_signal_dict = fine_sync.fine_sync(rx_matched_signal_dict)
    print(f"同步偏移：{rx_sampled_signal_dict['sync_offset']}") 
    print(f"同步信号长度：{len(rx_sampled_signal_dict['signal_stream'])}")
    # 粗频偏估计与补偿
    cfo_estimator = CFOEstimator(transmitter)
    result_dict = cfo_estimator.estimate_and_compensate_cfo_coarse(rx_sampled_signal_dict) 
    print(f"估计频偏：{result_dict['cfo_estimates_Hz'][0]:.2f} Hz，真实频偏：{channel.cfo.freq_offset} Hz，估计误差：{abs(result_dict['cfo_estimates_Hz'][0] - channel.cfo.freq_offset):.2f} Hz")

    rx_signal_compensate = result_dict['signal_stream']
    # 估计补偿后频偏
    cfo_est_after = cfo_estimator.estimate_cfo_coarse(rx_signal_compensate, fs=rx_matched_signal_dict['sample_rate_Hz'])
    print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # 下采样
    downsampler = Downsampler(transmitter)
    rx_downsampled_signal_dict = downsampler.recover_symbol(result_dict)
    print(f"下采样后信号长度：{len(rx_downsampled_signal_dict['signal_stream'])}")

    # 细频偏估计与补偿
    rx_signal_compensate_dict = cfo_estimator.estimate_and_compensate_cfo_fine(rx_downsampled_signal_dict)
    print(f"细频偏估计：{rx_signal_compensate_dict['cfo_estimates_Hz'][0]:.2f} Hz")
    rx_signal_compensate = rx_signal_compensate_dict['signal_stream']
    # 估计补偿后频偏
    cfo_est_after = cfo_estimator.estimate_cfo_fine(rx_signal_compensate, fs=rx_downsampled_signal_dict['sample_rate_Hz'])
    print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # 初始化信道估计器
    channel_estimator = ChannelEstimator(transmitter)
    # ========== 执行CES互相关信道估计（对齐MATLAB逻辑） ==========
    rx_estimated_dict = channel_estimator.channel_estimate(rx_signal_compensate_dict)    

    # 估计噪声方差
    noise_estimator = NoiseEstimator(transmitter)
    result_dict = noise_estimator.noise_estimate(rx_estimated_dict)
    noise_var_est_list = result_dict['noise_var']
    noise_var_est = noise_var_est_list[5]
    print(f"估计噪声方差：{noise_var_est:.6f}")

    # 计算并输出信噪比
    snr_linear, snr_dB = calculate_snr(rx_estimated_dict['signal_stream'], noise_var_est)
    print(f"设定信噪比（SNR）：{params.get('SNRdB'):.2f} dB")
    print(f"估计信噪比（SNR）：{snr_dB:.2f} dB")