import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
import matplotlib.pyplot as plt

class CoarseSync:
    """
    粗同步：对齐MATLAB Frame_detect逻辑
    基于重复SYNC序列的自相关检测，实现帧起始定位
    支持short/long两种SYNC类型，分块处理、功率归一化、模式匹配筛选
    """
    def __init__(self, tx_preamble, sync_threshold=0.1, scaling_factor=5):
        # 核心参数（对齐MATLAB）
        self.base_sequences_length = len(tx_preamble.a128)  # a128序列长度
        self.sync_threshold = sync_threshold  # 相关性阈值（0~1）
        self.scaling_factor = scaling_factor    # 连续高相关区间长度阈值（倍base_sequences_length）
        self.sync_type = tx_preamble.preamble_type  # SYNC类型 'short'/'long'
        self.p_noise = np.finfo(np.float64).eps  # 防除零极小值

    def _pattern_match(self, data, pattern):
        """
        辅助函数：模式匹配（对齐MATLAB patternMatch）
        :param data: 待匹配序列（一维）
        :param pattern: 目标模式（如[0,1]表示0→1的跳变）
        :return: 所有匹配起始索引
        """
        pattern_len = len(pattern)
        data_len = len(data)
        match_indices = []
        
        # 遍历所有可能的起始位置
        for i in range(data_len - pattern_len + 1):
            # 逐元素对比模式
            match_flag = True
            for j in range(pattern_len):
                if data[i + j] != pattern[j]:
                    match_flag = False
                    break
            if match_flag:
                match_indices.append(i)
        return np.array(match_indices)

    def _correlate_samples(self, rx_sig, symbol_length, threshold):
        """
        核心自相关计算（对齐MATLAB correlateSamples）
        :param rx_sig: 单块输入信号（N×C，C为通道数）
        :param symbol_length: a128序列长度
        :param threshold: 相关性阈值
        :return: packet_start（块内帧偏移）, Mn（归一化相关性统计值）
        """
        # 信号延迟：错开symbol_length个采样点
        if rx_sig.shape[0] <= symbol_length:
            return [], np.array([])  # 信号过短，直接返回空
        rx_delayed = rx_sig[symbol_length:, :]  # 延迟信号
        rx = rx_sig[:-symbol_length, :]         # 原始信号

        # 1. 计算自相关（共轭相乘 + 滑动求和 + 多通道求和）
        weights = np.ones((symbol_length, 1))   # 滑动窗（全1）
        # 共轭相乘
        corr_raw = np.conj(rx_delayed) * rx
        # 滑动求和（等价于MATLAB filter(weights,1,x)）
        # C = np.zeros((corr_raw.shape[0] - symbol_length + 1, corr_raw.shape[1]))
        # for c in range(corr_raw.shape[1]):
        #     C[:, c] = np.convolve(corr_raw[:, c].real, weights.flatten(), mode='valid') + \
        #               1j * np.convolve(corr_raw[:, c].imag, weights.flatten(), mode='valid')
        C = []
        for c in range(corr_raw.shape[1]):
            # 复信号直接卷积，保留相位信息
            conv_res = np.convolve(corr_raw[:, c], np.ones(symbol_length), mode='valid')
            C.append(conv_res)
        C = np.stack(C, axis=1)
        # 多通道求和 + 归一化
        C_sum = np.sum(C, axis=1)
        CS = C_sum / symbol_length

        # 2. 计算接收信号功率（滑动求和 + 多通道求和）
        power_raw = np.abs(rx_delayed) ** 2
        P = []
        for c in range(power_raw.shape[1]):
            conv_res = np.convolve(power_raw[:, c], np.ones(symbol_length), mode='valid')
            P.append(conv_res)
        P = np.stack(P, axis=1)
        P_sum = np.sum(P, axis=1)
        PS = (P_sum / symbol_length) + self.p_noise

        # 3. 功率归一化的相关性统计值
        Mn = (np.abs(CS) ** 2) / (PS)
        # 归一化到0~1（避免数值溢出）
        Mn_max = np.max(Mn) if len(Mn) > 0 else self.p_noise
        Mn_norm = Mn / Mn_max

        # 4. 阈值判决 + 模式匹配筛选连续高相关区间
        if len(Mn_norm) == 0:
            return [], Mn_norm
        N = (Mn_norm > threshold).astype(int)  # 二值化（1=超过阈值，0=未超过）
        packet_start = []

        # 放宽长度条件：测试场景下降低阈值更容易检测到
        min_valid_len = symbol_length * self.scaling_factor
        if np.sum(N) >= min_valid_len:
            # 构造扩展序列（对齐MATLAB：[N(1)~=1 N N(end)~=1]）
            N_ext_start = np.concatenate([[1 - N[0]], N])
            N_ext_end = np.concatenate([N, [1 - N[-1]]])
            # 找“0→1”起始位置、“1→0”结束位置
            start_ones = self._pattern_match(N_ext_start.tolist(), [0, 1])
            end_ones = self._pattern_match(N_ext_end.tolist(), [1, 0])

            if len(start_ones) > 0 and len(end_ones) > 0:
                # 计算每个连续区间的长度
                length_ones = end_ones - start_ones + 1
                # 筛选超过长度阈值的区间
                valid_idx = np.where(length_ones >= min_valid_len)[0]
                if len(valid_idx) > 0:
                    # 取第一个有效区间的起始位置
                    packet_start = [start_ones[valid_idx[0]] - 1]

        return packet_start, Mn_norm

    def detect_sync(self, rx_signal):
        """
        主检测函数（对齐MATLAB Frame_detect）
        :param rx_signal: 接收信号（N×C或N，C为通道数）
        :param sync_type: SYNC类型 'short'/'long'
        :return: coarse_offset（全局帧偏移）, Mn（相关性统计值）
        """
        # 0. 输入适配：转为2D数组（兼容单/多通道）
        if rx_signal.ndim == 1:
            rx_signal = rx_signal.reshape(-1, 1)
        inp_length = rx_signal.shape[0]
        num_channels = rx_signal.shape[1]

        # 1. SYNC参数初始化（对齐MATLAB）
        if self.sync_type == 'short':
            num_repetitions = 14  # short: a128重复14次
        elif self.sync_type == 'long':
            num_repetitions = 28  # long: a128重复28次
        else:
            raise ValueError("sync_type仅支持 'short' 或 'long'")
        
        len_sync = self.base_sequences_length * num_repetitions
        len_half_sync = len_sync // 2

        # 2. 补零处理：使输入长度为len_half_sync的整数倍
        if inp_length <= len_half_sync:
            num_pad_samples = len_sync - inp_length
        else:
            num_pad_samples = (np.ceil(inp_length / len_half_sync) * len_half_sync - inp_length).astype(int)
        
        pad_samples = np.zeros((num_pad_samples, num_channels), dtype=rx_signal.dtype)
        rx_padded = np.concatenate([rx_signal, pad_samples], axis=0)
        padded_length = rx_padded.shape[0]

        # 3. 分块处理初始化
        num_blocks = (padded_length) // len_half_sync
        # 预分配相关性统计数组
        ds_len = max(1, padded_length - 2 * self.base_sequences_length + 1)
        DS = np.zeros((ds_len, 1), dtype=np.float64)
        coarse_offset = None
        Mn = np.array([])

        # 4. 分块处理信号
        if num_blocks > 2:
            for n in range(num_blocks - 2):
                # 提取当前块（长度为len_sync）
                start_idx = n * len_half_sync
                end_idx = start_idx + len_sync
                if end_idx > rx_padded.shape[0]:
                    continue
                buffer = rx_padded[start_idx:end_idx, :]
                
                # 自相关检测
                start_offset, out = self._correlate_samples(buffer, self.base_sequences_length, self.sync_threshold)
                
                # 检测到帧：修正全局偏移并返回
                if len(start_offset) > 0:
                    coarse_offset = start_offset[0] + start_idx
                    Mn = out
                    return coarse_offset, Mn

            # 处理最后一个块
            blk_offset = len_half_sync * (num_blocks - 2)
            buffer = rx_padded[blk_offset:blk_offset + len_sync, :]
            # 补零到len_sync长度
            if buffer.shape[0] < len_sync:
                buffer = np.concatenate([buffer, np.zeros((len_sync - buffer.shape[0], num_channels), dtype=buffer.dtype)], axis=0)
            start_offset, out = self._correlate_samples(buffer, self.base_sequences_length, self.sync_threshold)
            
            if len(start_offset) > 0:
                coarse_offset = start_offset[0] + blk_offset
            Mn = out

        else:
            # 信号较短：直接处理补零后的完整信号
            buffer = np.concatenate([rx_signal, pad_samples], axis=0)[:len_sync, :]
            start_offset, out = self._correlate_samples(buffer, self.base_sequences_length, self.sync_threshold)
            if len(start_offset) > 0:
                coarse_offset = start_offset[0]
            Mn = out

        # 去除补零影响：裁剪Mn到原始信号长度
        if len(Mn) > inp_length:
            Mn = Mn[:inp_length]

        return coarse_offset, Mn

# 测试代码
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    tx_symbols = transmitter.tx_symbols
    tx_preamble = transmitter.preamble_gen
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # 恢复接收符号
    recovered_symbols = rx_matched_filter.recover_symbols(rx_signal)
    coarse_sync = CoarseSync(tx_preamble, sync_threshold=0.01, scaling_factor=5)

    # 检测SYNC
    offset, corr_norm = coarse_sync.detect_sync(recovered_symbols[:2000])
    print(f"粗同步偏移：{offset}")

     # ===================== 5. 可视化结果 =====================
    plt.figure(figsize=(12, 6))
    
    # 子图1：接收信号的幅度
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(recovered_symbols[:2000]), label='接收信号幅度')
    plt.axvline(x=params.get("delay"), color='r', linestyle='--', label=f'真实偏移：{params.get("delay")}')
    if offset is not None:
        plt.axvline(x=offset, color='g', linestyle='--', label=f'检测偏移：{offset}')
    plt.title('接收信号幅度')
    plt.xlabel('采样点')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 子图2：相关性统计值Mn（归一化）
    plt.subplot(2, 1, 2)
    plt.plot(corr_norm, label='Mn归一化值')
    plt.axhline(y=coarse_sync.sync_threshold, color='r', linestyle='-', label=f'阈值：{coarse_sync.sync_threshold}')
    if offset is not None:
        plt.axvline(x=offset, color='g', linestyle='--', label=f'检测偏移：{offset}')
    plt.title('归一化相关性统计值Mn')
    plt.xlabel('采样点')
    plt.ylabel('Mn (0~1)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()