import numpy as np

class CoarseSync:
    """
    粗同步：对齐MATLAB Frame_detect逻辑
    基于重复SYNC序列的自相关检测，实现帧起始定位
    支持short/long两种SYNC类型，分块处理、功率归一化、模式匹配筛选
    """
    def __init__(self, params):
        self.params = params
        # 核心参数（对齐MATLAB）
        self.symbol_length = 128  # a128序列长度
        self.sync_threshold = 0.8  # 相关性阈值（0~1）
        self.scaling_factor = 5    # 连续高相关区间长度阈值（倍symbol_length）
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
        rx_delayed = rx_sig[symbol_length:, :]  # 延迟信号
        rx = rx_sig[:-symbol_length, :]         # 原始信号

        # 1. 计算自相关（共轭相乘 + 滑动求和 + 多通道求和）
        weights = np.ones((symbol_length, 1))   # 滑动窗（全1）
        # 共轭相乘
        corr_raw = np.conj(rx_delayed) * rx
        # 滑动求和（等价于MATLAB filter(weights,1,x)）
        C = np.zeros((corr_raw.shape[0] - symbol_length + 1, corr_raw.shape[1]))
        for c in range(corr_raw.shape[1]):
            C[:, c] = np.convolve(corr_raw[:, c].real, weights.flatten(), mode='valid') + \
                      1j * np.convolve(corr_raw[:, c].imag, weights.flatten(), mode='valid')
        # 多通道求和 + 归一化
        C_sum = np.sum(C, axis=1)
        CS = C_sum[symbol_length:] / symbol_length

        # 2. 计算接收信号功率（滑动求和 + 多通道求和）
        power_raw = np.abs(rx_delayed) ** 2 / symbol_length
        P = np.zeros((power_raw.shape[0] - symbol_length + 1, power_raw.shape[1]))
        for c in range(power_raw.shape[1]):
            P[:, c] = np.convolve(power_raw[:, c], weights.flatten(), mode='valid')
        P_sum = np.sum(P, axis=1)
        PS = P_sum[symbol_length:] + self.p_noise  # 防除零

        # 3. 功率归一化的相关性统计值
        Mn = (np.abs(CS) ** 2) / (PS ** 2)
        # 归一化到0~1
        Mn_norm = Mn / (np.max(Mn) + self.p_noise)

        # 4. 阈值判决 + 模式匹配筛选连续高相关区间
        N = (Mn_norm > threshold).astype(int)  # 二值化（1=超过阈值，0=未超过）
        packet_start = []

        if np.sum(N) >= symbol_length * self.scaling_factor:
            # 构造扩展序列（对齐MATLAB：[N(1)~=1 N.' N(end)~=1]）
            N_ext_start = np.concatenate([[1 - N[0]], N])
            N_ext_end = np.concatenate([N, [1 - N[-1]]])
            # 找“0→1”起始位置、“1→0”结束位置
            start_ones = self._pattern_match(N_ext_start.tolist(), [0, 1])
            end_ones = self._pattern_match(N_ext_end.tolist(), [1, 0])

            if len(start_ones) > 0 and len(end_ones) > 0:
                # 计算每个连续区间的长度
                length_ones = end_ones - start_ones + 1
                # 筛选超过长度阈值的区间
                valid_idx = np.where(length_ones > symbol_length * self.scaling_factor)[0]
                if len(valid_idx) > 0:
                    # 取第一个最长有效区间的起始位置
                    max_len_idx = valid_idx[np.argmax(length_ones[valid_idx])]
                    packet_start = start_ones[max_len_idx] - 1  # 块内偏移

        return packet_start, Mn_norm

    def detect_sync(self, rx_signal, sync_type='short'):
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
        if sync_type == 'short':
            num_repetitions = 14  # short: a128重复14次
        elif sync_type == 'long':
            num_repetitions = 28  # long: a128重复28次
        else:
            raise ValueError("sync_type仅支持 'short' 或 'long'")
        
        len_sync = self.symbol_length * num_repetitions
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
        # 预分配相关性统计数组（对齐MATLAB DS）
        ds_len = padded_length - 2 * self.symbol_length + 1
        DS = np.zeros((ds_len, 1), dtype=np.float64)
        coarse_offset = []
        Mn = np.array([])

        # 4. 分块处理信号
        if num_blocks > 2:
            for n in range(num_blocks - 2):
                # 提取当前块（长度为len_sync）
                start_idx = n * len_half_sync
                end_idx = start_idx + len_sync
                buffer = rx_padded[start_idx:end_idx, :]
                
                # 自相关检测
                start_offset, out = self._correlate_samples(buffer, self.symbol_length, self.sync_threshold)
                
                # 保存相关性统计值
                ds_start = start_idx
                ds_end = start_idx + len_half_sync
                DS[ds_start:ds_end, 0] = out[:len_half_sync]

                # 检测到帧：修正全局偏移并返回
                if len(start_offset) > 0:
                    coarse_offset = start_offset + start_idx
                    DS[start_idx:start_idx + len(out), 0] = out
                    Mn = DS[:start_idx + len(out), 0]
                    return coarse_offset, Mn

            # 处理最后一个块
            blk_offset = len_half_sync * (num_blocks - 2)
            buffer = np.concatenate([rx_padded[blk_offset:, :], pad_samples], axis=0)[:len_sync, :]
            start_offset, out = self._correlate_samples(buffer, self.symbol_length, self.sync_threshold)
            
            if len(start_offset) > 0:
                coarse_offset = start_offset + blk_offset
            DS[blk_offset:blk_offset + len(out), 0] = out
            Mn = DS[:padded_length - num_pad_samples, 0]  # 去除补零部分

        else:
            # 信号较短：直接处理补零后的完整信号
            buffer = np.concatenate([rx_signal, pad_samples], axis=0)[:len_sync, :]
            start_offset, out = self._correlate_samples(buffer, self.symbol_length, self.sync_threshold)
            coarse_offset = start_offset
            Mn = out

        # 适配返回格式（无检测结果时返回空，否则返回数值）
        coarse_offset = coarse_offset[0] if isinstance(coarse_offset, (list, np.ndarray)) and len(coarse_offset) > 0 else None
        return coarse_offset, Mn

# 测试代码
if __name__ == "__main__":
    # 模拟PHY参数类（替代实际PHYParams）
    class PHYParams:
        pass

    params = PHYParams()
    coarse_sync = CoarseSync(params)

    # 生成测试信号：含重复a128序列的SYNC + 噪声
    symbol_length = 128
    # 生成ZC序列（a128）
    sync_seq = np.exp(-1j * np.pi * 3 * np.arange(symbol_length) * (np.arange(symbol_length) + 1) / symbol_length)
    # short SYNC：a128重复14次
    sync_short = np.tile(sync_seq, 14)
    # 构造接收信号：前导零 + SYNC + 噪声
    rx_signal = np.concatenate([
        np.zeros(500, dtype=np.complex128),  # 前导零（真实偏移500）
        sync_short,                          # short SYNC序列
        np.random.randn(1000) + 1j * np.random.randn(1000)  # 噪声
    ])

    # 检测short类型SYNC
    offset, corr_norm = coarse_sync.detect_sync(rx_signal, sync_type='short')
    print(f"粗同步偏移：{offset}（真实偏移：500）")