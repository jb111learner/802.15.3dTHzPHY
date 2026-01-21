import numpy as np

class BaseReceiver:
    """
    接收机基类：定义接收端的统一接口
    子类需重写：coarse_sync() 粗同步
              fine_sync() 细同步
              estimate_and_compensate_cfo() 频偏估计与补偿
              estimate_channel() 信道估计
              estimate_noise_var() 噪声方差估计
              equalize() 均衡
    """
    def __init__(self, params, transmitter):
        self.params = params  # 参数对象
        self.transmitter = transmitter  # 发射机对象（用于获取前导码等）
    
    def matched_filter(self):
        """匹配滤波"""
        pass

    def remove_gi(self):
        """去除GI"""
        pass

    def coarse_sync(self):
        """粗同步"""
        pass

    def fine_sync(self):
        """细同步"""
        pass

    def estimate_and_compensate_cfo(self):
        """频偏估计与补偿"""
        pass

    def estimate_channel(self):
        """信道估计"""
        pass

    def estimate_noise_var(self):
        """噪声方差估计"""
        pass

    def equalize(self):
        """均衡"""
        pass

    def demodulate(self):
        """解调"""
        pass
    
    def decision(self):
        """判决"""
        pass  # 默认不实现

    def decode(self):
        """解码"""
        pass  # 默认不实现

    def run(self, rx_signal_dict):
        """执行完整接收流程"""
        self.rx_signal_dict = rx_signal_dict
        self.matched_filter()
        self.coarse_sync()
        self.estimate_and_compensate_cfo()
        self.fine_sync()
        self.estimate_channel()
        self.estimate_noise_var()
        self.equalize()
        self.demodulate()
        self.decision()
        data = self.decode()
        return data

# # 使用示例（子类实现）
# class THzReceiver(BaseReceiver):
#     def __init__(self, params, preamble):
#         super().__init__(params, preamble)
#         # 拆分前导码
#         self.sync = preamble[:128]  # SYNC序列（前128个符号）
#         self.sfd = preamble[128:192]  # SFD序列（中间64个符号）
#         self.ces = preamble[192:]  # CES序列（后128个符号）

#     def coarse_sync(self):
#         """粗同步：互相关检测SYNC序列"""
#         rx_signal = self.rx_signal
#         sync = self.sync
#         # 计算互相关
#         corr = np.correlate(rx_signal, sync, mode="valid")
#         corr_mag = np.abs(corr)
#         # 找到峰值位置
#         self.sync_offset = np.argmax(corr_mag)
#         print(f"粗同步偏移：{self.sync_offset}")

#     def fine_sync(self):
#         """细同步：互相关检测SFD序列"""
#         rx_signal = self.rx_signal[self.sync_offset + len(self.sync):]
#         sfd = self.sfd
#         # 计算互相关
#         corr = np.correlate(rx_signal, sfd, mode="valid")
#         corr_mag = np.abs(corr)
#         # 找到峰值位置
#         fine_offset = np.argmax(corr_mag)
#         self.sync_offset += len(self.sync) + fine_offset
#         print(f"细同步偏移：{fine_offset}，总同步偏移：{self.sync_offset}")

#     def estimate_and_compensate_cfo(self):
#         """频偏估计与补偿（基于SYNC序列）"""
#         rx_signal = self.rx_signal
#         sync_offset = self.sync_offset
#         sync_field = rx_signal[sync_offset : sync_offset + len(self.sync)]
#         # 估计频偏：相邻符号相位差
#         phase_diff = np.angle(sync_field[1:] * np.conj(sync_field[:-1]))
#         fs = self.params.get("fs")
#         freq_offset = np.mean(phase_diff) / (2 * np.pi / fs)
#         print(f"估计频偏：{freq_offset:.2f} Hz")
#         # 补偿频偏
#         n = np.arange(len(rx_signal))
#         cfo_factor = np.exp(-1j * 2 * np.pi * freq_offset * n / fs)
#         self.rx_signal = rx_signal * cfo_factor

#     def estimate_channel(self):
#         """信道估计（基于CES序列）"""
#         rx_signal = self.rx_signal
#         sync_offset = self.sync_offset
#         ces_offset = sync_offset + len(self.sync) + len(self.sfd)
#         ces_field = rx_signal[ces_offset : ces_offset + len(self.ces)]
#         # 频域信道估计（LS估计）
#         ces_fft = np.fft.fft(self.ces)
#         ces_field_fft = np.fft.fft(ces_field)
#         self.chan_est = ces_field_fft / ces_fft
#         print(f"信道估计长度：{len(self.chan_est)}")

#     def estimate_noise_var(self):
#         """噪声方差估计（基于SYNC序列）"""
#         rx_signal = self.rx_signal
#         sync_offset = self.sync_offset
#         sync_field = rx_signal[sync_offset : sync_offset + len(self.sync)]
#         # 噪声 = 接收SYNC - 理想SYNC
#         noise = sync_field - self.sync
#         self.noise_var = np.mean(np.abs(noise)**2)
#         print(f"估计噪声方差：{self.noise_var:.6f}")

#     def equalize(self):
#         """频域均衡（迫零均衡）"""
#         rx_signal = self.rx_signal
#         sync_offset = self.sync_offset
#         preamble_length = len(self.preamble)
#         data_signal = rx_signal[sync_offset + preamble_length:]
#         N = self.params.get("N")
#         cp_length = max(self.params.get("chan_delays")) if len(self.params.get("chan_delays")) > 0 else 16
#         # 分块均衡
#         equalized_blocks = []
#         for i in range(0, len(data_signal), N + cp_length):
#             block = data_signal[i : i + N + cp_length]
#             if len(block) < N + cp_length:
#                 break
#             # 去除CP
#             block_no_cp = block[cp_length:]
#             # 频域转换
#             block_fft = np.fft.fft(block_no_cp)
#             # 迫零均衡
#             equalized_fft = block_fft / self.chan_est[:N]
#             # 时域转换
#             equalized_block = np.fft.ifft(equalized_fft)
#             equalized_blocks.append(equalized_block)
#         self.equalized_data = np.concatenate(equalized_blocks)
#         print(f"均衡后数据长度：{len(self.equalized_data)}")

# # 测试
# if __name__ == "__main__":
#     params = PHYParams()
#     transmitter = THzTransmitter(params)
#     tx_signal = transmitter.run()
    
#     channel = THzChannel(params)
#     rx_signal = channel.run(tx_signal)
    
#     receiver = THzReceiver(params, transmitter.preamble)
#     equalized_data = receiver.run(rx_signal)