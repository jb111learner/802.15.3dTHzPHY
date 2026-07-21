import numpy as np

class BaseChannel:
    """
    信道基类：定义信道的统一接口
    子类需重写：apply_multipath() 应用多径
              add_awgn() 添加噪声
              add_cfo() 添加频偏
    """
    def __init__(self, params):
        self.params = params  # 参数对象
        self.rx_signal = None  # 经过信道后的信号
        self.chan_impulse = None  # 信道冲激响应
        self._init_channel()  # 初始化信道

    def _init_channel(self):
        """初始化信道（子类可重写）"""
        # # 生成信道冲激响应
        # chan_delays = self.params.get("chan_delays")
        # chan_gains = self.params.get("chan_gains")
        # max_delay = max(chan_delays) if len(chan_delays) > 0 else 0
        # self.chan_impulse = np.zeros(max_delay + 1, dtype=complex)
        # for delay, gain in zip(chan_delays, chan_gains):
        #     self.chan_impulse[delay] = gain

    def apply_multipath(self, signal):
        """应用多径效应（子类可重写）"""

    def add_awgn(self, signal):
        """添加高斯白噪声（子类可重写）"""

    def add_cfo(self, signal):
        """添加频偏（子类可重写）"""

    def run(self, tx_signal):
        """执行完整信道流程（统一调度）"""
        signal_after_multipath = self.apply_multipath(tx_signal)
        signal_after_awgn = self.add_awgn(signal_after_multipath)
        self.rx_signal = self.add_cfo(signal_after_awgn)
        return self.rx_signal

# # 使用示例（子类实现）
# class THzChannel(BaseChannel):
#     def apply_multipath(self, signal):
#         """应用多径效应（线性卷积）"""
#         return np.convolve(signal, self.chan_impulse, mode="same")

#     def add_awgn(self, signal):
#         """添加高斯白噪声"""
#         snr_db = self.params.get("SNRdB")
#         signal_power = np.mean(np.abs(signal)**2)
#         noise_power = signal_power / (10 ** (snr_db / 10))
#         noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
#         return signal + noise

#     def add_cfo(self, signal):
#         """添加频偏"""
#         fs = self.params.get("fs")
#         ppm = self.params.get("ppm", 30)
#         fc = self.params.get("fc", 100e9)
#         freq_offset = ppm * 1e-6 * fc  # 实际频偏值
#         n = np.arange(len(signal))
#         cfo_factor = np.exp(1j * 2 * np.pi * freq_offset * n / fs)
#         return signal * cfo_factor

# # 测试
# if __name__ == "__main__":
#     params = PHYParams()
#     transmitter = THzTransmitter(params)
#     tx_signal = transmitter.run()
    
#     channel = THzChannel(params)
#     rx_signal = channel.run(tx_signal)
#     print(f"接收信号长度：{len(rx_signal)}")
#     print(f"信道冲激响应长度：{len(channel.chan_impulse)}")