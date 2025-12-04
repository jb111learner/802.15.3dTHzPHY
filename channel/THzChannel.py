import numpy as np
from core.BaseChannel import BaseChannel
from .MultipathChannel import MultipathChannel
from .AWGN import AWGN
from .CFO import CFO

class THzChannel(BaseChannel):
    """
    THz信道主类：调度子模块完成“多径→噪声→频偏”的非理想效应叠加
    可选添加相位噪声、时延等其他信道特性
    """
    def __init__(self, params):
        super().__init__(params)
        # 初始化子模块
        self.multipath_chan = MultipathChannel(params)
        self.awgn = AWGN(params)
        self.cfo = CFO(params)
        self.chan_true_fft = None  # 真实信道频域响应（用于校验估计精度）

    def _init_channel(self):
        """重写基类方法：初始化真实信道频域响应"""
        super()._init_channel()
        # 计算信道频域响应（FFT）
        N = self.params.get("N", 64)
        self.chan_true_fft = np.fft.fft(self.multipath_chan.chan_impulse, N)

    def apply_multipath(self, signal):
        """应用多径效应（调用MultipathChannel）"""
        return self.multipath_chan.apply_multipath(signal)

    def add_awgn(self, signal):
        """添加AWGN噪声（调用AWGN）"""
        return self.awgn.add_awgn(signal)

    def add_cfo(self, signal):
        """应用频偏（调用CFO）"""
        return self.cfo.apply_cfo(signal)

    def apply_phase_noise(self, signal):
        """可选：添加相位噪声（简化版）"""
        phase_noise_std = self.params.get("phase_noise_std", 0.01)  # 相位噪声标准差
        phase_noise = np.random.randn(len(signal)) * phase_noise_std
        signal_with_phase_noise = signal * np.exp(1j * phase_noise)
        return signal_with_phase_noise

    def apply_delay(self, signal):
        """可选：添加传输时延（前置零符号）"""
        delay = self.params.get("delay", 500)  # 与发射机延迟区分，可选
        signal_with_delay = np.concatenate([np.zeros(delay, dtype=complex), signal])
        return signal_with_delay

    def run(self, tx_signal):
        """执行完整信道流程：多径→噪声→频偏（可选添加相位噪声/时延）"""
        # 1. 多径效应
        signal = self.apply_multipath(tx_signal)
        # 2. 添加AWGN噪声
        signal = self.add_awgn(signal)
        # 3. 应用频偏
        signal = self.add_cfo(signal)
        # 4. 可选：添加相位噪声
        if self.params.get("enable_phase_noise", False):
            signal = self.apply_phase_noise(signal)
        # 5. 可选：添加传输时延
        if self.params.get("enable_delay", False):
            signal = self.apply_delay(signal)
        # 保存接收信号
        self.rx_signal = signal
        return self.rx_signal

# 测试
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    print(f"发射信号长度：{len(tx_signal)}, 接收信号长度：{len(rx_signal)}")
    print(f"真实信道频域响应：{np.abs(channel.chan_true_fft)[:10]}")  # 打印前10个点