import numpy as np
import numpy as np
from transmitter.THzTransmitter import THzTransmitter
from params.PHYParams import PHYParams
from channel.MultipathChannel import MultipathChannel
class AWGN:
    """
    加性高斯白噪声（AWGN）模型：根据信噪比（SNR）添加噪声
    支持复信号（基带）和实信号（射频）
    """
    def __init__(self, params):
        self.params = params

    def calculate_noise_power(self, signal_power, snr_db):
        """根据信号功率和SNR计算噪声功率"""
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        return noise_power

    def add_awgn(self, signal):
        """为信号添加AWGN噪声"""
        snr_db = self.params.get("SNRdB")
        # 计算信号功率（复信号：实部+虚部分开计算）
        signal_power = np.mean(np.abs(signal) ** 2)
        # 计算噪声功率
        noise_power = self.calculate_noise_power(signal_power, snr_db)
        # 生成复高斯噪声（实部和虚部分别服从N(0, noise_power/2)）
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        # 添加噪声
        signal_with_noise = signal + noise
        return signal_with_noise

# 测试
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    multipath_chan = MultipathChannel(params)
    signal_with_multipath = multipath_chan.apply_multipath(tx_signal)
    awgn = AWGN(params)
    signal_with_noise = awgn.add_awgn(signal_with_multipath)
    # 计算实际SNR（验证）
    signal_power = np.mean(np.abs(signal_with_multipath) ** 2)
    noise_power = np.mean(np.abs(signal_with_noise - signal_with_multipath) ** 2)
    actual_snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"目标SNR：{params.get('SNRdB')} dB, 实际SNR：{actual_snr_db:.2f} dB")