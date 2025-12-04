import numpy as np
from scipy.signal import lfilter
from transmitter.Pulseshaper import PulseShaper 

class RxMatchedFilter:
    """
    接收端匹配滤波类：实现接收信号的匹配滤波和下采样，恢复原始符号
    与发送端脉冲成型滤波器共轭匹配，最大化接收信噪比
    """
    def __init__(self, tx_pulse_shaper):
        """
        初始化接收端匹配滤波器（直接复用发送端参数，保证匹配）
        :param tx_pulse_shaper: 发送端TxPulseShaper实例
        """
        self.tx_params = tx_pulse_shaper
        self.upsample_factor = tx_pulse_shaper.upsample_factor
        self.filter_type = tx_pulse_shaper.filter_type
        self.filter_coeffs = self._design_rx_matched_filter()

    def _design_rx_matched_filter(self):
        """设计接收端匹配滤波器（与发送端滤波器共轭匹配）"""
        if self.filter_type == "rrc":
            # 根升余弦匹配滤波：与发送端相同（RRC*RRC=RC）
            matched_coeffs = self.tx_params.filter_coeffs
        else:
            # 其他滤波器：共轭反转（保证相位匹配）
            matched_coeffs = np.conj(self.tx_params.filter_coeffs[::-1])
        return matched_coeffs

    def matched_filter(self, rx_signal):
        """对接收信号进行匹配滤波"""
        filtered_signal = lfilter(self.filter_coeffs, 1, rx_signal)
        # 去除滤波器延迟
        filtered_signal = filtered_signal[len(self.filter_coeffs)//2:]
        return filtered_signal

    def downsample_symbols(self, filtered_signal):
        """对滤波后的信号下采样，恢复原始符号"""
        # 下采样：取每个符号周期的最佳采样点（同步后）
        downsampled = filtered_signal[::self.upsample_factor]
        return downsampled

    def recover_symbols(self, rx_signal):
        """完整接收处理流程：匹配滤波 + 下采样"""
        filtered_signal = self.matched_filter(rx_signal)
        recovered_symbols = self.downsample_symbols(filtered_signal)
        return recovered_symbols
    
