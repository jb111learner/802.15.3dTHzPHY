import numpy as np

class MultipathChannel:
    """
    多径信道模型：生成信道冲激响应（CIR），实现信号的多径卷积
    支持自定义多径时延、增益，或生成随机多径（如瑞利衰落）
    """
    def __init__(self, params):
        self.params = params
        self.chan_impulse = None  # 信道冲激响应（CIR）
        self._init_channel()

    def _init_channel(self):
        """初始化信道冲激响应：基于参数配置的多径时延和增益"""
        chan_delays = self.params.get("chan_delays")  # 多径时延（采样点单位）
        chan_gains = self.params.get("chan_gains")    # 多径增益（复数值）
        
        # 验证时延和增益长度一致
        assert len(chan_delays) == len(chan_gains), "多径时延和增益长度必须一致"
        
        # 生成信道冲激响应（索引对应时延，值对应增益）
        max_delay = max(chan_delays) if len(chan_delays) > 0 else 0
        self.chan_impulse = np.zeros(max_delay + 1, dtype=complex)
        for delay, gain in zip(chan_delays, chan_gains):
            self.chan_impulse[delay] = gain

    def generate_rayleigh_fading(self, num_paths=4, max_delay=12):
        """生成瑞利衰落多径信道（可选：随机多径）"""
        # 随机生成多径时延（0~max_delay）
        chan_delays = np.sort(np.random.choice(max_delay + 1, num_paths, replace=False))
        # 瑞利衰落增益（幅度服从瑞利分布，相位均匀分布）
        chan_gains = (np.random.randn(num_paths) + 1j * np.random.randn(num_paths)) / np.sqrt(2)
        chan_gains /= np.sqrt(np.sum(np.abs(chan_gains)**2))
        # 更新参数和冲激响应
        self.params.update(chan_delays=chan_delays, chan_gains=chan_gains)
        self._init_channel()

    def apply_multipath(self, signal):
        """应用多径效应：信号与信道冲激响应的线性卷积"""
        # 卷积（保持输出长度与输入一致，mode="same"）
        signal_with_multipath = np.convolve(signal, self.chan_impulse, mode="full")
        return signal_with_multipath

# 测试
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    params = PHYParams()
    multipath_chan = MultipathChannel(params)
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    signal_with_multipath = multipath_chan.apply_multipath(signal)
    print(f"信道冲激响应：{multipath_chan.chan_impulse}")
    print(f"原始信号长度：{len(signal)}, 多径后长度：{len(signal_with_multipath)}")