import numpy as np

class CFO:
    """
    载波频率偏移（CFO）模型：生成频偏并应用到信号
    支持ppm（百万分比）或直接指定频偏值
    """
    def __init__(self, params):
        self.params = params
        self.freq_offset = None  # 实际频偏值（Hz）
        self._init_cfo()

    def _init_cfo(self):
        """初始化频偏：基于ppm和载波频率计算"""
        ppm = self.params.get("ppm")        # 频率偏差（ppm）
        fc = self.params.get("fc")       # 载波频率（Hz，THz系统典型值）
        self.freq_offset = ppm * 1e-6 * fc      # 实际频偏值（Hz）

    def set_freq_offset(self, freq_offset):
        """直接设置频偏值（覆盖ppm计算）"""
        self.freq_offset = freq_offset

    def apply_cfo(self, signal):
        """应用频偏：信号乘以线性相位因子"""
        fs = self.params.get("symbol_rate") * self.params.get("oversampling")      # 采样率（码片速率）
        n = np.arange(len(signal))              # 采样索引
        # 频偏相位因子：exp(j*2π*freq_offset*n/fs)
        cfo_factor = np.exp(1j * 2 * np.pi * self.freq_offset * n / fs)
        signal_with_cfo = signal * cfo_factor
        return signal_with_cfo

# 测试
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    params = PHYParams()
    cfo = CFO(params)
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    signal_with_cfo = cfo.apply_cfo(signal)
    print(f"频偏值：{cfo.freq_offset} Hz")
    print(f"原始信号长度：{len(signal)}, 频偏后长度：{len(signal_with_cfo)}")