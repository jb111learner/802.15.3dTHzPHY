# import numpy as np

# class CFO:
#     """
#     载波频率偏移（CFO）模型：生成频偏并应用到信号
#     支持ppm（百万分比）或直接指定频偏值
#     """
#     def __init__(self, params):
#         self.params = params
#         self.freq_offset = None  # 实际频偏值（Hz）
#         self._init_cfo()

#     def _init_cfo(self):
#         """初始化频偏：基于ppm和载波频率计算"""
#         ppm = self.params.get("ppm")        # 频率偏差（ppm）
#         fc = self.params.get("fc")       # 载波频率（Hz，THz系统典型值）
#         self.freq_offset = ppm * 1e-6 * fc      # 实际频偏值（Hz）

#     def set_freq_offset(self, freq_offset):
#         """直接设置频偏值（覆盖ppm计算）"""
#         self.freq_offset = freq_offset

#     def apply_cfo(self, signal):
#         """应用频偏：信号乘以线性相位因子"""
#         fs = self.params.get("symbol_rate") * self.params.get("oversampling")      # 采样率（码片速率）
#         n = np.arange(len(signal))              # 采样索引
#         # 频偏相位因子：exp(j*2π*freq_offset*n/fs)
#         cfo_factor = np.exp(1j * 2 * np.pi * self.freq_offset * n / fs)
#         signal_with_cfo = signal * cfo_factor
#         return signal_with_cfo

# # 测试
# if __name__ == "__main__":
#     from params.PHYParams import PHYParams
#     params = PHYParams()
#     cfo = CFO(params)
#     signal = np.random.randn(1000) + 1j * np.random.randn(1000)
#     signal_with_cfo = cfo.apply_cfo(signal)
#     print(f"频偏值：{cfo.freq_offset} Hz")
#     print(f"原始信号长度：{len(signal)}, 频偏后长度：{len(signal_with_cfo)}")
import numpy as np
from scipy.signal import butter, lfilter, welch

class CFO:
    """
    载波频率偏移（CFO） + 相位噪声 模型
    - 频偏：基于ppm或直接指定
    - 相位噪声：一阶IIR低通滤波生成有色噪声（模拟1/f特性）
    """
    def __init__(self, params):
        self.params = params
        """初始化频偏：基于ppm和载波频率计算"""
        self.phase_noise_state = None    # 用于IIR滤波的状态
        self.ppm = self.params.get("ppm")
        self.fc = self.params.get("fc")   # 默认1GHz载波
        self.freq_offset = self.ppm * 1e-6 * self.fc

        # 输出参数      
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.signal_length = None  # 信号长度
        self.padding_bit_num = 0   # 补零的比特数

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 信号流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 信号长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.signal_length = data_dict["signal_length"]
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]      

    def _init_phase_noise(self, fs):
        """初始化相位噪声生成器（一阶IIR低通滤波）"""
        # 相位噪声参数（可从params读取，带默认值）
        self.enable_phase_noise = self.params.get("enable_phase_noise")
        self.phase_noise_std = self.params.get("phase_noise_std")   # 弧度（RMS）
        self.phase_noise_bw = self.params.get("phase_noise_bw")    # 低通带宽（Hz）

        if self.enable_phase_noise:
            # 设计一阶IIR滤波器：H(z) = b0 / (1 - a1*z^-1)
            # 截止频率（-3dB） fc = fs/(2π) * (1-a1)  近似
            # 所以 a1 = 1 - 2π*fc/fs
            a1 = 1 - 2 * np.pi * self.phase_noise_bw / fs
            # 保证稳定性
            a1 = max(0.0, min(a1, 0.9999))
            # 为了使输出噪声的稳态方差 = phase_noise_std^2
            # 输出方差 = b0^2 / (1 - a1^2)  => b0 = phase_noise_std * sqrt(1 - a1^2)
            b0 = self.phase_noise_std * np.sqrt(1 - a1**2)
            self.phase_noise_b = [b0]
            self.phase_noise_a = [1, -a1]
            self.phase_noise_state = np.zeros(len(self.phase_noise_a)-1, dtype=float)
        else:
            self.phase_noise_b = [1.0]
            self.phase_noise_a = [1.0]
            self.phase_noise_state = None

    def set_freq_offset(self, freq_offset):
        """直接设置频偏值（覆盖ppm计算）"""
        self.freq_offset = freq_offset

    def _generate_phase_noise(self, nsamples):
        """生成有色相位噪声序列（低通滤波后的高斯噪声）"""
        if not self.enable_phase_noise:
            return np.zeros(nsamples)
        # 生成高斯白噪声
        white_noise = np.random.normal(0, 1, nsamples)
        # 通过一阶IIR滤波器
        filtered, self.phase_noise_state = lfilter(self.phase_noise_b, self.phase_noise_a,
                                                   white_noise, zi=self.phase_noise_state)
        return filtered

    def apply_cfo(self, signal_dict):
        """
        同时应用CFO和相位噪声到信号
        signal: 复基带信号
        """
        # 校验输入数据合法性
        self._verification_data(signal_dict)
        fs = signal_dict["sample_rate_Hz"]
        signal = signal_dict["signal_stream"]

        ns = len(signal)
        self._init_phase_noise(fs)
        # 1. 生成相位噪声序列（弧度）
        self.phase_noise = self._generate_phase_noise(ns)
        # 2. 生成CFO线性相位因子
        n = np.arange(ns)
        cfo_phase = 2 * np.pi * self.freq_offset * n / fs
        # 3. 总相位扰动
        total_phase = cfo_phase + self.phase_noise
        # 4. 应用
        result_signal = signal * np.exp(1j * total_phase)

        # 校验输出信号长度
        if len(result_signal) != self.signal_length:
            raise ValueError("加噪后信号长度不匹配:预期长度={}, 实际长度={}".format(self.signal_length, len(result_signal)))
        
        results_dict = {
            "signal_stream": result_signal,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return results_dict

# 测试
if __name__ == "__main__":
    from channel.AWGN import AWGN
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter
    import matplotlib.pyplot as plt   # 可选，若无可注释绘图部分

    # 初始化参数和发射机
    params = PHYParams()

    transmitter = THzTransmitter(params)
    tx_signal_dict = transmitter.run()
    awgn = AWGN(params)
    signal_with_noise_dict = awgn.add_awgn(tx_signal_dict)

    # 计算实际SNR
    signal_power = signal_with_noise_dict["signal_power"]
    noise_power = signal_with_noise_dict["noise_power"]
    actual_snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"目标SNR：{params.get('SNRdB')} dB, 实际SNR：{actual_snr_db:.2f} dB")  

    # 应用 CFO + 相位噪声
    cfo = CFO(params) 
    signal_with_cfo_dict = cfo.apply_cfo(signal_with_noise_dict)
    
    # ========== 绘制星座图验证 ==========

    # 取前2048个点以避免过密（若信号很长）
    num_points = min(2048, len(tx_signal_dict["signal_stream"]))

    # 原始发射信号（无噪声、无损伤）
    orig_signal = tx_signal_dict["signal_stream"][:num_points]

    # 经过AWGN但无CFO/相位噪声的信号
    awgn_signal = signal_with_noise_dict["signal_stream"][:num_points]

    # 经过AWGN+CFO+相位噪声的信号
    cfo_signal = signal_with_cfo_dict["signal_stream"][:num_points]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 子图1：原始发射信号
    axes[0].scatter(orig_signal.real, orig_signal.imag, s=1, alpha=0.6)
    axes[0].set_title("原始发射信号星座图")
    axes[0].set_xlabel("同相分量 (I)")
    axes[0].set_ylabel("正交分量 (Q)")
    axes[0].grid(True)
    axes[0].axis("equal")

    # 子图2：AWGN后（无频偏）
    axes[1].scatter(awgn_signal.real, awgn_signal.imag, s=1, alpha=0.6)
    axes[1].set_title("BPSK调制、RRC滤波、AWGN后星座图")
    axes[1].set_xlabel("同相分量 (I)")
    axes[1].set_ylabel("正交分量 (Q)")
    axes[1].grid(True)
    axes[1].axis("equal")

    # 子图3：AWGN+CFO+相位噪声
    axes[2].scatter(cfo_signal.real, cfo_signal.imag, s=1, alpha=0.6)
    axes[2].set_title("CFO+相位噪声后星座图\n(可见旋转与扩散)")
    axes[2].set_xlabel("同相分量 (I)")
    axes[2].set_ylabel("正交分量 (Q)")
    axes[2].grid(True)
    axes[2].axis("equal")

    plt.tight_layout()
    plt.show()
