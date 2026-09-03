import numpy as np

class AWGN:
    """
    加性高斯白噪声（AWGN）模型，支持两种配置方式：
    1. 基于噪声温度+带宽（物理层热噪声）
    2. 直接SNR配置（SNR定义为系统带宽内的信噪比）
    核心公式：热噪声功率 Pn = k*T*B（考虑噪声系数时 Pn = k*T_sys*B）

    对复基带离散信号，白噪声覆盖整个采样带宽 Fs。SNR 模式下每个
    复采样点的噪声方差需按 Fs/B 放大，避免过采样使匹配滤波后的
    有效 SNR 人为提高 10*log10(Fs/B)。
    """
    # 物理常数定义
    BOLTZMANN_CONST = 1.38e-23  # 玻尔兹曼常数，单位J/K

    def __init__(self, params):
        self.params = params
        self.noise_temp = self.params.get("noise_temperature") # 噪声温度，默认290K
        self.bandwidth = self.params.get("bandwidth")  # 系统等效噪声带宽，单位Hz（必填）
        self.noise_figure_db = self.params.get("noise_figure_db")  # 噪声系数，默认0dB（无损耗）
        
        # 2. 兼容原有SNR配置（可选，优先级低于温度+带宽）
        self.snr_db = self.params.get("SNRdB")

        # 校验必填参数
        if self.bandwidth is None and self.snr_db is None and self.noise_temp is None and self.noise_figure_db is None:
            raise ValueError("必须配置bandwidth（带宽）、SNRdB（信噪比）、noise_temperature（噪声温度）、noise_figure_db（噪声系数），至少一项")
        
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

    def _calculate_system_noise_temp(self):
        """计算系统等效噪声温度（考虑噪声系数）"""
        # 噪声系数从dB转线性值
        noise_figure_linear = 10 ** (self.noise_figure_db / 10)
        # 系统等效噪声温度 = 物理噪声温度 * 噪声系数（简化模型，适配太赫兹通信场景）
        # 完整模型：T_sys = T_ant + (F-1)*T0，若需天线温度可扩展params添加
        sys_noise_temp = self.noise_temp * noise_figure_linear
        return sys_noise_temp

    def _noise_bandwidth(self):
        """返回 SNR 定义所使用的等效噪声带宽。"""
        if self.bandwidth is None:
            # 未显式配置带宽时，将 SNR 解释为全采样带宽内的 SNR。
            return float(self.sample_rate)

        bandwidth = float(self.bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth 必须为有限正数")
        if bandwidth > float(self.sample_rate) * (1.0 + 1e-12):
            raise ValueError(
                f"复基带噪声带宽 bandwidth={bandwidth} Hz 不能超过采样率 "
                f"sample_rate_Hz={self.sample_rate} Hz"
            )
        return bandwidth

    def calculate_noise_power(self, signal_power=None):
        """
        计算噪声功率（两种模式自动切换）
        :param signal_power: 信号功率（仅SNR模式需要）
        :return: noise_power: 每个复采样点的噪声方差（线性值）
        """
        # 模式1：基于噪声温度+带宽（物理热噪声，优先）
        if self.bandwidth and self.noise_temp and self.noise_figure_db is not None:
            sys_temp = self._calculate_system_noise_temp()
            noise_power = self.BOLTZMANN_CONST * sys_temp * self.bandwidth
        # 模式2：兼容原有SNR模式（备用）
        elif self.snr_db is not None and signal_power is not None:
            snr_linear = 10 ** (self.snr_db / 10)
            noise_bandwidth = self._noise_bandwidth()
            # SNRdB 是带宽 B 内的 SNR；生成的离散白噪声覆盖 Fs，故其
            # 每采样点总方差应乘 Fs/B。该写法对任意过采样率保持一致。
            noise_power = (
                signal_power
                * float(self.sample_rate)
                / noise_bandwidth
                / snr_linear
            )
        else:
            raise ValueError("计算噪声功率需要：要么配置bandwidth+noise_temperature，要么传入signal_power+配置SNRdB")
        return noise_power

    def add_awgn(self, signal_dict):
        """
        给输入信号添加加性高斯白噪声
        :param signal_dict: 包含信号流等信息的字典
        :return: signal_with_noise: 加噪后的复信号
        """
        # 校验输入数据合法性
        self._verification_data(signal_dict)
        signal = signal_dict["signal_stream"]

        # 1. 计算信号功率（SNR模式需要，温度+带宽模式可选，用于校验）
        signal_power = np.mean(np.abs(signal) ** 2)
        # 2. 计算噪声功率
        noise_power = self.calculate_noise_power(signal_power)
        # 3. 生成复高斯噪声（实部和虚部分别满足高斯分布，功率各占noise_power/2）
        # 噪声方差 = 噪声功率/2（复信号的实/虚部分开）
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        print(f"设定噪声方差：{noise_power:.6f}")
        # 4. 添加噪声
        signal_with_noise = signal + noise
        noise_bandwidth = self._noise_bandwidth()
        sample_snr_db = 10 * np.log10(signal_power / noise_power)
        # 将全采样带宽噪声折算回配置带宽，报告与 UI 参数一致的带内 SNR。
        in_band_noise_power = noise_power * noise_bandwidth / self.sample_rate
        self.snr_db = 10 * np.log10(signal_power / in_band_noise_power)
        # 校验输出信号长度
        if len(signal_with_noise) != self.signal_length:
            raise ValueError("加噪后信号长度不匹配:预期长度={}, 实际长度={}".format(self.signal_length, len(signal_with_noise)))
        
        results_dict = dict(signal_dict)
        results_dict.update({
            "signal_stream": signal_with_noise,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "signal_power": signal_power,
            "noise_power": noise_power,
            "SNRdB": self.snr_db,
            "sample_SNRdB": sample_snr_db,
            "noise_bandwidth_Hz": noise_bandwidth,
            "padding_bit_num": self.padding_bit_num,
        })
        
        return results_dict

# 测试
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter

    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal_dict = transmitter.run()
    # multipath_chan = MultipathChannel(params)
    # signal_with_multipath = multipath_chan.apply_multipath(tx_signal)
    awgn = AWGN(params)
    signal_with_noise_dict = awgn.add_awgn(tx_signal_dict)
    # 计算实际SNR
    signal_power = signal_with_noise_dict["signal_power"]
    noise_power = signal_with_noise_dict["noise_power"]
    actual_snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"目标SNR：{params.get('SNRdB')} dB, 实际SNR：{actual_snr_db:.2f} dB")
