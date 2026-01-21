import numpy as np
from scipy.signal import lfilter
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class RxMatchedFilter:
    def __init__(self, tx_pulse_shaper):
        self.upsample = tx_pulse_shaper.sps
        self.chan_max_delay = tx_pulse_shaper.chan_max_delay
        self.filter_type = tx_pulse_shaper.filter_type
        self.filter_coeffs = tx_pulse_shaper.filter_coeffs
        self.filter_length_span = tx_pulse_shaper.filter_length 
        
        # 匹配滤波器
        if self.filter_type == "rrc":
            self.h_rx = self.filter_coeffs  # RRC对称，共轭翻转后和原系数一致
        else:
            self.h_rx = np.conj(self.filter_coeffs[::-1])

        # 线性相位滤波器的延迟
        self.filter_delay = (len(self.h_rx) - 1) // 2  

        # 输出参数    
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.symbol_length = None  # 符号长度
        self.padding_bit_num = 0   # 补零的比特数   

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 信号流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 信号长度,
            "signal_power": 信号功率,
            "noise_power": 噪声功率,
            "SNRdB": self.snr_db,
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
        
        self.sample_rate = data_dict["sample_rate_Hz"] / self.upsample
        self.symbol_length = data_dict["signal_length"] / self.upsample
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]        

    # def matched_filter(self, rx):
    #     """滤波"""
    #     # full卷积 
    #     y = np.convolve(rx, self.h_rx, 'full')
    #     # 去除滤波器引入的线性相位延迟
    #     y = y[self.filter_delay:]
    #     return y
    
    # def downsample(self, y):
    #     """下采样"""
    #     start = 0
    #     return y[start::self.upsample]

    # def recover_symbols(self, rx_signal_dict):
    #     """恢复符号（匹配滤波+下采样+截断边缘冗余）"""
    #     # 校验输入数据合法性
    #     self._verification_data(rx_signal_dict)
    #     rx_signal = rx_signal_dict["signal_stream"]
    #     # 匹配滤波
    #     y = self.matched_filter(rx_signal)
    #     # 下采样
    #     s = self.downsample(y)
    #     # 截断边缘冗余
    #     span = int(self.filter_length_span / self.upsample)
    #     if len(s) > 2*span:
    #         s = s[:-span] 

    #     if len(s) != self.symbol_length:
    #         raise ValueError("匹配滤波后信号长度不匹配:预期长度={}, 实际长度={}".format(self.symbol_length, len(s)))
        
    #     result_dict = {
    #         "symbol_stream": s,
    #         "sample_rate_Hz": self.sample_rate,
    #         "duration_seconds": self.duration,
    #         "symbol_length": self.symbol_length,
    #         "padding_bit_num": self.padding_bit_num,
    #     }
    #     return result_dict
    def matched_filter(self, rx):
        """滤波：精确计算卷积和延迟补偿"""
        # full卷积 (长度 = len(rx) + len(h_rx) - 1)
        y = np.convolve(rx, self.h_rx, 'full')
        
        # 精确移除滤波器引入的线性相位延迟
        # 处理整数/半整数延迟：先取整，后续下采样时再微调
        delay_int = int(np.round(self.filter_delay))
        y = y[delay_int:]
        
        # 截断到原始信号长度（去除卷积扩展的部分）
        y = y[:len(rx)]
        
        return y
    
    def downsample(self, y):
        """下采样：精确对齐符号时钟"""
        # 计算下采样起始偏移（补偿半整数延迟）
        offset = int((self.filter_delay - np.floor(self.filter_delay)) * self.upsample)
        start = offset % self.upsample
        
        # 下采样并确保长度正确
        downsampled = y[start::self.upsample]
        
        return downsampled

    def recover_symbols(self, rx_signal_dict):
        """恢复符号（匹配滤波+下采样+精确长度截断）"""
        # 校验输入数据合法性
        self._verification_data(rx_signal_dict)
        rx_signal = rx_signal_dict["signal_stream"]
        
        # 1. 匹配滤波（含延迟补偿和长度截断）
        self.y_filtered = self.matched_filter(rx_signal)
        
        # 2. 下采样
        s = self.downsample(self.y_filtered)
        
        # 3. 精确截断到预期符号长度（核心修复）
        # 移除边缘冗余并确保长度严格匹配
        if len(s) > self.symbol_length:
            # 计算需要截断的长度
            trim_length = len(s) - self.symbol_length
            # 前后均分截断（避免单边截断导致的信号失真）
            trim_front = trim_length // 2
            trim_back = trim_length - trim_front
            s = s[trim_front:-trim_back] if trim_back > 0 else s[trim_front:]
        elif len(s) < self.symbol_length:
            # 补零（应对极少数长度不足情况）
            pad_length = self.symbol_length - len(s)
            s = np.pad(s, (0, pad_length), mode='constant')
        
        # 最终长度校验
        if not np.isclose(len(s), self.symbol_length):
            raise ValueError(
                f"匹配滤波后信号长度不匹配:预期长度={self.symbol_length}, 实际长度={len(s)}"
            )
        
        result_dict = {
            "symbol_stream": s,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "symbol_length": self.symbol_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return result_dict

# 测试
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # === 进行匹配滤波并获得中间信号 ===
    rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)
    print(f"发射信号长度：{len(tx_signal_dict['signal_stream'])}, 接收信号长度：{len(rx_matched_dict['symbol_stream'])}")

    # 提取信号用于可视化
    tx_signal = tx_signal_dict["signal_stream"]
    rx_signal = rx_signal_dict["signal_stream"]
    y_filtered = rx_matched_filter.y_filtered
    tx_symbols = transmitter.data_with_gi_dict["symbol_stream"]
    y_matched = rx_matched_dict["symbol_stream"]
    # ================================
    #        可视化绘图部分
    # ================================

    plt.figure(figsize=(10, 8))

    # ---- 1. 时域波形（匹配滤波前）----
    plt.subplot(3, 3, 1)
    plt.plot(np.real(tx_signal[:200]))
    plt.title("发射信号时域波形（实部）")
    plt.xlabel("样本点")
    plt.ylabel("幅度")

    # ---- 2. 时域波形（匹配滤波后）----
    plt.subplot(3, 3, 2)
    plt.plot(np.real(rx_signal[:200]))
    plt.title("接受信号时域波形（实部）")
    plt.xlabel("样本点")
    plt.ylabel("幅度")

    # ---- 3. 频域（匹配滤波前）----
    plt.subplot(3, 3, 3)
    RX = np.fft.fftshift(np.fft.fft(rx_signal))
    plt.plot(20*np.log10(np.abs(RX) + 1e-12))
    plt.title("接收信号（匹配滤波前）频谱")
    plt.xlabel("频率Bin")
    plt.ylabel("幅度 (dB)")

    # ---- 4. 频域（匹配滤波后）----
    plt.subplot(3, 3, 4)
    MF = np.fft.fftshift(np.fft.fft(y_filtered))
    plt.plot(20*np.log10(np.abs(MF) + 1e-12))
    plt.title("匹配滤波后信号频谱")
    plt.xlabel("频率Bin")
    plt.ylabel("幅度 (dB)")

    # ---- 5. 时域（原始符号）----
    plt.subplot(3, 3, 5)
    plt.plot(np.real(tx_symbols[:100]))
    plt.title("发射符号时域波形（实部）")
    plt.xlabel("样本点")
    plt.ylabel("幅度")    

    # ---- 6. 时域（恢复符号）----
    plt.subplot(3, 3, 6)
    plt.plot(np.real(y_matched[:100]))
    plt.title("恢复符号时域波形（实部）")
    plt.xlabel("样本点")
    plt.ylabel("幅度")


    plt.tight_layout()
    plt.show()
    
