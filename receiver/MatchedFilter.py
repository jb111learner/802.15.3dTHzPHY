import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
# from receiver.CFOEstimator import CFOEstimator
# from receiver.sync.FineSync import FineSync
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class RxMatchedFilter:
    def __init__(self, transmitter):
        self.sps = transmitter.oversampling
        # self.chan_max_delay = transmitter.chan_max_delay
        self.filter_type = transmitter.pulse_shaper.filter_type
        self.filter_length = transmitter.pulse_shaper.filter_length
        self.link_mode = transmitter.params.get("link_mode").lower()

        # 获取（或生成）TX滤波器系数，统一从TX推导RX
        tx_coeffs = transmitter.pulse_shaper.filter_coeffs
        if tx_coeffs is None:
            tx_coeffs = transmitter.pulse_shaper._design_tx_filter()

        if self.link_mode == "sc-fde":
            # RRC/RC匹配滤波器：时间反转共轭
            self.h_rx = np.conj(tx_coeffs[::-1])
        elif self.link_mode == "ofdm":
            # OFDM低通滤波器：与TX相同形状，DC增益=1（TX的DC增益=sps）
            # TX滤波器 = firwin / sum(h) * sps，除以sps即得单位DC增益的RX滤波器
            self.h_rx = tx_coeffs / self.sps

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
        self.signal_length = data_dict["signal_length"] + 2 * self.filter_delay
        self.duration = self.signal_length / self.sample_rate
        self.padding_bit_num = data_dict["padding_bit_num"]

    def matched_filter(self, signal_dict):
        """匹配滤波"""
        self._verification_data(signal_dict)
        y = np.convolve(signal_dict["signal_stream"], self.h_rx, 'full')
        # 最终长度校验··
        if not np.isclose(len(y), self.signal_length):
            raise ValueError(
                f"匹配滤波后信号长度不匹配:预期长度={self.signal_length}, 实际长度={len(y)}"
            )
        
        result_dict = {
            "signal_stream": y,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return result_dict

# 测试
if __name__ == "__main__":
    # # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)


    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter)
    # === 进行匹配滤波并获得中间信号 ===
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)
    print(f"匹配滤波后信号长度：{len(rx_matched_signal_dict['signal_stream'])}, 预期长度：{rx_matched_filter.signal_length}")
    print(f"采样率：{rx_matched_signal_dict['sample_rate_Hz']} Hz，时长：{rx_matched_signal_dict['duration_seconds']} 秒")
    preamble = transmitter.preamble
    filter_delay = rx_matched_filter.filter_delay
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].plot(transmitter.tx_signal_dict['signal_stream'][:500])
    axes[0].set_title("发射信号（前500点）")
    axes[0].set_xlabel("采样点索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(rx_matched_signal_dict['signal_stream'][filter_delay:filter_delay+500], color='orange')
    axes[1].set_title("接收滤波后信号（前500点）")
    axes[1].set_xlabel("采样点索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    # 提取信号用于可视化
    tx_signal = tx_signal_dict["signal_stream"]
    rx_signal = rx_signal_dict["signal_stream"]
    y_filtered = rx_matched_signal_dict["signal_stream"]
    tx_symbols = transmitter.data_with_preamble_dict["signal_stream"]
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

    # # ---- 6. 时域（恢复符号）----
    # plt.subplot(3, 3, 6)
    # plt.plot(np.real(y_matched[:100]))
    # plt.title("恢复符号时域波形（实部）")
    # plt.xlabel("样本点")
    # plt.ylabel("幅度")

    # # ---- 7. 相位（恢复符号）----
    # plt.subplot(3, 3, 7)
    # plt.plot(np.angle(y_matched[:200]))
    # plt.title("恢复符号相位")
    # plt.xlabel("样本点")
    # plt.ylabel("相位 (弧度)")

    # ---- 8. 相位（原始符号）----
    plt.subplot(3, 3, 8)
    plt.plot(np.angle(tx_symbols[:200]))
    plt.title("原始符号相位")
    plt.xlabel("样本点")
    plt.ylabel("相位 (弧度)")

    plt.tight_layout()
    plt.show()
    
