# import numpy as np
# from scipy.signal import lfilter
# from transmitter.Pulseshaper import TxPulseShaper 
# from params.PHYParams import PHYParams
# from transmitter.THzTransmitter import THzTransmitter
# from channel.THzChannel import THzChannel
# import matplotlib.pyplot as plt

# # 设置中文显示
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False

# class RxMatchedFilter:
#     def __init__(self, tx_pulse_shaper):
#         self.upsample = tx_pulse_shaper.oversampling
#         self.chan_max_delay = tx_pulse_shaper.chan_max_delay
#         self.filter_type = tx_pulse_shaper.filter_type
#         h = tx_pulse_shaper.filter_coeffs        # Tx RRC
        
#         # 匹配滤波器 = 共轭翻转
#         if self.filter_type == "rrc":
#             self.h_rx = h      
#         else:
#             self.h_rx = np.conj(h[::-1])

#         # 发送 + 信道 + 接收总延迟
#         # self.total_delay = (len(h) - 1) + self.chan_max_delay
#         self.total_delay = (len(h) - 1)


#     def matched_filter(self, rx):
#         # full 卷积才正确
#         y = np.convolve(rx, self.h_rx, 'full')
#         # 去除总延迟
#         y = y[self.total_delay:]
#         return y
    
#     def downsample(self, y):
#         # 理论最佳采样点是 h 的中心
#         start = (self.upsample - 1)
#         return y[start::self.upsample]

#     def recover_symbols(self, rx_signal):
#         y = self.matched_filter(rx_signal)
#         s = self.downsample(y)
#         return s
import numpy as np
from scipy.signal import lfilter
from transmitter.Pulseshaper import TxPulseShaper 
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class RxMatchedFilter:
    def __init__(self, tx_pulse_shaper):
        self.upsample = tx_pulse_shaper.oversampling  # 对应MATLAB的sps
        self.chan_max_delay = tx_pulse_shaper.chan_max_delay
        self.filter_type = tx_pulse_shaper.filter_type
        self.filter_coeffs = tx_pulse_shaper.filter_coeffs  # Tx滤波器系数
        self.filter_length_span = tx_pulse_shaper.filter_length  # 对应MATLAB的span
        
        # 匹配滤波器 = 共轭翻转（RRC无需翻转，因为线性相位滤波器对称）
        if self.filter_type == "rrc":
            self.h_rx = self.filter_coeffs  # RRC对称，共轭翻转后和原系数一致
        else:
            self.h_rx = np.conj(self.filter_coeffs[::-1])

        # 线性相位滤波器的延迟 = (滤波器长度-1)/2 个采样点（核心修正）
        self.filter_delay = (len(self.h_rx) - 1) // 2  

    def matched_filter(self, rx):
        """等效MATLAB upfirdn(rxData,Filter,1,1)：仅滤波，不下采样"""
        # full卷积 = MATLAB conv的full模式，与upfirdn的滤波逻辑一致
        y = np.convolve(rx, self.h_rx, 'full')
        # 去除滤波器引入的线性相位延迟（对齐符号）
        y = y[self.filter_delay:]
        return y
    
    def downsample(self, y):
        """等效MATLAB upfirdn的q=sps下采样：从第0个点（Python下标）开始，步长upsample"""
        # 修正：MATLAB下标1 → Python下标0，起始点为0（而非upsample-1）
        start = 0  # 核心修正：下采样起始点对齐MATLAB
        return y[start::self.upsample]

    def recover_symbols(self, rx_signal):
        """完整等效MATLAB：匹配滤波 → 下采样 → 截断边缘冗余"""
        # 1. 匹配滤波（等效upfirdn滤波步骤）
        y = self.matched_filter(rx_signal)
        # 2. 下采样（等效upfirdn的q=sps下采样）
        s = self.downsample(y)
        # 3. 截断边缘冗余（等效MATLAB的span+1:end-span）
        # 修正：Python下标从0开始，span对应截断长度
        span = int(self.filter_length_span / self.upsample)
        if len(s) > 2*span:
            s = s[:-span]  # 核心修正：对齐MATLAB的截断逻辑
        return s

# 测试
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    tx_symbols = transmitter.tx_symbols
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # === 进行匹配滤波并获得中间信号 ===
    y_filtered = rx_matched_filter.matched_filter(rx_signal)
    y_matched = rx_matched_filter.recover_symbols(rx_signal)
    print(f"发射信号长度：{len(tx_symbols)}, 接收信号长度：{len(y_matched)}")


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
    
