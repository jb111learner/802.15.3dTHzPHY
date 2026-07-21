import numpy as np
from scipy.signal import find_peaks, resample_poly
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.sync.FineSync import FineSync
# from receiver.CFOEstimator import CFOEstimator
import matplotlib.pyplot as plt

class Downsampler:
    """
    下采样器
    用于将上采样后的信号转换为原始采样率，并根据细同步修正量进行偏移调整
    """

    def __init__(self, transmitter):
        self.params = transmitter.params
        self.oversampling = self.params.get("oversampling")
        self.filter_delay = (len(transmitter.pulse_shaper.filter_coeffs) - 1) // 2


        # 输出参数（两种模式共用）
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0
   
    def _verification_data(self, data_dict):
        """
        校验输入数据字典，并根据模式计算输出参数
        """
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")

        # 考虑滤波延迟
        self.sample_rate = data_dict["sample_rate_Hz"] / self.oversampling
        self.signal_length = int((data_dict["signal_length"] - self.filter_delay) / self.oversampling)
        self.duration = self.signal_length / self.sample_rate
        self.padding_bit_num = data_dict["padding_bit_num"]
    
    def downsample(self, y, symbol_length=None, start_index=None):
        """
        下采样：从匹配滤波后的信号中抽取符号
        :param y: 输入已上采样的接收信号（匹配滤波后）。
        :param symbol_length: 期望符号长度(可选)。
        :return: downsampled: 抽取后的符号序列，长度 = symbol_length（若指定）。
        """
        downsampled = y[0::self.oversampling]
        if symbol_length is not None:
            if len(downsampled) > symbol_length:
                # 只保留前 symbol_length 个符号，保持起始对齐
                downsampled = downsampled[:symbol_length]
            elif len(downsampled) < symbol_length:
                pad_length = symbol_length - len(downsampled)
                downsampled = np.pad(downsampled, (0, pad_length), mode='constant')

        return downsampled

    # # ---------- OFDM 模式：抗混叠下采样 ----------
    # def ofdm_downsample(self, y, start_index=0):
    #     """
    #     OFDM 专用下采样：抗混叠低通滤波 + 抽取
    #     恢复至原始符号速率，无匹配滤波概念。
    #     """
    #     # 确保长度为 oversampling 整数倍
    #     y_sfd = y[start_index:]
    #     num_symbols = len(y_sfd) // self.oversampling
    #     y_trunc = y_sfd[:num_symbols * self.oversampling]
    #     # resample_poly 内置抗混叠滤波和抽取，输出长度精确为 num_symbols
    #     downsampled = resample_poly(y_trunc, 1, self.oversampling)
    #     return downsampled    
    
    # ---------- 统一入口 ----------
    def recover_symbol(self, signal_dict):
        """
        下采样到符号率
        """
        self._verification_data(signal_dict)
        y = signal_dict["signal_stream"]
        self.rx_symbols = self.downsample(y, symbol_length=self.signal_length)
        result_dict = {
            "signal_stream": self.rx_symbols,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return result_dict
    
if __name__ == "__main__": 
    # # # 初始化参数和发射机
    # params = PHYParams()
    # transmitter = THzTransmitter(params)    
    # # 执行完整发射流程  
    # tx_signal_dict = transmitter.run() 
    # # 初始化信道并生成接收信号
    # channel = THzChannel(params)
    # rx_signal_dict = channel.run(tx_signal_dict)

    # # 初始化接收端匹配滤波器
    # rx_matched_filter = RxMatchedFilter(transmitter)
    # # === 进行匹配滤波并获得中间信号 ===
    # rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)
    # fine_sync = FineSync(transmitter)
    # rx_sampled_signal_dict = fine_sync.recover_symbol(rx_matched_signal_dict)
    # print(f"定时采样后信号长度：{len(rx_sampled_signal_dict['signal_stream'])}, 预期长度：{fine_sync.signal_length}")
    # print(f"采样率：{rx_sampled_signal_dict['sample_rate_Hz']} Hz，时长：{rx_sampled_signal_dict['duration_seconds']} 秒")
    # print(f"细同步修正量（采样点数）：{rx_sampled_signal_dict['fine_offset']}")

    # fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    # axes[0].plot(transmitter.data_with_preamble_dict['signal_stream'][:100])
    # axes[0].set_title("发射端原始符号（前500点）")
    # axes[0].set_xlabel("符号索引")
    # axes[0].set_ylabel("幅度")
    # axes[0].grid(True)
    # axes[1].plot(rx_sampled_signal_dict['signal_stream'][:100], color='orange')
    # axes[1].set_title("定时同步采样的符号（前500点）")
    # axes[1].set_xlabel("符号索引")
    # axes[1].set_ylabel("幅度")
    # axes[1].grid(True)
    # plt.tight_layout()
    # plt.show()

    # OFDM测试
    params = PHYParams()
    # params.update(link_mode = "ofdm")
    # params.update(enable_window_filter = False)

    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    # 初始化接收端滤波器
    rx_matched_filter = RxMatchedFilter(transmitter)
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)

    fine_sync = FineSync(transmitter)
    rx_sampled_signal_dict = fine_sync.fine_sync(rx_matched_signal_dict)
    downsampler = Downsampler(transmitter)
    rx_downsampled_signal_dict = downsampler.recover_symbol(rx_sampled_signal_dict)
    print(f"下采样后信号长度：{len(rx_downsampled_signal_dict['signal_stream'])}")
    print(f"采样率：{rx_downsampled_signal_dict['sample_rate_Hz']} Hz，时长：{rx_downsampled_signal_dict['duration_seconds']} 秒")
    print(f"细同步修正量（采样点数）：{rx_sampled_signal_dict['sync_offset']}")    
    preamble = transmitter.preamble
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].plot(transmitter.tx_signal_dict['signal_stream'][len(preamble)*4:len(preamble)*4+500])
    axes[0].set_title("发射信号（前500点）")
    axes[0].set_xlabel("采样点索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(rx_signal_dict['signal_stream'][len(preamble)*4:len(preamble)*4+500], color='orange')
    axes[1].set_title("接受信号（前500点）")
    axes[1].set_xlabel("采样点索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].plot(transmitter.data_with_preamble_dict['signal_stream'][:100])
    axes[0].set_title("发射端原始符号（前1000点）")
    axes[0].set_xlabel("符号索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(rx_downsampled_signal_dict['signal_stream'][:100], color='orange')
    axes[1].set_title("下采样的符号（前1000点）")
    axes[1].set_xlabel("符号索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()