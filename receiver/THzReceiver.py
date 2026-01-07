import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from core.BaseReceiver import BaseReceiver
from receiver.sync.CoarseSync import CoarseSync
from receiver.sync.FineSync import FineSync
from receiver.sync.CFOEstimator import CFOEstimator
from receiver.ChannelEstimator import ChannelEstimator
from receiver.NoiseEstimator import NoiseEstimator
from transmitter.CPInserter import CPInserter
from receiver.DeModulator import THzDemodulator
from receiver.MatchedFilter import RxMatchedFilter

class THzReceiver(BaseReceiver):
    """
    THz接收机主类：调度子模块完成“同步→频偏补偿→信道估计→均衡”全流程
    """
    def __init__(self, params, transmitter):
        super().__init__(params, transmitter)
        self.params = params
        self.transmitter = transmitter
        # 拆分前导码
        self.sync_seq = self.transmitter.sync   # SYNC序列
        self.sfd_seq = self.transmitter.sfd     # SFD序列
        self.ces_seq = self.transmitter.ces     # CES序列
        # 初始化子模块
        self.demodulator = THzDemodulator(params)
        self.rx_matched_filter = RxMatchedFilter(self.transmitter.pulse_shaper)
        # self.coarse_sync = CoarseSync(params)
        # self.fine_sync = FineSync(params)
        # self.cfo_estimator = CFOEstimator(params)
        # self.channel_estimator = ChannelEstimator(params)
        # self.noise_estimator = NoiseEstimator(params)
        # self.equalizer = Equalizer(params)
        self.cp_inserter = self.transmitter.cp_inserter
        self.coder = self.transmitter.coder
        self.scrembler = self.transmitter.scrambler

    # def coarse_sync(self):
    #     """粗同步（重写基类方法）"""
    #     self.coarse_offset, self.corr_norm = self.coarse_sync.detect_sync(self.rx_signal, self.sync_seq)
    #     return self.coarse_offset

    # def fine_sync(self):
    #     """细同步（重写基类方法）"""
    #     self.fine_offset = self.fine_sync.detect_sfd(self.rx_signal, self.sfd_seq, self.coarse_offset)
    #     self.sync_offset = self.fine_offset  # 总同步偏移
    #     return self.sync_offset

    # def estimate_and_compensate_cfo(self):
    #     """频偏估计与补偿（重写基类方法）"""
    #     # 提取SYNC字段
    #     sync_field = self.rx_signal[self.coarse_offset : self.coarse_offset + len(self.sync_seq)]
    #     # 估计频偏
    #     self.cfo_est = self.cfo_estimator.estimate_cfo(sync_field)
    #     # 补偿频偏
    #     self.rx_signal_compensated = self.cfo_estimator.compensate_cfo(self.rx_signal, self.cfo_est)
    #     return self.cfo_est

    # def estimate_channel(self):
    #     """信道估计（重写基类方法）"""
    #     # 提取CES字段（补偿频偏后）
    #     ces_offset = self.sync_offset + len(self.sync_seq) + len(self.sfd_seq)
    #     ces_field = self.rx_signal_compensated[ces_offset : ces_offset + len(self.ces_seq)]
    #     # LS信道估计
    #     self.chan_est_fft, self.chan_est_time = self.channel_estimator.ls_estimate(ces_field, self.ces_seq)
    #     return self.chan_est_fft

    # def estimate_noise_var(self):
    #     """噪声方差估计（重写基类方法）"""
    #     # 提取SYNC字段（补偿频偏后）
    #     sync_field = self.rx_signal_compensated[self.coarse_offset : self.coarse_offset + len(self.sync_seq)]
    #     self.noise_var = self.noise_estimator.estimate_noise_var(sync_field, self.sync_seq)
    #     return self.noise_var

    # def equalize(self):
    #     """均衡（重写基类方法）"""
    #     # 提取数据段（补偿频偏+同步后）
    #     data_offset = self.sync_offset + len(self.preamble)
    #     rx_data = self.rx_signal_compensated[data_offset:]
    #     # MMSE均衡（或ZF均衡）
    #     self.equalized_data = self.equalizer.mmse_equalize(rx_data, self.chan_est_fft, self.noise_var)
    #     # 可选：ZF均衡
    #     # self.equalized_data = self.equalizer.zf_equalize(rx_data, self.chan_est_fft)
    #     return self.equalized_data
    
    def matched_filter(self):
        """匹配滤波（重写基类方法）"""
        self.rx_symbols_cp = self.rx_matched_filter.recover_symbols(self.rx_signal)
        self.rx_symbols = self.cp_inserter.remove_cp(self.rx_symbols_cp)
        return self.rx_symbols
    
    def demodulate(self):
        """解调（重写基类方法）"""
        snr_db = self.params.get("SNRdB")    
        snr_linear = 10 ** (snr_db / 10)
        sigma = np.sqrt(1.00 / (2 * snr_linear))
        self.llr = self.demodulator.demodulate(self.rx_symbols, sigma)
        return self.llr

    def decision(self):
        """判决（重写基类方法）"""
        # LLR转比特
        self.rx_bits = self.demodulator.llr_to_bits(self.llr)
        return self.rx_bits

    def decode(self):
        """解码（重写基类方法）"""
        self.decoded_bits = self.coder.decode(self.rx_bits)
        self.data_bits = self.scrembler.descramble(self.decoded_bits)
        return self.data_bits

    def run(self, rx_signal):
        """执行完整接收流程（重写基类方法）"""
        self.rx_signal = rx_signal
        # # 粗同步
        # self.coarse_sync()
        # # 频偏估计与补偿
        # self.estimate_and_compensate_cfo()
        # # 细同步（补偿频偏后）
        # self.fine_sync()
        # # 噪声方差估计
        # self.estimate_noise_var()
        # # 信道估计
        # self.estimate_channel()
        # # 均衡
        # self.equalize()
        self.matched_filter()
        self.demodulate()
        self.decision()
        data = self.decode()
        return data

# 测试
if __name__ == "__main__":
    # 初始化参数、发射机、信道
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    # 初始化接收机并恢复数据
    receiver = THzReceiver(params, transmitter)
    rx_data = receiver.run(rx_signal)
    rx_bits = receiver.rx_bits
    decoded_bits = receiver.decoded_bits
    print(f"\n===== 解调结果分析 =====")
    print(f"接收比特长度：{len(rx_bits)}, 原始比特长度：{len(transmitter.coded_bits)}")
    print(f"原始比特（前10）：{transmitter.coded_bits[0:10]}")
    print(f"解调比特（前10）：{rx_bits[0:10]}")
    print(f"比特错误数：{np.sum(transmitter.coded_bits != rx_bits)}")
    print(f"误码率：{np.sum(transmitter.coded_bits != rx_bits)/len(transmitter.coded_bits):.6f}")

    print(f"\n===== 解码结果分析 =====")
    print(f"接收比特长度：{len(rx_data)}, 原始比特长度：{len(transmitter.data_bits)}")
    print(f"原始比特（前10）：{transmitter.data_bits[0:10]}")
    print(f"解调比特（前10）：{rx_data[0:10]}")
    print(f"比特错误数：{np.sum(transmitter.data_bits != rx_data)}")
    print(f"误码率：{np.sum(transmitter.data_bits != rx_data)/len(transmitter.data_bits):.6f}")