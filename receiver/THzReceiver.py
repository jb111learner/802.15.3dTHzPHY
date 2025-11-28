import numpy as np
from core.BaseReceiver import BaseReceiver
from .sync.CoarseSync import CoarseSync
from .sync.FineSync import FineSync
from .sync.CFOEstimator import CFOEstimator
from .ChannelEstimator import ChannelEstimator
from .NoiseEstimator import NoiseEstimator
from .Equalizer import Equalizer
from transmitter.CPInserter import CPInserter

class THzReceiver(BaseReceiver):
    """
    THz接收机主类：调度子模块完成“同步→频偏补偿→信道估计→均衡”全流程
    """
    def __init__(self, params, preamble):
        super().__init__(params, preamble)
        # 拆分前导码
        self.sync_seq = preamble[:128]    # SYNC序列
        self.sfd_seq = preamble[128:192]  # SFD序列
        self.ces_seq = preamble[192:]     # CES序列
        # 初始化子模块
        self.coarse_sync = CoarseSync(params)
        self.fine_sync = FineSync(params)
        self.cfo_estimator = CFOEstimator(params)
        self.channel_estimator = ChannelEstimator(params)
        self.noise_estimator = NoiseEstimator(params)
        self.equalizer = Equalizer(params)
        self.cp_inserter = CPInserter(params)
        self.equalizer.set_cp_inserter(self.cp_inserter)

    def coarse_sync(self):
        """粗同步（重写基类方法）"""
        self.coarse_offset, self.corr_norm = self.coarse_sync.detect_sync(self.rx_signal, self.sync_seq)
        return self.coarse_offset

    def fine_sync(self):
        """细同步（重写基类方法）"""
        self.fine_offset = self.fine_sync.detect_sfd(self.rx_signal, self.sfd_seq, self.coarse_offset)
        self.sync_offset = self.fine_offset  # 总同步偏移
        return self.sync_offset

    def estimate_and_compensate_cfo(self):
        """频偏估计与补偿（重写基类方法）"""
        # 提取SYNC字段
        sync_field = self.rx_signal[self.coarse_offset : self.coarse_offset + len(self.sync_seq)]
        # 估计频偏
        self.cfo_est = self.cfo_estimator.estimate_cfo(sync_field)
        # 补偿频偏
        self.rx_signal_compensated = self.cfo_estimator.compensate_cfo(self.rx_signal, self.cfo_est)
        return self.cfo_est

    def estimate_channel(self):
        """信道估计（重写基类方法）"""
        # 提取CES字段（补偿频偏后）
        ces_offset = self.sync_offset + len(self.sync_seq) + len(self.sfd_seq)
        ces_field = self.rx_signal_compensated[ces_offset : ces_offset + len(self.ces_seq)]
        # LS信道估计
        self.chan_est_fft, self.chan_est_time = self.channel_estimator.ls_estimate(ces_field, self.ces_seq)
        return self.chan_est_fft

    def estimate_noise_var(self):
        """噪声方差估计（重写基类方法）"""
        # 提取SYNC字段（补偿频偏后）
        sync_field = self.rx_signal_compensated[self.coarse_offset : self.coarse_offset + len(self.sync_seq)]
        self.noise_var = self.noise_estimator.estimate_noise_var(sync_field, self.sync_seq)
        return self.noise_var

    def equalize(self):
        """均衡（重写基类方法）"""
        # 提取数据段（补偿频偏+同步后）
        data_offset = self.sync_offset + len(self.preamble)
        rx_data = self.rx_signal_compensated[data_offset:]
        # MMSE均衡（或ZF均衡）
        self.equalized_data = self.equalizer.mmse_equalize(rx_data, self.chan_est_fft, self.noise_var)
        # 可选：ZF均衡
        # self.equalized_data = self.equalizer.zf_equalize(rx_data, self.chan_est_fft)
        return self.equalized_data

    def run(self, rx_signal):
        """执行完整接收流程（重写基类方法）"""
        self.rx_signal = rx_signal
        # 1. 粗同步
        self.coarse_sync()
        # 2. 频偏估计与补偿
        self.estimate_and_compensate_cfo()
        # 3. 细同步（补偿频偏后）
        self.fine_sync()
        # 4. 噪声方差估计
        self.estimate_noise_var()
        # 5. 信道估计
        self.estimate_channel()
        # 6. 均衡
        self.equalize()
        return self.equalized_data

# 测试
if __name__ == "__main__":
    from params.PHYParams import PHYParams
    from transmitter.THzTransmitter import THzTransmitter
    from channel.THzChannel import THzChannel
    # 初始化参数、发射机、信道
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    # 初始化接收机并恢复数据
    receiver = THzReceiver(params, transmitter.preamble)
    equalized_data = receiver.run(rx_signal)
    print(f"均衡后数据长度：{len(equalized_data)}")
    print(f"原始数据长度：{len(transmitter.modulated_data)}")