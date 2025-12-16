import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.sync.CoarseSync import CoarseSync
import matplotlib.pyplot as plt

class CFOEstimator:
    """
    载波频率偏移（CFO）估计与补偿类
    完全对齐MATLAB CFO_estimation/Add_CFO逻辑，适配自研CFO仿真程序的参数体系
    核心：重复段互相关相位差法，支持ppm/直接频偏两种场景
    """
    def __init__(self,tx_preamble, params):
        self.params = params
        # 1. 采样率计算（对齐CFO仿真程序的fs逻辑）
        self.symbol_rate = params.get("symbol_rate")    # 码片速率
        self.oversampling = params.get("oversampling")    # 过采样率
        self.fs = self.symbol_rate * self.oversampling       # 实际采样率（Hz）
        # 2. CFO估计核心参数（与MATLAB L=128对齐）
        self.L = len(tx_preamble.a128) * self.oversampling  # 重复段长度（与MATLAB的L=128一致）
        # 3. 频偏缓存（可选：记录估计的频偏值）
        self.estimated_cfo = None

    def estimate_cfo(self, sync_field):
        """
        估计频偏（对齐MATLAB CFO_estimation函数）
        :param sync_field: 接收的SYNC序列段（需包含至少L个采样点的重复结构）
        :return: 估计的频偏值（Hz）
        """
        # 边界检查1：信号维度适配（支持1D/2D，兼容多通道）
        if sync_field.ndim > 2:
            raise ValueError(f"仅支持1D/2D信号，当前维度：{sync_field.ndim}")
        # 统一转为2D处理（行=采样点，列=通道）
        if sync_field.ndim == 1:
            sync_field = sync_field.reshape(-1, 1)
        
        # 边界检查2：确保SYNC字段长度大于重复段长度L
        n_samples = sync_field.shape[0]
        if n_samples <= self.L:
            raise ValueError(
                f"SYNC字段长度必须大于重复段长度L={self.L}！"
                f"当前长度：{n_samples}"
            )
        
        # 步骤1：拆分信号为两段重复部分（对齐MATLAB索引逻辑）
        # MATLAB: cx = x(1:end-L,:); sx = x(L+1:end,:)
        # Python索引适配（0起始）：
        cx = sync_field[:-self.L, :]   # 前半段：去掉最后L个采样点
        sx = sync_field[self.L:, :]    # 后半段：从第L个采样点开始

        # 步骤2：计算共轭互相关和（对齐MATLAB res = cx'*sx）
        # 对每个通道分别计算，再求和（多通道融合提升鲁棒性）
        res = 0
        for ch in range(sync_field.shape[1]):
            res += np.sum(np.conj(cx[:, ch]) * sx[:, ch])
        
        # 步骤3：提取相位差并归一化（修正相位折叠）
        phase = np.angle(res)                      # 提取总相位差（弧度）
        offset = phase / (2 * np.pi)              # 归一化到周期数
        offset = np.mod(offset + 0.5, 1) - 0.5    # 限制在[-0.5, 0.5]，避免折叠

        # 步骤4：计算最终频偏估计值（对齐MATLAB foffset = offset * fs/L）
        self.estimated_cfo = offset * self.fs / self.L
        return self.estimated_cfo
    
    def estimate_cfo_multi_segment(self, sync_field, segment_num=5):
        """多段重复序列平均估计，提升抗噪性"""
        seg_len = self.L * 2
        cfo_list = []
        for i in range(segment_num):
            start = i * self.L
            end = start + seg_len
            if end > len(sync_field):
                break
            seg = sync_field[start:end]
            cfo_list.append(self.estimate_cfo(seg))
        self.estimated_cfo = np.mean(cfo_list)
        return self.estimated_cfo    

    def compensate_cfo(self, rx_signal, cfo_est=None):
        """
        补偿频偏（对齐MATLAB Add_CFO函数，反向抵消频偏）
        :param rx_signal: 接收信号（1D/2D数组，支持多通道）
        :param cfo_est: 估计的频偏值（None则使用类内缓存的estimated_cfo）
        :return: 补偿后的信号（与输入同维度）
        """
        # 校验频偏值
        if cfo_est is None:
            if self.estimated_cfo is None:
                raise ValueError("未估计频偏！请先调用estimate_cfo或传入cfo_est")
            cfo_est = self.estimated_cfo
        
        # 统一转为2D处理（行=采样点，列=通道）
        is_1d = False
        if rx_signal.ndim == 1:
            is_1d = True
            rx_signal = rx_signal.reshape(-1, 1)
        
        # 生成时间索引（对齐CFO仿真程序的n/fs逻辑）
        n_samples = rx_signal.shape[0]
        n = np.arange(n_samples)[:, np.newaxis]  # 列向量，适配多通道
        
        # 步骤1：生成反向频偏相位因子（抵消CFO）
        # CFO仿真程序相位因子：exp(1j*2π*freq_offset*n/fs)
        # 补偿时取负：exp(-1j*2π*cfo_est*n/fs)
        cfo_factor = np.exp(-1j * 2 * np.pi * cfo_est * n / self.fs)
        
        # 步骤2：应用相位因子（逐元素相乘）
        rx_compensated = rx_signal * cfo_factor
        
        # 还原原始维度（1D/2D）
        if is_1d:
            rx_compensated = rx_compensated.flatten()
        
        return rx_compensated

# 测试代码
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    tx_symbols = transmitter.tx_symbols
    tx_preamble = transmitter.preamble_gen
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # 恢复接收信号
    recovered_symbols = rx_matched_filter.matched_filter(rx_signal)
    coarse_sync = CoarseSync(tx_preamble, sync_threshold=0.9, scaling_factor=5)

    # 检测SYNC
    offset, corr_norm = coarse_sync.detect_sync(recovered_symbols)
    print(f"粗同步偏移：{offset}")
    cfo_estimator = CFOEstimator(tx_preamble, params)
    # 提取SYNC字段
    sync_start = offset
    sync_field = recovered_symbols[sync_start:sync_start + len(tx_preamble.sync) * params.get("oversampling")]

    # ========== 原有功能：频偏估计与结果输出 ==========
    # 估计频偏
    cfo_est = cfo_estimator.estimate_cfo(sync_field)

    # 补偿频偏
    rx_signal_compensated = cfo_estimator.compensate_cfo(recovered_symbols, cfo_est)

    sync_field = rx_signal_compensated[sync_start:sync_start + len(tx_preamble.sync) * params.get("oversampling")]

    cfo_est_again = cfo_estimator.estimate_cfo(sync_field)

    # 结果验证
    print(f"\n=== 频偏估计与补偿结果 ===")
    print(f"真实频偏：{channel.cfo.freq_offset} Hz")
    print(f"估计频偏：{cfo_est:.2f} Hz")
    print(f"频偏估计误差：{abs(cfo_est - channel.cfo.freq_offset):.2f} Hz")
    print(f"补偿后再次估计频偏：{cfo_est_again:.2f} Hz")


