import numpy as np
from scipy.signal import find_peaks
from scipy.signal import resample
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.sync.CoarseSync import CoarseSync
from receiver.sync.CFOEstimator import CFOEstimator
import matplotlib.pyplot as plt

class FineSync:
    """
    细同步：基于SFD序列的互相关检测（适配“仅符号级SFD序列”场景）
    核心改进：
    1. 自动对符号级SFD序列做上采样，匹配接收信号采样率
    2. 完全对齐MATLAB的共轭翻转、卷积、峰值筛选逻辑
    3. 兼容符号级/采样级SFD序列输入
    4. 新增卷积结果可视化，定位偏移异常问题
    """
    def __init__(self, tx_preamble, search_window=100, min_peak_ratio=0.5):
        self.params = tx_preamble.params
        self.search_window = search_window          # 粗峰值前后搜索范围
        self.sfd_symbol_sequence = tx_preamble.sfd  # 符号级SFD序列
        self.sfd_symbol_length = len(self.sfd_symbol_sequence)    # SFD符号长度
        self.min_peak_ratio = min_peak_ratio       # MATLAB中的MinPeakHeight=0.5*max(abs(Mn))
        self.oversampling = self.params.get("oversampling")  
        self.sync_length = len(tx_preamble.sync) * self.oversampling

    def _upsample_sfd(self, sfd_symbol_seq):
        """符号级SFD序列上采样（匹配接收信号的采样率）"""
        upsampled_length = self.sfd_symbol_length * self.oversampling
        upsampled = resample(sfd_symbol_seq, upsampled_length)
        return upsampled

    def detect_sfd(self, rx_signal, coarse_offset, plot_flag=True):
        """
        检测SFD序列位置（输入为符号级SFD序列）
        :param rx_signal: 接收信号（已上采样，如4倍）
        :param coarse_offset: 粗同步偏移（基于采样级的SYNC起始位置）
        :param plot_flag: 是否绘制卷积结果可视化图
        :return: 细同步修正量（整体偏移=粗同步+该值）
        """
        # ========== 1. SFD序列上采样（核心适配） ==========
        sfd_seq = self._upsample_sfd(self.sfd_symbol_sequence)
        sfd_sample_length = len(sfd_seq)

        # ========== 2. 截取SFD候选段（采样级，对齐MATLAB逻辑） ==========
        sfd_candidate_start = coarse_offset + self.sync_length
        sfd_candidate_len = sfd_sample_length + 2 * self.search_window
        start_idx = max(0, sfd_candidate_start - self.search_window)
        end_idx = min(len(rx_signal), sfd_candidate_start + sfd_candidate_len)
        rx_segment = rx_signal[start_idx:end_idx]

        # ========== 3. 构造匹配滤波器 + 卷积 ==========
        sfd_rev = np.flip(np.conj(sfd_seq))
        Mn = np.convolve(rx_segment, sfd_rev, mode="full")
        Mn = Mn[sfd_sample_length:]  # 截断前导无效区
        abs_Mn = np.abs(Mn)

        # ========== 4. 异常保护 ==========
        if len(abs_Mn) == 0:
            if plot_flag:
                plt.figure(figsize=(12, 6))
                plt.title("卷积结果为空（候选段长度不足）")
                plt.show()
            return 0

        # ========== 5. 峰值检测 ==========
        idx_peak = np.argmax(abs_Mn)
        max_corr = np.max(abs_Mn)

        # 局部搜索区域
        search_start = max(0, idx_peak - self.search_window)
        search_end = min(len(Mn), idx_peak + self.search_window + 1)
        Mn_region = Mn[search_start:search_end]
        abs_Mn_region = np.abs(Mn_region)

        # 筛选有效峰值
        peaks, _ = find_peaks(abs_Mn_region, height=self.min_peak_ratio * max_corr)
        if len(peaks) == 0:
            peak_pos_in_segment = idx_peak
        else:
            peak_pos_in_segment = search_start + peaks[0]

        # ========== 6. 计算细同步修正量 ==========
        fine_offset = peaks[0] - self.search_window + idx_peak - 1

        # ========== 7. 可视化卷积结果（核心定位问题） ==========
        if plot_flag:
            self._plot_convolution_result(
                abs_Mn, idx_peak, search_start, search_end,
                peaks, peak_pos_in_segment, start_idx, sfd_candidate_start
            )

        return fine_offset

    def _plot_convolution_result(self, abs_Mn, idx_peak, search_start, search_end, peaks, peak_pos_in_segment, start_idx, sfd_candidate_start):
        """
        绘制卷积结果可视化图，包含：
        1. 整体卷积幅值曲线
        2. 全局峰值、局部搜索区域、有效峰值标记
        3. 候选段起始位置标注
        """
        plt.figure(figsize=(15, 8))
        
        # 子图1：整体卷积幅值
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(abs_Mn, color='blue', label='卷积幅值（Mn）')
        # 标记全局峰值
        ax1.scatter(idx_peak, abs_Mn[idx_peak], color='red', s=50, label='全局峰值')
        # 标记局部搜索区域
        ax1.axvspan(search_start, search_end, alpha=0.2, color='yellow', label='局部搜索区域')
        # 标记有效峰值
        if len(peaks) > 0:
            ax1.scatter(peak_pos_in_segment, abs_Mn[peak_pos_in_segment], color='green', s=80, marker='*', label='有效峰值')
        # 标注关键信息
        ax1.axhline(y=self.min_peak_ratio * np.max(abs_Mn), color='orange', linestyle='--', label=f'峰值阈值（{self.min_peak_ratio}×max）')
        ax1.set_title('SFD卷积结果（Mn）幅值曲线', fontsize=12, fontweight='bold')
        ax1.set_xlabel('卷积序列索引')
        ax1.set_ylabel('幅值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：局部搜索区域放大
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(abs_Mn[search_start:search_end], color='blue')
        # 标记局部区域内的峰值
        if len(peaks) > 0:
            ax2.scatter(peaks, abs_Mn[search_start:search_end][peaks], color='green', s=80, marker='*', label='有效峰值')
        ax2.axhline(y=self.min_peak_ratio * np.max(abs_Mn), color='orange', linestyle='--', label=f'峰值阈值')
        ax2.set_title('局部搜索区域放大', fontsize=12, fontweight='bold')
        ax2.set_xlabel('局部区域索引')
        ax2.set_ylabel('幅值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 标注候选段信息
        plt.figtext(0.5, 0.01, 
                    f'候选段起始（采样级）：{sfd_candidate_start} | 候选段截取范围：[{start_idx}, {start_idx+len(abs_Mn)}] | 全局峰值索引：{idx_peak}',
                    ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

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
    recovered_symbols = rx_matched_filter.matched_filter(rx_signal)
    
    # 粗同步
    coarse_sync = CoarseSync(tx_preamble, sync_threshold=0.5, scaling_factor=5)
    coarse_offset, corr_norm = coarse_sync.detect_sync(recovered_symbols)
    
    # CFO估计与补偿
    cfo_estimator = CFOEstimator(tx_preamble, params)
    sync_start = coarse_offset
    sync_field = recovered_symbols[sync_start:sync_start + len(tx_preamble.sync) * params.get("oversampling")]
    cfo_est = cfo_estimator.estimate_cfo(sync_field)
    rx_signal_compensated = cfo_estimator.compensate_cfo(recovered_symbols, cfo_est)
    
    # 再次估计频偏（验证补偿效果）
    sync_field_comp = rx_signal_compensated[sync_start:sync_start + len(tx_preamble.sync) * params.get("oversampling")]
    cfo_est_again = cfo_estimator.estimate_cfo(sync_field_comp)
    
    # 结果验证
    print(f"\n=== 频偏估计与补偿结果 ===")
    print(f"真实频偏：{channel.cfo.freq_offset} Hz")
    print(f"估计频偏：{cfo_est:.2f} Hz")
    print(f"频偏估计误差：{abs(cfo_est - channel.cfo.freq_offset):.2f} Hz") 
    print(f"补偿后再次估计频偏：{cfo_est_again:.2f} Hz")   
    
    # 细同步（开启可视化）
    fine_sync = FineSync(tx_preamble, search_window=16, min_peak_ratio=0.5) 
    fine_correction = fine_sync.detect_sfd(rx_signal_compensated, coarse_offset, plot_flag=True)
    total_offset = coarse_offset + fine_correction

    print(f"\n=== 同步偏移结果 ===")
    print(f"粗同步偏移：{coarse_offset}")
    print(f"细同步修正量：{fine_correction}")
    print(f"最终整体同步偏移：{total_offset}")
