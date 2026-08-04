import numpy as np
from scipy.signal import correlate, find_peaks, resample_poly
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
# from receiver.CFOEstimator import CFOEstimator
import matplotlib.pyplot as plt

class FineSync:
    """
    细同步与定时恢复（支持 SC‑FDE / OFDM 双模式）

    SC‑FDE 模式：基于 SFD 互相关获得最佳采样点，进行下采样。
    OFDM  模式：仅执行抗混叠低通滤波 + 抽取，恢复符号速率。
    """

    def __init__(self, transmitter, search_window=50, min_peak_ratio=0.5):
        self.params = transmitter.params
        self.oversampling = self.params.get("oversampling")
        self.link_mode = self.params.get("link_mode").lower()  # 新增
        self.search_window = search_window
        self.min_peak_ratio = min_peak_ratio
        self.sfd_upsampled = transmitter.sfd_upsampled
        self.sync_upsampled_length = len(transmitter.sync_upsampled)
        self.sfd_upsampled_length = len(transmitter.sfd_upsampled)

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
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.signal_length = data_dict["signal_length"]
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]

    def detect_sfd(self, rx_signal, coarse_sync_start=0, plot_flag=False):
        """
        检测SFD序列位置（输入为符号级SFD序列）
        :param rx_signal: 接收信号（已上采样，如4倍）
        :param sfd_seq: SFD序列（已上采样）
        :param plot_flag: 是否绘制卷积结果可视化图
        :return: 细同步修正量（整体偏移=粗同步+该值）
        """
        # ========== 1. 截取SFD候选段（采样级，对齐MATLAB逻辑） ==========
        sfd_candidate_start = int(coarse_sync_start) + self.sync_upsampled_length
        sfd_candidate_len = self.sfd_upsampled_length + 2 * self.search_window
        start_idx = max(0, sfd_candidate_start - self.search_window)
        end_idx = min(len(rx_signal), start_idx + sfd_candidate_len)
        rx_segment = rx_signal[start_idx:end_idx]

        # ========== 3. 构造匹配滤波器 + 卷积 ==========
        # 高过采样率下直接时域卷积复杂度为 O(N²)，4x SFD 会非常慢。
        # FFT 相关与原来的 conj+reverse 卷积数学等价。
        Mn = correlate(
            rx_segment, self.sfd_upsampled, mode="full", method="fft"
        )
        Mn = Mn[self.sfd_upsampled_length:]  # 截断前导无效区
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
        # fine_offset = peaks[0] - self.search_window + idx_peak - 1
        fine_offset = peak_pos_in_segment + 1 - self.search_window

        # ========== 7. 可视化卷积结果（核心定位问题） ==========
        if plot_flag:
            self._plot_convolution_result(
                abs_Mn, idx_peak, search_start, search_end,
                peaks, peak_pos_in_segment, start_idx, sfd_candidate_start
            )
        return fine_offset
    
    # def downsample(self, y, symbol_length=None, start_index=None):
    #     """
    #     下采样：从匹配滤波后的信号中抽取符号
    #     :param y: 输入已上采样的接收信号（匹配滤波后）。
    #     :param symbol_length: 期望符号长度(可选)。
    #     :param start_index: 自定义下采样起始索引（即第一个符号的最佳采样点在 y 中的位置）。
    #     :return: downsampled: 抽取后的符号序列，长度 = symbol_length（若指定）。
    #     """
    #     # ===== 使用外部指定的最佳采样点 =====
    #     # 确保起始索引在有效范围内
    #     if start_index < 0 or start_index >= len(y):
    #         raise ValueError(f"start_index ({start_index}) 超出信号范围 [0, {len(y)})")

    #     downsampled = y[start_index::self.oversampling]
    #     # 当已知确切起点时，多余符号一定在尾部，只需截断尾部或补零
    #     if symbol_length is not None:
    #         if len(downsampled) > symbol_length:
    #             # 只保留前 symbol_length 个符号，保持起始对齐
    #             downsampled = downsampled[:symbol_length]
    #         elif len(downsampled) < symbol_length:
    #             pad_length = symbol_length - len(downsampled)
    #             downsampled = np.pad(downsampled, (0, pad_length), mode='constant')

    #     return downsampled

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
    def fine_sync(self, signal_dict):
        """
        SFD互相关帧定界（SC-FDE / OFDM 通用）

        CoarseSync 已将信号对齐到 SYNC 起始，SFD 在固定偏移处。
        detect_sfd 搜索 SFD 位置并返回微调偏移量：
          - 若 CoarseSync 对齐准确 → offset ≈ 0
          - 若有残余偏差 → offset 自动补偿
        """
        self._verification_data(signal_dict)
        y = signal_dict["signal_stream"]

        coarse_start = int(
            signal_dict.get("coarse_sync_start", signal_dict.get("sync_offset", 0))
        )
        fine_offset = self.detect_sfd(
            y, coarse_sync_start=coarse_start, plot_flag=False
        )
        frame_start = coarse_start + fine_offset
        if frame_start < 0 or frame_start >= len(y):
            raise ValueError(
                f"合并同步偏移超出信号范围：coarse={coarse_start}, "
                f"fine={fine_offset}, frame_start={frame_start}"
            )
        aligned = y[frame_start:]
        result_dict = {
            "signal_stream": aligned,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": len(aligned) / self.sample_rate,
            "signal_length": len(aligned),
            "padding_bit_num": self.padding_bit_num,
            "sync_offset": fine_offset,
            "coarse_sync_start": coarse_start,
            "frame_start": frame_start,
        }
        return result_dict

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
    print(f"定时采样后信号长度：{len(rx_sampled_signal_dict['signal_stream'])}, 预期长度：{fine_sync.signal_length}")
    print(f"采样率：{rx_sampled_signal_dict['sample_rate_Hz']} Hz，时长：{rx_sampled_signal_dict['duration_seconds']} 秒")
    print(f"细同步修正量（采样点数）：{rx_sampled_signal_dict['fine_offset']}")
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
    axes[0].plot(transmitter.data_with_preamble_dict['signal_stream'][:1000])
    axes[0].set_title("发射端原始符号（前1000点）")
    axes[0].set_xlabel("符号索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(rx_sampled_signal_dict['signal_stream'][:1000], color='orange')
    axes[1].set_title("定时同步采样的符号（前1000点）")
    axes[1].set_xlabel("符号索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()



