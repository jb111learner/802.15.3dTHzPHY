import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.sync.FineSync import FineSync
from receiver.Downsampler import Downsampler
import matplotlib.pyplot as plt
from scipy import signal

class CFOEstimator:
    """
    载波频率偏移（CFO）估计与补偿类
    支持多帧信号：每帧独立估计并补偿自身 CFO。
    """
    def __init__(self, transmitter):
        self.params = transmitter.params
        self.link_mode = self.params.get("link_mode").lower()
        subframe_len = self.params.get("subframe_length")
        gi_len = self.params.get("gi_length")
        preamble_len = len(transmitter.preamble)
        if self.link_mode == "ofdm":
            subframe_count = self.params.get("subframe_ofdm_num")
        else:
            subframe_count = self.params.get("subframe_num")
        self.frame_symbol_num = (subframe_len + gi_len) * subframe_count + preamble_len
        self.oversampling = self.params.get("oversampling")
        self.frame_sample_num = self.frame_symbol_num * self.oversampling

        self.L = len(transmitter.preamble_gen.a128)                # 符号级重复段长度
        self.sync_len = len(transmitter.sync)                     # 符号级 SYNC 长度
        self.L_oversample = self.L * self.oversampling            # 采样级重复段长度
        self.sync_len_oversample = self.sync_len * self.oversampling

        # 输出缓存
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0
        self.estimated_cfo = None   # 设为平均值

    def _verification_data(self, data_dict):
        required_keys = ["signal_stream", "sample_rate_Hz", "duration_seconds",
                         "signal_length", "padding_bit_num"]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("采样率与时长不匹配")
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.signal_length = data_dict["signal_length"]
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]

    def _estimate_cfo_from_sync(self, sync_field, fs, L):
        """
        核心估计函数：利用两个相距 L 的重复段计算 CFO
        :param sync_field: 包含重复结构的信号片段（长度 >= 2*L）
        :param fs: 采样率 (Hz)
        :param L: 重复间隔（采样点数）
        :return: 估计的频偏 (Hz)
        """
        if len(sync_field) <= L:
            raise ValueError(f"sync_field 长度 ({len(sync_field)}) 必须大于 L ({L})")
        cx = sync_field[:-L]
        sx = sync_field[L:]
        # 多通道处理：如果 sync_field 是 2D (采样点 x 通道)，则按通道求和
        res = np.sum(np.conj(cx) * sx)          # 自动广播，结果为标量（所有通道求和）
        phase = np.angle(res)
        offset = phase / (2 * np.pi)            # 归一化周期数
        # 可选相位折叠修正（保留注释，暂不启用）
        # offset = np.mod(offset + 0.5, 1) - 0.5
        cfo = offset * fs / L
        return cfo

    def estimate_cfo_coarse(self, rx_signal, fs=None):
        """粗估计：使用过采样信号"""
        if fs is None:
            fs = self.sample_rate
        sync_field = rx_signal[:self.sync_len_oversample]
        return self._estimate_cfo_from_sync(sync_field, fs, self.L_oversample)

    def estimate_cfo_fine(self, rx_signal, fs=None):
        """细估计：使用符号率信号（定时后）"""
        if fs is None:
            fs = self.sample_rate
        sync_field = rx_signal[:self.sync_len]
        return self._estimate_cfo_from_sync(sync_field, fs, self.L)

    def compensate_cfo(self, rx_signal, fs, cfo_est=None, start_idx=0):
        """
        补偿频偏（支持指定全局起始索引）
        :param rx_signal: 1D 或 2D
        :param fs: 采样率
        :param cfo_est: 频偏估计值 (Hz)，若 None 则使用 self.estimated_cfo
        :param start_idx: 该信号段在全局时间轴上的起始采样索引（从0开始）
        """
        if cfo_est is None:
            if self.estimated_cfo is None:
                raise ValueError("未提供 CFO 估计值且类内未缓存")
            cfo_est = self.estimated_cfo

        is_1d = False
        if rx_signal.ndim == 1:
            is_1d = True
            rx_signal = rx_signal.reshape(-1, 1)

        n_samples = rx_signal.shape[0]
        # 使用全局索引：从 start_idx 到 start_idx + n_samples - 1
        n = np.arange(start_idx, start_idx + n_samples)[:, np.newaxis]
        cfo_factor = np.exp(-1j * 2 * np.pi * cfo_est * n / fs)
        compensated = rx_signal * cfo_factor

        if is_1d:
            compensated = compensated.flatten()
        return compensated


    def _split_into_frames(self, rx_signal, frame_len, tail_mode='discard'):
        """
        将一维/二维信号分割为帧列表
        :param rx_signal: 输入信号 (1D 或 2D)
        :param tail_mode: 'discard' 丢弃不足一帧的尾部，'zero_pad' 补零至整帧
        :return: (frames, num_frames, processed_len)
        """
        total_len = len(rx_signal)
        if frame_len <= 0:
            raise ValueError("帧长度必须大于 0")
        num_frames = total_len // frame_len
        remainder = total_len % frame_len

        if remainder == 0:
            processed_signal = rx_signal
            num_frames = total_len // frame_len
        else:
            if tail_mode == 'discard':
                processed_signal = rx_signal[:num_frames * frame_len]
                print(f"丢弃尾部 {remainder} 个采样点（不足一帧）")
            elif tail_mode == 'zero_pad':
                pad_len = frame_len - remainder
                if rx_signal.ndim == 1:
                    pad = np.zeros(pad_len, dtype=rx_signal.dtype)
                else:
                    pad = np.zeros((pad_len, rx_signal.shape[1]), dtype=rx_signal.dtype)
                processed_signal = np.concatenate([rx_signal, pad], axis=0)
                num_frames = num_frames + 1
                print(f"尾部补零 {pad_len} 个采样点")
            else:
                raise ValueError("tail_mode 必须为 'discard' 或 'zero_pad'")

        frames = np.split(processed_signal, num_frames, axis=0)
        return frames, num_frames, len(processed_signal)

    def estimate_and_compensate_cfo_coarse(self, rx_signal_dict, tail_mode='zero_pad'):
        self._verification_data(rx_signal_dict)
        rx_signal = rx_signal_dict["signal_stream"]
        fs = self.sample_rate

        frames, num_frames, _ = self._split_into_frames(rx_signal, self.frame_sample_num, tail_mode)
        cfo_list = []
        compensated_frames = []
        global_idx = 0  # 全局采样索引

        for frame in frames:
            sync_field = frame[:self.sync_len_oversample]
            cfo_est = self._estimate_cfo_from_sync(sync_field, fs, self.L_oversample)
            cfo_list.append(cfo_est)
            # 传入全局起始索引
            comp_frame = self.compensate_cfo(frame, fs, cfo_est, start_idx=global_idx)
            compensated_frames.append(comp_frame)
            global_idx += len(frame)   # 更新全局索引（已处理样本数）

        rx_compensated = np.concatenate(compensated_frames, axis=0)
        avg_cfo = np.mean(cfo_list) if cfo_list else 0.0
        self.estimated_cfo = avg_cfo

        result_dict = {
            "signal_stream": rx_compensated,
            "sample_rate_Hz": fs,
            "duration_seconds": len(rx_compensated) / fs,
            "signal_length": len(rx_compensated),
            "padding_bit_num": self.padding_bit_num,
            "estimated_cfo_Hz": avg_cfo,
            "cfo_estimates_Hz": cfo_list,
        }
        return result_dict

    def estimate_and_compensate_cfo_fine(self, rx_signal_dict, tail_mode='zero_pad'):
        self._verification_data(rx_signal_dict)
        rx_signal = rx_signal_dict["signal_stream"]
        fs = self.sample_rate

        frames, num_frames, _ = self._split_into_frames(rx_signal, self.frame_symbol_num, tail_mode)
        cfo_list = []
        compensated_frames = []
        global_idx = 0  # 全局采样索引

        for frame in frames:
            sync_field = frame[:self.sync_len]
            cfo_est = self._estimate_cfo_from_sync(sync_field, fs, self.L)
            cfo_list.append(cfo_est)
            # 传入全局起始索引
            comp_frame = self.compensate_cfo(frame, fs, cfo_est, start_idx=global_idx)
            compensated_frames.append(comp_frame)
            global_idx += len(frame)   # 更新全局索引（已处理样本数）

        rx_compensated = np.concatenate(compensated_frames, axis=0)
        # 计算平均频偏，丢弃最后一个帧的频偏估计
        avg_cfo = np.mean(cfo_list[:-1]) if cfo_list else 0.0   
        self.estimated_cfo = avg_cfo

        result_dict = {
            "signal_stream": rx_compensated,
            "sample_rate_Hz": fs,
            "duration_seconds": len(rx_compensated) / fs,
            "signal_length": len(rx_compensated),
            "padding_bit_num": self.padding_bit_num,
            "estimated_cfo_Hz": avg_cfo,
            "cfo_estimates_Hz": cfo_list,
        }
        return result_dict

# 测试代码
if __name__ == "__main__":
    # # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    print(f"发射信号长度：{len(tx_signal_dict['signal_stream'])}, 接收信号长度：{len(rx_signal_dict['signal_stream'])}")
    print(f"采样率：{rx_signal_dict['sample_rate_Hz']} Hz，时长：{rx_signal_dict['duration_seconds']} 秒")

    # 接收端匹配滤波
    rx_matched_filter = RxMatchedFilter(transmitter)
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)
    # 细同步恢复符号
    fine_sync = FineSync(transmitter)
    rx_sampled_signal_dict = fine_sync.fine_sync(rx_matched_signal_dict)
    print(f"同步偏移：{rx_sampled_signal_dict['sync_offset']}") 
    print(f"同步信号长度：{len(rx_sampled_signal_dict['signal_stream'])}")
    # 粗频偏估计与补偿
    cfo_estimator = CFOEstimator(transmitter)
    result_dict = cfo_estimator.estimate_and_compensate_cfo_coarse(rx_sampled_signal_dict) 
    print(f"估计频偏：{result_dict['cfo_estimates_Hz'][0]:.2f} Hz，真实频偏：{channel.cfo.freq_offset} Hz，估计误差：{abs(result_dict['cfo_estimates_Hz'][0] - channel.cfo.freq_offset):.2f} Hz")

    rx_signal_compensate = result_dict['signal_stream']
    # 估计补偿后频偏
    cfo_est_after = cfo_estimator.estimate_cfo_coarse(rx_signal_compensate, fs=rx_matched_signal_dict['sample_rate_Hz'])
    print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # 下采样
    downsampler = Downsampler(transmitter)
    rx_downsampled_signal_dict = downsampler.recover_symbol(result_dict)
    print(f"下采样后信号长度：{len(rx_downsampled_signal_dict['signal_stream'])}")

    # 细频偏估计与补偿
    result_dict = cfo_estimator.estimate_and_compensate_cfo_fine(rx_downsampled_signal_dict)
    print(f"细频偏估计：{result_dict['cfo_estimates_Hz'][0]:.2f} Hz")
    rx_signal_compensate = result_dict['signal_stream']
    # 估计补偿后频偏
    cfo_est_after = cfo_estimator.estimate_cfo_fine(rx_signal_compensate, fs=rx_downsampled_signal_dict['sample_rate_Hz'])
    print(f"补偿后估计频偏：{cfo_est_after:.2f} Hz")

    # # 相躁功率谱估计
    # phase_noise_seq = cfo_estimator.extract_phase_noise_from_pilots(rx_signal_compensate[:len(transmitter.preamble_gen.sync) * params.get("oversampling")], tx_signal_dict['signal_stream'][:len(transmitter.preamble_gen.sync) * params.get("oversampling")])
    # len_preamble = len(transmitter.preamble_gen.sync) * params.get("oversampling")
    # nperseg = min(256, len_preamble// 8)
    # noverlap = nperseg // 2
    # print(f"前导码信号长度（过采样）：{len_preamble}，nperseg={nperseg}, noverlap={noverlap}")
    # f, psd = cfo_estimator.estimate_phase_noise_psd(phase_noise_seq, pilot_rate_hz=rx_signal_dict['sample_rate_Hz'], nperseg=nperseg, noverlap=noverlap)

    # # 转换为 dBc/Hz (单边带，已由 return_onesided=True 给出)
    # psd_db = 10 * np.log10(psd + 1e-12)

    # # # 绘图
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # # 子图0：相位噪声功率谱密度
    # axes[0].semilogx(f, psd_db)
    # axes[0].grid(True)
    # axes[0].set_xlabel('Frequency (Hz)')
    # axes[0].set_ylabel('Phase Noise PSD (dBc/Hz)')
    # axes[0].set_title('相位噪声功率谱密度')
    # ========== 绘制星座图验证 ==========

    # 取前2048个点以避免过密（若信号很长）
    num_points = max(2048, len(result_dict["signal_stream"]))
    len_preamble = len(transmitter.preamble)
    # 经过CFO补偿前的信号
    orig_signal = rx_downsampled_signal_dict["signal_stream"][len_preamble:len_preamble + num_points]

    # 经过CFO补偿的信号
    cfo_signal = result_dict["signal_stream"][len_preamble:len_preamble + num_points]
    # # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 子图1：补偿前星座图
    axes[0].scatter(orig_signal.real, orig_signal.imag, s=1, alpha=0.6)
    axes[0].set_title("CFO补偿前信号星座图")
    axes[0].set_xlabel("同相分量 (I)")
    axes[0].set_ylabel("正交分量 (Q)")
    axes[0].grid(True)
    axes[0].axis("equal")
    # 子图2：补偿后星座图
    axes[1].scatter(cfo_signal.real, cfo_signal.imag, s=1, alpha=0.6)
    axes[1].set_title("CFO补偿后信号星座图")
    axes[1].set_xlabel("同相分量 (I)")
    axes[1].set_ylabel("正交分量 (Q)")
    axes[1].grid(True)
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     # 初始化参数和发射机
#     params = PHYParams()
#     transmitter = THzTransmitter(params)
#     # 执行完整发射流程
#     tx_signal_dict = transmitter.run()
#     # 初始化信道并生成接收信号
#     channel = THzChannel(params)
#     rx_signal_dict = channel.run(tx_signal_dict)
#     print(f"发射信号长度：{len(tx_signal_dict['signal_stream'])}, 接收信号长度：{len(rx_signal_dict['signal_stream'])}")
#     print(f"采样率：{rx_signal_dict['sample_rate_Hz']} Hz，时长：{rx_signal_dict['duration_seconds']} 秒")

#     # 接收端匹配滤波
#     rx_matched_filter = RxMatchedFilter(transmitter)
#     rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)

#     # ------------------ 正确流程：先粗频偏估计（基于过采样信号） ------------------
#     cfo_estimator = CFOEstimator(transmitter)

#     # 1. 利用已知的过采样 SYNC 波形（发射机提供的本地参考）与匹配滤波后的信号做互相关，
#     #    找到 SYNC 的精确起始位置，这一步模拟实际系统中的粗定时同步。
#     #    在真实场景中，这一步应通过延迟自相关或匹配滤波等粗同步算法完成。
#     local_sync_upsampled = transmitter.sync_upsampled  # 发射机内部存储的过采样 SYNC 序列
#     corr_with_local = np.correlate(rx_matched_signal_dict['signal_stream'], local_sync_upsampled, mode='valid')
#     sync_start = np.argmax(np.abs(corr_with_local))    # 最佳匹配位置（匹配滤波器延迟已包含）
#     print(f"粗定时：{sync_start}")

#     # 2. 截取从 sync_start 开始的完整过采样 SYNC 段
#     fs = rx_matched_signal_dict['sample_rate_Hz']
#     sync_len_oversample = len(local_sync_upsampled)
#     sync_segment = rx_matched_signal_dict['signal_stream'][sync_start : sync_start + sync_len_oversample]

#     # 3. 验证该 SYNC 段的延迟自相关特性（粗频偏估计的核心）
#     L_oversample = len(transmitter.preamble_gen.a128) * transmitter.oversampling  # 重复段长度
#     seg1 = sync_segment[:-L_oversample]
#     seg2 = sync_segment[L_oversample:]
#     prod = np.conj(seg1) * seg2
#     corr_sum = np.sum(prod)
#     phase = np.angle(corr_sum)
#     est_cfo_coarse = phase / (2 * np.pi) * fs / L_oversample

#     # 可视化相关性
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(prod.real, prod.imag, '.', alpha=0.5)
#     plt.axis('equal')
#     plt.grid(True)
#     plt.title('粗频偏估计：共轭乘积散点图')
#     plt.xlabel('实部'), plt.ylabel('虚部')
#     plt.subplot(1, 2, 2)
#     plt.plot([0, np.real(corr_sum)], [0, np.imag(corr_sum)], 'r-o', lw=2)
#     plt.axis('equal')
#     plt.grid(True)
#     plt.title(f'相关和：相位={phase:.4f} rad (估计CFO={est_cfo_coarse:.2f} Hz)')
#     plt.tight_layout()
#     plt.show()

#     print(f"粗频偏估计（采样级验证）：{est_cfo_coarse:.2f} Hz，真实频偏：{channel.cfo.freq_offset:.2f} Hz，"
#           f"误差：{abs(est_cfo_coarse - channel.cfo.freq_offset):.2f} Hz")

#     # 4. 对整个匹配滤波后的信号进行粗频偏补偿
#     #    构建一个临时的字典，信号从 sync_start 开始（补偿整个接收信号）
#     #    更简单的做法是对整个 rx_matched_signal_dict 进行补偿，因为延迟自相关估计出的频偏
#     #    在整个信号上都是适用的。
#     compensated_coarse = cfo_estimator.compensate_cfo(
#         rx_matched_signal_dict['signal_stream'],
#         fs,
#         est_cfo_coarse
#     )
#     # 更新字典
#     rx_coarse_comp_dict = {
#         'signal_stream': compensated_coarse,
#         'sample_rate_Hz': fs,
#         'duration_seconds': rx_matched_signal_dict['duration_seconds'],
#         'signal_length': len(compensated_coarse),
#         'padding_bit_num': rx_matched_signal_dict['padding_bit_num'],
#         'estimated_cfo_Hz': est_cfo_coarse,
#     }

#     # 验证补偿后 SYNC 段的相关性（相位应接近 0）
#     sync_segment_comp = compensated_coarse[sync_start : sync_start + sync_len_oversample]
#     seg1_comp = sync_segment_comp[:-L_oversample]
#     seg2_comp = sync_segment_comp[L_oversample:]
#     prod_comp = np.conj(seg1_comp) * seg2_comp
#     corr_sum_comp = np.sum(prod_comp)
#     phase_comp = np.angle(corr_sum_comp)
#     print(f"补偿后 SYNC 段相关和相位：{phase_comp:.4f} rad （应接近 0）")

#     # ------------------ 细同步（符号恢复） ------------------
#     fine_sync = FineSync(transmitter)
#     # 注意：FineSync 内部会寻找最佳采样点并下采样，输入应是匹配滤波后的过采样信号
#     # 我们这里输入粗补偿后的过采样信号
#     rx_sampled_signal_dict = fine_sync.fine_sync(rx_coarse_comp_dict)
#     print(f"同步后信号长度：{len(rx_sampled_signal_dict['signal_stream'])}")

#     # ------------------ 细频偏估计与补偿（符号速率） ------------------
#     result_fine = cfo_estimator.estimate_and_compensate_cfo_fine(rx_sampled_signal_dict)
#     print(f"细频偏估计：{result_fine['estimated_cfo_Hz']:.2f} Hz")
#     print(f"补偿后残余频偏测试：", end=' ')
#     # 对补偿后信号再进行一次细估计，应接近 0
#     test_after = cfo_estimator.estimate_cfo_fine(result_fine['signal_stream'],
#                                                  fs=result_fine['sample_rate_Hz'])
#     print(f"{test_after:.2f} Hz")









