import numpy as np
from receiver.IQRealParallelCalibrator import IQRealParallelCalibrator

# 测试用导入
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.CFOEstimator import CFOEstimator
from receiver.sync.FineSync import FineSync
from receiver.ChannelEstimator import ChannelEstimator
from receiver.NoiseEstimator import NoiseEstimator
from receiver.Equalizer import FreqDomainEqualizer
from receiver.MatchedFilter import RxMatchedFilter
from receiver.Downsampler import Downsampler
import matplotlib.pyplot as plt


class IQCompensator:
    """
    RX IQ 不平衡补偿器（符号率、多帧支持）

    利用每帧的 CES 字段（收发双方均已知）估计 RX 端 I/Q 损伤滤波器，
    设计后补偿滤波器并应用到整个帧（含前导码），使下游信道估计/均衡
    模块看到干净的 I/Q 信号。

    算法核心（来自 IQRealParallelCalibrator）：
        rI = e1·yI + e2·yQ + dI       ← RX 损伤模型
        rQ = e3·yI + e4·yQ + dQ
        ycI = f1·rI + f2·rQ + cI      ← 后补偿模型
        ycQ = f3·rI + f4·rQ + cQ

    在 THzReceiver 流程中的推荐位置：
        fine CFO 补偿之后、信道估计之前
    """

    def __init__(self, transmitter, filter_len=5, ridge_lambda=0.0,
                 compensation_mode='per_frame'):
        """
        :param transmitter: THzTransmitter 实例（提供 preamble/CES 结构信息）
        :param filter_len: IQ 损伤 FIR 滤波器阶数（≥1，默认 5）
        :param ridge_lambda: LS 估计的岭回归正则化系数（0 表示伪逆，>0 启用岭回归）
        :param compensation_mode: 'per_frame' 逐帧独立估计/补偿，
                                  'first_frame' 仅第一帧估计，其余复用
        """
        self.params = transmitter.params
        self.transmitter = transmitter

        # IQ 补偿超参数
        self.filter_len = self._validate_filter_len(filter_len)
        self.ridge_lambda = float(ridge_lambda)
        if not np.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda 必须为非负有限值")
        if compensation_mode not in ('per_frame', 'first_frame'):
            raise ValueError("compensation_mode 必须为 'per_frame' 或 'first_frame'")
        self.compensation_mode = compensation_mode

        # 从前导码结构中提取关键长度信息
        self.sync_len = len(transmitter.sync)       # SYNC 符号数
        self.sfd_len = len(transmitter.sfd)         # SFD 符号数
        self.tx_ces = transmitter.ces               # 本地已知 CES 序列（符号级）
        self.ces_len = len(self.tx_ces)             # CES 符号数
        self.preamble_len = len(transmitter.preamble)  # 前导码总符号数

        # 计算完整帧长度（符号级），用于分帧
        subframe_len = self.params.get("subframe_length")
        gi_len = self.params.get("gi_length")
        self.frame_symbol_num = (
            (subframe_len + gi_len) * self.params.get("subframe_num")
            + self.preamble_len
        )

        # 输出缓存
        self.sample_rate = None
        self.duration = None
        self.signal_length = None
        self.padding_bit_num = 0
        self.compensation_filters = None   # 每帧补偿滤波器列表

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_filter_len(filter_len):
        """校验并转换 filter_len 为整数"""
        try:
            filter_len = int(filter_len)
        except (TypeError, ValueError) as exc:
            raise ValueError("filter_len 必须为正整数") from exc
        if filter_len <= 0:
            raise ValueError("filter_len 必须为正整数")
        return filter_len

    def _verification_data(self, data_dict):
        """校验输入数据字典的完整性"""
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds",
            "signal_length", "padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        # 采样率 × 时长 ≅ 信号长度 的一致性校验
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")

        self.sample_rate = data_dict["sample_rate_Hz"]
        self.signal_length = data_dict["signal_length"]
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]

    def _split_into_frames(self, rx_signal, frame_len, tail_mode='discard'):
        """
        将一维信号分割为帧列表

        :param rx_signal: 1D 复信号
        :param frame_len: 每帧符号数
        :param tail_mode: 'discard' 丢弃不足一帧的尾部，
                          'zero_pad' 补零至整帧
        :return: (frames_list, num_frames, processed_len)
        """
        total_len = len(rx_signal)
        if frame_len <= 0:
            raise ValueError("帧长度必须大于 0")

        num_frames = total_len // frame_len
        remainder = total_len % frame_len

        if remainder == 0:
            processed_signal = rx_signal
        else:
            if tail_mode == 'discard':
                processed_signal = rx_signal[:num_frames * frame_len]
                print(f"[IQCompensator] 丢弃尾部 {remainder} 个采样点（不足一帧）")
            elif tail_mode == 'zero_pad':
                pad_len = frame_len - remainder
                if rx_signal.ndim == 1:
                    pad = np.zeros(pad_len, dtype=rx_signal.dtype)
                else:
                    pad = np.zeros((pad_len, rx_signal.shape[1]), dtype=rx_signal.dtype)
                processed_signal = np.concatenate([rx_signal, pad], axis=0)
                num_frames += 1
                print(f"[IQCompensator] 尾部补零 {pad_len} 个采样点")
            else:
                raise ValueError("tail_mode 必须为 'discard' 或 'zero_pad'")

        frames = np.split(processed_signal, num_frames, axis=0)
        return frames, num_frames, len(processed_signal)

    def _estimate_and_design_filters(self, rx_ces):
        """
        从单帧的 RX CES 和本地 TX CES 估计损伤并设计后补偿滤波器

        :param rx_ces: 接收端受损 CES 序列（1D 复信号）
        :return: dict {f1, f2, f3, f4, cI, cQ} 补偿滤波器系数
        """
        # 步骤1：估计 RX 端损伤滤波器 e1~e4 及 DC 偏移 dI/dQ
        rx_est = IQRealParallelCalibrator.estimate_rx_impairment_filters(
            y=self.tx_ces,
            r=rx_ces,
            filter_len=self.filter_len,
            ridge_lambda=self.ridge_lambda
        )

        # 步骤2：设计后补偿 FIR 滤波器 f1~f4
        post_filters = IQRealParallelCalibrator.design_postcompensation_filters(
            rx_est['e1'], rx_est['e2'], rx_est['e3'], rx_est['e4'],
            filter_len=self.filter_len
        )

        # 步骤3：设计后补偿 DC 偏移 cI/cQ
        dc_offset = IQRealParallelCalibrator.design_postcompensation_dc_offset(
            post_filters['f1'], post_filters['f2'],
            post_filters['f3'], post_filters['f4'],
            rx_est['dI'], rx_est['dQ']
        )

        return {
            'f1': post_filters['f1'],
            'f2': post_filters['f2'],
            'f3': post_filters['f3'],
            'f4': post_filters['f4'],
            'cI': dc_offset['cI'],
            'cQ': dc_offset['cQ'],
            # 附加诊断信息
            'residual_norm_I': rx_est['residual_norm_I'],
            'residual_norm_Q': rx_est['residual_norm_Q'],
            'design_condition_number': rx_est['design_condition_number'],
        }

    # ------------------------------------------------------------------
    # 主入口：IQ 补偿
    # ------------------------------------------------------------------

    def compensate(self, data_dict, tail_mode='discard'):
        """
        多帧 IQ 不平衡补偿入口

        流程：
          1. 将输入信号按帧分割
          2. 逐帧提取 CES → 估计损伤 → 设计补偿滤波器
          3. 对整帧应用后补偿（含前导码 + 数据部分）
          4. 拼接所有帧输出

        :param data_dict: 输入信号字典（符号率，细 CFO 补偿之后）
        :param tail_mode: 尾部处理方式 'discard' / 'zero_pad'
        :return: result_dict 包含补偿后的信号流及补偿滤波器信息
        """
        self._verification_data(data_dict)
        rx_signal = data_dict["signal_stream"]
        fs = self.sample_rate

        # 分帧
        frames, num_frames, processed_len = self._split_into_frames(
            rx_signal, self.frame_symbol_num, tail_mode
        )
        if num_frames == 0:
            raise ValueError("[IQCompensator] 信号不包含完整帧，无法进行 IQ 补偿")

        compensated_frames = []
        filter_info_list = []

        # 用于 'first_frame' 模式：缓存第一帧的滤波器
        first_filters = None

        for i, frame in enumerate(frames):
            # 提取接收帧中的 CES 字段
            ces_start = self.sync_len + self.sfd_len
            rx_ces = frame[ces_start : ces_start + self.ces_len]

            # 根据模式决定是重新估计还是复用
            if self.compensation_mode == 'per_frame' or (
                self.compensation_mode == 'first_frame' and i == 0
            ):
                filters = self._estimate_and_design_filters(rx_ces)
                if self.compensation_mode == 'first_frame':
                    first_filters = filters
            else:
                filters = first_filters

            # 对整帧应用后补偿（前导码 + 数据全部补偿，
            # 使下游信道估计模块看到 IQ 干净的 CES）
            compensated_frame = IQRealParallelCalibrator.apply_postcompensation(
                frame,
                filters['f1'], filters['f2'],
                filters['f3'], filters['f4'],
                filters['cI'], filters['cQ']
            )

            compensated_frames.append(compensated_frame)
            filter_info_list.append(filters)

        # 缓存补偿滤波器
        self.compensation_filters = filter_info_list

        # 拼接所有帧
        rx_compensated = np.concatenate(compensated_frames)

        # 更新时长和长度
        self.duration = len(rx_compensated) / fs
        self.signal_length = len(rx_compensated)

        result_dict = {
            "signal_stream": rx_compensated,
            "sample_rate_Hz": fs,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
            # IQ 补偿特有的输出字段
            "iq_compensation_filters": filter_info_list,
        }
        return result_dict


# ======================================================================
# 测试代码：验证 IQ 补偿在接收链路中的效果
# ======================================================================
if __name__ == "__main__":
    # ---- 1. 配置参数（开启 RX 端 IQ 不平衡） ----
    params = PHYParams()
    params.apply_dict({
        "enable_iq_imbalance": True,
        "iq_imbalance_position": "rx",
        "iq_imbalance_model": "fd",           # 频率相关 IQ 不平衡
        "iq_gain_imbalance_db": 2.0,
        "iq_phase_imbalance_deg": 5.0,
        "iq_gI_taps": [1.0, 0.08, -0.03],
        "iq_gQ_taps": [1.0, -0.12, 0.04],
        "iq_power_normalize": False,
    })

    # ---- 2. 发射 → 信道 → 接收前端（匹配滤波 + 同步 + CFO） ----
    transmitter = THzTransmitter(params)
    tx_signal_dict = transmitter.run()

    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    print(f"发射信号长度: {len(tx_signal_dict['signal_stream'])}, "
          f"接收信号长度: {len(rx_signal_dict['signal_stream'])}")

    # 匹配滤波
    rx_matched_filter = RxMatchedFilter(transmitter)
    rx_matched_signal_dict = rx_matched_filter.matched_filter(rx_signal_dict)

    # 帧定时同步
    fine_sync = FineSync(transmitter)
    rx_sampled_signal_dict = fine_sync.fine_sync(rx_matched_signal_dict)
    print(f"同步偏移: {rx_sampled_signal_dict['sync_offset']}")

    # 粗频偏估计与补偿
    cfo_estimator = CFOEstimator(transmitter)
    coarse_result = cfo_estimator.estimate_and_compensate_cfo_coarse(rx_sampled_signal_dict)
    print(f"粗频偏估计: {coarse_result['cfo_estimates_Hz'][0]:.2f} Hz")

    # 下采样
    downsampler = Downsampler(transmitter)
    rx_downsampled_dict = downsampler.recover_symbol(coarse_result)

    # 细频偏估计与补偿
    fine_cfo_dict = cfo_estimator.estimate_and_compensate_cfo_fine(rx_downsampled_dict)
    print(f"细频偏估计: {fine_cfo_dict['cfo_estimates_Hz'][0]:.2f} Hz")

    # ---- 3. ★ IQ 补偿（在细 CFO 之后、信道估计之前）★ ----
    iq_compensator = IQCompensator(
        transmitter,
        filter_len=5,
        ridge_lambda=0.0,
        compensation_mode='per_frame'
    )
    iq_compensated_dict = iq_compensator.compensate(fine_cfo_dict)

    # 打印诊断信息
    for i, filt in enumerate(iq_compensated_dict['iq_compensation_filters']):
        print(f"\n--- 第 {i+1} 帧 IQ 补偿滤波器 ---")
        print(f"  f1: {filt['f1']}")
        print(f"  f2: {filt['f2']}")
        print(f"  f3: {filt['f3']}")
        print(f"  f4: {filt['f4']}")
        print(f"  DC (cI, cQ): ({filt['cI']:.4f}, {filt['cQ']:.4f})")
        print(f"  残差范数 (I/Q): ({filt['residual_norm_I']:.4f}, {filt['residual_norm_Q']:.4f})")

    # ---- 4. 信道估计（在 IQ 补偿后） ----
    channel_estimator = ChannelEstimator(transmitter)
    rx_estimated_dict = channel_estimator.channel_estimate(iq_compensated_dict)
    H_est_list = rx_estimated_dict['channel_freq_response']

    # ---- 5. 噪声估计 ----
    noise_estimator = NoiseEstimator(transmitter)
    noise_result = noise_estimator.noise_estimate(rx_estimated_dict)

    # ---- 6. 均衡 ----
    equalizer = FreqDomainEqualizer(transmitter)
    equalized_dict = equalizer.equalize(noise_result)

    # ---- 7. 可视化对比 ----
    num_points = min(2048, len(equalized_dict['signal_stream']))
    eq_signal = equalized_dict['signal_stream'][:num_points]

    # IQ 补偿前后对比结果
    rx_iq_compensated = iq_compensated_dict['signal_stream'][:num_points]
    rx_no_iq = fine_cfo_dict['signal_stream'][:num_points]

    # 参考：无 IQ 补偿的均衡结果
    # （跳过 IQ 补偿，直接用细 CFO 结果走信道估计 → 均衡）
    rx_no_iq_est = channel_estimator.channel_estimate(fine_cfo_dict)
    rx_no_iq_noise = noise_estimator.noise_estimate(rx_no_iq_est)
    eq_no_iq_dict = equalizer.equalize(rx_no_iq_noise)
    eq_no_iq_signal = eq_no_iq_dict['signal_stream'][:num_points]

    # 原始发射星座点
    tx_data = transmitter.modulated_data_dict['signal_stream'][:num_points]

    fig, axes = plt.subplots(1, 5, figsize=(30, 5))

    axes[0].plot(tx_data.real, tx_data.imag, 'x', markersize=3, alpha=0.6)
    axes[0].set_title("TX 原始星座图")
    axes[0].set_xlabel("I"); axes[0].set_ylabel("Q")
    axes[0].axis('equal'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eq_no_iq_signal.real, eq_no_iq_signal.imag, '.', markersize=2, alpha=0.6)
    axes[1].set_title("无 IQ 补偿 → 均衡后星座图")
    axes[1].set_xlabel("I"); axes[1].set_ylabel("Q")
    axes[1].axis('equal'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eq_signal.real, eq_signal.imag, '.', markersize=2, alpha=0.6)
    axes[2].set_title("IQ 补偿后 → 均衡后星座图")
    axes[2].set_xlabel("I"); axes[2].set_ylabel("Q")
    axes[2].axis('equal'); axes[2].grid(True, alpha=0.3)

    axes[3].plot(rx_no_iq.real, rx_no_iq.imag, 'r.', markersize=2, alpha=0.6)
    axes[3].set_title("IQ补偿前接收信号星座图")
    axes[3].set_xlabel("I"); axes[3].set_ylabel("Q")
    axes[3].axis('equal'); axes[3].grid(True, alpha=0.3)

    axes[4].plot(rx_iq_compensated.real, rx_iq_compensated.imag, 'g.', markersize=2, alpha=0.6)
    axes[4].set_title("IQ补偿后接收信号星座图")
    axes[4].set_xlabel("I"); axes[4].set_ylabel("Q")
    axes[4].axis('equal'); axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 频域 CSI 对比（第一帧）
    if H_est_list:
        H_iq = H_est_list[0]
        H_no_iq = rx_no_iq_est['channel_freq_response']
        if isinstance(H_no_iq, list):
            H_no_iq = H_no_iq[0]

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        nfft = len(H_iq)
        axes2[0].plot(np.abs(H_iq), 'b-', label='w/ IQ补偿')
        axes2[0].plot(np.abs(H_no_iq), 'r--', label='w/o IQ补偿')
        axes2[0].set_title(f"频域 CSI 幅值对比 ({nfft}-FFT)")
        axes2[0].set_xlabel("子载波索引"); axes2[0].set_ylabel("|H|")
        axes2[0].legend(); axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(np.angle(H_iq), 'b-', label='w/ IQ补偿')
        axes2[1].plot(np.angle(H_no_iq), 'r--', label='w/o IQ补偿')
        axes2[1].set_title(f"频域 CSI 相位对比 ({nfft}-FFT)")
        axes2[1].set_xlabel("子载波索引"); axes2[1].set_ylabel("∠H (rad)")
        axes2[1].legend(); axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("\n===== IQ 补偿测试完成 =====")
