import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import resample
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.sync.CoarseSync import CoarseSync
from receiver.sync.CFOEstimator import CFOEstimator
from receiver.sync.FineSync import FineSync
from receiver.NoiseEstimator import NoiseEstimator

class ChannelEstimator:
    """
    信道估计器：完全对齐MATLAB逻辑
    核心功能：
    1. 基于CES序列互相关估计时域CIR（冲激响应）
    2. 估计细同步偏移（fine_offset）
    3. 转换为频域信道响应（CSI）
    4. 兼容原有LS/MMSE估计接口
    """
    def __init__(self, tx_preamble, Lh=256, nfft=256):
        self.params = tx_preamble.params
        self.oversampling = tx_preamble.params.get("oversampling")
        self.N_ces = len(tx_preamble.a512) * self.oversampling # a512/b512序列长度（协议固定）
        self.Lh = Lh  # CIR有效长度（多径最大时延扩展）
        self.nfft = nfft  # FFT点数（频域CSI维度）
        self.len_b128 = len(tx_preamble.b128) * self.oversampling  # CES前导b128长度（协议固定）
    
    def _upsample_ces(self, ces_symbol_seq):
        """符号级CES序列上采样（匹配接收信号的采样率）"""
        upsampled_length = len(ces_symbol_seq) * self.oversampling
        upsampled = resample(ces_symbol_seq, upsampled_length)
        return upsampled

    def ces_corr_estimate(self, ces_field, ces_seq):
        """
        核心方法：对齐MATLAB的CES互相关信道估计逻辑
        :param ces_field: 接收的完整CES序列
        :param ces_seq: 本地完整CES序列
        :return: H_est（频域CSI）, fine_offset（细偏移）, h_est（时域CIR）
        """
        # ========== 步骤1：提取本地CES的a512/b512子序列（对齐MATLAB） ==========
        # ces_seq = self._upsample_ces(ces_seq)  # 上采样至采样级
        # 提取a512（跳过前128的b128）
        start_a512 = self.len_b128
        a512 = ces_seq[start_a512 : start_a512 + self.N_ces]
        # 提取b512（a512之后）
        start_b512 = start_a512 + self.N_ces
        b512 = ces_seq[start_b512 : start_b512 + self.N_ces]

        # ========== 步骤2：提取接收CES的a512/b512对应段 ==========
        ra512 = ces_field[start_a512 : start_a512 + self.N_ces]
        rb512 = ces_field[start_b512 : start_b512 + self.N_ces]

        # ========== 步骤3：互相关计算（对齐MATLAB conv + flipud + conj） ==========
        # 互相关：conv(x, flip(conj(y))) 等价于 MATLAB conv(x, flipud(conj(y)))
        ra = np.convolve(ra512, np.flip(np.conj(a512)), mode='full')
        rb = np.convolve(rb512, np.flip(np.conj(b512)), mode='full')
        c = (ra + rb) / (2 * self.N_ces)  # 归一化（对齐MATLAB）

        # ========== 步骤4：相关峰检测（粗峰值位置） ==========
        abs_c = np.abs(c)
        center = np.argmax(abs_c)  # 最大相关峰位置

        # ========== 步骤5：截取峰值附近区间，提取有效峰 ==========
        # 截取center±Lh/2区间（对齐MATLAB startregion/endregion）
        start_region = int(center - self.Lh / 2)
        end_region = int(center + self.Lh / 2)
        # 边界保护（避免下标越界）
        start_region = max(0, start_region)
        end_region = min(len(c)-1, end_region)
        c_region = c[start_region:end_region+1]
        abs_c_region = np.abs(c_region)

        # 找有效峰值（MinPeakHeight=0.25*max(abs(c))）
        peaks, _ = find_peaks(abs_c_region, height=0.25 * abs_c.max())
        
        if len(peaks) == 0:
            # 无有效峰时，取区间内最大值
            peaks = [np.argmax(abs_c_region)]

        # ========== 步骤6：计算细偏移和CIR下标 ==========
        # 映射回原始相关序列的下标（对齐MATLAB idx0）
        idx0 = peaks + start_region
        # 取第一个峰值（主径）计算细偏移（对齐MATLAB fine_offset = idx0(1)-512）
        fine_offset = idx0[0] - self.N_ces
        # CIR相对下标（1-based → 0-based适配Python）
        idx_h = idx0 - idx0[0]

        # ========== 步骤7：构建时域CIR（对齐MATLAB h_est） ==========
        h_est = np.zeros((self.Lh,), dtype=np.complex128)
        # 仅在有效峰值位置填充相关值（边界保护）
        valid_idx = [i for i in idx_h if 0 <= i < self.Lh]
        valid_c = [c[idx0[k]] for k in range(len(idx0)) if 0 <= idx_h[k] < self.Lh]
        h_est[valid_idx] = valid_c

        # ========== 步骤8：转换为频域CSI（对齐MATLAB fftshift(fft(cirEst,nfft))） ==========
        cir_est = h_est
        H_est = np.fft.fftshift(np.fft.fft(cir_est, self.nfft))

        return H_est, fine_offset, h_est


# 测试：完全复现MATLAB逻辑的验证
if __name__ == "__main__":
    params = PHYParams()
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
    
    # 细同步
    fine_sync = FineSync(tx_preamble, search_window=32, min_peak_ratio=0.4) 
    fine_correction = fine_sync.detect_sfd(rx_signal_compensated, coarse_offset, plot_flag=False)
    total_offset = coarse_offset + fine_correction

    print(f"\n=== 同步偏移结果 ===")
    print(f"粗同步偏移：{coarse_offset}")
    print(f"细同步修正量：{fine_correction}")
    print(f"最终整体同步偏移：{total_offset}")

    # 噪声方差估计测试
    sync_field_sync = rx_signal_compensated[total_offset:total_offset + len(tx_preamble.sync) * params.get("oversampling")]
    noise_estimator = NoiseEstimator(tx_preamble)
    # 估计噪声方差
    noise_var_est = noise_estimator.estimate_noise_var_with_gi(sync_field_sync)
    print(f"\n=== 噪声与信噪比结果 ===")
    print(f"估计噪声方差：{noise_var_est:.6f}")

    # 信道估计测试
    # 获取真实信道参数（用于对比）
    nfft = 256  # 与ChannelEstimator默认nfft一致
    channel_estimator = ChannelEstimator(tx_preamble, nfft=nfft)
    
    # 1. 获取真实信道的时域冲激响应h_true
    # 从THzChannel中提取真实CIR（需确保THzChannel类暴露h属性，若未暴露可替换为以下模拟方式）
    # 方式1：若THzChannel有h属性（真实信道CIR）
    h_true = channel.chan_true

    # 2. 计算真实信道的频域响应H_true（对齐MATLAB：fftshift(fft(h,nfft))）
    H_true = np.fft.fftshift(np.fft.fft(h_true, nfft))

    # 3. 提取CES序列并执行信道估计
    # ces_seq = tx_preamble.ces  # 本地CES序列
    ces_seq = tx_signal[
        len(tx_preamble.sync)*params.get("oversampling") + len(tx_preamble.sfd)*params.get("oversampling") :
        len(transmitter.preamble)* params.get("oversampling")
    ]
    ces_field = rx_signal_compensated[
        total_offset + len(tx_preamble.sync)* params.get("oversampling") + len(tx_preamble.sfd)* params.get("oversampling") :
        total_offset + len(transmitter.preamble)* params.get("oversampling")
    ]

    # ========== 执行CES互相关信道估计（对齐MATLAB逻辑） ==========
    H_est, fine_offset, h_est = channel_estimator.ces_corr_estimate(ces_field, ces_seq)

    # ========== 新增：计算频域NMSE（dB）（对齐MATLAB逻辑） ==========
    # norm(x)^2 等价于 np.linalg.norm(x)**2
    numerator = np.linalg.norm(H_true - H_est) ** 2
    denominator = np.linalg.norm(H_true) ** 2 + np.finfo(float).eps  # +eps避免除零
    nmse_freq = numerator / denominator
    nmse_freq_dB = 10 * np.log10(nmse_freq)
    print(f"\n=== 信道估计精度评估 ===")
    print(f"频域NMSE = {nmse_freq_dB:.2f} dB")

    # ========== 输出关键结果 ==========
    print(f"\n=== 信道估计结果 ===")
    print(f"细同步偏移：{fine_offset}")
    print(f"时域CIR非零值位置：{np.where(np.abs(h_est) > 1e-3)[0]}")
    print(f"时域CIR非零值幅值：{np.abs(h_est)[np.where(np.abs(h_est) > 1e-3)]}")
    print(f"频域CSI长度：{len(H_est)}")

    # ========== 重构可视化：使用ax对象绘图（修复use_line_collection报错） ==========
    # 图1：真实CIR的实部
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    markerline, stemlines, baseline = ax1.stem(np.real(h_true), linefmt='b-', basefmt='b-')
    ax1.set_title('CIR real (magnitude)')
    ax1.set_xlabel('n')
    ax1.set_ylabel('|h_real|')
    ax1.grid(alpha=0.3)

    # 图2：频域CSI实部对比（H_true vs H_est）
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(np.real(H_true), 'r-', label='H_true')
    ax2.plot(np.real(H_est), 'b-', label='H_estimation')
    ax2.set_title(f'H-frequency domain (real) {nfft}-fft')
    ax2.set_xlabel('n')
    ax2.set_ylabel('real(H)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 图3：频域CSI虚部对比（H_true vs H_est）
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(np.imag(H_true), 'r-', label='H_true')
    ax3.plot(np.imag(H_est), 'b-', label='H_estimation')
    ax3.set_title(f'H-frequency domain (imag) {nfft}-fft')
    ax3.set_xlabel('n')
    ax3.set_ylabel('imag(H)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 额外：时域CIR幅值对比
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    # 绘制真实CIR
    markerline1, stemlines1, baseline1 = ax4.stem(np.abs(h_true), linefmt='r-', basefmt='r-', label='h_true')
    # 绘制估计CIR
    markerline2, stemlines2, baseline2 = ax4.stem(np.abs(h_est), linefmt='b-', basefmt='b-', label='h_est', markerfmt='bo')
    ax4.set_title('CIR Magnitude Comparison')
    ax4.set_xlabel('n')
    ax4.set_ylabel('|h|')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()