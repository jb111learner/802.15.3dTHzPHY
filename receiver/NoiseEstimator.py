# import numpy as np
# from params.PHYParams import PHYParams
# from transmitter.THzTransmitter import THzTransmitter
# from channel.THzChannel import THzChannel
# from receiver.MatchedFilter import RxMatchedFilter
# from receiver.sync.CoarseSync import CoarseSync
# from receiver.sync.CFOEstimator import CFOEstimator
# from receiver.sync.FineSync import FineSync
# import matplotlib.pyplot as plt

# class NoiseEstimator:
#     """
#     噪声方差估计器：完全对齐MATLAB逻辑
#     核心原理：利用SYNC序列的128符号重复块结构，通过块间差分消去信号、仅保留噪声，实现无偏噪声方差估计
#     """
#     def __init__(self, tx_preamble):
#         self.oversampling = tx_preamble.params.get("oversampling")
#         self.base_sequences_length = len(tx_preamble.a128) * self.oversampling

#     def estimate_noise_var(self, rx_sync_seq):
#         """
#         估计噪声方差（仅输入接收的SYNC序列，无需理想序列，对齐MATLAB逻辑）
#         :param rx_sync_seq: 同步后的接收SYNC序列（一维数组，复信号/实信号均可）
#         :return: 噪声方差估计值（无偏）
#         """
#         # ========== 步骤1：计算SYNC序列中的完整块数（向下取整） ==========
#         Nblks = np.floor(len(rx_sync_seq) / self.base_sequences_length).astype(int)
#         if Nblks < 2:  # 至少需要2个块才能做差分
#             # 块数不足时退化为直接方差估计（保底逻辑）
#             return np.mean(np.abs(rx_sync_seq) ** 2)
        
#         # ========== 步骤2：重塑为[块长度×块数]的矩阵（对齐MATLAB reshape） ==========
#         # 截取完整块的部分，避免不完整块干扰
#         rx_sync_valid = rx_sync_seq[:self.base_sequences_length * Nblks]
#         sync_blk = np.reshape(rx_sync_valid, (self.base_sequences_length, Nblks))  # 行=块长度，列=块数

#         # ========== 步骤3：按列差分（对齐MATLAB diff(syncBlk,[],2)） ==========
#         # 差分后：相邻块相减，消去SYNC信号，仅保留噪声
#         diff_blk = np.diff(sync_blk, axis=1)  # axis=1对应MATLAB的列维度差分

#         # ========== 步骤4：计算差分后的总功率（对齐MATLAB sum(sum(...))） ==========
#         total_power = np.sum(np.abs(diff_blk) ** 2)

#         # ========== 步骤5：无偏归一化（对齐MATLAB分母 L*(Nblks-1)*2） ==========
#         # 归一化因子说明：
#         # - L*(Nblks-1)：差分后的总样本数
#         # - ×2：复信号实部/虚部分别贡献噪声功率，实信号需删除×2
#         is_complex = np.iscomplexobj(rx_sync_seq)
#         norm_factor = self.base_sequences_length * (Nblks - 1) * (2 if is_complex else 1)
        
#         # ========== 步骤6：计算噪声方差（无偏估计） ==========
#         noise_var = total_power / norm_factor

#         return noise_var

#     def estimate_noise_var_with_gi(self, rx_sync_seq, skip_first=True, skip_last=True):
#         """
#         进阶版：支持跳过第一个块（GI）和最后一个块（对齐MATLAB注释逻辑）
#         :param rx_sync_seq: 接收的SYNC序列
#         :param skip_first: 是否跳过第一个块（作为GI）
#         :param skip_last: 是否跳过最后一个块
#         :return: 噪声方差估计值
#         """
#         Nblks = np.floor(len(rx_sync_seq) / self.base_sequences_length).astype(int)
#         if Nblks < 3:  # 跳过GI后至少需要2个有效块
#             return self.estimate_noise_var(rx_sync_seq)
        
#         # 截取有效块（跳过第一个/最后一个）
#         start_idx = 1 if skip_first else 0
#         end_idx = Nblks - 1 if skip_last else Nblks
#         rx_sync_valid = rx_sync_seq[self.base_sequences_length * start_idx : self.base_sequences_length * end_idx]
        
#         # 复用基础估计逻辑
#         return self.estimate_noise_var(rx_sync_valid)

# # 测试
# if __name__ == "__main__":
#     # 初始化参数和发射机
#     params = PHYParams()
#     transmitter = THzTransmitter(params)
#     tx_signal = transmitter.run()
#     tx_symbols = transmitter.tx_symbols
#     tx_preamble = transmitter.preamble_gen
    
#     # 初始化信道并生成接收信号
#     channel = THzChannel(params)
#     rx_signal = channel.run(tx_signal)
    
#     # 初始化接收端匹配滤波器
#     rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
#     recovered_symbols = rx_matched_filter.matched_filter(rx_signal)
    
#     # 粗同步
#     coarse_sync = CoarseSync(tx_preamble, sync_threshold=0.5, scaling_factor=5)
#     coarse_offset, corr_norm = coarse_sync.detect_sync(recovered_symbols)
    
#     # CFO估计与补偿
#     cfo_estimator = CFOEstimator(tx_preamble, params)
#     sync_start = coarse_offset
#     sync_field = recovered_symbols[sync_start:sync_start + len(tx_preamble.sync) * params.get("oversampling")]
#     cfo_est = cfo_estimator.estimate_cfo(sync_field)
#     rx_signal_compensated = cfo_estimator.compensate_cfo(recovered_symbols, cfo_est)
    
#     # 再次估计频偏（验证补偿效果）
#     sync_field_comp = rx_signal_compensated[sync_start:sync_start + len(tx_preamble.sync) * params.get("oversampling")]
#     cfo_est_again = cfo_estimator.estimate_cfo(sync_field_comp)
    
#     # 结果验证
#     print(f"\n=== 频偏估计与补偿结果 ===")
#     print(f"真实频偏：{channel.cfo.freq_offset} Hz")
#     print(f"估计频偏：{cfo_est:.2f} Hz")
#     print(f"频偏估计误差：{abs(cfo_est - channel.cfo.freq_offset):.2f} Hz") 
#     print(f"补偿后再次估计频偏：{cfo_est_again:.2f} Hz")   
    
#     # 细同步（开启可视化）
#     fine_sync = FineSync(tx_preamble, search_window=32, min_peak_ratio=0.4) 
#     fine_correction = fine_sync.detect_sfd(rx_signal_compensated, coarse_offset, plot_flag=False)
#     total_offset = coarse_offset + fine_correction

#     print(f"\n=== 同步偏移结果 ===")
#     print(f"粗同步偏移：{coarse_offset}")
#     print(f"细同步修正量：{fine_correction}")
#     print(f"最终整体同步偏移：{total_offset}")

#     # 噪声方差估计测试
#     sync_field_sync = rx_signal_compensated[total_offset:total_offset + len(tx_preamble.sync) * params.get("oversampling")]
#     noise_estimator = NoiseEstimator(tx_preamble)
#     # 估计噪声方差
#     noise_var_est = noise_estimator.estimate_noise_var(sync_field_sync)
#     print(f"估计噪声方差：{noise_var_est:.6f}")
import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.sync.CoarseSync import CoarseSync
from receiver.sync.CFOEstimator import CFOEstimator
from receiver.sync.FineSync import FineSync
import matplotlib.pyplot as plt

class NoiseEstimator:
    """
    噪声方差估计器：完全对齐MATLAB逻辑
    核心原理：利用SYNC序列的128符号重复块结构，通过块间差分消去信号、仅保留噪声，实现无偏噪声方差估计
    """
    def __init__(self, tx_preamble):
        self.oversampling = tx_preamble.params.get("oversampling")
        self.base_sequences_length = len(tx_preamble.a128) * self.oversampling

    def estimate_noise_var(self, rx_sync_seq):
        """
        估计噪声方差（仅输入接收的SYNC序列，无需理想序列，对齐MATLAB逻辑）
        :param rx_sync_seq: 同步后的接收SYNC序列（一维数组，复信号/实信号均可）
        :return: 噪声方差估计值（无偏）
        """
        # ========== 步骤1：计算SYNC序列中的完整块数（向下取整） ==========
        Nblks = np.floor(len(rx_sync_seq) / self.base_sequences_length).astype(int)
        if Nblks < 2:  # 至少需要2个块才能做差分
            # 块数不足时退化为直接方差估计（保底逻辑）
            return np.mean(np.abs(rx_sync_seq) ** 2)
        
        # ========== 步骤2：重塑为[块长度×块数]的矩阵（对齐MATLAB reshape） ==========
        # 截取完整块的部分，避免不完整块干扰
        rx_sync_valid = rx_sync_seq[:self.base_sequences_length * Nblks]
        sync_blk = np.reshape(rx_sync_valid, (self.base_sequences_length, Nblks))  # 行=块长度，列=块数

        # ========== 步骤3：按列差分（对齐MATLAB diff(syncBlk,[],2)） ==========
        # 差分后：相邻块相减，消去SYNC信号，仅保留噪声
        diff_blk = np.diff(sync_blk, axis=1)  # axis=1对应MATLAB的列维度差分

        # ========== 步骤4：计算差分后的总功率（对齐MATLAB sum(sum(...))） ==========
        total_power = np.sum(np.abs(diff_blk) ** 2)

        # ========== 步骤5：无偏归一化（对齐MATLAB分母 L*(Nblks-1)*2） ==========
        # 归一化因子说明：
        # - L*(Nblks-1)：差分后的总样本数
        # - ×2：复信号实部/虚部分别贡献噪声功率，实信号需删除×2
        is_complex = np.iscomplexobj(rx_sync_seq)
        norm_factor = self.base_sequences_length * (Nblks - 1) * (2 if is_complex else 1)
        
        # ========== 步骤6：计算噪声方差（无偏估计） ==========
        noise_var = total_power / norm_factor

        return noise_var

    def estimate_noise_var_with_gi(self, rx_sync_seq, skip_first=True, skip_last=True):
        """
        进阶版：支持跳过第一个块（GI）和最后一个块（对齐MATLAB注释逻辑）
        :param rx_sync_seq: 接收的SYNC序列
        :param skip_first: 是否跳过第一个块（作为GI）
        :param skip_last: 是否跳过最后一个块
        :return: 噪声方差估计值
        """
        Nblks = np.floor(len(rx_sync_seq) / self.base_sequences_length).astype(int)
        if Nblks < 3:  # 跳过GI后至少需要2个有效块
            return self.estimate_noise_var(rx_sync_seq)
        
        # 截取有效块（跳过第一个/最后一个）
        start_idx = 1 if skip_first else 0
        end_idx = Nblks - 1 if skip_last else Nblks
        rx_sync_valid = rx_sync_seq[self.base_sequences_length * start_idx : self.base_sequences_length * end_idx]
        
        # 复用基础估计逻辑
        return self.estimate_noise_var(rx_sync_valid)

def calculate_snr(rx_sync_seq, noise_var):
    """
    计算信噪比（SNR）：基于SYNC序列的总功率和估计的噪声方差
    :param rx_sync_seq: 同步后的接收SYNC序列
    :param noise_var: 估计的噪声方差
    :return: snr_linear（线性域SNR）, snr_dB（分贝域SNR）
    """
    # 步骤1：计算SYNC序列的总功率（信号功率 + 噪声功率）
    total_power = np.mean(np.abs(rx_sync_seq) ** 2)
    
    # 步骤2：计算纯信号功率（总功率 - 噪声功率）
    # 注：由于噪声方差=噪声功率（平稳噪声），因此直接相减即可
    signal_power = total_power - noise_var
    
    # 防止信号功率为负（极端低信噪比场景）
    signal_power = max(signal_power, 1e-10)
    
    # 步骤3：计算线性域和分贝域SNR
    snr_linear = signal_power / noise_var
    snr_dB = 10 * np.log10(snr_linear)
    
    return snr_linear, snr_dB

# 测试
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
    
    # 计算并输出信噪比
    snr_linear, snr_dB = calculate_snr(sync_field_sync, noise_var_est)
    print(f"线性域信噪比：{snr_linear:.2f}")
    print(f"分贝域信噪比（SNR）：{snr_dB:.2f} dB")