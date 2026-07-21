import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler
from transmitter.Encoder import Encoder
from transmitter.Modulator import THzModulator
from utils.Seq_gen import Generator
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class TxOFDMProcesser:
    """
    OFDM处理器类：按子帧分帧 → 插入块状导频 → 串并转换 → IFFT → 加CP → 并串转换
    """

    def __init__(self, params):
        self.subwave_num = params.get("subwave_num")                   # 子载波数 (N_fft = 512)
        self.subframe_ofdm_num = params.get("subframe_ofdm_num")       # 每个子帧的OFDM符号总数 (48)
        self.pilot_block_indexes = params.get("pilot_block_indexes")   # 块状导频索引 [0, 16, 32]

        # 每个子帧中数据OFDM符号个数 = 总数 - 导频数
        self.num_data_ofdm_per_subframe = self.subframe_ofdm_num - len(self.pilot_block_indexes)  # 45

        # 导频序列生成（512点BPSK，填充整个块状导频OFDM符号）
        self.seq_gen = Generator(params)
        self.pilot_seq, _ = self.seq_gen.generate_512_sequences()      # a512作为导频序列

        # 输出参数
        self.sample_rate = None
        self.duration = None
        self.symbol_length = None
        self.padding_bit_num = 0
        self.frame_symbol_num = None  # 每帧符号数  
        self.frame_num = None  # 帧数

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性，计算输出参数

        :param data_dict: 输入数据字典，包含：
            "signal_stream": 复数调制符号流 (ndarray, complex)
            "sample_rate_Hz": 符号速率 (Hz)
            "duration_seconds": 信号时长 (s)
            "signal_length": 符号流长度
            "padding_bit_num": 补零比特数
        """
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num","frame_symbol_num","frame_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")

        # 校验采样率与时长一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")

        # 校验数据长度是每子帧数据符号数的整数倍
        if data_dict["frame_num"] != data_dict["signal_length"] // data_dict["frame_symbol_num"]:
            raise ValueError("输入数据字典中的帧数与符号长度不匹配")
        
        #  校验输入字典参数是否与物理层参数一致
        if data_dict["frame_symbol_num"] != self.num_data_ofdm_per_subframe * self.subwave_num:
            raise ValueError("输入数据字典中的每帧符号数与物理层参数不匹配")

        self.symbol_length = (data_dict["frame_symbol_num"] + len(self.pilot_block_indexes) * self.subwave_num) * data_dict["frame_num"]
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.duration = self.symbol_length / self.sample_rate
        self.padding_bit_num = data_dict["padding_bit_num"]
        self.frame_symbol_num = data_dict["frame_symbol_num"] + len(self.pilot_block_indexes) * self.subwave_num
        self.frame_num = data_dict["frame_num"]

    def ofdm_process(self, signal_dict):
        """
        OFDM发射处理主流程：

        1. 按子帧分帧：将调制符号 reshape 为 (num_subframes, 45, 512)
        2. 插入块状导频：在 pilot_block_indexes 位置插入 pilot_seq
        3. 串并转换 + IFFT：每列一个512点OFDM符号
        4. 添加循环前缀 (CP)
        5. 并串转换 (列优先 / Fortran顺序)

        :param signal_dict: 输入数据字典（格式同 _verification_data）
        :return: result_dict — 包含 OFDM 时域信号及元信息
        """
        self._verification_data(signal_dict)
        data_stream = signal_dict["signal_stream"]

        N_SC = self.subwave_num                        # 512
        N_DATA = self.num_data_ofdm_per_subframe       # 45
        N_SYM = self.subframe_ofdm_num                 # 48
        N_SUB = self.frame_num

        # ==================== 1. 分帧：reshape 为 3D ====================
        # data_3d: (num_subframes, 45, 512)
        #   轴0: 子帧索引
        #   轴1: 子帧内数据OFDM符号索引 (0..44)
        #   轴2: 子载波索引 (0..511)
        data_3d = data_stream.reshape(N_SUB, N_DATA, N_SC)

        # ==================== 2. 插入块状导频 ====================
        # 创建完整OFDM网格 (num_subframes, 48, 512)
        ofdm_grid = np.zeros((N_SUB, N_SYM, N_SC), dtype=np.complex128)

        # 导频列索引 [0, 16, 32] — 填入 pilot_seq（广播到所有子帧）
        for pilot_idx in self.pilot_block_indexes:
            ofdm_grid[:, pilot_idx, :] = self.pilot_seq  # broadcast: (512,) -> (N_SUB, 512)

        # 数据列索引 [1..15, 17..31, 33..47] — 共45列
        data_col_indices = np.setdiff1d(np.arange(N_SYM), self.pilot_block_indexes)
        ofdm_grid[:, data_col_indices, :] = data_3d

        # ==================== 3. 串并转换 + IFFT ====================
        # (N_SUB, 48, 512) → (512, N_SYM, N_SUB) → (512, N_SUB * 48)
        # transpose(2,1,0): 轴2(子载波)→第0维，轴1(符号)→第1维，轴0(子帧)→第2维
        # reshape后每列 = 一个OFDM符号的全部512子载波
        ofdm_2d = ofdm_grid.transpose(2, 1, 0).reshape(N_SC, -1)
        # 每列是一个OFDM符号（频域），512点IFFT（正交归一化，功率守恒）
        ifft_out = np.fft.ifft(ofdm_2d, axis=0, norm='ortho')  # (512, total_ofdm_symbols)

        # ==================== 4. 并串转换 (列优先) ====================
        tx_signal = ifft_out.ravel(order='F')

        # ortho模式下IFFT功率守恒：输入功率=输出功率（≈1.0），无需额外归一化

        # ==================== 校验输出长度 ====================
        if len(tx_signal) != self.symbol_length:
            raise ValueError(
                f"OFDM处理后信号长度不匹配: 预期长度={self.symbol_length}, "
                f"实际长度={len(tx_signal)}"
            )

        result_dict = {
            "signal_stream": tx_signal,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.symbol_length,
            "padding_bit_num": self.padding_bit_num,
            "frame_symbol_num": self.frame_symbol_num,
            "frame_num": self.frame_num,
        }
        return result_dict


if __name__ == "__main__":
    params = PHYParams()
    p1 = BitStreamProcessor(params)
    res1 = p1.run()
    scrambler = Scrambler(params)
    scrambled_data_dict = scrambler.scramble(res1)
    coder = Encoder(params)
    encoded_data_dict = coder.encode(scrambled_data_dict)
    modulator = THzModulator(params)
    symbols_dict = modulator.modulate(encoded_data_dict)
    print(f"符号长度：{symbols_dict['signal_length']}")
    print(f"采样率：{symbols_dict['sample_rate_Hz']} Hz，时长：{symbols_dict['duration_seconds']} 秒")
    print(f"补零数量：{symbols_dict['padding_bit_num']} bit")

    ofdm_processer = TxOFDMProcesser(params)
    symbols_ofdm_dict = ofdm_processer.ofdm_process(symbols_dict)

    tx_signal = symbols_ofdm_dict["signal_stream"]
    N_fft = ofdm_processer.subwave_num
    gi_len = ofdm_processer.gi_length
    base_len = N_fft + gi_len            # 544
    num_blocks = len(tx_signal) // base_len
    num_subframes = ofdm_processer.num_subframes

    print(f"\n===== OFDM 发射处理结果 =====")
    print(f"子帧数量: {num_subframes}")
    print(f"OFDM符号总数: {num_blocks} (= {num_subframes}子帧 × 48符号/子帧)")
    print(f"OFDM输出信号长度: {len(tx_signal)} (= {num_blocks} × {base_len})")
    print(f"输出采样率: {symbols_ofdm_dict['sample_rate_Hz']} Hz")
    print(f"输出时长: {symbols_ofdm_dict['duration_seconds']:.6e} s")

    # ==================== 1. 理想解调测试 ====================
    print(f"\n===== 1. 理想解调验证 =====")
    # 发送端原始调制符号
    tx_symbols = symbols_dict["signal_stream"]
    # 提取数据OFDM符号（跳过导频符号）
    rx_data_symbols = []
    pilot_block_set = set(ofdm_processer.pilot_block_indexes)

    for b in range(num_blocks):
        block_idx_in_subframe = b % ofdm_processer.subframe_ofdm_num
        start = b * base_len
        # 取 CP 后的 N_fft 个样点 → FFT 回频域
        block = tx_signal[start + gi_len : start + gi_len + N_fft]
        rx_block_freq = np.fft.fft(block, norm='ortho')

        if block_idx_in_subframe in pilot_block_set:
            # 导频符号：验证 pilot_seq
            pilot_error = rx_block_freq - ofdm_processer.pilot_seq
            pilot_mse = np.mean(np.abs(pilot_error) ** 2)
            if b < 3:  # 只打印前3个导频符号
                print(f"  导频符号 #{b} (子帧内索引{block_idx_in_subframe}): MSE vs pilot_seq = {pilot_mse:.2e}")
        else:
            # 数据符号：收集用于MSE比较
            rx_data_symbols.append(rx_block_freq)

    rx_data_symbols = np.concatenate(rx_data_symbols)
    # 截取与发送符号相同长度进行比较
    compare_len = min(len(tx_symbols), len(rx_data_symbols))
    error = rx_data_symbols[:compare_len] - tx_symbols[:compare_len]
    mse = np.mean(np.abs(error) ** 2)
    print(f"数据符号理想解调 MSE: {mse:.2e} (应接近 0)")

    # ==================== 2. 星座图对比 ====================
    print(f"\n===== 2. 星座图 =====")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(tx_symbols.real[:500], tx_symbols.imag[:500], 'b.', markersize=2)
    axes[0].set_title('发送端星座图 (前500符号)')
    axes[0].axis('equal')
    axes[0].grid(True)

    axes[1].plot(rx_data_symbols.real[:500], rx_data_symbols.imag[:500], 'r.', markersize=2)
    axes[1].set_title('接收端理想解调星座图 (前500符号)')
    axes[1].axis('equal')
    axes[1].grid(True)
    fig.suptitle('OFDM 理想解调验证')
    plt.tight_layout()
    plt.show()

    # ==================== 3. PAPR 统计 ====================
    print(f"\n===== 3. PAPR CCDF =====")
    papr_vals = []
    for b in range(num_blocks):
        blk = tx_signal[b * base_len : b * base_len + base_len]
        power = np.abs(blk) ** 2
        papr = np.max(power) / np.mean(power)
        papr_vals.append(10 * np.log10(papr))

    papr_vals = np.array(papr_vals)
    papr_vals_sorted = np.sort(papr_vals)
    ccdf = 1.0 - np.arange(len(papr_vals_sorted)) / len(papr_vals_sorted)

    plt.figure(figsize=(8, 5))
    plt.semilogy(papr_vals_sorted, ccdf)
    plt.title('PAPR CCDF (每个OFDM符号)')
    plt.xlabel('PAPR [dB]')
    plt.ylabel('CCDF')
    plt.grid(True)
    plt.show()

    print(f"\n===== 所有验证完成 =====")
