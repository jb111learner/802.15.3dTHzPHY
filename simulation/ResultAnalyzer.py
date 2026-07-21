# import numpy as np
# import matplotlib.pyplot as plt
# from params.PHYParams import PHYParams
# from channel.THzChannel import THzChannel
# from transmitter.Modulator import THzModulator as Modulator
# from transmitter.GIInserter import GIInserter
# from transmitter.DataProcesser import BitStreamProcessor
# from transmitter.Encoder import Encoder 
# from receiver.GIremover import GIRemover
# from receiver.DeModulator import THzDemodulator
# from receiver.Decoder import Decoder
# from receiver.DeScrambler import DeScrambler
# from params.PHYParams import PHYParams
# from transmitter.Scrambler import Scrambler 
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', "KaiTi", "SimSun"]  # 多字体兜底
# plt.rcParams['axes.unicode_minus'] = False  # 强制显示负号
# plt.rcParams['font.family'] = 'sans-serif'

# # 误码率仿真主函数
# def simulate_ber_vs_snr_decode():
#     # 设置仿真参数
#     ebn0_range = np.arange(6.0, 7.0, 0.1)  # Eb/N0范围：6.0——7.0dB，步长1dB
#     modes = np.array(["hdd", "lcc"])
#     num_trials = 1  # 每个SNR点的仿真次数（次数越多结果越稳定）
#     # 初始化基础模块
#     params = PHYParams()
#     params.update(NCBPS=1)
#     params.update(MCS=1) #设置调制方案为BPSK

#     ber_decode_hdd = []
#     ber_decode_lcc = []
    
#     for mode in modes:
#         params.update(decode_mode=mode)
#         # 遍历每个SNR值进行仿真
#         for ebn0_db in ebn0_range:
#             # 线性域转换：SNR_linear = EbN0_linear * spectral_efficiency
#             ebn0_linear = 10 ** (ebn0_db / 10)
#             snr_linear = ebn0_linear * 1
#             snr_db = 10 * np.log10(snr_linear)
#             print(f"正在仿真 Eb/N0 = {ebn0_db} dB ...")
#             params.update(SNRdB=snr_db)  # 更新信噪比参数
#             packet_size = params.get("rs_packet_size")  # 单个数据帧原始大小（bit）
            
#             total_err_decode = 0 # 解码后总错误比特数
#             total_bits = 0       # 总比特数
            
#             for _ in range(num_trials):
#                 # 发射端生成信号
#                 modulator = Modulator(params)
#                 coder = Encoder(params)
#                 assembler = BitStreamProcessor(params)  
#                 x1 = assembler.run()
#                 x2 = coder.encode(x1)
#                 tx_signal_dict = modulator.modulate(x2)
#                 channel = THzChannel(params)
#                 # 信道传输
#                 rx_signal_dict = channel.run(tx_signal_dict)
#                 # 接收端处理
#                 demodulator = THzDemodulator(params)
#                 decoder = Decoder(params)
#                 sigma = np.sqrt(1.00 / (2 * snr_linear))
#                 y1 = demodulator.demodulate(rx_signal_dict, sigma)
#                 rx_data_dict = decoder.decode(y1)
#                 # 统计误码
#                 total_bits = len(x1['signal_stream'])
#                 total_err_decode += np.sum(x1['signal_stream'] != rx_data_dict['signal_stream'])
#             # 计算平均误码率
#             avg_ber_decode = total_err_decode / (num_trials * total_bits)
#             if mode == "hdd":
#                 ber_decode_hdd.append(avg_ber_decode)
#             elif mode == "lcc":
#                 ber_decode_lcc.append(avg_ber_decode)
    
#     # 绘制误码率曲线
#     plt.figure(figsize=(10, 6))
#     plt.semilogy(ebn0_range, ber_decode_hdd, 'o-', color='blue', linewidth=2, markersize=8, label='hdd误码率')
#     plt.semilogy(ebn0_range, ber_decode_lcc, 's-', color='orange', linewidth=2, markersize=8, label='lcc误码率')
#     plt.xlabel('Eb/N0 / dB', fontsize=12)
#     plt.ylabel('误码率 (BER)', fontsize=12)
#     plt.title('太赫兹通信系统误码率随信噪比变化曲线', fontsize=14, fontweight='bold')
#     plt.grid(True, which="both", ls="--", alpha=0.7)
#     plt.legend(fontsize=12)
#     plt.savefig('ber_vs_snr.png', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == "__main__":
#     simulate_ber_vs_snr_decode()

import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from channel.THzChannel import THzChannel
from transmitter.Modulator import THzModulator as Modulator
from transmitter.GIInserter import GIInserter
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Encoder import Encoder 
from receiver.GIremover import GIRemover
from receiver.DeModulator import THzDemodulator
from receiver.Decoder import Decoder
from receiver.DeScrambler import DeScrambler
from params.PHYParams import PHYParams
from transmitter.Scrambler import Scrambler 

# 全局绘图样式配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', "KaiTi", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 6
plt.rcParams['lines.linewidth'] = 2

# 误帧率仿真主函数（仅统计阶段分帧）
def simulate_fer_vs_snr_decode():
    # ===================== 仿真参数配置 =====================
    ebn0_range = np.arange(6.0, 7.0, 0.1)  # Eb/N0范围：6.0~7.0dB，步长0.1dB
    modes = ["hdd", "lcc"]
    num_trials = 1  # 每个SNR点的仿真次数
    
    # 初始化基础模块
    params = PHYParams()
    params.update(NCBPS=1)
    params.update(MCS=1)  # BPSK调制
    packet_bit_size = params.get("rs_packet_size") * 8  # 单帧比特数（字节→比特）
    
    # 初始化误帧率存储数组
    fer_results = {
        "hdd": np.zeros_like(ebn0_range, dtype=np.float64),
        "lcc": np.zeros_like(ebn0_range, dtype=np.float64)
    }
    
    # ===================== 主仿真循环 =====================
    for mode in modes:
        params.update(decode_mode=mode)        
        print(f"\n========== 开始仿真 {mode.upper()} 译码模式 ==========")
        
        for idx, ebn0_db in enumerate(ebn0_range):
            # SNR线性域转换（适配BPSK）
            ebn0_linear = 10 ** (ebn0_db / 10)
            snr_linear = ebn0_linear * 1  # BPSK频谱效率=1
            snr_db = 10 * np.log10(snr_linear)
            params.update(SNRdB=snr_db)
            
            print(f"  Eb/N0 = {ebn0_db:.1f} dB (SNR = {snr_db:.2f} dB) ...")
            
            # 统计变量：错误数据包数/总数据包数
            total_err_packets = 0
            total_packets = 0
            
            for _ in range(num_trials):
                # -------- 发射端：生成长比特流（编译码内部自动分帧） --------
                coder = Encoder(params)
                modulator = Modulator(params)
                assembler = BitStreamProcessor(params)                
                
                # 生成原始长比特流（包含多帧）
                tx_long_dict = assembler.run()
                tx_long_bits = tx_long_dict['signal_stream']
                # 编码（内部自动分帧，无需手动处理）
                tx_encoded_long = coder.encode(tx_long_dict)
                # 调制长比特流
                tx_signal = modulator.modulate(tx_encoded_long)
                
                # -------- 信道传输 --------
                channel = THzChannel(params)
                rx_signal = channel.run(tx_signal)
                
                # -------- 接收端：解调+译码（内部自动分帧） --------
                demodulator = THzDemodulator(params)
                decoder = Decoder(params)
                
                sigma = np.sqrt(1.0 / (2 * snr_linear))  # 噪声标准差
                rx_llrs = demodulator.demodulate(rx_signal, sigma)  # 解调得到LLR长序列
                # 译码（内部自动分帧，无需手动处理）
                rx_decoded_long = decoder.decode(rx_llrs)['signal_stream']
                
                # -------- 仅统计阶段分帧计算FER（核心修正） --------
                # 1. 对齐发射/接收比特流长度（避免截断不一致）
                min_len = min(len(tx_long_bits), len(rx_decoded_long))
                tx_aligned = tx_long_bits[:min_len]
                rx_aligned = rx_decoded_long[:min_len]
                
                # 2. 按单帧大小分帧（仅用于统计，不干预编译码）
                num_frames = min_len // packet_bit_size
                # 截断到整数帧长度
                tx_frames = np.array_split(tx_aligned[:num_frames * packet_bit_size], num_frames)
                rx_frames = np.array_split(rx_aligned[:num_frames * packet_bit_size], num_frames)
                
                # 3. 逐帧统计错误（任意1比特错误=帧错误）
                for tx_frame, rx_frame in zip(tx_frames, rx_frames):
                    frame_error = not np.array_equal(tx_frame, rx_frame)
                    if frame_error:
                        total_err_packets += 1
                    total_packets += 1
            
            # 计算当前Eb/N0下的误帧率（鲁棒性处理：避免除以0）
            fer = total_err_packets / total_packets if total_packets > 0 else 0.0
            fer_results[mode][idx] = fer
            print(f"    误帧率(FER) = {fer:.6f} (总帧数：{total_packets}，错误帧数：{total_err_packets})")
    
    # ===================== 绘图展示 =====================
    plt.figure(figsize=(12, 7))
    
    # 绘制HDD/LCC误帧率曲线
    plt.semilogy(ebn0_range, fer_results["hdd"], 'o-', color='#2E86AB', 
                 label='HDD 硬判决', markerfacecolor='white', markeredgewidth=1.5)
    plt.semilogy(ebn0_range, fer_results["lcc"], 's-', color='#E63946', 
                 label='LCC 软判决', markerfacecolor='white', markeredgewidth=1.5)
    
    # 图表标注与样式
    plt.xlabel('Eb/N0 (dB)', fontsize=14, labelpad=8)
    plt.ylabel('误帧率 (FER)', fontsize=14, labelpad=8)
    plt.title('太赫兹通信系统误帧率随信噪比变化曲线', fontsize=16, fontweight='bold', pad=15)
    
    # 坐标轴范围与刻度
    plt.xlim(ebn0_range.min(), ebn0_range.max())
    plt.xticks(np.arange(ebn0_range.min(), ebn0_range.max()+0.1, 0.1), fontsize=12)
    plt.yticks(fontsize=12)
    
    # 网格、图例、背景
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True)
    plt.gca().set_facecolor('#F8F9FA')
    
    # 保存与显示
    plt.tight_layout()
    plt.savefig('fer_vs_snr.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    simulate_fer_vs_snr_decode()