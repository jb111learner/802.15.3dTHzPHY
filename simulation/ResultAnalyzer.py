import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from core.BaseReceiver import BaseReceiver
from receiver.sync.CoarseSync import CoarseSync
from receiver.sync.FineSync import FineSync
from receiver.sync.CFOEstimator import CFOEstimator
from receiver.ChannelEstimator import ChannelEstimator
from receiver.NoiseEstimator import NoiseEstimator
from transmitter.CPInserter import CPInserter
from receiver.DeModulator import THzDemodulator
from receiver.MatchedFilter import RxMatchedFilter
from receiver.THzReceiver import THzReceiver
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', "KaiTi", "SimSun"]  # 多字体兜底
plt.rcParams['axes.unicode_minus'] = False  # 强制显示负号
plt.rcParams['font.family'] = 'sans-serif'

# 误码率仿真主函数
def simulate_ber_vs_snr():
    # 1. 设置仿真参数
    snr_range = np.arange(-10, 11, 2)  # 信噪比范围：-10~10dB，步长2dB
    ncbps = np.array([2, 4])
    num_trials = 10  # 每个SNR点的仿真次数（次数越多结果越稳定）
    # 初始化基础模块（只需初始化一次）
    params = PHYParams()

    ber_demod_QPSk = []
    ber_decode_QPSk = []
    ber_demod_16QAM = []
    ber_decode_16QAM = []
    
    for mcs in ncbps:
        params.update(NCBPS=mcs)
        # 遍历每个SNR值进行仿真
        for snr_db in snr_range:
            print(f"正在仿真 SNR = {snr_db} dB ...")
            params.update(SNRdB=snr_db)  # 更新信噪比参数
            
            total_err_demod = 0  # 解调后总错误比特数
            total_err_decode = 0 # 解码后总错误比特数
            total_bits = 0       # 总比特数
            
            # 多次仿真取平均
            for _ in range(num_trials):
                transmitter = THzTransmitter(params)
                channel = THzChannel(params)
                # 发射端生成信号
                tx_signal = transmitter.run()
                # 信道传输
                rx_signal = channel.run(tx_signal)
                # 接收端处理
                receiver = THzReceiver(params, transmitter)
                rx_data = receiver.run(rx_signal)
                
                # 统计误码
                total_bits = len(transmitter.coded_bits)
                total_err_demod += np.sum(transmitter.coded_bits != receiver.rx_bits)
                total_err_decode += np.sum(transmitter.data_bits != rx_data)
            
            # 计算平均误码率
            avg_ber_demod = total_err_demod / (num_trials * total_bits)
            avg_ber_decode = total_err_decode / (num_trials * total_bits)
            
            if mcs == 2:
                ber_demod_QPSk.append(avg_ber_demod)
                ber_decode_QPSk.append(avg_ber_decode)
            elif mcs == 4:
                ber_demod_16QAM.append(avg_ber_demod)
                ber_decode_16QAM.append(avg_ber_decode)
    
    # 4. 绘制误码率曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_demod_QPSk, 'o-', color='blue', linewidth=2, markersize=8, label='QPSK解调后误码率')
    plt.semilogy(snr_range, ber_decode_QPSk, 's-', color='orange', linewidth=2, markersize=8, label='QPSK解码后误码率')
    plt.semilogy(snr_range, ber_demod_16QAM, 'o-', color='green', linewidth=2, markersize=8, label='16QAM解调后误码率')
    plt.semilogy(snr_range, ber_decode_16QAM, 's-', color='red', linewidth=2, markersize=8, label='16QAM解码后误码率')
    
    # 设置图表样式
    plt.xlabel('信噪比 (SNR) / dB', fontsize=12)
    plt.ylabel('误码率 (BER)', fontsize=12)
    plt.title('太赫兹通信系统误码率随信噪比变化曲线', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(snr_range)
    plt.xlim(-11, 11)
    
    # 保存图片（可选）
    plt.savefig('ber_vs_snr.png', dpi=300, bbox_inches='tight')
    # 显示图片
    plt.show()

# 执行仿真
if __name__ == "__main__":
    simulate_ber_vs_snr()