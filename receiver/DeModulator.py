import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from params.PHYParams import PHYParams
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class THzDemodulator:
    """保留你原有的解调器类代码（无需修改）"""
    def __init__(self, params):
        self.params = params
        self.MCS = self.params.get("MCS")
        self.NCBPS = self.params.get("NCBPS")
        self.Rate = self.params.get("Rate")
        
        if self.NCBPS not in [1,2,3,4,6] and self.MCS != "0":
            raise ValueError(f"NCBPS仅支持[1,2,3,4,6]")

    def _find_min(self, A, B):
        A_exp = A[:, np.newaxis]
        B_exp = B[np.newaxis, :]
        D = A_exp - B_exp
        M = np.abs(D) ** 2
        y = np.min(M, axis=1)
        return y

    def _dbpsk_demodulate(self, rx_symbols, sigma):
        N0 = (sigma ** 2) / 32
        N = N0 + (N0 / 2) ** 2
        
        L = len(rx_symbols)
        llr = np.zeros(L, dtype=np.float64)
        llr[0] = np.real(rx_symbols[0])
        
        for i in range(1, L):
            llr[i] = 2 * np.real(np.conj(rx_symbols[i]) * rx_symbols[i-1]) / N
        
        llr = -llr
        
        return llr

    def _bpsk_demodulate(self, rx_symbols, sigma):
        L = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L) / 2)
        s = rx_symbols * phase
        llr = -4 * np.real(s) / (sigma ** 2)       
        return llr

    def _qpsk_demodulate(self, rx_symbols, sigma):
        L_sym = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L_sym) / 2)
        s = rx_symbols * phase
        s = s * np.exp(1j * np.pi / 4)
        llr_b0 = -2 * np.sqrt(2) * np.real(s) / (sigma ** 2)
        llr_b1 = -2 * np.sqrt(2) * np.imag(s) / (sigma ** 2)
        
        llr = np.vstack((llr_b0, llr_b1)).T.reshape(-1)
        
        return llr

    def _8psk_demodulate(self, rx_symbols, sigma):
        L_sym = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L_sym) / 2)
        s = rx_symbols * phase
        constellation = np.exp(1j * np.pi * np.arange(8) / 4)
        S1_b0 = np.array([1, (1-1j)/np.sqrt(2), -1j, (-1-1j)/np.sqrt(2)], dtype=np.complex128)
        S0_b0 = np.array([-1, (-1+1j)/np.sqrt(2), 1j, (1+1j)/np.sqrt(2)], dtype=np.complex128)
        S1_b1 = np.array([1j, (1+1j)/np.sqrt(2), 1, (1-1j)/np.sqrt(2)], dtype=np.complex128)
        S0_b1 = np.array([-1j, (-1-1j)/np.sqrt(2), -1, (-1+1j)/np.sqrt(2)], dtype=np.complex128)
        S1_b2 = np.array([1j, (-1+1j)/np.sqrt(2), -1j, (1-1j)/np.sqrt(2)], dtype=np.complex128)
        S0_b2 = np.array([1, (1+1j)/np.sqrt(2), -1, (-1-1j)/np.sqrt(2)], dtype=np.complex128)
        
        xopt1_b0 = self._find_min(s, S1_b0)
        xopt0_b0 = self._find_min(s, S0_b0)
        llr_b0 = (xopt1_b0 - xopt0_b0) / (sigma ** 2)
        
        xopt1_b1 = self._find_min(s, S1_b1)
        xopt0_b1 = self._find_min(s, S0_b1)
        llr_b1 = (xopt1_b1 - xopt0_b1) / (sigma ** 2)
        
        xopt1_b2 = self._find_min(s, S1_b2)
        xopt0_b2 = self._find_min(s, S0_b2)
        llr_b2 = (xopt1_b2 - xopt0_b2) / (sigma ** 2)
        
        llr = np.vstack((llr_b0, llr_b1, llr_b2)).T.reshape(-1)
    
        return llr

    def _16qam_demodulate(self, rx_symbols, sigma):
        L_sym = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L_sym) / 2)
        s = rx_symbols * phase
        
        P = np.sqrt(10)
        br = np.real(s)
        bi = np.imag(s)
        T = np.vstack((br, bi)).T
        
        A = np.zeros_like(T)
        B = np.zeros_like(T)
        
        mask = T >= 2 / P
        A[mask] = 0.8 - 8 * T[mask] / P
        mask = (T < 2 / P) & (T >= -2 / P)
        A[mask] = -4 * T[mask] / P
        mask = T < -2 / P
        A[mask] = -0.8 - 8 * T[mask] / P
        
        mask = T >= 0
        B[mask] = -0.8 + 4 * T[mask] / P
        mask = T < 0
        B[mask] = -0.8 - 4 * T[mask] / P
        
        llr = np.vstack((A[:,0], B[:,0], A[:,1], B[:,1])).T.reshape(-1)
        llr = llr / (sigma ** 2)
        
        return llr

    def _64qam_demodulate(self, rx_symbols, sigma):
        L_sym = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L_sym) / 2)
        s = rx_symbols * phase
        
        P = np.sqrt(42)
        br = np.real(s)
        bi = np.imag(s)
        T = np.vstack((br, bi)).T
        
        A = np.zeros_like(T)
        B = np.zeros_like(T)
        C = np.zeros_like(T)
        
        mask = T >= 6 / P
        A[mask] = 48/42 - 16 * T[mask] / P
        mask = (T < 6 / P) & (T >= 4 / P)
        A[mask] = 24/42 - 12 * T[mask] / P
        mask = (T < 4 / P) & (T >= 2 / P)
        A[mask] = 8/42 - 8 * T[mask] / P
        mask = (T < 2 / P) & (T >= -2 / P)
        A[mask] = -4 * T[mask] / P
        mask = (T < -2 / P) & (T >= -4 / P)
        A[mask] = -8/42 - 8 * T[mask] / P
        mask = (T < -4 / P) & (T >= -6 / P)
        A[mask] = -24/42 - 12 * T[mask] / P
        mask = T < -6 / P
        A[mask] = -48/42 - 16 * T[mask] / P
        
        mask = T >= 6 / P
        B[mask] = -40/42 + 8 * T[mask] / P
        mask = (T < 6 / P) & (T >= 2 / P)
        B[mask] = -16/42 + 4 * T[mask] / P
        mask = (T < 2 / P) & (T >= 0)
        B[mask] = -24/42 + 8 * T[mask] / P
        mask = (T < 0) & (T >= -2 / P)
        B[mask] = -24/42 - 8 * T[mask] / P
        mask = (T < -2 / P) & (T >= -6 / P)
        B[mask] = -16/42 - 4 * T[mask] / P
        mask = T < -6 / P
        B[mask] = -40/42 - 8 * T[mask] / P
        
        mask = T >= 4 / P
        C[mask] = -24/42 + 4 * T[mask] / P
        mask = (T < 4 / P) & (T >= 0)
        C[mask] = 8/42 - 4 * T[mask] / P
        mask = (T < 0) & (T >= -4 / P)
        C[mask] = 8/42 + 4 * T[mask] / P
        mask = T < -4 / P
        C[mask] = -24/42 - 4 * T[mask] / P
        
        llr = np.vstack((A[:,0], B[:,0], C[:,0], A[:,1], B[:,1], C[:,1])).T.reshape(-1)
        llr = llr / (sigma ** 2)
        
        return llr

    def demodulate(self, rx_symbols, sigma):
        if self.MCS == "0":
            llr = self._dbpsk_demodulate(rx_symbols, sigma)
        else:
            if self.NCBPS == 1:
                llr = self._bpsk_demodulate(rx_symbols, sigma)
            elif self.NCBPS == 2:
                llr = self._qpsk_demodulate(rx_symbols, sigma)
            elif self.NCBPS == 3:
                llr = self._8psk_demodulate(rx_symbols, sigma)
            elif self.NCBPS == 4:
                llr = self._16qam_demodulate(rx_symbols, sigma)
            elif self.NCBPS == 6:
                llr = self._64qam_demodulate(rx_symbols, sigma)
            else:
                raise ValueError(f"不支持的NCBPS：{self.NCBPS}")
        
        if np.isclose(self.Rate, 13/16):
            llr = llr / 2
        
        return llr

    def llr_to_bits(self, llr):
        bits = (llr < 0).astype(np.uint8)
        return bits

def snr_to_sigma(snr_db, signal_power=1.0):
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(signal_power / (2 * snr_linear))
    return sigma

# ========== 新增：全链路波形分析函数 ==========
def analyze_link_waveforms(tx_symbols, tx_signal, rx_signal, y_matched, y_cp_del, params):
    """
    可视化全链路关键节点波形/星座图，定位误码原因
    tx_symbols: 发射复符号
    rx_signal: 信道输出接收信号
    y_matched: 匹配滤波后信号
    y_cp_del: 去CP后信号
    params: PHY参数
    """
    # 1. 绘制发射/接收时域波形（取前1000个采样点，避免图形过大）
    plt.figure(figsize=(15, 10))
    
    # 子图1：发射符号实部（时域）
    plt.subplot(3, 3, 1)
    plt.plot(np.real(tx_symbols[:min(100, len(tx_symbols))]), 'b-', label='发射符号实部')
    plt.plot(np.imag(tx_symbols[:min(100, len(tx_symbols))]), 'r--', label='发射符号虚部')
    plt.title('发射复符号（前100个）')
    plt.xlabel('符号索引')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 子图2：接收信号时域波形
    plt.subplot(3, 3, 2)
    plt.plot(np.real(rx_signal[:min(1000, len(rx_signal))]), 'b-', label='接收信号实部')
    plt.plot(np.imag(rx_signal[:min(1000, len(rx_signal))]), 'r--', label='接收信号虚部')
    plt.title('信道输出接收信号（前1000采样点）')
    plt.xlabel('采样索引')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 子图3：匹配滤波后信号
    plt.subplot(3, 3, 3)
    plt.plot(np.real(y_matched[:min(100, len(y_matched))]), 'b-', label='匹配滤波实部')
    plt.plot(np.imag(y_matched[:min(100, len(y_matched))]), 'r--', label='匹配滤波虚部')
    plt.title('匹配滤波后符号（前100个）')
    plt.xlabel('符号索引')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 子图4：去CP后信号
    plt.subplot(3, 3, 4)
    plt.plot(np.real(y_cp_del[:min(100, len(y_cp_del))]), 'b-', label='去CP实部')
    plt.plot(np.imag(y_cp_del[:min(100, len(y_cp_del))]), 'r--', label='去CP虚部')
    plt.title('去循环前缀后符号（前100个）')
    plt.xlabel('符号索引')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 2. 绘制星座图（核心定位误码）
    # 子图5：发射符号星座图
    plt.subplot(3, 3, 5)
    plt.scatter(np.real(tx_symbols), np.imag(tx_symbols), c='blue', s=20, alpha=0.8)
    plt.title('发射符号星座图')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    # 子图6：匹配滤波后星座图
    plt.subplot(3, 3, 6)
    plt.scatter(np.real(y_matched), np.imag(y_matched), c='green', s=20, alpha=0.8)
    plt.title('匹配滤波后星座图')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    # 子图7：去CP后星座图
    plt.subplot(3, 3, 7)
    plt.scatter(np.real(y_cp_del), np.imag(y_cp_del), c='red', s=20, alpha=0.8)
    plt.title('去CP后星座图')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.axis('equal')
    plt.grid(True)
    
    # 3. 绘制功率分布（检测幅度异常）
    # 子图8：发射符号功率分布
    plt.subplot(3, 3, 8)
    tx_power = np.abs(tx_symbols) ** 2
    plt.hist(tx_power, bins=20, color='blue', alpha=0.7)
    plt.title(f'发射符号功率分布（均值：{np.mean(tx_power):.4f}）')
    plt.xlabel('功率')
    plt.ylabel('频次')
    plt.grid(True)
    
    # 子图9：去CP后符号功率分布
    plt.subplot(3, 3, 9)
    cp_del_power = np.abs(y_cp_del) ** 2
    plt.hist(cp_del_power, bins=20, color='red', alpha=0.7)
    plt.title(f'去CP后符号功率分布（均值：{np.mean(cp_del_power):.4f}）')
    plt.xlabel('功率')
    plt.ylabel('频次')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ========== 关键参数打印（定位异常） ==========
    print("\n===== 全链路关键参数分析 =====")   
    # 2. 功率归一化检查
    tx_power_mean = np.mean(np.abs(tx_symbols)**2)
    cp_del_power_mean = np.mean(np.abs(y_cp_del)**2)
    print(f"发射符号平均功率：{tx_power_mean:.6f}（理想值=1）")
    print(f"去CP后符号平均功率：{cp_del_power_mean:.6f}（理想值≈1）")
    if abs(tx_power_mean - 1) > 0.1:
        print(f"⚠️ 发射符号功率未归一化！均值={tx_power_mean}")
    if abs(cp_del_power_mean - tx_power_mean) > 0.2:
        print(f"⚠️ 链路功率衰减/放大异常！发射={tx_power_mean}, 接收={cp_del_power_mean}")
    
    # 3. 噪声强度检查
    snr_db = params.get("SNRdB")
    sigma_calc = snr_to_sigma(snr_db, tx_power_mean)
    print(f"配置SNR：{snr_db}dB → 理论噪声标准差：{sigma_calc:.6f}")
    # 计算实际噪声功率（假设发射功率=1）
    noise_power = np.mean(np.abs(rx_signal - tx_signal)**2)
    actual_snr = 10 * np.log10(tx_power_mean / noise_power)
    print(f"实际链路SNR：{actual_snr:.2f}dB（与配置{snr_db}dB对比）")
    if abs(actual_snr - snr_db) > 5:
        print(f"⚠️ 实际SNR与配置偏差过大！配置={snr_db}dB, 实际={actual_snr}dB")

if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)
    tx_signal = transmitter.run()
    tx_symbols = transmitter.tx_symbols
    
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal = channel.run(tx_signal)
    print(f"发射信号长度：{len(tx_signal)}, 接收信号长度：{len(rx_signal)}")

    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    y_matched = rx_matched_filter.recover_symbols(rx_signal)
    print(f"发射符号长度：{len(tx_symbols)}, 接收符号长度：{len(y_matched)}")

    # 去除GI
    y_cp_del = transmitter.cp_inserter.remove_cp(y_matched)
    print(f"去除循环前缀后符号长度：{len(y_cp_del)}")
    
    # ========== 新增：全链路波形分析 ==========
    analyze_link_waveforms(tx_symbols, tx_signal,rx_signal, y_matched, y_cp_del, params)
    
    # 解调
    snr_db = params.get("SNRdB")    
    # 修正sigma计算（匹配信号功率）
    tx_power_mean = np.mean(np.abs(tx_symbols)**2)
    print(f"发射符号平均功率用于sigma计算：{tx_power_mean:.6f}")
    sigma = snr_to_sigma(snr_db, tx_power_mean)  # 改用归一化后的sigma计算
    
    demodulator = THzDemodulator(params)
    llr = demodulator.demodulate(y_cp_del, sigma)

    # LLR转比特
    rec_bits = demodulator.llr_to_bits(llr)

    # 截断比特到相同长度（避免维度不匹配）
    min_bit_len = min(len(transmitter.coded_bits), len(rec_bits))
    tx_bits_trunc = transmitter.coded_bits[:min_bit_len]
    rec_bits_trunc = rec_bits[:min_bit_len]
    
    print(f"\n===== 解调结果分析 =====")
    print(f"接收比特长度：{len(rec_bits)}, 原始比特长度：{len(transmitter.coded_bits)}")
    print(f"原始比特（前10）：{tx_bits_trunc[0:10]}")
    print(f"解调比特（前10）：{rec_bits_trunc[0:10]}")
    print(f"比特错误数：{np.sum(tx_bits_trunc != rec_bits_trunc)}")
    print(f"误码率：{np.sum(tx_bits_trunc != rec_bits_trunc)/min_bit_len:.6f}")
    
    # 额外：LLR分布分析
    plt.figure(figsize=(8, 4))
    plt.hist(llr, bins=50, color='purple', alpha=0.7)
    plt.title(f'LLR值分布（均值：{np.mean(llr):.4f}，标准差：{np.std(llr):.4f}）')
    plt.xlabel('LLR值')
    plt.ylabel('频次')
    plt.grid(True)
    plt.show()