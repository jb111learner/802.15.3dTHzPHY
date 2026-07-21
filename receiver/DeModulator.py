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
    """解调器类"""
    def __init__(self, params):
        self.params = params
        self.MCS = self.params.get("MCS")
        self.NCBPS = self.params.get("NCBPS")
        self.Rate = self.params.get("Rate")
        
        if self.MCS == 1:
            if self.NCBPS not in [1,2,3,4,6,8] and self.MCS != "0":
                raise ValueError(f"NCBPS仅支持[1,2,3,4,6,8] for QAM")
        elif self.MCS == 2:
            # APSK 缓存（只在 APSK 模式时生效）
            rings = self.params.get("APSK_RINGS")
            offsets = self.params.get("APSK_PHASE_OFFSETS")
            if rings is None or offsets is None:
                rings, offsets = self._default_apsk_rings(2 ** int(self.NCBPS))
            total_pts = sum(int(n) for n, _ in rings)
            if total_pts != int(2 ** int(self.NCBPS)):
                raise ValueError(f"APSK_RINGS 点数总和({total_pts}) 必须等于 2**NCBPS({self.NCBPS})")
            if len(offsets) != len(rings):
                raise ValueError(f"APSK_PHASE_OFFSETS 数量({len(offsets)}) 必须等于 rings 数量({len(rings)})") 

            const, bits, k = self._build_apsk_table_from(2 ** int(self.NCBPS), rings, offsets)
            self._apsk_const = const
            self._apsk_bits = bits
            self._apsk_k = int(k)      

            # 预切片，避免 demodulate 时反复筛选
            self._apsk_S0 = []
            self._apsk_S1 = []
            for j in range(self._apsk_k):
                mask1 = (bits[:, j] == 1)
                mask0 = ~mask1
                self._apsk_S1.append(const[mask1])
                self._apsk_S0.append(const[mask0])
        
        # 输出参数      
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.bit_length = None  # 比特长度
        self.padding_bit_num = 0   # 补零的比特数

    @staticmethod
    def _gray_int(x: np.ndarray) -> np.ndarray:
        return x ^ (x >> 1)

    @staticmethod
    def _int_to_bits(idx: np.ndarray, k: int) -> np.ndarray:
        idx = idx.astype(np.int64)
        return ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)
    
    def _find_min(self, A, B):
        """
        A: (N,) complex
        B: (M,) complex
        return: (N,) min |A - B|^2
        """
        A_exp = A[:, np.newaxis]
        B_exp = B[np.newaxis, :]
        D = A_exp - B_exp
        M = np.abs(D) ** 2
        return np.min(M, axis=1)
        
    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 符号流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 符号长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        

        if self.MCS == "0":
            self.sample_rate = data_dict["sample_rate_Hz"]
            self.bit_length = data_dict["signal_length"]
        else:
            self.sample_rate = data_dict["sample_rate_Hz"] * self.NCBPS
            self.bit_length = data_dict["signal_length"] * self.NCBPS
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]  

    # -------------------- APSK (A/B) --------------------
    def _default_apsk_rings(self, M: int):
        """
        默认 rings/offsets（可按你需要调整半径比）
        """
        if M == 16:
            rings = [(4, 1.0), (12, 2.85)]
        elif M == 32:
            rings = [(4, 1.0), (12, 2.84), (16, 5.27)]
        elif M == 64:
            rings = [(4, 1.0), (12, 2.73), (20, 4.52), (28, 6.31)]
        else:
            raise ValueError(f"未提供 APSK_RINGS，且无 {M}APSK 默认 rings。请手动设置 APSK_RINGS/APSK_PHASE_OFFSETS")
        offsets = [0.0] * len(rings)
        return rings, offsets 

    def _build_apsk_table_from(self, M: int, rings, offsets):
        """
        构造 APSK constellation + (binary index -> constellation) + labels
        - 先按 ring 顺序拼接 const
        - 归一化到 Es=1
        - 用 Gray permute：symbols_by_b[b] = const[gray(b)]
        - bits_by_b[b] = binary bits of b
        """
        k = int(np.log2(M))
        if 2 ** k != M:
            raise ValueError("APSK_M 必须是 2 的幂（16/32/64/256...）")

        const = []
        for (n, r), phi0 in zip(rings, offsets):
            ang = phi0 + 2 * np.pi * np.arange(int(n)) / int(n)
            const.append(float(r) * np.exp(1j * ang))
        const = np.concatenate(const).astype(np.complex128)

        # Es 归一化到 1
        const = const / np.sqrt(np.mean(np.abs(const) ** 2))

        b = np.arange(M, dtype=np.int64)
        g = self._gray_int(b)                 # gray index
        symbols_by_b = const[g]               # binary index -> symbol
        bits_by_b = self._int_to_bits(b, k)   # binary index -> bits label
        return symbols_by_b, bits_by_b, k    

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

    @staticmethod
    def _gray_labels_16():
        """
        返回 16 个电平对应的 Gray 码比特标签（MSB->LSB）
        """
        i = np.arange(16, dtype=np.int32)
        g = i ^ (i >> 1)
        b0 = (g >> 3) & 1
        b1 = (g >> 2) & 1
        b2 = (g >> 1) & 1
        b3 = (g >> 0) & 1
        return np.vstack([b0, b1, b2, b3]).T.astype(np.int8)

    @staticmethod
    def _min_masked(dist2, mask):
        tmp = np.where(mask[None, :], dist2, np.inf)
        return np.min(tmp, axis=1)

    def _256qam_demodulate(self, rx_symbols, sigma):
        L_sym = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L_sym) / 2)
        s = rx_symbols * phase

        P = np.sqrt(170.0)
        levels = (2 * np.arange(16) - 15) / P
        labels = self._gray_labels_16()

        br = np.real(s)
        bi = np.imag(s)

        distI = (br[:, None] - levels[None, :]) ** 2
        distQ = (bi[:, None] - levels[None, :]) ** 2

        llrI, llrQ = [], []
        for kk in range(4):
            mask1 = labels[:, kk] == 1
            mask0 = ~mask1

            d1I = self._min_masked(distI, mask1)
            d0I = self._min_masked(distI, mask0)
            llrI.append((d1I - d0I) / (sigma ** 2))

            d1Q = self._min_masked(distQ, mask1)
            d0Q = self._min_masked(distQ, mask0)
            llrQ.append((d1Q - d0Q) / (sigma ** 2))

        llr = np.vstack((llrI[0], llrI[1], llrI[2], llrI[3],
                         llrQ[0], llrQ[1], llrQ[2], llrQ[3])).T.reshape(-1)
        return llr

    # -------------------- APSK demodulate (cached) --------------------
    def _apsk_demodulate(self, rx_symbols, sigma):
        """
        Max-Log LLR:
          llr_j = (min_{s in S1} |y-s|^2 - min_{s in S0} |y-s|^2) / sigma^2
        与你的 llr_to_bits 规则一致：llr<0 => bit=1
        """

        L_sym = len(rx_symbols)
        phase = np.exp(-1j * np.pi * np.arange(L_sym) / 2)
        y = rx_symbols * phase

        llrs = []
        for j in range(self._apsk_k):
            S1 = self._apsk_S1[j]
            S0 = self._apsk_S0[j]
            d1 = self._find_min(y, S1)
            d0 = self._find_min(y, S0)
            llrs.append((d1 - d0) / (sigma ** 2))

        return np.vstack(llrs).T.reshape(-1)

    def demodulate(self, rx_symbols_dict, sigma):
        self._verification_data(rx_symbols_dict)
        rx_symbols = rx_symbols_dict["signal_stream"]
        if self.MCS == 0:
            llr = self._dbpsk_demodulate(rx_symbols, sigma)
        elif self.MCS == 2:
            llr = self._apsk_demodulate(rx_symbols, sigma)
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
            elif self.NCBPS == 8:
                llr = self._256qam_demodulate(rx_symbols, sigma)
            else:
                raise ValueError(f"不支持的NCBPS：{self.NCBPS}")
        
        if np.isclose(self.Rate, 13/16):
            llr = llr / 2

        if len(llr) != self.bit_length:
            raise ValueError(f"解调后比特长度不匹配:预期长度={self.bit_length}, 实际长度={len(llr)}")

        result_dict = {
            "signal_stream": llr,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.bit_length,
            "padding_bit_num": self.padding_bit_num
        }
        
        return result_dict

    def llr_to_bits(self, llr_dict):
        if round(llr_dict["sample_rate_Hz"] * llr_dict["duration_seconds"]) != llr_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")   
        if len(llr_dict["signal_stream"]) != llr_dict["signal_length"]:
            raise ValueError(f"输入LLR长度不匹配:预期长度={llr_dict['signal_length']}, 实际长度={len(llr_dict['signal_stream'])}")     
        bits = (llr_dict["signal_stream"] < 0).astype(np.uint8)
        result_dict = {
            "signal_stream": bits,
            "sample_rate_Hz": llr_dict["sample_rate_Hz"],
            "duration_seconds": llr_dict["duration_seconds"],
            "signal_length": llr_dict["signal_length"],
            "padding_bit_num": llr_dict["padding_bit_num"]
        }
        return result_dict

def snr_to_sigma(snr_db, signal_power=1.0):
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(signal_power / (2 * snr_linear))
    return sigma

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
    from receiver.MatchedFilter import RxMatchedFilter
    from receiver.sync.CoarseSync import CoarseSync
    from receiver.sync.FineSync import FineSync
    from receiver.CFOEstimator import CFOEstimator
    from receiver.Downsampler import Downsampler
    from receiver.ChannelEstimator import ChannelEstimator
    from receiver.RxOFDMProcesser import RxOFDMProcesser
    from receiver.Equalizer import FreqDomainEqualizer
    from channel.THzChannel import THzChannel

    for mode in ["ofdm"]:
        print(f"\n{'='*55}")
        print(f"     {mode.upper()} 模式 BER 测试")
        print(f"{'='*55}")

        params = PHYParams()
        params.update(link_mode=mode)
        tx = THzTransmitter(params)
        tx.run()
        ch = THzChannel(params)
        rx = ch.run(tx.tx_signal_dict)

        sps = params.get("oversampling")

        # ---- RX: MF -> CoarseSync -> CFOc -> FineSync -> DS -> CFOf ----
        mf = RxMatchedFilter(tx)
        mf_out = mf.matched_filter(rx)

        cs = CoarseSync(tx)
        sync_out = cs.detect_sync(mf_out)

        # ---- RX: MF -> CoarseSync -> CFOc -> FineSync -> DS -> CFOf ----
        cfo = CFOEstimator(tx)
        mf_fs = sync_out["sample_rate_Hz"]
        cfo_c = cfo.estimate_cfo_coarse(sync_out["signal_stream"], fs=mf_fs)
        sig_c = cfo.compensate_cfo(sync_out["signal_stream"], fs=mf_fs, cfo_est=cfo_c)
        cd = dict(sync_out); cd["signal_stream"] = sig_c; cd["signal_length"] = len(sig_c)

        fsync = FineSync(tx)
        fout = fsync.fine_sync(cd)

        ds = Downsampler(tx)
        sout = ds.recover_symbol(fout)

        cfo_f = cfo.estimate_cfo_fine(sout["signal_stream"], fs=sout["sample_rate_Hz"])
        rx_cfo = cfo.compensate_cfo(sout["signal_stream"],
                                     fs=sout["sample_rate_Hz"], cfo_est=cfo_f)
        rd = dict(sout); rd["signal_stream"] = rx_cfo

        if mode == "ofdm":
            # ---- OFDM: RxOFDMProcesser (内置导频LS+时域加窗+相位插值) ----
            rx_ofdm = RxOFDMProcesser(tx)
            ofdm_r = rx_ofdm.ofdm_demodulate(rd, H_init_list=None)
            rx_sym_out = ofdm_r["signal_stream"]
            sym_fs = ofdm_r["sample_rate_Hz"]
        else:
            # ---- SC-FDE: ChannelEstimator → FreqDomainEqualizer ----
            ch_est = ChannelEstimator(tx)
            ch_result = ch_est.channel_estimate(rd)
            eq = FreqDomainEqualizer(tx)
            eq_out = eq.equalize(ch_result)
            rx_sym_out = eq_out["signal_stream"]
            sym_fs = eq_out["sample_rate_Hz"]

        # 构造 DeModulator 兼容的 dict
        rx_symbols_dict = {
            "signal_stream": rx_sym_out,
            "sample_rate_Hz": sym_fs,
            "signal_length": len(rx_sym_out),
            "duration_seconds": len(rx_sym_out) / sym_fs,
            "padding_bit_num": rd["padding_bit_num"],
        }

        # ---- 解调 ----
        snr_db = params.get("SNRdB")
        sigma = snr_to_sigma(snr_db)
        demodulator = THzDemodulator(params)
        llr_dict = demodulator.demodulate(rx_symbols_dict, sigma)
        rec_bits_dict = demodulator.llr_to_bits(llr_dict)

        # ---- BER ----
        tx_bits = tx.coded_bits_dict["signal_stream"]
        rx_bits = rec_bits_dict["signal_stream"]
        min_len = min(len(tx_bits), len(rx_bits))
        tx_b = tx_bits[:min_len]; rx_b = rx_bits[:min_len]
        ber = np.sum(tx_b != rx_b) / min_len
        print(f"  coded bits: tx={len(tx_bits)} rx={len(rx_bits)}")
        print(f"  BER = {ber:.2e}  ({np.sum(tx_b != rx_b)} / {min_len})")

    # ---- 可视化 ----
    plt.figure(figsize=(8, 4))
    plt.hist(llr_dict["signal_stream"], bins=50, color='purple', alpha=0.7)
    plt.title(f'{mode.upper()} LLR Distribution '
              f'(mean={np.mean(llr_dict["signal_stream"]):.3f} '
              f'std={np.std(llr_dict["signal_stream"]):.3f})')
    plt.xlabel('LLR'); plt.ylabel('count'); plt.grid(True)
    plt.show()

    print("\nDone")