import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler 
from transmitter.Encoder import Encoder
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class THzModulator:
    """
    太赫兹通信调制器类：支持多种调制方式
    """
    # -------------------- utils --------------------
    @staticmethod
    def _gray_int(x: np.ndarray) -> np.ndarray:
        """binary int -> gray int"""
        return x ^ (x >> 1)

    @staticmethod
    def _gray_to_binary(g: np.ndarray) -> np.ndarray:
        """Gray int -> Binary int（通用写法）"""
        b = g.copy()
        shift = 1
        while True:
            t = b >> shift
            if np.all(t == 0):
                break
            b ^= t
            shift <<= 1
        return b

    @staticmethod
    def _int_to_bits(idx: np.ndarray, k: int) -> np.ndarray:
        """idx shape (N,) -> bits shape (N,k), MSB->LSB"""
        idx = idx.astype(np.int64)
        return ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)

    # -------------------- init --------------------
    def __init__(self, params):
        self.params = params
        self.MCS = self.params.get("MCS")
        self.NCBPS = self.params.get("NCBPS")

        # 这些 scale 和你当前实现对齐（用于 QAM/PSK 的基础缩放）
        self.const_scale = {
            1: 1.0,          # BPSK
            2: np.sqrt(2),   # QPSK
            3: 1.0,          # 8PSK
            4: np.sqrt(10),  # 16QAM
            6: np.sqrt(42),  # 64QAM
            8: np.sqrt(170)  # 256QAM
        }

        if self.MCS == 1:
            if self.NCBPS not in self.const_scale:
                raise ValueError(f"NCBPS仅支持{list(self.const_scale.keys())}，当前={self.NCBPS}")
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

            self._apsk_symbols_by_b, self._apsk_bits_by_b, self._apsk_k = self._build_apsk_table_from(2 ** int(self.NCBPS), rings, offsets)  

        # 输出参数 
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.symbol_length = None  # 符号长度
        self.padding_bit_num = 0   # 补零的比特数


    # -------------------- APSK (A/B) --------------------
    def _default_apsk_rings(self, M: int):
        """
        若用户没提供 APSK_RINGS / APSK_PHASE_OFFSETS，则给出常用默认配置
        你可按需要替换半径比
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
        
    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "bit_stream": 比特流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "bit_length": 比特长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "bit_stream", "sample_rate_Hz", "duration_seconds", "bit_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["bit_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        if self.MCS != "0" and data_dict["bit_length"] % self.NCBPS != 0:
            raise ValueError("输入比特流长度必须是NCBPS的整数倍")
        

        if self.MCS == "0":
            self.sample_rate = data_dict["sample_rate_Hz"]
            self.symbol_length = data_dict["bit_length"]
        else:
            self.sample_rate = data_dict["sample_rate_Hz"] / self.NCBPS
            self.symbol_length = data_dict["bit_length"] / self.NCBPS
        self.duration = data_dict["duration_seconds"]
        self.padding_bit_num = data_dict["padding_bit_num"]  


    def _dbpsk_modulate(self, bits):
        bits = bits.astype(np.int8)
        L = len(bits)
        d = np.zeros(L, dtype=np.complex128)
        c = 2 * bits - 1  # 0→-1，1→1
        d[0] = c[0]
        for i in range(1, L):
            d[i] = d[i-1] * c[i]
        # # 强制功率归一化到1
        # d = d / np.sqrt(np.mean(np.abs(d)**2))
        return d

    def _bpsk_modulate(self, bits):
        bits = bits.astype(np.int8)
        L = len(bits)
        c = 2 * bits - 1
        phase = np.exp(1j * np.pi * np.arange(L) / 2)
        d = c.astype(np.complex128) * phase
        # # 强制功率归一化到1
        # d = d / np.sqrt(np.mean(np.abs(d)**2))
        return d

    def _qpsk_modulate(self, bits):
        bits = bits.astype(np.int8)
        c = bits.reshape(-1, 2)
        L_sym = len(c)
        # MATLAB对齐的QPSK映射：pi/4旋转补偿 + 基础缩放
        real_part = 2 * c[:, 0] - 1
        imag_part = 2 * c[:, 1] - 1
        s = (real_part + 1j * imag_part) * np.exp(-1j * np.pi / 4) / self.const_scale[2]
        # pi/2旋转
        phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
        d = s * phase
        # # 强制功率归一化到1（核心修正）
        # d = d / np.sqrt(np.mean(np.abs(d)**2))
        return d

    def _8psk_modulate(self, bits):
        bits = bits.astype(np.int8)     
        c = bits.reshape(-1, 3)
        L_sym = len(c)
        c1, c2, c3 = c[:, 0], c[:, 1], c[:, 2]
        # MATLAB原版8PSK相位映射公式
        phase_factor = (c1 - 3*c2 - c3 - 2*c1*c2 + 2*c2*c3 + 2*c1*c3 + 4*c1*c2*c3 + 4) / 4
        s = np.exp(1j * np.pi * phase_factor)
        # pi/2旋转
        phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
        d = s * phase
        # # 强制功率归一化到1
        # d = d / np.sqrt(np.mean(np.abs(d)**2))
        return d

    def _16qam_modulate(self, bits):
        bits = bits.astype(np.int8)     
        c = bits.reshape(-1, 4)
        L_sym = len(c)
        c1, c2, c3, c4 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        real_part = 4*c1 - 2 - (2*c1 - 1)*(2*c2 - 1)
        imag_part = 4*c3 - 2 - (2*c3 - 1)*(2*c4 - 1)
        s = (real_part + 1j * imag_part) / self.const_scale[4]
        phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
        d = s * phase
        # # 强制功率归一化到1
        # d = d / np.sqrt(np.mean(np.abs(d)**2))
        return d

    def _64qam_modulate(self, bits):
        bits = bits.astype(np.int8)    
        c = bits.reshape(-1, 6)
        L_sym = len(c)
        c1, c2, c3, c4, c5, c6 = c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
        real_part = 8*c1 - 4 - (2*c1 - 1)*(4*c2 - 2) + (2*c1 - 1)*(2*c2 - 1)*(2*c3 - 1)
        imag_part = 8*c4 - 4 - (2*c4 - 1)*(4*c5 - 2) + (2*c4 - 1)*(2*c5 - 1)*(2*c6 - 1)
        s = (real_part + 1j * imag_part) / self.const_scale[6]
        phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
        d = s * phase
        # # 强制功率归一化到1
        # d = d / np.sqrt(np.mean(np.abs(d)**2))
        return d

    def _256qam_modulate(self, bits):
        """
        256QAM：I/Q 各 4bit，16-PAM Gray 映射
        幅度集合：-15,-13,...,+15（步进2）
        归一化：除以 sqrt(170) 使平均功率=1
        """
        bits = bits.astype(np.int8)
        c = bits.reshape(-1, 8)
        L_sym = len(c)

        i_bits = c[:, 0:4]
        q_bits = c[:, 4:8]

        gI = (i_bits[:, 0] << 3) | (i_bits[:, 1] << 2) | (i_bits[:, 2] << 1) | i_bits[:, 3]
        gQ = (q_bits[:, 0] << 3) | (q_bits[:, 1] << 2) | (q_bits[:, 2] << 1) | q_bits[:, 3]

        bI = self._gray_to_binary(gI)
        bQ = self._gray_to_binary(gQ)

        I = 2 * bI - 15
        Q = 2 * bQ - 15

        s = (I + 1j * Q) / self.const_scale[8]
        phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
        return s * phase

    # -------------------- APSK modulate --------------------
    def _apsk_modulate(self, bits):
        bits = bits.astype(np.uint8)
        k = int(self._apsk_k)
        if len(bits) % k != 0:
            raise ValueError(f"APSK: bits长度必须是{k}的整数倍")

        bmat = bits.reshape(-1, k)
        idx = (bmat * (2 ** np.arange(k - 1, -1, -1))).sum(axis=1).astype(np.int64)

        s = self._apsk_symbols_by_b[idx]

        # 与其他调制一致：pi/2 旋转
        phase = np.exp(1j * np.pi * np.arange(len(s)) / 2)
        return s * phase

    def modulate(self, data_dict):
        """主调制函数（修正版）"""
        self._verification_data(data_dict)
        bits = data_dict["bit_stream"]
        if not np.all(np.isin(bits, [0, 1])):
            raise ValueError("输入必须是二进制比特数组（0/1）")
        
        if self.MCS == 2:
            symbols_modulated = self._apsk_modulate(bits)
        elif self.MCS == 0:
            symbols_modulated = self._dbpsk_modulate(bits)
        else:
            if self.NCBPS == 1:
                symbols_modulated = self._bpsk_modulate(bits)
            elif self.NCBPS == 2:
                symbols_modulated = self._qpsk_modulate(bits)
            elif self.NCBPS == 3:
                symbols_modulated = self._8psk_modulate(bits)
            elif self.NCBPS == 4:
                symbols_modulated = self._16qam_modulate(bits)
            elif self.NCBPS == 6:
                symbols_modulated = self._64qam_modulate(bits)
            elif self.NCBPS == 8:
                symbols_modulated = self._256qam_modulate(bits)
            else:
                raise ValueError(f"不支持的NCBPS：{self.NCBPS}")
            
        result_dict = {
            "symbol_stream": symbols_modulated,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "symbol_length": round(self.symbol_length),
            "padding_bit_num": self.padding_bit_num,
        }   
        return result_dict
        

    def plot_constellation(self, modulated_symbols=None, MCS=1):
        """星座图绘制（保留）"""
        if modulated_symbols is None:
            test_bits = np.arange(0, self.NCBPS*8) % 2
            modulated_symbols = self.modulate(test_bits)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(modulated_symbols.real, modulated_symbols.imag, c='blue', s=100)
        
        for i, (x, y) in enumerate(zip(modulated_symbols.real, modulated_symbols.imag)):
            plt.text(x, y, str(i), ha='center', va='center', color='white', fontweight='bold')
        
        plt.grid(True)
        plt.axis('equal')
        
        if MCS == 1:
            mod_name = {1:"BPSK",2:"QPSK",3:"8PSK",4:"16QAM",6:"64QAM"}.get(self.NCBPS)
            plt.title(f'pi/2-{mod_name} Constellation')
            plt.xlabel('In-phase Component')
            plt.ylabel('Quadrature Component')
        elif MCS == 2:
            plt.title(f'pi/2-APSK 星座图')
            plt.xlabel('同相分量')
            plt.ylabel('正交分量')
        
        plt.show()

# ====================== 使用示例 ======================
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
    signal_power = np.mean(np.abs(symbols_dict["symbol_stream"]) ** 2)
    print(f"调制后信号功率：{signal_power}")
    modulator.plot_constellation(symbols_dict["symbol_stream"][:200], MCS=params.get("MCS"))