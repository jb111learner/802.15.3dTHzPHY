# import numpy as np
# from scipy import special
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from params.PHYParams import PHYParams

# # 设置中文字体和编码
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# class QAMModulator:
#     """
#     QAM调制器类，实现MATLAB qammod函数的功能
    
#     支持：
#     - 比特输入模式
#     - 单位平均功率归一化
#     - 不同阶数的QAM调制（2^k，k为偶数）
#     """
    
#     def __init__(self, params, input_type='bit', unit_average_power=True):
#         """
#         初始化QAM调制器
#         """
#         self.params = params
#         self.M = self.params.get("M")  # 调制阶数

#         self.input_type = input_type.lower()
#         self.unit_average_power = unit_average_power
        
#         # 检查M是否为2的幂
#         if not (self.M > 0 and (self.M & (self.M - 1)) == 0):
#             raise ValueError("M必须是2的幂")
        
#         # 计算每符号的比特数
#         self.bits_per_symbol = int(np.log2(self.M))
        
#         # 生成QAM星座图
#         self.constellation = self._generate_constellation()
    
#     def _generate_constellation(self):
#         """生成QAM星座图"""
#         # 计算星座图的维度（假设为方形QAM）
#         n = int(np.sqrt(self.M))
        
#         if n * n != self.M:
#             raise ValueError("目前仅支持方形QAM调制（M为4, 16, 64, 256等）")
        
#         # 生成星座点坐标
#         x = np.arange(-(n-1), n, 2)
#         y = np.arange(-(n-1), n, 2)
#         X, Y = np.meshgrid(x, y)
        
#         # 展平为一维数组
#         constellation = X.flatten() + 1j * Y.flatten()
        
#         # 如果需要单位平均功率，进行归一化
#         if self.unit_average_power:
#             avg_power = np.mean(np.abs(constellation) ** 2)
#             constellation /= np.sqrt(avg_power)
        
#         return constellation
    
#     def _bits_to_symbols(self, bits):
#         """将比特序列转换为符号索引"""
#         # 确保比特数是每符号比特数的整数倍
#         if len(bits) % self.bits_per_symbol != 0:
#             raise ValueError(f"比特数必须是{self.bits_per_symbol}的整数倍")
        
#         # 重塑比特数组
#         bits_reshaped = bits.reshape(-1, self.bits_per_symbol)
        
#         # 将每比特组转换为十进制索引
#         symbols = np.zeros(len(bits_reshaped), dtype=int)
#         for i in range(self.bits_per_symbol):
#             symbols += bits_reshaped[:, i] * (2 ** (self.bits_per_symbol - 1 - i))
        
#         return symbols
    
#     def modulate(self, data):
#         """
#         执行QAM调制
        
#         参数：
#         data: 输入数据，可以是比特数组或符号索引数组
        
#         返回：
#         调制后的复数QAM符号
#         """
#         if self.input_type == 'bit':
#             # 确保输入是二进制数组
#             if not np.all(np.isin(data, [0, 1])):
#                 raise ValueError("比特输入必须只包含0和1")
            
#             # 将比特转换为符号索引
#             symbols = self._bits_to_symbols(data)
#         elif self.input_type == 'symbol':
#             # 确保符号索引有效
#             if np.any(data < 0) or np.any(data >= self.M):
#                 raise ValueError(f"符号索引必须在0到{self.M-1}之间")
#             symbols = data
#         else:
#             raise ValueError("input_type必须是'bit'或'symbol'")
        
#         # 执行调制
#         modulated = self.constellation[symbols]
        
#         return modulated
    
#     def plot_constellation(self, use_english=False):
#         """
#         绘制星座图
        
#         参数：
#         use_english: 是否使用英文显示标签
#         """
#         plt.figure(figsize=(8, 8))
#         plt.scatter(self.constellation.real, self.constellation.imag, c='blue', s=100)
        
#         # 添加星座点标签
#         for i, (x, y) in enumerate(zip(self.constellation.real, self.constellation.imag)):
#             plt.text(x, y, str(i), ha='center', va='center', color='white', fontweight='bold')
        
#         plt.grid(True)
#         plt.axis('equal')
        
#         if use_english:
#             plt.title(f'{self.M}-QAM Constellation')
#             plt.xlabel('In-phase Component')
#             plt.ylabel('Quadrature Component')
#         else:
#             plt.title(f'{self.M}-QAM 星座图')
#             plt.xlabel('同相分量')
#             plt.ylabel('正交分量')
        
#         plt.show()


# # 使用示例
# if __name__ == "__main__":
#     params = PHYParams()
#     # 创建16-QAM调制器，比特输入，单位平均功率
#     qam16 = QAMModulator(params)
    
#     # 生成随机比特序列
#     data_bits = np.random.randint(0, 2, 600)  # 640比特
    
#     # 进行调制
#     modulated_data = qam16.modulate(data_bits)
    
#     print("调制后的数据（前10个符号）：")
#     print(modulated_data[:10])
    
#     print(f"\n星座图平均功率：{np.mean(np.abs(qam16.constellation)**2):.6f}")
    
#     # 绘制星座图（使用英文避免乱码）
#     qam16.plot_constellation()

# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = ["SimHei"]
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# class THzModulator:
#     """
#     太赫兹通信调制器类（严格对照MATLAB DataModulate函数）
#     支持：
#     - 控制信道：DBPSK差分调制
#     - 数据信道：pi/2旋转的BPSK/QPSK/8PSK/16QAM/64QAM
#     - 能量归一化（与MATLAB完全对齐）
#     - 自动补零：输入比特数不满足要求时末尾补零
#     """
    
#     def __init__(self, params):
#         """
#         初始化调制器
#         params参数需包含：
#         - MCS: 调制编码方案（"0"=控制信道，其他=数据信道）
#         - NCBPS: 每符号比特数（1=BPSK,2=QPSK,3=8PSK,4=16QAM,6=64QAM）
#         """
#         self.params = params
#         self.MCS = self.params.get("MCS")  # 默认控制信道
#         self.NCBPS = self.params.get("NCBPS")  # 默认BPSK
        
#         # 预定义能量归一化因子（与MATLAB完全对齐）
#         self.norm_factors = {
#             1: 1.0,          # BPSK无额外归一化（pi/2旋转后能量仍为1）
#             2: np.sqrt(2),   # QPSK: 1/sqrt(2)
#             3: 1.0,          # 8PSK: 单位模长
#             4: np.sqrt(10),  # 16QAM: 1/sqrt(10)
#             6: np.sqrt(42)   # 64QAM: 1/sqrt(42)
#         }
        
#         # 验证NCBPS合法性
#         if self.NCBPS not in self.norm_factors.keys() and self.MCS != "0":
#             raise ValueError(f"NCBPS仅支持{list(self.norm_factors.keys())}")

#     def _dbpsk_modulate(self, bits):
#         """
#         控制信道：DBPSK差分调制（对照MATLAB MCS="0"逻辑）
#         bits: 二进制比特数组（0/1）
#         返回：差分调制后的复符号
#         """
#         L = len(bits)
#         d = np.zeros(L, dtype=np.complex128)
#         # 比特→双极性序列（0→-1，1→1）
#         c = 2 * bits - 1
#         # 差分调制：第一个符号为参考，后续符号=前一符号×当前双极性比特
#         d[0] = c[0]
#         for i in range(1, L):
#             d[i] = d[i-1] * c[i]
#         return d

#     def _bpsk_modulate(self, bits):
#         """
#         数据信道：pi/2-BPSK调制（NCBPS=1）
#         BPSK无需补零（1比特/符号，任意长度都满足）
#         """
#         L = len(bits)
#         # 比特→双极性序列
#         c = 2 * bits - 1
#         # 生成pi/2递增旋转相位：exp(1j*pi*(0:L-1)/2)
#         phase = np.exp(1j * np.pi * np.arange(L) / 2)
#         # BPSK符号 × 旋转相位
#         d = c.astype(np.complex128) * phase
#         return d

#     def _qpsk_modulate(self, bits):
#         """
#         数据信道：pi/2-QPSK调制（NCBPS=2）
#         新增：自动补零至比特数为2的整数倍
#         """
#         # ====== 新增补零逻辑 ======
#         remainder = len(bits) % 2
#         if remainder != 0:
#             # 末尾补零，使总长度为2的整数倍
#             bits = np.pad(bits, (0, 2 - remainder), mode='constant', constant_values=0)
#             print(f"QPSK调制：输入比特数{len(bits)-2+remainder}，补{2-remainder}个零后长度{len(bits)}")
#         # =========================
#         c = bits.reshape(-1, 2)
#         L_sym = len(c)
#         # QPSK基础映射 + pi/4旋转补偿
#         real_part = 2 * c[:, 0] - 1  # 比特0→实部
#         imag_part = 2 * c[:, 1] - 1  # 比特1→虚部
#         s = (real_part + 1j * imag_part) * np.exp(-1j * np.pi / 4) / self.norm_factors[2]
#         # pi/2递增旋转
#         phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
#         d = s * phase
#         return d

#     def _8psk_modulate(self, bits):
#         """
#         数据信道：pi/2-8PSK调制（NCBPS=3）
#         新增：自动补零至比特数为3的整数倍
#         """
#         # ====== 新增补零逻辑 ======
#         remainder = len(bits) % 3
#         if remainder != 0:
#             # 末尾补零，使总长度为3的整数倍
#             bits = np.pad(bits, (0, 3 - remainder), mode='constant', constant_values=0)
#             print(f"8PSK调制：输入比特数{len(bits)-3+remainder}，补{3-remainder}个零后长度{len(bits)}")
#         # =========================
#         c = bits.reshape(-1, 3)
#         L_sym = len(c)
#         # 8PSK相位映射公式（严格对照MATLAB）
#         c1, c2, c3 = c[:, 0], c[:, 1], c[:, 2]
#         phase_factor = (c1 - 3*c2 - c3 - 2*c1*c2 + 2*c2*c3 + 2*c1*c3 + 4*c1*c2*c3 + 4) / 4
#         s = np.exp(1j * np.pi * phase_factor)
#         # pi/2递增旋转
#         phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
#         d = s * phase
#         return d

#     def _16qam_modulate(self, bits):
#         """
#         数据信道：pi/2-16QAM调制（NCBPS=4）
#         新增：自动补零至比特数为4的整数倍
#         """
#         # ====== 新增补零逻辑 ======
#         remainder = len(bits) % 4
#         if remainder != 0:
#             # 末尾补零，使总长度为4的整数倍
#             bits = np.pad(bits, (0, 4 - remainder), mode='constant', constant_values=0)
#             print(f"16QAM调制：输入比特数{len(bits)-4+remainder}，补{4-remainder}个零后长度{len(bits)}")
#         # =========================
#         c = bits.reshape(-1, 4)
#         L_sym = len(c)
#         # 16QAM实部/虚部分别映射（严格对照MATLAB）
#         c1, c2, c3, c4 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
#         # 实部映射（比特0+1）
#         real_part = 4*c1 - 2 - (2*c1 - 1)*(2*c2 - 1)
#         # 虚部映射（比特2+3）
#         imag_part = 4*c3 - 2 - (2*c3 - 1)*(2*c4 - 1)
#         s = (real_part + 1j * imag_part) / self.norm_factors[4]
#         # pi/2递增旋转
#         phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
#         d = s * phase
#         return d

#     def _64qam_modulate(self, bits):
#         """
#         数据信道：pi/2-64QAM调制（NCBPS=6）
#         新增：自动补零至比特数为6的整数倍
#         """
#         # ====== 新增补零逻辑 ======
#         remainder = len(bits) % 6
#         if remainder != 0:
#             # 末尾补零，使总长度为6的整数倍
#             bits = np.pad(bits, (0, 6 - remainder), mode='constant', constant_values=0)
#             print(f"64QAM调制：输入比特数{len(bits)-6+remainder}，补{6-remainder}个零后长度{len(bits)}")
#         # =========================
#         c = bits.reshape(-1, 6)
#         L_sym = len(c)
#         # 64QAM实部/虚部分别映射（严格对照MATLAB）
#         c1, c2, c3, c4, c5, c6 = c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
#         # 实部映射（比特0+1+2）
#         real_part = 8*c1 - 4 - (2*c1 - 1)*(4*c2 - 2) + (2*c1 - 1)*(2*c2 - 1)*(2*c3 - 1)
#         # 虚部映射（比特3+4+5）
#         imag_part = 8*c4 - 4 - (2*c4 - 1)*(4*c5 - 2) + (2*c4 - 1)*(2*c5 - 1)*(2*c6 - 1)
#         s = (real_part + 1j * imag_part) / self.norm_factors[6]
#         # pi/2递增旋转
#         phase = np.exp(1j * np.pi * np.arange(L_sym) / 2)
#         d = s * phase
#         return d

#     def modulate(self, bits):
#         """
#         主调制函数（统一入口）
#         bits: 二进制比特数组（0/1）
#         返回：调制后的复符号
#         """
#         # 确保输入是二进制数组
#         if not np.all(np.isin(bits, [0, 1])):
#             raise ValueError("输入必须是二进制比特数组（0/1）")
        
#         # 控制信道：DBPSK
#         if self.MCS == "0":
#             return self._dbpsk_modulate(bits)
#         # 数据信道：按NCBPS选择调制方式
#         else:
#             if self.NCBPS == 1:
#                 return self._bpsk_modulate(bits)
#             elif self.NCBPS == 2:
#                 return self._qpsk_modulate(bits)
#             elif self.NCBPS == 3:
#                 return self._8psk_modulate(bits)
#             elif self.NCBPS == 4:
#                 return self._16qam_modulate(bits)
#             elif self.NCBPS == 6:
#                 return self._64qam_modulate(bits)
#             else:
#                 raise ValueError(f"不支持的NCBPS：{self.NCBPS}")

#     def plot_constellation(self, modulated_symbols=None, use_english=False):
#         """
#         绘制调制符号星座图
#         modulated_symbols: 可选，已调制的符号数组；若为None则生成测试符号绘制
#         """
#         if modulated_symbols is None:
#             # 生成测试比特并调制
#             test_bits = np.arange(0, self.NCBPS*8) % 2  # 测试比特序列
#             modulated_symbols = self.modulate(test_bits)
        
#         plt.figure(figsize=(8, 8))
#         plt.scatter(modulated_symbols.real, modulated_symbols.imag, c='blue', s=100)
        
#         # 添加符号标签
#         for i, (x, y) in enumerate(zip(modulated_symbols.real, modulated_symbols.imag)):
#             plt.text(x, y, str(i), ha='center', va='center', color='white', fontweight='bold')
        
#         plt.grid(True)
#         plt.axis('equal')
        
#         if use_english:
#             mod_name = {1:"BPSK",2:"QPSK",3:"8PSK",4:"16QAM",6:"64QAM"}.get(self.NCBPS, "DBPSK")
#             plt.title(f'pi/2-{mod_name} Constellation')
#             plt.xlabel('In-phase Component')
#             plt.ylabel('Quadrature Component')
#         else:
#             mod_name = {1:"BPSK",2:"QPSK",3:"8PSK",4:"16QAM",6:"64QAM"}.get(self.NCBPS, "DBPSK")
#             plt.title(f'pi/2-{mod_name} 星座图')
#             plt.xlabel('同相分量')
#             plt.ylabel('正交分量')
        
#         plt.show()
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class THzModulator:
    """
    太赫兹通信调制器类（修正功率归一化，严格对齐MATLAB）
    核心修正：所有调制符号平均功率强制归一化到1
    """
    
    def __init__(self, params):
        self.params = params
        self.MCS = self.params.get("MCS")  
        self.NCBPS = self.params.get("NCBPS")  
        
        # 修正：仅用于星座点基础缩放，最终强制归一化到功率1
        self.const_scale = {
            1: 1.0,          # BPSK: 基础幅度1
            2: np.sqrt(2),   # QPSK: 基础幅度√2（(1+1)/2=1）
            3: 1.0,          # 8PSK: 单位幅度
            4: np.sqrt(10),  # 16QAM: 基础幅度√10（(1²+3²)*4/16=2.5 → 除以√10后0.25，需再缩放）
            6: np.sqrt(42)   # 64QAM: 基础幅度√42
        }
        
        if self.NCBPS not in self.const_scale.keys() and self.MCS != "0":
            raise ValueError(f"NCBPS仅支持{list(self.const_scale.keys())}")

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

    def modulate(self, bits):
        """主调制函数（修正版）"""
        if not np.all(np.isin(bits, [0, 1])):
            raise ValueError("输入必须是二进制比特数组（0/1）")
        
        if self.MCS == "0":
            return self._dbpsk_modulate(bits)
        else:
            if self.NCBPS == 1:
                return self._bpsk_modulate(bits)
            elif self.NCBPS == 2:
                return self._qpsk_modulate(bits)
            elif self.NCBPS == 3:
                return self._8psk_modulate(bits)
            elif self.NCBPS == 4:
                return self._16qam_modulate(bits)
            elif self.NCBPS == 6:
                return self._64qam_modulate(bits)
            else:
                raise ValueError(f"不支持的NCBPS：{self.NCBPS}")

    def plot_constellation(self, modulated_symbols=None, use_english=False):
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
        
        if use_english:
            mod_name = {1:"BPSK",2:"QPSK",3:"8PSK",4:"16QAM",6:"64QAM"}.get(self.NCBPS, "DBPSK")
            plt.title(f'pi/2-{mod_name} Constellation')
            plt.xlabel('In-phase Component')
            plt.ylabel('Quadrature Component')
        else:
            mod_name = {1:"BPSK",2:"QPSK",3:"8PSK",4:"16QAM",6:"64QAM"}.get(self.NCBPS, "DBPSK")
            plt.title(f'pi/2-{mod_name} 星座图')
            plt.xlabel('同相分量')
            plt.ylabel('正交分量')
        
        plt.show()

# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 模拟PHY参数（替代原PHYParams类）
    class PHYParams:
        def __init__(self, mcs="0", ncbps=1):
            self.mcs = mcs
            self.ncbps = ncbps
        def get(self, key, default=None):
            if key == "MCS":
                return self.mcs
            elif key == "NCBPS":
                return self.ncbps
            return default
    
    # 示例1：控制信道DBPSK调制
    print("=== 控制信道DBPSK调制 ===")
    dbpsk_params = PHYParams(mcs="0", ncbps=1)
    dbpsk_mod = THzModulator(dbpsk_params)
    dbpsk_bits = np.array([0, 1, 0, 1, 1])
    dbpsk_syms = dbpsk_mod.modulate(dbpsk_bits)
    print("DBPSK调制符号：", dbpsk_syms)
    dbpsk_mod.plot_constellation(dbpsk_syms)

    # 示例2：数据信道64QAM调制（pi/2旋转）
    print("\n=== 数据信道64QAM调制 ===")
    qam64_params = PHYParams(mcs="1", ncbps=6)
    qam64_mod = THzModulator(qam64_params)
    qam64_bits = np.random.randint(0, 2, 6000)  # 64QAM符号
    qam64_syms = qam64_mod.modulate(qam64_bits)
    print("64QAM调制符号（前5个）：", qam64_syms[:5])
    print(f"64QAM符号平均功率：{np.mean(np.abs(qam64_syms)**2):.6f}")
    qam64_mod.plot_constellation(qam64_syms)