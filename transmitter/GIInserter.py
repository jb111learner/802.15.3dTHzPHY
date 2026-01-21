import numpy as np
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler 
from utils.Coder import RSCoder
from transmitter.Modulator import THzModulator
class GIInserter:
    """
    循环前缀（GI）插入器：为每个数据块添加GI，消除多径ISI
    - 支持通过params直接指定数据块长度（N）和GI长度（gi_length）
    """
    def __init__(self, params):
        self.params = params
        self.subframe_length = self.params.get("subframe_length")           # 数据子帧长度
        self.gi_length = self.params.get("gi_length")                       # GI长度
        self.is_cp = self.params.get("is_cp")                               # 是否使用CP

        # 输出参数    
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.symbol_length = None  # 符号长度
        self.padding_bit_num = 0   # 补零的比特数      

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "symbol_stream": 符号流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "symbol_length": 符号长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "symbol_stream", "sample_rate_Hz", "duration_seconds", "symbol_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["symbol_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
        if data_dict["symbol_length"] % self.subframe_length != 0:
            raise ValueError("输入符号流长度必须是subframe_length的整数倍")
        
        if self.is_cp:
            self.symbol_length = data_dict["symbol_length"] + (self.gi_length * (data_dict["symbol_length"] // self.subframe_length))
        else:
            self.symbol_length = data_dict["symbol_length"] + (self.gi_length * (data_dict["symbol_length"] // self.subframe_length + 1))
        self.duration = data_dict["duration_seconds"]
        self.sample_rate = self.symbol_length / self.duration
        self.padding_bit_num = data_dict["padding_bit_num"]     

    def wlanGolaySequence(self,N, output_format='bipolar'):
        """
        复刻MATLAB的wlanGolaySequence函数，生成符合IEEE 802.11标准的Golay互补序列对
        参考MATLAB文档：https://www.mathworks.com/help/wlan/ref/olangolaysequence.html
        
        参数:
            N : int
                Golay序列长度，必须为2的幂（2^n, n≥1），如2,4,8,16,32,64,128等
            output_format : str, 可选
                输出格式：
                - 'bipolar' (默认): 双极性序列，元素为+1/-1（符合IEEE 802.11标准）
                - 'binary': 二进制序列，元素为0/1（0对应-1，1对应+1）
        
        返回:
            A : np.array
                Golay互补序列对的第一个序列
            B : np.array
                Golay互补序列对的第二个序列
        
        异常:
            ValueError: 当N不是2的幂或小于2时抛出
        """
        # 输入合法性校验
        if not (isinstance(N, int) and N >= 2 and (N & (N - 1)) == 0):
            raise ValueError(f"序列长度N必须是2的幂且≥2，当前输入：{N}")
        
        # 计算递推阶数n (N=2^n)
        n = int(np.log2(N))
        
        # 递推生成Golay互补序列对（双极性初始值）
        # 初始条件（n=1，长度2）：A1=[1], B1=[1] → 扩展为A1=[1,1], B1=[1,-1]
        A = np.array([1], dtype=np.complex128)
        B = np.array([1], dtype=np.complex128)
        
        # 递推规则（IEEE 802.11标准定义）：
        # A_{k+1} = [A_k, B_k]
        # B_{k+1} = [A_k, -B_k]
        for _ in range(n):
            A = np.concatenate([A, B])
            B = np.concatenate([A[:len(A)//2], -B])
        
        # 4. 格式转换（对齐MATLAB的输出格式）
        if output_format == 'binary':
            # 双极性转二进制：+1→1，-1→0
            A = np.where(A == 1, 1, 0).astype(np.uint8)
            B = np.where(B == 1, 1, 0).astype(np.uint8)
        elif output_format != 'bipolar':
            raise ValueError(f"输出格式仅支持'bipolar'/'binary'，当前输入：{output_format}")
        
        return A, B

    def insert_gi(self, data_dict):
        """
        为数据插入循环前缀（GI）
        """
        self._verification_data(data_dict)
        data = data_dict["symbol_stream"]

        if self.is_cp:
            # 重塑为[N, num_blocks]的矩阵（每列一个数据块）
            data_blocks = data.reshape(-1, self.subframe_length).T  # 转置为列块结构

            # 插入GI：取每个块的最后gi_length个元素作为前缀，拼接在块前
            gi_blocks = np.concatenate([data_blocks[-self.gi_length:, :], data_blocks], axis=0)
            
            # 展平为一维信号（按列优先）
            data_with_gi = gi_blocks.T.flatten()
            if len(data_with_gi) != self.symbol_length:
                raise ValueError("插入GI后数据长度不匹配:预期长度={}, 实际长度={}".format(self.symbol_length, len(data_with_gi)))
        else:
            self.Ga, self.Gb = self.wlanGolaySequence(self.gi_length)
            # 重塑为[N, num_blocks]的矩阵（每列一个数据块）
            num_blocks = len(data) // self.subframe_length
            data_blocks = data.reshape(-1, self.subframe_length).T  # 转置为列块结构

            # 插入GI：在每个块前添加Golay序列Ga
            gi_blocks = np.concatenate([np.expand_dims(self.Ga, axis=1).repeat(num_blocks, axis=1), data_blocks], axis=0)
            data_with_gi = gi_blocks.T.flatten()
            # 展平为一维信号（按列优先）
            data_with_gi = np.concatenate([data_with_gi, self.Ga])  # 在最后添加Ga作为结尾GI
            if len(data_with_gi) != self.symbol_length:
                raise ValueError("插入GI后数据长度不匹配:预期长度={}, 实际长度={}".format(self.symbol_length, len(data_with_gi)))

        result_dict = {
            "symbol_stream": data_with_gi, 
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "symbol_length": self.symbol_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return result_dict
        # # ====== 新增：能量归一化 ======
        # # 归一化到原始数据功率（或目标功率）
        # data_with_gi = self._normalize_power(data_with_gi)

if __name__ == "__main__":
    params = PHYParams()
    p1 = BitStreamProcessor(params)
    res1 = p1.run()
    scrambler = Scrambler(params)
    scrambled_data_dict = scrambler.scramble(res1)
    coder = RSCoder(params)
    encoded_data_dict = coder.encode(scrambled_data_dict)
    modulator = THzModulator(params)
    symbols_dict = modulator.modulate(encoded_data_dict)
    print(f"符号长度：{symbols_dict['symbol_length']}")
    print(f"采样率：{symbols_dict['sample_rate_Hz']} Hz，时长：{symbols_dict['duration_seconds']} 秒")
    print(f"补零数量：{symbols_dict['padding_bit_num']} bit")    
    gi_inserter = GIInserter(params)
    data_with_gi_dict = gi_inserter.insert_gi(symbols_dict)
    signal_power = np.mean(np.abs(data_with_gi_dict["symbol_stream"]) ** 2)
    print(f"插入GI后信号功率：{signal_power}")
    print(f"符号长度：{data_with_gi_dict['symbol_length']}")
    print(f"采样率：{data_with_gi_dict['sample_rate_Hz']} Hz，时长：{data_with_gi_dict['duration_seconds']} 秒")
    print(f"补零数量：{data_with_gi_dict['padding_bit_num']} bit")

# # 测试
# if __name__ == "__main__":
#     # 模拟PHYParams类（替代原有导入）
#     class PHYParams:
#         def __init__(self):
#             self.params = {
#                 "subframe_length": 480,
#                 "cp_length": 32,
#                 "target_power": 1.0,  # 新增目标功率参数
#                 "chan_delays": []
#             }
#         def get(self, key, default=None):
#             return self.params.get(key, default)
    
#     params1 = PHYParams()
#     gi_inserter1 = GIInserter(params1)
#     # 生成测试复数据（1000个符号，平均功率≈1）
#     data1 = (np.random.randn(1000) + 1j * np.random.randn(1000)) / np.sqrt(2)
#     print("="*50)
#     print("测试用例1（自定义subframe_length=480，gi_length=32）：")
#     print(f"原始数据长度：{len(data1)}")
#     print(f"原始数据平均功率：{np.mean(np.abs(data1)**2):.6f}")
    
#     # 插入GI
#     data_with_gi1 = gi_inserter1.insert_gi(data1)
#     print(f"插入GI后长度：{len(data_with_gi1)}")  # 480 + 32 = 512
#     print(f"插入GI后平均功率：{np.mean(np.abs(data_with_gi1)**2):.6f}")

#     # 移除GI
#     data_removed_gi1 = gi_inserter1.remove_gi(data_with_gi1)
#     print(f"移除GI后长度：{len(data_removed_gi1)}")
#     print(f"移除GI后平均功率：{np.mean(np.abs(data_removed_gi1)**2):.6f}")

#     # 验证数据一致性（忽略补零/GI部分）
#     print(f"原始数据与移除GI后数据前1000位是否一致：{np.allclose(data1, data_removed_gi1[:1000])}")
#     print(f"GI长度：{gi_inserter1.gi_length}，数据块长度：{gi_inserter1.N}")