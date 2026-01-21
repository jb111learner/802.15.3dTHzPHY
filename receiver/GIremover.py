import numpy as np
from params.PHYParams import PHYParams
from transmitter.THzTransmitter import THzTransmitter
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter

class GIRemover:
    """
    循环前缀（GI）移除器：从每个数据块中移除GI，消除多径ISI
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
        
        if self.is_cp:
            if data_dict["symbol_length"] % (self.subframe_length + self.gi_length) != 0:
                raise ValueError("输入符号流长度必须是subframe_length + gi_length的整数倍")
            else:
                self.symbol_length = self.subframe_length * (data_dict["symbol_length"] // (self.subframe_length + self.gi_length))
        else:
            if (data_dict["symbol_length"] - self.gi_length) % (self.subframe_length + self.gi_length) != 0:
                raise ValueError("输入符号流长度减去结尾GI后，必须是subframe_length + gi_length的整数倍")
            else:
                self.symbol_length = self.subframe_length * ((data_dict["symbol_length"] - self.gi_length) // (self.subframe_length + self.gi_length))
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


    def remove_gi(self, data_dict):
        """
        移除数据中的保护间隔（GI），支持两种模式：
        1. 移除循环前缀（CP）：对应is_cp=True的插入逻辑
        2. 移除Golay序列GI：对应is_cp=False的插入逻辑
        """
        # 校验输入数据合法性
        self._verification_data(data_dict)
        data_with_gi = data_dict["symbol_stream"]
        gi_length = self.gi_length
        subframe_length = self.subframe_length

        # 移除GI
        if self.is_cp:
            # --------------------------
            # 模式1：移除循环前缀（CP）
            # --------------------------
            num_blocks = len(data_with_gi) // (subframe_length + gi_length)
            # 校验总长度合法性
            if len(data_with_gi) != num_blocks * (subframe_length + gi_length):
                raise ValueError(f"带CP的数据长度非法：预期为{(subframe_length + gi_length)*num_blocks}，实际为{len(data_with_gi)}")
            # 重塑为列块结构：行=单块总长度，列=块数
            data_blocks_with_gi = data_with_gi.reshape(num_blocks, -1).T  # 转置后：行=gi+subframe，列=num_blocks
            # 移除GI
            data_blocks = data_blocks_with_gi[gi_length:, :]
            # 展平为一维原始信号
            data_without_gi = data_blocks.T.flatten()
        else:
            # --------------------------
            # 模式2：移除Golay序列GI
            # --------------------------
            # 移除结尾的Ga序列
            data_without_end_gi = data_with_gi[:-gi_length]
            # 计算块数并重塑为列块结构
            # 单块带GI长度 = gi_length + subframe_length
            single_block_with_gi_len = gi_length + subframe_length
            num_blocks = len(data_without_end_gi) // single_block_with_gi_len
            # 重塑为列块结构：行=gi+subframe，列=num_blocks
            data_blocks_with_gi = data_without_end_gi.reshape(num_blocks, -1).T
            # 移除每个块前的Golay序列GI
            data_blocks = data_blocks_with_gi[gi_length:, :]
            # 展平为一维原始信号（按列优先）
            data_without_gi = data_blocks.T.flatten()

        # 校验移除GI后的长度（与原始subframe总长度匹配）
        if len(data_without_gi) != self.symbol_length:
            raise ValueError(f"移除GI后数据长度不匹配:预期长度={self.symbol_length}, 实际长度={len(data_without_gi)}")

        result_dict = {
            "symbol_stream": data_without_gi,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "symbol_length": self.symbol_length,
            "padding_bit_num": self.padding_bit_num,
        }
        return result_dict
    
if __name__ == "__main__":
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # === 进行匹配滤波并获得中间信号 ===
    rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)

    # 初始化GI移除器
    gi_remover = GIRemover(params)
    # === 进行GI移除 ===
    rx_data_no_gi_dict = gi_remover.remove_gi(rx_matched_dict)
    print(f"发射符号长度：{len(transmitter.modulated_data_dict['symbol_stream'])}, 接收符号长度：{len(rx_data_no_gi_dict['symbol_stream'])}")
    print(f"发射符号采样率：{transmitter.modulated_data_dict['sample_rate_Hz']} Hz, 时长：{transmitter.modulated_data_dict['duration_seconds']} 秒")
    print(f"接收符号采样率：{rx_data_no_gi_dict['sample_rate_Hz']} Hz, 时长：{rx_data_no_gi_dict['duration_seconds']} 秒")


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