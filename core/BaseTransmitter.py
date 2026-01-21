import numpy as np

class BaseTransmitter:
    """
    发射机基类：定义发射端的统一接口
    子类需重写：generate_header() 生成头部
              generate_preamble() 生成前导码
              modulate() 调制
              insert_cp() 插入循环前缀
              assemble_signal() 组装信号
    """
    def __init__(self, params):
        self.params = params  # 参数对象

    def generate_preamble(self):
        """生成前导码"""
        pass
    
    def assemble_frame(self):
        """组装帧"""
        pass

    def modulate(self):
        """调制"""
        pass

    def scramble_data(self):
        """扰码数据"""
        pass
    
    def channel_encode(self):
        """信道编码"""
        pass

    def insert_gi(self):
        """插入循环前缀"""
        pass    

    def pulse_shaping(self):
        """脉冲成型"""
        pass

    def run(self,data_bits=None):
        """执行完整发射流程"""
        self.generate_preamble()
        self.assemble_frame()
        self.scramble_data()
        self.channel_encode()
        self.modulate()
        self.insert_gi()
        self.pulse_shaping()
        return self.tx_signal

# # 使用示例（子类实现）
# class THzTransmitter(BaseTransmitter):
#     def generate_header(self):
#         """生成PHY头部（示例：简单编码）"""
#         mcs = self.params.get("mcs", 5)
#         self.header_bits = np.array([int(bit) for bit in bin(mcs)[2:].zfill(4)])  # 4bit MCS

#     def generate_preamble(self):
#         """生成前导码（SYNC+SFD+CES）"""
#         sync = np.random.randn(128) + 1j * np.random.randn(128)  # 示例：随机生成SYNC
#         sfd = np.random.randn(64) + 1j * np.random.randn(64)    # 示例：随机生成SFD
#         ces = np.random.randn(128) + 1j * np.random.randn(128)  # 示例：随机生成CES
#         self.preamble = np.concatenate([sync, sfd, ces])

#     def modulate(self, data_bits):
#         """4-QAM调制"""
#         M = self.params.get("M")
#         bits_per_symbol = int(np.log2(M))
#         # 分组（每2bit一组）
#         symbols = data_bits.reshape(-1, bits_per_symbol)
#         # 映射（00→1+1j, 01→-1+1j, 10→1-1j, 11→-1-1j）
#         qam_symbols = 2 * symbols[:, 0] - 1 + 1j * (2 * symbols[:, 1] - 1)
#         # 功率归一化
#         return qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2))

#     def insert_cp(self, data):
#         """插入循环前缀（CP长度=信道最大时延）"""
#         chan_delays = self.params.get("chan_delays")
#         cp_length = max(chan_delays) if len(chan_delays) > 0 else 16
#         cp = data[-cp_length:]  # 取数据尾部作为CP
#         return np.concatenate([cp, data])

#     def assemble_signal(self, data_with_cp):
#         """组装信号：延迟+前导码+带CP数据"""
#         delay = self.params.get("delay", 500)
#         delay_signal = np.zeros(delay, dtype=complex)
#         self.tx_signal = np.concatenate([delay_signal, self.preamble, data_with_cp])

# # 测试
# if __name__ == "__main__":
#     params = PHYParams()
#     transmitter = THzTransmitter(params)
#     tx_signal = transmitter.run()
#     print(f"发射信号长度：{len(tx_signal)}")
#     print(f"前导码长度：{len(transmitter.preamble)}")
#     print(f"调制后数据长度：{len(transmitter.modulated_data)}")