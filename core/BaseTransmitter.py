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
        self.tx_signal = None  # 最终发射信号
        self.header_bits = None  # 头部比特
        self.preamble = None  # 前导码（SYNC+SFD+CES）
        self.modulated_data = None  # 调制后数据

    def add_header(self, data_bits):
        """生成头部（子类必须重写）"""
        raise NotImplementedError("子类必须实现 add_header() 方法")

    def generate_preamble(self):
        """生成前导码（子类必须重写）"""
        raise NotImplementedError("子类必须实现 generate_preamble() 方法")

    def _generate_random_data(self, length=int(10e5)):
        """生成随机二进制数据（内部辅助方法）"""
        return np.random.randint(0, 2, length, dtype=np.uint8)

    def modulate(self, data_bits):
        """调制（子类必须重写）"""
        raise NotImplementedError("子类必须实现 modulate() 方法")
    
    def channel_encode(self, data_bits):
        """信道编码（子类可选重写）"""
        return data_bits  # 默认不编码

    def insert_cp(self, data):
        """插入循环前缀（子类必须重写）"""
        raise NotImplementedError("子类必须实现 insert_cp() 方法")
    
    def pulse_shaping(self):
        """脉冲成型（子类可选重写）"""

    def assemble_signal(self):
        """组装发射信号（子类必须重写）"""
        raise NotImplementedError("子类必须实现 assemble_signal() 方法")

    def run(self,data_bits=None):
        """执行完整发射流程（统一调度）"""
        self.generate_preamble()
        if data_bits is None:
            data_bits = self._generate_random_data()
        data_bits = self.add_header(data_bits)
        data_bits = self.channel_encode(data_bits)
        self.modulated_data = self.modulate(data_bits)
        data_with_cp = self.insert_cp(self.modulated_data)
        self.assemble_signal()
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