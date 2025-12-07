import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from typing import List, Optional, Tuple
# 若PHYParams未定义，这里提供简易实现以便测试
try:
    from params.PHYParams import PHYParams
except ImportError:
    class PHYParams:
        def __init__(self):
            self.rs_nsym = 16    # 校验符号数
            self.rs_c_exp = 8    # 有限域指数
            self.rs_packet_size = 32  # 编码包大小（字节）
        
        def get(self, key, default=None):
            return getattr(self, key, default)

class RSCoder:
    """
    封装 reedsolo.RSCodec 的类，支持按字节分包编码/解码。

    参数:
        params: 配置参数对象，需包含：
            - rs_nsym: 校验符号数量（bytes）。纠错能力约为 nsym//2 个字节错误
            - rs_c_exp: 有限域指数（可选），默认8（GF(2^8)）
            - rs_packet_size: 编码包大小（字节），即每个分包的原始数据大小
    """

    def __init__(self, params):
        self.params = params
        self.nsym = self.params.get("rs_nsym")
        self.c_exp = self.params.get("rs_c_exp")
        self.packet_size = self.params.get("rs_packet_size")  # 分包大小（字节）
        
        # 初始化RS编解码器
        self.rsc = RSCodec(self.nsym) if self.c_exp is None else RSCodec(self.nsym, c_exp=self.c_exp)
        
        # 编码后每个包的大小 = 原始包大小 + 校验符号数
        self.encoded_packet_size = self.packet_size + self.nsym

    # ---------------------------------------------------------
    # 工具函数：bit array <-> byte array
    # ---------------------------------------------------------

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """
        输入：bit 数组 (0/1)
        输出：bytes
        """
        if bits.dtype != np.uint8:
            raise TypeError("bits 必须是 np.uint8 类型")
        if bits.ndim != 1:
            raise ValueError("bits 必须是一维数组")

        # 补齐到8的倍数
        pad = (-len(bits)) % 8
        if pad != 0:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        # 每8 bit 转成1 byte
        bits = bits.reshape(-1, 8)
        vals = np.packbits(bits, axis=1)
        return vals.flatten().tobytes()

    @staticmethod
    def bytes_to_bits(b: bytes) -> np.ndarray:
        arr = np.frombuffer(b, dtype=np.uint8)
        return np.unpackbits(arr)

    # ---------------------------------------------------------
    # 工具函数：字节数据分包/合包
    # ---------------------------------------------------------

    def split_bytes_to_packets(self, data_bytes: bytes) -> List[bytes]:
        """
        将字节数据按包大小拆分，最后一包不足时补0填充
        """
        packets = []
        total_len = len(data_bytes)
        
        # 拆分完整包
        for i in range(0, total_len, self.packet_size):
            packet = data_bytes[i:i+self.packet_size]
            # 最后一包不足时补0
            if len(packet) < self.packet_size:
                packet += b'\x00' * (self.packet_size - len(packet))
            packets.append(packet)
        
        return packets

    def combine_packets_to_bytes(self, packets: List[bytes], original_len: int) -> bytes:
        """
        将分包数据合并，去除填充的0并截断到原始长度
        """
        combined = b''.join(packets)
        # 截断到原始长度（去除填充）
        return combined[:original_len]

    # ---------------------------------------------------------
    # 单包编码/解码（底层接口）
    # ---------------------------------------------------------

    def encode_single_packet(self, packet_bytes: bytes) -> bytes:
        """
        对单个数据包进行RS编码
        """
        if len(packet_bytes) != self.packet_size:
            raise ValueError(f"单包大小必须为 {self.packet_size} 字节，当前为 {len(packet_bytes)} 字节")
        return self.rsc.encode(packet_bytes)

    def decode_single_packet(self, encoded_packet_bytes: bytes, erase_pos_bytes: Optional[List[int]] = None) -> bytes:
        """
        对单个编码包进行RS解码
        """
        if len(encoded_packet_bytes) != self.encoded_packet_size:
            raise ValueError(f"编码包大小必须为 {self.encoded_packet_size} 字节，当前为 {len(encoded_packet_bytes)} 字节")
        
        try:
            if erase_pos_bytes is None:
                res = self.rsc.decode(encoded_packet_bytes)
            else:
                res = self.rsc.decode(encoded_packet_bytes, erase_pos=erase_pos_bytes)
            # 返回解码后的原始包数据
            return bytes(res[0])
        except ReedSolomonError as e:
            raise ReedSolomonError(f"单包解码失败: {str(e)}")

    # ---------------------------------------------------------
    # 分包编码：输入 bit → 输出 bit
    # ---------------------------------------------------------

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        输入：bit np.ndarray(uint8)
        输出：bit np.ndarray(uint8)（按包编码后拼接）
        """
        # 转换为字节并记录原始长度
        data_bytes = self.bits_to_bytes(bits)
        original_len = len(data_bytes)
        
        # 分包
        packets = self.split_bytes_to_packets(data_bytes)
        
        # 逐包编码
        encoded_packets = []
        for packet in packets:
            encoded_packet = self.encode_single_packet(packet)
            encoded_packets.append(encoded_packet)
        
        # 合并编码包并转换为bit
        encoded_bytes = b''.join(encoded_packets)
        encoded_bits = self.bytes_to_bits(encoded_bytes)
        
        # 保存原始长度（用于解码时恢复）
        self._last_encoded_original_len = original_len
        
        return encoded_bits

    # ---------------------------------------------------------
    # 分包解码：输入 bit → 输出 bit
    # ---------------------------------------------------------

    def decode(self, bits: np.ndarray, erase_pos_bytes: Optional[List[int]] = None) -> Tuple[np.ndarray, dict]:
        """
        输入：bit数组（可能含错误）
        可选：erase_pos_bytes (按字节索引，不是按bit)
        输出：(bit数组, debug信息)
        """
        # 转换为字节
        coded_bytes = self.bits_to_bytes(bits)
        
        # 检查编码后数据长度是否合法
        if len(coded_bytes) % self.encoded_packet_size != 0:
            raise ValueError(f"编码数据长度必须是 {self.encoded_packet_size} 的倍数，当前为 {len(coded_bytes)} 字节")
        
        # 拆分编码包
        encoded_packets = []
        for i in range(0, len(coded_bytes), self.encoded_packet_size):
            encoded_packet = coded_bytes[i:i+self.encoded_packet_size]
            encoded_packets.append(encoded_packet)
        
        try:
            # 逐包解码
            decoded_packets = []
            for idx, packet in enumerate(encoded_packets):
                # 计算当前包的擦除位置偏移
                packet_erase_pos = None
                if erase_pos_bytes is not None:
                    # 转换为相对当前包的偏移
                    start = idx * self.encoded_packet_size
                    end = start + self.encoded_packet_size
                    packet_erase_pos = [pos - start for pos in erase_pos_bytes if start <= pos < end]
                
                decoded_packet = self.decode_single_packet(packet, packet_erase_pos)
                decoded_packets.append(decoded_packet)
            
            # 合并解码包并恢复原始长度
            decoded_bytes = self.combine_packets_to_bytes(decoded_packets, getattr(self, '_last_encoded_original_len', len(decoded_packets)*self.packet_size))
            decoded_bits = self.bytes_to_bits(decoded_bytes)

            debug = {
                "status": "ok",
                "nsym": self.nsym,
                "packet_size": self.packet_size,
                "encoded_packet_size": self.encoded_packet_size,
                "packet_count": len(encoded_packets),
                "original_length": self._last_encoded_original_len
            }
            return decoded_bits, debug

        except ReedSolomonError as e:
            debug = {"status": "fail", "error": str(e)}
            raise ReedSolomonError(debug)

    # ---------------------------------------------------------
    # 错误注入（兼容分包场景）
    # ---------------------------------------------------------

    def introduce_random_byte_errors(self, bits: np.ndarray, n_byte_errors: int) -> np.ndarray:
        """
        在字节层注入错误，安全且完全符合 RS 纠错模型。
        输入：bit 流 (np.ndarray, 0/1)
        输出：bit 流
        """
        # bit → bytes
        data_bytes = self.bits_to_bytes(bits)
        arr = np.frombuffer(data_bytes, dtype=np.uint8).copy()

        length = len(arr)
        if n_byte_errors > length:
            raise ValueError("字节错误数大于字节总数")
        
        # 随机选择 n_byte_errors 个字节位置
        positions = np.random.choice(length, n_byte_errors, replace=False)
        
        for pos in positions:
            old_val = arr[pos]
            # 将该字节替换为另一个不同的值（确保存“字节错误”）
            new_val = np.random.randint(0, 256)
            while new_val == old_val:
                new_val = np.random.randint(0, 256)
            arr[pos] = new_val
        
        # bytes → bit
        return self.bytes_to_bits(arr.tobytes())

    # ---------------------------------------------------------
    # 属性
    # ---------------------------------------------------------

    @property
    def max_correctable_bytes(self):
        return self.nsym // 2

    @property
    def packet_count(self):
        """获取最后一次编码的包数量（需先执行encode）"""
        if hasattr(self, '_last_encoded_original_len'):
            return (self._last_encoded_original_len + self.packet_size - 1) // self.packet_size
        return 0


# ---------- 测试示例 ----------
if __name__ == "__main__":
    import numpy as np
    
    # 初始化参数和编码器
    params = PHYParams()
    # params.rs_packet_size = 8  # 每个包8字节原始数据
    # params.rs_nsym = 16        # 每个包16字节校验
    wrapper = RSCoder(params)

    # 测试数据（大于单个包大小，验证分包功能）
    msg_str = b"HELLO RS 1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"原始消息: {msg_str}")
    print(f"原始消息长度: {len(msg_str)} 字节")
    
    # 转换为bit数组
    msg_bits = np.unpackbits(np.frombuffer(msg_str, dtype=np.uint8))
    
    # 编码（自动分包）
    coded_bits = wrapper.encode(msg_bits)
    coded_bytes_len = len(coded_bits) // 8
    print(f"\n编码后总长度: {coded_bytes_len} 字节")
    print(f"每个编码包大小: {wrapper.encoded_packet_size} 字节")
    print(f"分包数量: {wrapper.packet_count}")

    # 注入错误（小于最大可纠错数）
    max_error = wrapper.max_correctable_bytes
    corrupted_bits = wrapper.introduce_random_byte_errors(coded_bits, max_error - 2)

    # 解码（自动分包包解码）
    decoded_bits, info = wrapper.decode(corrupted_bits)
    
    # 恢复原始字节数据
    decoded_bytes = wrapper.bits_to_bytes(decoded_bits)[:len(msg_str)]
    print(f"\n解码后消息: {decoded_bytes}")
    print(f"是否恢复正确：{decoded_bytes == msg_str}")
    print(f"解码信息: {info}")

