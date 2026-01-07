# import numpy as np
# from reedsolo import RSCodec, ReedSolomonError
# from typing import List, Optional, Tuple
# from params.PHYParams import PHYParams

# class RSCoder:
#     """
#     封装 reedsolo.RSCodec 的类，支持按字节分包编码/解码。

#     参数:
#         params: 配置参数对象，需包含：
#             - rs_nsym: 校验符号数量（bytes）。纠错能力约为 nsym//2 个字节错误
#             - rs_c_exp: 有限域指数（可选），默认8（GF(2^8)）
#             - rs_packet_size: 编码包大小（字节），即每个分包的原始数据大小
#     """

#     def __init__(self, params):
#         self.params = params
#         self.nsym = self.params.get("rs_nsym")
#         self.c_exp = self.params.get("rs_c_exp")
#         self.packet_size = self.params.get("rs_packet_size")  # 分包大小（字节）
        
#         # 初始化RS编解码器
#         self.rsc = RSCodec(self.nsym) if self.c_exp is None else RSCodec(self.nsym, c_exp=self.c_exp)
        
#         # 编码后每个包的大小 = 原始包大小 + 校验符号数
#         self.encoded_packet_size = self.packet_size + self.nsym

#     # ---------------------------------------------------------
#     # 工具函数：bit array <-> byte array
#     # ---------------------------------------------------------

#     @staticmethod
#     def bits_to_bytes(bits: np.ndarray) -> bytes:
#         """
#         输入：bit 数组 (0/1)
#         输出：bytes
#         """
#         if bits.dtype != np.uint8:
#             raise TypeError("bits 必须是 np.uint8 类型")
#         if bits.ndim != 1:
#             raise ValueError("bits 必须是一维数组")

#         # 补齐到8的倍数
#         pad = (-len(bits)) % 8
#         if pad != 0:
#             bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

#         # 每8 bit 转成1 byte
#         bits = bits.reshape(-1, 8)
#         vals = np.packbits(bits, axis=1)
#         return vals.flatten().tobytes()

#     @staticmethod
#     def bytes_to_bits(b: bytes) -> np.ndarray:
#         arr = np.frombuffer(b, dtype=np.uint8)
#         return np.unpackbits(arr)

#     # ---------------------------------------------------------
#     # 工具函数：字节数据分包/合包
#     # ---------------------------------------------------------

#     def split_bytes_to_packets(self, data_bytes: bytes) -> List[bytes]:
#         """
#         将字节数据按包大小拆分，最后一包不足时补0填充
#         """
#         packets = []
#         total_len = len(data_bytes)
        
#         # 拆分完整包
#         for i in range(0, total_len, self.packet_size):
#             packet = data_bytes[i:i+self.packet_size]
#             # 最后一包不足时补0
#             if len(packet) < self.packet_size:
#                 packet += b'\x00' * (self.packet_size - len(packet))
#             packets.append(packet)
        
#         return packets

#     def combine_packets_to_bytes(self, packets: List[bytes], original_len: int) -> bytes:
#         """
#         将分包数据合并，去除填充的0并截断到原始长度
#         """
#         combined = b''.join(packets)
#         # 截断到原始长度（去除填充）
#         return combined[:original_len]

#     # ---------------------------------------------------------
#     # 单包编码/解码（底层接口）
#     # ---------------------------------------------------------

#     def encode_single_packet(self, packet_bytes: bytes) -> bytes:
#         """
#         对单个数据包进行RS编码
#         """
#         x = len(packet_bytes)
#         if len(packet_bytes) != self.packet_size:
#             raise ValueError(f"单包大小必须为 {self.packet_size} 字节，当前为 {len(packet_bytes)} 字节")
#         packet_bytes_encoded = self.rsc.encode(packet_bytes)
#         x_encoded = len(packet_bytes_encoded)
#         return packet_bytes_encoded

#     def decode_single_packet(self, encoded_packet_bytes: bytes, erase_pos_bytes: Optional[List[int]] = None) -> bytes:
#         """
#         对单个编码包进行RS解码
#         """
#         if len(encoded_packet_bytes) != self.encoded_packet_size:
#             raise ValueError(f"编码包大小必须为 {self.encoded_packet_size} 字节，当前为 {len(encoded_packet_bytes)} 字节")
        
        
#         if erase_pos_bytes is None:
#             res = self.rsc.decode(encoded_packet_bytes)
#         else:
#             res = self.rsc.decode(encoded_packet_bytes, erase_pos=erase_pos_bytes)
#         # 返回解码后的原始包数据
#         return bytes(res[0])

#     # ---------------------------------------------------------
#     # 分包编码：输入 bit → 输出 bit
#     # ---------------------------------------------------------

#     def encode(self, bits: np.ndarray) -> np.ndarray:
#         """
#         输入：bit np.ndarray(uint8)
#         输出：bit np.ndarray(uint8)（按包编码后拼接）
#         """
#         # 转换为字节并记录原始长度
#         data_bytes = self.bits_to_bytes(bits)
#         original_len = len(data_bytes)
        
#         # 分包
#         packets = self.split_bytes_to_packets(data_bytes)
        
#         # 逐包编码
#         encoded_packets = []
#         for packet in packets:
#             encoded_packet = self.encode_single_packet(packet)
#             encoded_packets.append(encoded_packet)
        
#         # 合并编码包并转换为bit
#         encoded_bytes = b''.join(encoded_packets)
#         encoded_bits = self.bytes_to_bits(encoded_bytes)
        
#         # 保存原始长度（用于解码时恢复）
#         self._last_encoded_original_len = original_len
        
#         return encoded_bits

#     # ---------------------------------------------------------
#     # 分包解码：输入 bit → 输出 bit
#     # ---------------------------------------------------------

#     def decode(self, bits: np.ndarray, erase_pos_bytes: Optional[List[int]] = None) -> Tuple[np.ndarray, dict]:
#         """
#         输入：bit数组（可能含错误）
#         可选：erase_pos_bytes (按字节索引，不是按bit)
#         输出：(bit数组, debug信息)
#         """
#         # 转换为字节
#         coded_bytes = self.bits_to_bytes(bits)
        
#         # 检查编码后数据长度是否合法
#         if len(coded_bytes) % self.encoded_packet_size != 0:
#             raise ValueError(f"编码数据长度必须是 {self.encoded_packet_size} 的倍数，当前为 {len(coded_bytes)} 字节")
        
#         # 拆分编码包
#         encoded_packets = []
#         for i in range(0, len(coded_bytes), self.encoded_packet_size):
#             encoded_packet = coded_bytes[i:i+self.encoded_packet_size]
#             encoded_packets.append(encoded_packet)
        
#             # 逐包解码
#             decoded_packets = []
#             for idx, packet in enumerate(encoded_packets):
#                 # 计算当前包的擦除位置偏移
#                 packet_erase_pos = None
#                 if erase_pos_bytes is not None:
#                     # 转换为相对当前包的偏移
#                     start = idx * self.encoded_packet_size
#                     end = start + self.encoded_packet_size
#                     packet_erase_pos = [pos - start for pos in erase_pos_bytes if start <= pos < end]
                
#                 decoded_packet = self.decode_single_packet(packet, packet_erase_pos)
#                 decoded_packets.append(decoded_packet)
            
#         # 合并解码包并恢复原始长度
#         decoded_bytes = self.combine_packets_to_bytes(decoded_packets, getattr(self, '_last_encoded_original_len', len(decoded_packets)*self.packet_size))
#         decoded_bits = self.bytes_to_bits(decoded_bytes)
#         return decoded_bits

#     # ---------------------------------------------------------
#     # 错误注入（兼容分包场景）
#     # ---------------------------------------------------------

#     def introduce_random_byte_errors(self, bits: np.ndarray, n_byte_errors: int) -> np.ndarray:
#         """
#         在字节层注入错误，安全且完全符合 RS 纠错模型。
#         输入：bit 流 (np.ndarray, 0/1)
#         输出：bit 流
#         """
#         # bit → bytes
#         data_bytes = self.bits_to_bytes(bits)
#         arr = np.frombuffer(data_bytes, dtype=np.uint8).copy()

#         length = len(arr)
#         if n_byte_errors > length:
#             raise ValueError("字节错误数大于字节总数")
        
#         # 随机选择 n_byte_errors 个字节位置
#         positions = np.random.choice(length, n_byte_errors, replace=False)
        
#         for pos in positions:
#             old_val = arr[pos]
#             # 将该字节替换为另一个不同的值（确保存“字节错误”）
#             new_val = np.random.randint(0, 256)
#             while new_val == old_val:
#                 new_val = np.random.randint(0, 256)
#             arr[pos] = new_val
        
#         # bytes → bit
#         return self.bytes_to_bits(arr.tobytes())

#     # ---------------------------------------------------------
#     # 属性
#     # ---------------------------------------------------------

#     @property
#     def max_correctable_bytes(self):
#         return self.nsym // 2

#     @property
#     def packet_count(self):
#         """获取最后一次编码的包数量（需先执行encode）"""
#         if hasattr(self, '_last_encoded_original_len'):
#             return (self._last_encoded_original_len + self.packet_size - 1) // self.packet_size
#         return 0
import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from reedsolo import rs_correct_msg  # 导入底层纠错函数（用于提取中间结果）
from typing import List, Optional, Tuple

class RSCoder:
    """
    封装 reedsolo.RSCodec 的类，支持按字节分包编码/解码。
    专为误码率仿真优化：错误超限时不抛异常，返回最优恢复结果并输出详细纠错状态。
    """
    def __init__(self, params):
        self.params = params
        self.nsym = self.params.get("rs_nsym")          # 校验符号数量（bytes）
        self.c_exp = self.params.get("rs_c_exp")        # 有限域指数，默认GF(2^8)
        self.packet_size = self.params.get("rs_packet_size")  # 单个数据包原始大小（字节）
        
        # 初始化RS编解码器
        self.rsc = RSCodec(self.nsym, c_exp=self.c_exp)
        # 编码后每个包的大小 = 原始包大小 + 校验符号数
        self.encoded_packet_size = self.packet_size + self.nsym
        
        # 保存编码时的原始长度信息（用于解码后对齐）
        self._original_bit_len = 0
        self._pad_bits = 0

    # ---------------------------------------------------------
    # 工具函数：bit <-> byte 转换（严格对齐，避免填充干扰误码率）
    # ---------------------------------------------------------
    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> Tuple[bytes, int]:
        """
        输入：bit数组 (0/1, uint8)
        输出：(转换后的bytes, 填充的比特数)
        """
        if bits.dtype != np.uint8:
            raise TypeError("bits 必须是 np.uint8 类型")
        if bits.ndim != 1:
            raise ValueError("bits 必须是一维数组")
        
        # 补齐到8的倍数，记录填充数
        pad_bits = (-len(bits)) % 8
        if pad_bits != 0:
            bits = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])
        
        bits_2d = bits.reshape(-1, 8)
        vals = np.packbits(bits_2d, axis=1)
        return vals.flatten().tobytes(), pad_bits

    @staticmethod
    def bytes_to_bits(b: bytes, pad_bits: int = 0) -> np.ndarray:
        """
        输入：bytes + 填充比特数
        输出：去除填充后的bit数组
        """
        arr = np.frombuffer(b, dtype=np.uint8)
        bits = np.unpackbits(arr)
        # 去除填充的比特
        if pad_bits > 0 and len(bits) >= pad_bits:
            bits = bits[:-pad_bits]
        return bits

    # ---------------------------------------------------------
    # 工具函数：字节数据分包/合包
    # ---------------------------------------------------------
    def split_bytes_to_packets(self, data_bytes: bytes) -> List[bytes]:
        """将字节数据按包大小拆分，最后一包不足时补0填充"""
        packets = []
        total_len = len(data_bytes)
        for i in range(0, total_len, self.packet_size):
            packet = data_bytes[i:i+self.packet_size]
            if len(packet) < self.packet_size:
                packet += b'\x00' * (self.packet_size - len(packet))
            packets.append(packet)
        return packets

    def combine_packets_to_bytes(self, packets: List[bytes], original_len: int) -> bytes:
        """合并分包数据，截断到原始长度（去除填充）"""
        combined = b''.join(packets)
        return combined[:original_len]

    # ---------------------------------------------------------
    # 单包编码/解码（核心：异常捕获 + 最优恢复）
    # ---------------------------------------------------------
    def encode_single_packet(self, packet_bytes: bytes) -> bytes:
        """对单个数据包进行RS编码"""
        if len(packet_bytes) != self.packet_size:
            raise ValueError(f"单包大小必须为 {self.packet_size} 字节，当前为 {len(packet_bytes)} 字节")
        return self.rsc.encode(packet_bytes)

    def decode_single_packet(self, encoded_packet_bytes: bytes, 
                           erase_pos_bytes: Optional[List[int]] = None) -> Tuple[bytes, str]:
        """
        单包解码（容错版）：错误超限时返回最优恢复结果
        返回：(解码后数据, 纠错状态)
        纠错状态：'success'（完全纠错）/'partial_correct'（部分纠错）/'failed'（完全失败）
        """
        if len(encoded_packet_bytes) != self.encoded_packet_size:
            raise ValueError(f"编码包大小必须为 {self.encoded_packet_size} 字节")

        erase_pos = erase_pos_bytes or []
        try:
            # 正常纠错：返回完全修复的结果
            res = self.rsc.decode(encoded_packet_bytes, erase_pos=erase_pos)
            return bytes(res[0]), 'success'
        
        except ReedSolomonError:
            # 错误超限：尝试提取底层纠错的中间结果（最优恢复）
            try:
                msg_in = bytearray(encoded_packet_bytes)
                # 调用底层纠错函数，不抛异常，获取部分纠错结果
                msg_out, num_err, err_pos = rs_correct_msg(
                    msg_in, self.nsym, fcr=self.rsc.fcr, 
                    generator=self.rsc.generator, erase_pos=erase_pos
                )
                # 提取原始数据部分（前packet_size字节）
                partial_data = bytes(msg_out[:self.packet_size])
                return partial_data, 'partial_correct'
            
            except Exception:
                # 完全无法纠错：返回编码包前packet_size字节作为兜底
                fallback_data = encoded_packet_bytes[:self.packet_size]
                return fallback_data, 'failed'

    # ---------------------------------------------------------
    # 整体编码/解码（适配误码率仿真）
    # ---------------------------------------------------------
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        输入：原始比特流（np.uint8）
        输出：RS编码后的比特流
        """
        # 转换为字节并记录填充数（用于解码还原）
        data_bytes, pad_bits = self.bits_to_bytes(bits)
        self._original_bit_len = len(bits)  # 保存原始比特长度
        self._pad_bits = pad_bits           # 保存填充比特数
        self._original_byte_len = len(data_bytes)  # 保存原始字节长度

        # 分包 + 逐包编码
        packets = self.split_bytes_to_packets(data_bytes)
        encoded_packets = [self.encode_single_packet(pkt) for pkt in packets]
        encoded_bytes = b''.join(encoded_packets)
        
        # 转换为比特流返回
        return self.bytes_to_bits(encoded_bytes)

    def decode(self, bits: np.ndarray, erase_pos_bytes: Optional[List[int]] = None) -> Tuple[np.ndarray, dict]:
        """
        输入：含错比特流
        输出：(最优恢复比特流, debug信息字典)
        debug信息包含：
            - packet_count: 总包数
            - success_packets: 完全纠错包索引
            - partial_packets: 部分纠错包索引
            - failed_packets: 完全失败包索引
            - original_bit_len: 原始比特长度（用于误码率计算）
        """
        # 初始化debug信息（核心：统计各状态包数量，方便误码率分析）
        debug_info = {
            "packet_count": 0,
            "success_packets": [],
            "partial_packets": [],
            "failed_packets": [],
            "original_bit_len": self._original_bit_len
        }

        # 比特转字节（保留填充，编码包长度固定）
        coded_bytes, _ = self.bits_to_bytes(bits)
        if len(coded_bytes) % self.encoded_packet_size != 0:
            raise ValueError(f"编码数据长度必须是 {self.encoded_packet_size} 的倍数")

        # 拆分编码包
        encoded_packets = []
        for i in range(0, len(coded_bytes), self.encoded_packet_size):
            encoded_packets.append(coded_bytes[i:i+self.encoded_packet_size])
        debug_info["packet_count"] = len(encoded_packets)

        # 逐包解码（异常容错 + 状态记录）
        decoded_packets = []
        for pkt_idx, pkt in enumerate(encoded_packets):
            # 计算当前包的相对擦除位置
            pkt_erase_pos = None
            if erase_pos_bytes is not None:
                start = pkt_idx * self.encoded_packet_size
                end = start + self.encoded_packet_size
                pkt_erase_pos = [p - start for p in erase_pos_bytes if start <= p < end]
            
            # 解码（捕获异常，返回最优结果）
            pkt_data, pkt_status = self.decode_single_packet(pkt, pkt_erase_pos)
            decoded_packets.append(pkt_data)

            # 记录纠错状态（用于后续误码率统计）
            if pkt_status == 'success':
                debug_info["success_packets"].append(pkt_idx)
            elif pkt_status == 'partial_correct':
                debug_info["partial_packets"].append(pkt_idx)
            else:
                debug_info["failed_packets"].append(pkt_idx)

        # 合并解码包并恢复原始长度（关键：保证误码率计算准确）
        original_byte_len = self._original_byte_len
        decoded_bytes = self.combine_packets_to_bytes(decoded_packets, original_byte_len)
        
        # 转换为比特流，去除编码时的填充比特
        decoded_bits = self.bytes_to_bits(decoded_bytes, self._pad_bits)

        # 严格截断到原始比特长度（避免填充比特干扰误码率）
        if len(decoded_bits) > self._original_bit_len:
            decoded_bits = decoded_bits[:self._original_bit_len]

        return decoded_bits

    # ---------------------------------------------------------
    # 错误注入（适配误码率仿真，可复现）
    # ---------------------------------------------------------
    def introduce_random_byte_errors(self, bits: np.ndarray, n_byte_errors: int, seed: Optional[int] = None) -> np.ndarray:
        """
        精准注入指定数量的字节错误，支持随机种子（复现仿真结果）
        """
        if seed is not None:
            np.random.seed(seed)  # 固定种子，保证仿真可复现

        data_bytes, pad_bits = self.bits_to_bytes(bits)
        arr = np.frombuffer(data_bytes, dtype=np.uint8).copy()

        if n_byte_errors < 0 or n_byte_errors > len(arr):
            raise ValueError(f"错误数必须在 0~{len(arr)} 之间")
        if n_byte_errors == 0:
            return self.bytes_to_bits(arr.tobytes(), pad_bits)

        # 随机选择错误位置，确保每个位置只改一次
        error_pos = np.random.choice(len(arr), n_byte_errors, replace=False)
        for pos in error_pos:
            # 随机替换为不同的字节值（确保产生错误）
            old_val = arr[pos]
            new_val = np.random.randint(0, 256)
            while new_val == old_val:
                new_val = np.random.randint(0, 256)
            arr[pos] = new_val

        # 转回比特流并去除填充，截断到原始长度
        corrupted_bits = self.bytes_to_bits(arr.tobytes(), pad_bits)
        return corrupted_bits[:len(bits)]

    # ---------------------------------------------------------
    # 工具属性（方便仿真）
    # ---------------------------------------------------------
    @property
    def max_correctable_bytes(self):
        """返回最大可纠正字节错误数"""
        return self.nsym // 2