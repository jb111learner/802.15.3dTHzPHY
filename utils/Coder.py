import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from typing import List, Optional, Tuple
from params.PHYParams import PHYParams

class RSCoder:
    """
    封装 reedsolo.RSCodec 的简单类。

    参数:
        nsym: 校验符号数量（bytes）。典型值如 32、16 等。纠错能力约为 nsym//2 个字节错误。
        c_exp: 有限域指数（可选），默认库内部通常为 8（GF(2^8)）。
    """

    def __init__(self, params):
        self.params = params
        self.nsym = self.params.get("rs_nsym")
        self.c_exp = self.params.get("rs_c_exp")
        self.rsc = RSCodec(self.nsym) if self.c_exp is None else RSCodec(self.nsym, c_exp=self.c_exp)

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
    # 编码：输入 bit → 输出 bit
    # ---------------------------------------------------------

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        输入：bit np.ndarray(uint8)
        输出：bit np.ndarray(uint8)
        """
        byte_data = self.bits_to_bytes(bits)
        coded_bytes = self.rsc.encode(byte_data)
        return self.bytes_to_bits(coded_bytes)

    # ---------------------------------------------------------
    # 解码：输入 bit → 输出 bit
    # ---------------------------------------------------------

    def decode(self, bits: np.ndarray, erase_pos_bytes: Optional[List[int]] = None) -> Tuple[np.ndarray, dict]:
        """
        输入：bit数组（可能含错误）
        可选：erase_pos_bytes (按字节索引，不是按bit)
        输出：(bit数组, debug信息)
        """
        coded_bytes = self.bits_to_bytes(bits)

        try:
            if erase_pos_bytes is None:
                res = self.rsc.decode(coded_bytes)
            else:
                res = self.rsc.decode(coded_bytes, erase_pos=erase_pos_bytes)

            # res = (decoded_message_bytearray, ecc_bytearray)
            decoded_bytes = bytes(res[0])
            decoded_bits = self.bytes_to_bits(decoded_bytes)

            debug = {
                "status": "ok",
                "nsym": self.nsym,
            }
            return decoded_bits, debug

        except ReedSolomonError as e:
            debug = {"status": "fail", "error": str(e)}
            raise ReedSolomonError(debug)

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

    @property
    def max_correctable_bytes(self):
        return self.nsym // 2



# # ---------- 简单使用示例 ----------
# if __name__ == "__main__":
#     import numpy as np

#     wrapper = RSCoder(nsym=16)

#     msg_str = b"HELLO RS 1234"
#     msg_bits = np.unpackbits(np.frombuffer(msg_str, dtype=np.uint8))

#     # 编码
#     coded_bits = wrapper.encode(msg_bits)

#     # 注入 5 个字节错误，保证 ≤ nsym//2 = 8
#     corrupted_bits = wrapper.introduce_random_byte_errors(coded_bits, 5)

#     # 解码
#     decoded_bits, info = wrapper.decode(corrupted_bits)

#     # 验证
#     decoded_bytes = np.packbits(decoded_bits).tobytes()[:len(msg_str)]
#     print("是否恢复正确：", decoded_bytes == msg_str)

