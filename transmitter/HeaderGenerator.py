# transmitter/HeaderGenerator.py
# -*- coding: utf-8 -*-

import numpy as np
from params.PHYParams import PHYParams


class HeaderGenerator:
    """
    THz-SC PHY头部生成器：生成37位头部比特序列
    字段分配：
    - b0-b3: MCS (4位)
    - b4-b7: 带宽编码 (4位)
    - b8-b11: 加扰器种子ID (4位)
    - b12-b13: PPRE (2位)
    - b14: PW (1位)
    - b15-b36: 帧长度 (22位)
    """

    def __init__(self, params: PHYParams):
        self.params = params
        self.mcs = self.params.get("mcs")  # 默认MCS=5
        self.bandwidth = self.params.get("bandwidth")    # 默认带宽2.16GHz
        self.scrambler_seed_id = self.params.get("scrambler_seed_id")  # 默认种子ID=3
        self.ppre = self.params.get("ppre")  # 默认PPRE=1
        self.pw = self.params.get("pw")      # 默认PW=0
        self.subframe_length = self.params.get("subframe_length")  # 默认帧长度=0

        # 初始化带宽映射表
        self.bandwidth_map = {
            2.16: 0,
            4.32: 1,
            8.64: 2,
            12.96: 3,
            14.28: 4,
            25.92: 5,
            51.84: 6,
            69.12: 7,
            34.56: 8
        }
        # 验证参数有效性
        self._validate_params()

    def _validate_params(self):
        """验证PHY参数的有效性"""
        # 验证MCS
        if not (0 <= self.mcs <= 15):
            raise ValueError(f"MCS必须在0-15范围内，当前值：{self.mcs}")
        
        # 验证带宽
        if self.bandwidth not in self.bandwidth_map:
            valid_bws = list(self.bandwidth_map.keys())
            raise ValueError(f"带宽值无效，有效值：{valid_bws} GHz")
        
        # 验证加扰器种子ID
        if not (0 <= self.scrambler_seed_id <= 15):
            raise ValueError(f"加扰器种子ID必须在0-15范围内，当前值：{self.scrambler_seed_id}")
        
        # 验证PPRE
        if not (0 <= self.ppre <= 3):
            raise ValueError(f"PPRE必须在0-3范围内，当前值：{self.ppre}")
        
        # 验证PW
        if self.pw not in (0, 1):
            raise ValueError(f"PW必须是0或1，当前值：{self.pw}")
        
        # 验证帧长度
        max_frame_len = 2**22 - 1
        if not (0 <= self.subframe_length <= max_frame_len):
            raise ValueError(f"帧长度必须在0-{max_frame_len}范围内，当前值：{self.frame_length}")

    def _dec_to_bin(self, value: int, num_bits: int, msb_first: bool = True) -> np.ndarray:
        """
        将十进制数转换为指定位数的二进制数组
        :param value: 十进制数值
        :param num_bits: 二进制位数
        :param msb_first: 是否高位在前（大端序）
        :return: 二进制数组（0/1）
        """
        # 转换为二进制字符串，去除前缀'0b'并补零到指定位数
        bin_str = bin(value)[2:].zfill(num_bits)
        # 转换为numpy数组
        bin_array = np.array([int(bit) for bit in bin_str], dtype=np.uint8)
        # 如果需要低位在前则反转
        if not msb_first:
            bin_array = bin_array[::-1]
        return bin_array

    def _get_bandwidth_code(self) -> int:
        """获取带宽对应的编码值"""
        return self.bandwidth_map[self.bandwidth]

    def display_header_info(self):
        """显示头部信息（类似MATLAB的display_header_info）"""
        bandwidth_code = self._get_bandwidth_code()
        print("THz-SC PHY 头部信息:")
        print(f"  MCS: {self.mcs} (0b{self.mcs:04b})")
        print(f"  带宽: {self.bandwidth:.2f} GHz (代码: {bandwidth_code}, 0b{bandwidth_code:04b})")
        print(f"  加扰器种子ID: {self.scrambler_seed_id} (0b{self.scrambler_seed_id:04b})")
        print(f"  PPRE: {self.ppre} (0b{self.ppre:02b})")
        print(f"  PW: {self.pw}")
        print(f"  帧长度: {self.subframe_length} 字节")

    def generate(self) -> np.ndarray:
        """生成37位PHY头部比特序列"""
        # 初始化37位头部数组
        header_bits = np.zeros(37, dtype=np.uint8)
        
        # 1. MCS字段 (b0-b3, 4位)
        header_bits[0:4] = self._dec_to_bin(self.mcs, 4)
        
        # 2. 带宽字段 (b4-b7, 4位)
        bandwidth_code = self._get_bandwidth_code()
        header_bits[4:8] = self._dec_to_bin(bandwidth_code, 4)
        
        # 3. 加扰器种子ID字段 (b8-b11, 4位)
        header_bits[8:12] = self._dec_to_bin(self.scrambler_seed_id, 4)
        
        # 4. PPRE字段 (b12-b13, 2位)
        header_bits[12:14] = self._dec_to_bin(self.ppre, 2)
        
        # 5. PW字段 (b14, 1位)
        header_bits[14] = self.pw
        
        # 6. 帧长度字段 (b15-b36, 22位)
        header_bits[15:37] = self._dec_to_bin(self.subframe_length, 22)
        
        # 显示头部信息
        self.display_header_info()
        
        return header_bits


# 测试代码（需确保PHYParams类包含所需字段）
if __name__ == "__main__":
    params = PHYParams()
    generator = HeaderGenerator(params)
    header_bits = generator.generate()
    
    print(f"\n生成的头部比特序列（共{len(header_bits)}位）:")
    print(header_bits)
