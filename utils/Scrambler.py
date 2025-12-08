# utils/Scrambler.py
import numpy as np

class Scrambler:
    """
    扰码器/解扰码器：基于线性反馈移位寄存器（LFSR）实现
    遵循802.15.3d协议的扰码规则（也可自定义多项式）
    多项式：x^7 + x^4 + 1（经典7阶扰码器，避免长0/长1）
    """
    def __init__(self, params):
        self.params = params
        # 扰码器初始状态
        self.init_state = self.params.get("scrambler_init")
        self.poly = [7,4]  # 反馈抽头（对应x^7和x^4）
        self.state = self.init_state.copy()  # 当前移位寄存器状态

    def _reset(self):
        """重置移位寄存器状态"""
        self.state = self.init_state.copy()

    def scramble(self, data_bits):
        """
        扰码：将输入比特流与LFSR生成的伪随机序列逐位异或
        :param data_bits: 输入二进制比特流（np.array，0/1）
        :return: 扰码后的比特流
        """
        self._reset()
        scrambled_bits = np.zeros_like(data_bits)
        
        for i in range(len(data_bits)):
            # 1. 计算反馈位（抽头位异或）
            feedback = self.state[self.poly[0]-1] ^ self.state[self.poly[1]-1]
            # 2. 输出位 = 输入位 ^ 移位寄存器最后一位
            scrambled_bits[i] = data_bits[i] ^ self.state[-1]
            # 3. 移位寄存器右移
            self.state[1:] = self.state[:-1]
            # 4. 反馈位填入第一位
            self.state[0] = feedback
        
        return scrambled_bits

    def descramble(self, scrambled_bits):
        """
        解扰码：与扰码使用完全相同的LFSR流程（自同步特性）
        :param scrambled_bits: 扰码后的比特流
        :return: 解扰后的原始比特流
        """
        return self.scramble(scrambled_bits)  # 扰码和解扰算法完全一致