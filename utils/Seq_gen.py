# utils/Scrambler.py
import numpy as np
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor


class Generator:
    """
    扰码器/解扰码器：严格遵循3GPP TS 38.211标准实现
    核心：两个31阶m序列异或生成Gold序列，完全匹配5G NR物理层规范
    """
    def __init__(self, params):
        self.params = params
        self.Nc = 1600  # 3GPP标准固定偏移
        self.c_init = params.get("c_init")  # 初始配置值
        
        # 初始化x1序列（固定初始状态：x1(0)=1，其余为0）
        self.x1_init = np.array([1] + [0]*30, dtype=np.uint8)
        # 初始化x2序列（由c_init生成初始状态）
        self.x2_init = self._cinit_to_x2_init(self.c_init)
        
        # x1的反馈抽头：x^31 = x^3 + x^0 → 对应索引[30, 2]（0基）
        self.x1_poly = [31, 3]
        # x2的反馈抽头：x^31 = x^3 + x^2 + x^1 + x^0 → 对应索引[30, 2, 1, 0]（0基）
        self.x2_poly = [31, 3, 2, 1]

    def _cinit_to_x2_init(self, c_init):
        """
        将c_init（整数）转换为x2序列的31位初始状态
        3GPP标准：c_init = Σ_{i=0}^{30} x2(i)·2^i
        """
        if not (0 <= c_init < 2**31):
            raise ValueError(f"c_init必须在[0, 2^31-1]范围内，当前值：{c_init}")
        # 转换为31位二进制数组，高位在前
        x2_init = np.array([(c_init >> (30 - i)) & 1 for i in range(31)], dtype=np.uint8)
        return x2_init

    def generate_m_sequence(self, init_state, poly, length):
        """生成指定长度的m序列"""
        state = init_state.copy()
        seq = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            seq[i] = state[-1]  # 输出最低位
            # 计算反馈位
            feedback = 0
            for tap in poly:
                feedback ^= state[tap - 1]  # poly是1基索引，转0基
            # 左移1位，反馈位填充到最低位
            state = np.roll(state, shift=-1)
            state[-1] = feedback
        return seq

    def generate_gold_sequence(self, length):
        """生成3GPP标准Gold序列"""
        # 生成x1和x2序列，长度为length + Nc
        x1_seq = self.generate_m_sequence(self.x1_init, self.x1_poly, length + self.Nc)
        x2_seq = self.generate_m_sequence(self.x2_init, self.x2_poly, length + self.Nc)
        # 跳过前Nc个比特，异或得到Gold序列
        c_seq = (x1_seq[self.Nc:] ^ x2_seq[self.Nc:]) % 2
        return c_seq





