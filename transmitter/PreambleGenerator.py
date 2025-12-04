import numpy as np

class PreambleGenerator:
    """
    前导码生成器：按802.15.3d协议生成SYNC+SFD+CES序列
    - SYNC：长训练序列（用于粗同步）
    - SFD：帧起始定界符（用于细同步）
    - CES：信道估计序列（用于信道估计）
    """
    def __init__(self, params):
        self.params = params
        self.preamble_type = self.params.get("Preamble_type")  # 默认long类型
        self._init_preamble_config()
        # 预生成基础序列
        self.a128, self.b128 = self._generate_base_sequences()
        self.a256, self.b256 = self._generate_256_sequences()
        self.a512, self.b512 = self._generate_512_sequences()

    def _init_preamble_config(self):
        """初始化前导码长度配置"""
        if self.preamble_type == "short":
            self.sync_repeat = 14  # short类型重复14次
        elif self.preamble_type == "long":
            self.sync_repeat = 28  # long类型重复28次
        else:
            raise ValueError(f"不支持的前导码类型：{self.preamble_type}")

    def _hex_to_bpsk(self, hex_str):
        """
        十六进制字符串转BPSK符号序列（0→-1，1→+1）
        :param hex_str: 十六进制字符串
        :return: BPSK符号序列（numpy数组）
        """
        # 十六进制转二进制字符串（MSB在前）
        bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
        # 二进制转BPSK符号
        bpsk = np.array([1 if bit == '1' else -1 for bit in bin_str], dtype=np.float64)
        return bpsk

    def _generate_base_sequences(self):
        """生成128位基础序列a128和b128"""
        # MATLAB中定义的十六进制值
        hex_a128 = '5A5599963C33FFF00F00CCC36966AAA5'
        hex_b128 = 'A5AA6669C3CC000F0F00CCC36966AAA5'
        
        a128 = self._hex_to_bpsk(hex_a128)
        b128 = self._hex_to_bpsk(hex_b128)
        return a128, b128

    def _generate_256_sequences(self):
        """生成256位序列a256和b256"""
        # a256 = [b128, a128]
        a256 = np.concatenate([self.b128, self.a128])
        # b256 = [-b128, a128]
        b256 = np.concatenate([-self.b128, self.a128])
        return a256, b256

    def _generate_512_sequences(self):
        """生成512位序列a512和b512"""
        # a512 = [b256, a256]
        a512 = np.concatenate([self.b256, self.a256])
        # b512 = [-b256, a256]
        b512 = np.concatenate([-self.b256, self.a256])
        return a512, b512

    def generate_sync(self):
        """生成SYNC序列（a128重复M次，转置为列向量）"""
        sync = np.tile(self.a128, (self.sync_repeat, 1)).T.flatten()  # 保持与MATLAB一致的维度
        # 若需要列向量形式，使用：sync = np.tile(self.a128, self.sync_repeat).reshape(-1, 1)
        return sync

    def generate_sfd(self):
        """生成SFD序列（a128符号反转，转置为列向量）"""
        sfd = (-self.a128).reshape(-1, 1).flatten()  # 保持与MATLAB一致的维度
        return sfd

    def generate_ces(self):
        """生成CES序列：[b128, a512, b512, a256]，转置为列向量"""
        ces = np.concatenate([self.b128, self.a512, self.b512, self.a256]).reshape(-1, 1).flatten()
        return ces

    def generate(self):
        """生成完整前导码（SYNC+SFD+CES）"""
        sync = self.generate_sync()
        sfd = self.generate_sfd()
        ces = self.generate_ces()
        return sync, sfd, ces

# 测试代码
if __name__ == "__main__":
    # 模拟PHYParams类（实际使用时替换为真实类）
    from params.PHYParams import PHYParams
    params = PHYParams()
    preamble_gen = PreambleGenerator(params)
    sync, sfd, ces = preamble_gen.generate()
    
    print(f"SYNC长度：{len(sync)}")    # short:14*128=1792; long:28*128=3584
    print(f"SFD长度：{len(sfd)}")     # 128
    print(f"CES长度：{len(ces)}")     # 128+512+512+256=1408