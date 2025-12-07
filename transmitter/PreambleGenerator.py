import numpy as np

class PreambleGenerator:
    """
    前导码生成器：按802.15.3d协议生成SYNC+SFD+CES序列，支持相位旋转
    - SYNC：长训练序列（用于粗同步）
    - SFD：帧起始定界符（用于细同步）
    - CES：信道估计序列（用于信道估计）
    输出序列为复数类型（带指定相位旋转），适配基带信号处理
    """
    def __init__(self, params):
        self.params = params
        self.preamble_type = self.params.get("Preamble_type")  # 默认long类型
        self.phase_rot = self.params.get("phase_rotation")  # 相位旋转角度（弧度）
        self._init_preamble_config()
        # 预生成基础序列（复数类型，带相位旋转）
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

    def _apply_phase_rotation(self, seq):
        """对复数序列应用相位旋转：seq * exp(j*phase_rot)"""
        rot_factor = np.exp(1j * self.phase_rot)  # 旋转因子 e^jθ
        return seq * rot_factor

    def _hex_to_bpsk(self, hex_str):
        """
        十六进制字符串转复数BPSK符号序列（0→-1+0j，1→+1+0j），并应用相位旋转
        :param hex_str: 十六进制字符串
        :return: 复数BPSK符号序列（numpy数组，dtype=np.complex128）
        """
        # 十六进制转二进制字符串（MSB在前）
        bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
        # 二进制转复数BPSK符号（实数±1转换为复数形式）
        bpsk_real = np.array([1 if bit == '1' else -1 for bit in bin_str], dtype=np.float64)
        bpsk_complex = bpsk_real.astype(np.complex128)
        # 应用相位旋转
        bpsk_rot = self._apply_phase_rotation(bpsk_complex)
        return bpsk_rot

    def _generate_base_sequences(self):
        """生成128位基础序列a128和b128（复数类型，带相位旋转）"""
        # MATLAB中定义的十六进制值
        hex_a128 = '5A5599963C33FFF00F00CCC36966AAA5'
        hex_b128 = 'A5AA6669C3CC000F0F00CCC36966AAA5'
        
        a128 = self._hex_to_bpsk(hex_a128)
        b128 = self._hex_to_bpsk(hex_b128)
        return a128, b128

    def _generate_256_sequences(self):
        """生成256位序列a256和b256（复数类型，带相位旋转）"""
        # a256 = [b128, a128]
        a256 = np.concatenate([self.b128, self.a128])
        # b256 = [-b128, a128]（复数取反后自动继承相位旋转）
        b256 = np.concatenate([-self.b128, self.a128])
        return a256, b256

    def _generate_512_sequences(self):
        """生成512位序列a512和b512（复数类型，带相位旋转）"""
        # a512 = [b256, a256]
        a512 = np.concatenate([self.b256, self.a256])
        # b512 = [-b256, a256]（复数取反后自动继承相位旋转）
        b512 = np.concatenate([-self.b256, self.a256])
        return a512, b512

    def generate_sync(self):
        """生成SYNC序列（a128重复M次，复数类型，带相位旋转）"""
        sync = np.tile(self.a128, (self.sync_repeat, 1)).T.flatten()
        return sync

    def generate_sfd(self):
        """生成SFD序列（a128符号反转，复数类型，带相位旋转）"""
        sfd = (-self.a128).reshape(-1, 1).flatten()
        return sfd

    def generate_ces(self):
        """生成CES序列：[b128, a512, b512, a256]，复数类型，带相位旋转"""
        ces = np.concatenate([self.b128, self.a512, self.b512, self.a256]).reshape(-1, 1).flatten()
        return ces

    def generate(self):
        """生成完整前导码（SYNC+SFD+CES），均为复数类型且带相位旋转"""
        sync = self.generate_sync()
        sfd = self.generate_sfd()
        ces = self.generate_ces()
        return sync, sfd, ces

# 测试代码
if __name__ == "__main__":
    # 模拟PHYParams类（实际使用时替换为真实类）
    try:
        from params.PHYParams import PHYParams
    except ImportError:
        # 备用模拟类（无PHYParams时使用）
        class PHYParams:
            def get(self, key, default=None):
                param_map = {
                    "Preamble_type": "long",
                    "phase_rotation": np.pi/4  # 旋转45度（π/4弧度）
                }
                return param_map.get(key, default)
        
    params = PHYParams()
    preamble_gen = PreambleGenerator(params)
    sync, sfd, ces = preamble_gen.generate()
    
    # 打印长度和类型信息
    print(f"相位旋转角度：{params.get('phase_rotation')/np.pi}π 弧度")
    print(f"SYNC长度：{len(sync)}，类型：{sync.dtype}，示例值：{sync[:5]}")    # short:1792; long:3584
    print(f"SFD长度：{len(sfd)}，类型：{sfd.dtype}，示例值：{sfd[:5]}")     # 128
    print(f"CES长度：{len(ces)}，类型：{ces.dtype}，示例值：{ces[:5]}")     # 1408
    
    # 验证相位旋转效果（虚部不再为0）
    print(f"\nSYNC虚部最大值：{np.max(np.imag(sync)):.4f}")
    print(f"SYNC实部最大值：{np.max(np.real(sync)):.4f}")