import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体（避免绘图中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PreambleGenerator:
    """
    前导码生成器：按802.15.3d协议生成SYNC+SFD+CES序列，支持相位旋转
    - 修复SYNC序列重复逻辑错误
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
        """十六进制字符串转复数BPSK符号序列，应用相位旋转"""
        bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
        bpsk_real = np.array([1 if bit == '1' else -1 for bit in bin_str], dtype=np.float64)
        bpsk_complex = bpsk_real.astype(np.complex128)
        bpsk_rot = self._apply_phase_rotation(bpsk_complex)
        return bpsk_rot

    def _generate_base_sequences(self):
        """生成128位基础序列a128和b128"""
        hex_a128 = '5A5599963C33FFF00F00CCC36966AAA5'
        hex_b128 = 'A5AA6669C3CC000F0F00CCC36966AAA5'
        a128 = self._hex_to_bpsk(hex_a128)
        b128 = self._hex_to_bpsk(hex_b128)
        return a128, b128

    def _generate_256_sequences(self):
        """生成256位序列a256和b256"""
        a256 = np.concatenate([self.b128, self.a128])
        b256 = np.concatenate([-self.b128, self.a128])
        return a256, b256

    def _generate_512_sequences(self):
        """生成512位序列a512和b512"""
        a512 = np.concatenate([self.b256, self.a256])
        b512 = np.concatenate([-self.b256, self.a256])
        return a512, b512

    def generate_sync(self):
        """生成SYNC序列（修复重复逻辑）"""
        self.sync = np.tile(self.a128, self.sync_repeat)  # 核心修复！
        return self.sync 

    def generate_sfd(self):
        """生成SFD序列"""
        self.sfd = -self.a128  # 无需reshape，一维数组直接取反
        return self.sfd

    def generate_ces(self):
        """生成CES序列"""
        self.ces = np.concatenate([self.b128, self.a512, self.b512, self.a256])
        return self.ces

    def generate(self):
        """生成完整前导码"""
        sync = self.generate_sync()
        sfd = self.generate_sfd()
        ces = self.generate_ces()
        return sync, sfd, ces

# ===================== 验证修复效果 =====================
def verify_sync_repetition():
    class PHYParams:
        def get(self, key, default=None):
            param_map = {
                "Preamble_type": "long",
                "phase_rotation": np.pi/4
            }
            return param_map.get(key, default)
    
    params = PHYParams()
    preamble_gen = PreambleGenerator(params)
    sync_seq = preamble_gen.generate_sync()
    a128 = preamble_gen.a128
    L = len(a128)
    repeat_times = preamble_gen.sync_repeat

    # 数值验证
    print("="*60)
    print(f"SYNC序列重复性验证（修复后）")
    print("="*60)
    print(f"基础a128长度：{L}")
    print(f"SYNC序列总长度：{len(sync_seq)}")
    print(f"理论重复次数：{repeat_times} | 实际重复次数：{len(sync_seq)/L}")
    
    max_error = 0
    for i in range(repeat_times):
        seg_start = i * L
        seg_end = (i+1) * L
        sync_seg = sync_seq[seg_start:seg_end]
        seg_error = np.mean(np.abs(sync_seg - a128))
        max_error = max(max_error, seg_error)
        if i < 5 or i == repeat_times-1:
            print(f"第{i+1}个重复段与a128的平均误差：{seg_error:.6f}")
    
    print(f"\n所有重复段的最大误差：{max_error:.6f}")
    if max_error < 1e-10:
        print("✅ 数值验证：SYNC序列重复性修复成功！")
    else:
        print("❌ 数值验证：仍存在重复段不一致问题！")

    # 可视化验证（可选，保留之前的可视化代码）
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # 子图1：SYNC整体波形
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(len(sync_seq)), np.real(sync_seq), label='SYNC实部', alpha=0.8, linewidth=0.8)
    ax1.plot(np.arange(len(sync_seq)), np.imag(sync_seq), label='SYNC虚部', alpha=0.8, linewidth=0.8)
    for i in range(1, repeat_times):
        ax1.axvline(x=i*L, color='red', linestyle='--', alpha=0.7, linewidth=1,
                    label='重复段分割线' if i==1 else "")
    ax1.set_title(f'SYNC序列整体波形（{repeat_times}次重复a128）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('幅值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2：前3个重复段实部对比
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['blue', 'green', 'orange']
    labels = ['第1段', '第2段', '第3段']
    for i in range(3):
        seg_start = i * L
        seg_end = (i+1) * L
        sync_seg = sync_seq[seg_start:seg_end]
        ax2.plot(np.arange(L), np.real(sync_seg), color=colors[i], 
                 label=labels[i], alpha=0.8, linewidth=1)
    ax2.set_title('前3个重复段实部对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('a128内采样点索引')
    ax2.set_ylabel('实部幅值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3：前3个重复段虚部对比
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(3):
        seg_start = i * L
        seg_end = (i+1) * L
        sync_seg = sync_seq[seg_start:seg_end]
        ax3.plot(np.arange(L), np.imag(sync_seg), color=colors[i], 
                 label=labels[i], alpha=0.8, linewidth=1)
    ax3.set_title('前3个重复段虚部对比', fontsize=12, fontweight='bold')
    ax3.set_xlabel('a128内采样点索引')
    ax3.set_ylabel('虚部幅值')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 子图4：互相关验证
    ax4 = fig.add_subplot(gs[1, 1])
    corr = np.correlate(np.real(sync_seq), np.real(a128), mode='full')
    corr = corr / np.max(corr)
    lag = np.arange(len(corr)) - (L - 1)
    
    ax4.plot(lag, corr, linewidth=1.5, color='darkorange')
    peak_lags = [i * L for i in range(repeat_times)]
    for i, peak_lag in enumerate(peak_lags):
        if peak_lag >= np.min(lag) and peak_lag <= np.max(lag):
            ax4.scatter(peak_lag, 1.0, color='red', s=50, zorder=5,
                        label='重复段峰值' if i==0 else "")
            ax4.axvline(x=peak_lag, color='red', linestyle=':', alpha=0.5)
    
    ax4.set_title('SYNC序列互相关验证（归一化）', fontsize=12, fontweight='bold')
    ax4.set_xlabel('延迟（lag）')
    ax4.set_ylabel('归一化互相关值')
    ax4.set_ylim([-0.1, 1.1])
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    fig.suptitle(f'SYNC序列重复性验证（修复后，相位旋转={params.get("phase_rotation")/np.pi:.2f}π）', 
                 fontsize=16, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    verify_sync_repetition()