import numpy as np
import matplotlib.pyplot as plt
from params.PHYParams import PHYParams
from utils.Seq_gen import Generator
# 设置中文字体（避免绘图中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PreambleInsertor:
    """
    前导码插入器：按802.15.3d协议插入SYNC+SFD+CES序列，支持相位旋转
    """
    def __init__(self, params):
        self.params = params
        self.generator = Generator(params)
        self.preamble_type = self.params.get("Preamble_type")  # 默认long类型
        self._init_preamble_config()
        # 预生成基础序列（复数类型，带相位旋转）
        self.a128, self.b128 = self.generator.generate_base_sequences()
        self.a256, self.b256 = self.generator.generate_256_sequences()
        self.a512, self.b512 = self.generator.generate_512_sequences()
        self.preamble = self._generate_preamble()

        # 输出参数 
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.signal_length = None  # 符号长度
        self.padding_bit_num = 0   # 补零的比特数
        self.frame_symbol_num = None  # 每帧符号数  
        self.frame_valid_symbol_num = None  # 每帧有效符号数
        self.frame_num = None  # 帧数

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 比特流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 比特长度,
            "padding_bit_num": 补零比特数,
            "frame_symbol_num": 单帧符号数,
            "frame_num": 帧数,
        """
        # 校验输入字典完整性
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num","frame_symbol_num","frame_num"    
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        if data_dict["frame_symbol_num"] * data_dict["frame_num"] != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的分帧信息不匹配")
        preamble_len = len(self.preamble)
        # 计算符号长度(每帧插入前导码)
        self.signal_length = (data_dict["frame_symbol_num"] + preamble_len) * data_dict["frame_num"]
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.duration = self.signal_length / self.sample_rate
        self.padding_bit_num = data_dict["padding_bit_num"]  
        self.frame_symbol_num = data_dict["frame_symbol_num"] + preamble_len
        self.frame_valid_symbol_num = data_dict["frame_symbol_num"]
        self.frame_num = data_dict["frame_num"]         

    def _init_preamble_config(self):
        """初始化前导码长度配置"""
        if self.preamble_type == "short":
            self.sync_repeat = 14  # short类型重复14次
        elif self.preamble_type == "long":
            self.sync_repeat = 28  # long类型重复28次
        else:
            raise ValueError(f"不支持的前导码类型：{self.preamble_type}")

    def _generate_sync(self):
        """生成SYNC序列（修复重复逻辑）"""
        self.sync = np.tile(self.a128, self.sync_repeat)  # 核心修复！
        return self.sync 

    def _generate_sfd(self):
        """生成SFD序列"""
        self.sfd = -self.a128  # 无需reshape，一维数组直接取反
        return self.sfd

    def _generate_ces(self):
        """生成CES序列"""
        self.ces = np.concatenate([self.b128, self.a512, self.b512, self.a256])
        return self.ces

    def _generate_preamble(self):
        """生成完整前导码"""
        self.sync = self._generate_sync()
        self.sfd = self._generate_sfd()
        self.ces = self._generate_ces()
        return np.concatenate([self.sync, self.sfd, self.ces])
    
    def insert_preamble(self, data_dict):
        """在符号序列中插入前导码"""
        self._verification_data(data_dict)
        symbols = data_dict["signal_stream"]
        # 将每一帧的前导码插入到符号序列开头
        frame_symbols = symbols.reshape(-1, int(self.frame_valid_symbol_num))
        num_frames = frame_symbols.shape[0]  # 帧的数量
        if num_frames != self.frame_num:
            return ValueError("输入数据字典中的分帧信息不匹配")
        # 将 preamble 扩展到每帧一行
        preamble_rows = np.tile(self.preamble, (num_frames, 1))  # shape (num_frames, len(preamble))
        # 沿列拼接
        frame_symbols = np.concatenate([preamble_rows, frame_symbols], axis=1)
        # 展平回一维
        symbols = frame_symbols.flatten()
        if len(symbols) != self.signal_length:
            raise ValueError("插入前导码后，符号序列长度与预期不匹配")
        
        result_dict = {
            "signal_stream": symbols,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.signal_length,
            "padding_bit_num": self.padding_bit_num,
            "frame_symbol_num": self.frame_symbol_num,
            "frame_valid_symbol_num": self.frame_valid_symbol_num,
            "frame_num": self.frame_num,
        } 
        return result_dict



# # ===================== 验证修复效果 =====================
# def verify_sync_repetition():
#     class PHYParams:
#         def get(self, key, default=None):
#             param_map = {
#                 "Preamble_type": "long",
#                 "phase_rotation": np.pi/4
#             }
#             return param_map.get(key, default)
    
#     params = PHYParams()
#     preamble_gen = PreambleGenerator(params)
#     sync_seq = preamble_gen.generate_sync()
#     a128 = preamble_gen.a128
#     L = len(a128)
#     repeat_times = preamble_gen.sync_repeat

#     # 数值验证
#     print("="*60)
#     print(f"SYNC序列重复性验证（修复后）")
#     print("="*60)
#     print(f"基础a128长度：{L}")
#     print(f"SYNC序列总长度：{len(sync_seq)}")
#     print(f"理论重复次数：{repeat_times} | 实际重复次数：{len(sync_seq)/L}")
    
#     max_error = 0
#     for i in range(repeat_times):
#         seg_start = i * L
#         seg_end = (i+1) * L
#         sync_seg = sync_seq[seg_start:seg_end]
#         seg_error = np.mean(np.abs(sync_seg - a128))
#         max_error = max(max_error, seg_error)
#         if i < 5 or i == repeat_times-1:
#             print(f"第{i+1}个重复段与a128的平均误差：{seg_error:.6f}")
    
#     print(f"\n所有重复段的最大误差：{max_error:.6f}")
#     if max_error < 1e-10:
#         print("✅ 数值验证：SYNC序列重复性修复成功！")
#     else:
#         print("❌ 数值验证：仍存在重复段不一致问题！")

#     # 可视化验证（可选，保留之前的可视化代码）
#     fig = plt.figure(figsize=(20, 12))
#     gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
#     # 子图1：SYNC整体波形
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.plot(np.arange(len(sync_seq)), np.real(sync_seq), label='SYNC实部', alpha=0.8, linewidth=0.8)
#     ax1.plot(np.arange(len(sync_seq)), np.imag(sync_seq), label='SYNC虚部', alpha=0.8, linewidth=0.8)
#     for i in range(1, repeat_times):
#         ax1.axvline(x=i*L, color='red', linestyle='--', alpha=0.7, linewidth=1,
#                     label='重复段分割线' if i==1 else "")
#     ax1.set_title(f'SYNC序列整体波形（{repeat_times}次重复a128）', fontsize=12, fontweight='bold')
#     ax1.set_xlabel('采样点')
#     ax1.set_ylabel('幅值')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # 子图2：前3个重复段实部对比
#     ax2 = fig.add_subplot(gs[0, 1])
#     colors = ['blue', 'green', 'orange']
#     labels = ['第1段', '第2段', '第3段']
#     for i in range(3):
#         seg_start = i * L
#         seg_end = (i+1) * L
#         sync_seg = sync_seq[seg_start:seg_end]
#         ax2.plot(np.arange(L), np.real(sync_seg), color=colors[i], 
#                  label=labels[i], alpha=0.8, linewidth=1)
#     ax2.set_title('前3个重复段实部对比', fontsize=12, fontweight='bold')
#     ax2.set_xlabel('a128内采样点索引')
#     ax2.set_ylabel('实部幅值')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
    
#     # 子图3：前3个重复段虚部对比
#     ax3 = fig.add_subplot(gs[1, 0])
#     for i in range(3):
#         seg_start = i * L
#         seg_end = (i+1) * L
#         sync_seg = sync_seq[seg_start:seg_end]
#         ax3.plot(np.arange(L), np.imag(sync_seg), color=colors[i], 
#                  label=labels[i], alpha=0.8, linewidth=1)
#     ax3.set_title('前3个重复段虚部对比', fontsize=12, fontweight='bold')
#     ax3.set_xlabel('a128内采样点索引')
#     ax3.set_ylabel('虚部幅值')
#     ax3.grid(True, alpha=0.3)
#     ax3.legend()
    
#     # 子图4：互相关验证
#     ax4 = fig.add_subplot(gs[1, 1])
#     corr = np.correlate(np.real(sync_seq), np.real(a128), mode='full')
#     corr = corr / np.max(corr)
#     lag = np.arange(len(corr)) - (L - 1)
    
#     ax4.plot(lag, corr, linewidth=1.5, color='darkorange')
#     peak_lags = [i * L for i in range(repeat_times)]
#     for i, peak_lag in enumerate(peak_lags):
#         if peak_lag >= np.min(lag) and peak_lag <= np.max(lag):
#             ax4.scatter(peak_lag, 1.0, color='red', s=50, zorder=5,
#                         label='重复段峰值' if i==0 else "")
#             ax4.axvline(x=peak_lag, color='red', linestyle=':', alpha=0.5)
    
#     ax4.set_title('SYNC序列互相关验证（归一化）', fontsize=12, fontweight='bold')
#     ax4.set_xlabel('延迟（lag）')
#     ax4.set_ylabel('归一化互相关值')
#     ax4.set_ylim([-0.1, 1.1])
#     ax4.grid(True, alpha=0.3)
#     ax4.legend()
    
#     fig.suptitle(f'SYNC序列重复性验证（修复后，相位旋转={params.get("phase_rotation")/np.pi:.2f}π）', 
#                  fontsize=16, fontweight='bold')
#     plt.show()

# if __name__ == "__main__":
#     params = PHYParams()
#     preamble_gen = PreambleGenerator(params)
#     sync, sfd, ces = preamble_gen.generate()
#     preamble = np.concatenate([sync, sfd, ces])
#     signal_power = np.mean(np.abs(preamble) ** 2)
#     print(f"前导码功率：{signal_power}")
#     print(f"前导码长度：{len(preamble)}")
