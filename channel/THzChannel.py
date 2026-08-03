import numpy as np
from pathlib import Path
from core.BaseChannel import BaseChannel
from channel.MultipathChannel import MultipathChannel
from channel.AWGN import AWGN
from channel.CFO import CFO
from channel.IQImbalance import IQImbalance
from channel.npa.PowerAmplifier import ModifiedRappPA
from channel.npa.RappParameterLoader import RappParameterLoader
from channel.MIMOChannel import MIMOChannel
from channel.MeasuredChannel import MeasuredChannel
from params.PHYParams import PHYParams
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class THzChannel(BaseChannel):
    """
    THz信道主类
    """
    def __init__(self, params):
        super().__init__(params)
        self.enable_mimo = bool(self.params.get("enable_mimo", False))
        self.use_measured_channel = (
            str(self.params.get("multipath_source", "simulated")).lower()
            == "measured"
        )
        # 初始化子模块
        self.mimo_channel = MIMOChannel(params) if self.enable_mimo else None
        self.measured_channel = (
            MeasuredChannel(params)
            if self.use_measured_channel and not self.enable_mimo
            else None
        )
        self.multipath_chan = (
            MultipathChannel(params)
            if self.params.get("enable_multipath")
            and not self.enable_mimo
            and not self.use_measured_channel
            else None
        )
        self.awgn = AWGN(params) if self.params.get("enable_awgn") and not self.enable_mimo else None
        self.cfo = CFO(params) if self.params.get("enable_cfo") and not self.enable_mimo else None
        self.iq_imbalance = IQImbalance(params) if self.params.get("enable_iq_imbalance") and not self.enable_mimo else None
        self.power_amplifier = self._create_power_amplifier() if self.params.get("enable_pa") and not self.enable_mimo else None
        self.pa_diagnostics = None
        self.chan_true = None  # 多径启用后保存等效信道冲激响应，便于后续信道估计/均衡验证

    def _create_power_amplifier(self):
        """创建功率放大器，并按配置选择数据集参数或手动参数。"""
        model = str(self.params.get("pa_model", "modified_rapp")).strip().lower()
        if model != "modified_rapp":
            raise ValueError(
                f"Unsupported PA model '{model}'. Available model: modified_rapp"
            )

        keys = (
            "pa_G", "pa_Vsat", "pa_p", "pa_A", "pa_B", "pa_q1", "pa_q2",
            "pa_phase_unit", "pa_load_ohm", "pa_auto_input_scaling",
            "pa_input_power_dbm",
        )
        pa_params = {key: self.params.get(key) for key in keys}

        if self.params.get("pa_load_params_from_dataset", False):
            dataset_dir = Path(
                self.params.get("pa_rapp_dataset_dir", "M260003_Rapp_dataset")
            ).expanduser()
            if not dataset_dir.is_absolute():
                cwd_candidate = Path.cwd() / dataset_dir
                module_candidate = Path(__file__).resolve().parent / "npa" / dataset_dir
                dataset_dir = cwd_candidate if cwd_candidate.exists() else module_candidate

            fc_hz = float(self.params.get("pa_fc_Hz", 300e9))
            loaded_params = RappParameterLoader(str(dataset_dir)).load_params_for_fc(
                fc_hz / 1e9,
                nearest=bool(self.params.get("pa_use_nearest_fc", False)),
            )
            pa_params.update(loaded_params)

        return ModifiedRappPA(pa_params)

    def apply_power_amplifier(self, signal_dict):
        """应用发射端功放非线性，并保留信号字典中的元数据。"""
        if not self.params.get("enable_pa") or self.power_amplifier is None:
            return signal_dict

        self.signal_pa_dict = self.power_amplifier.process_signal_dict(signal_dict)
        self.pa_diagnostics = self.signal_pa_dict.get("pa_diagnostics")
        return self.signal_pa_dict

    def apply_multipath(self, signal_dict):
        """应用多径效应，输入输出均为 signal_dict。"""
        if self.measured_channel is not None:
            self.signal_multipath_dict = self.measured_channel.apply(signal_dict)
            self.chan_true = self.measured_channel.channel_impulse_response[0, 0]
            return self.signal_multipath_dict

        if not self.params.get("enable_multipath") or self.multipath_chan is None:
            return signal_dict

        self.signal_multipath_dict = self.multipath_chan.apply(signal_dict)
        self.chan_true = self.multipath_chan.chan_impulse
        return self.signal_multipath_dict

    def add_awgn(self, signal_dict):
        """添加AWGN噪声"""
        if not self.params.get("enable_awgn") or self.awgn is None:
            return signal_dict

        self.signal_awgn_dict = self.awgn.add_awgn(signal_dict)
        return self.signal_awgn_dict

    def add_cfo(self, signal_dict):
        """应用频偏"""
        if not self.params.get("enable_cfo") or self.cfo is None:
            return signal_dict

        self.signal_cfo_dict = self.cfo.apply_cfo(signal_dict)
        return self.signal_cfo_dict
    
    def apply_iq_imbalance(self, signal_dict, stage="rx"):
        """应用 I/Q 不平衡"""
        if not self.params.get("enable_iq_imbalance") or self.iq_imbalance is None:
            return signal_dict

        self.signal_iq_imbalance_dict = self.iq_imbalance.apply(signal_dict, stage=stage)
        return self.signal_iq_imbalance_dict

    def reset_state(self):
        """
        重置信道层状态。

        目前主要用于清空多径信道的帧间输入历史。
        后续如果加入 Rayleigh/Rician 时变衰落，也可以在这里统一重置随机相位、
        多普勒过程、块衰落状态等。
        """
        if self.multipath_chan is not None:
            self.multipath_chan.reset_state()

    # def apply_delay(self, signal):
    #     """添加传输时延（前置零符号）"""
    #     delay = self.params.get("delay")
    #     signal_with_delay = np.concatenate([np.zeros(delay, dtype=complex), signal])
    #     return signal_with_delay

    def run(self, signal_dict):
        """执行完整信道流程：多径→噪声→频偏（可选添加相位噪声/时延）"""
        if self.enable_mimo:
            if self.params.get("enable_pa", False):
                raise ValueError("MIMO 分支暂不支持功率放大器非线性，请先关闭 enable_pa")
            self.rx_signal_dict = self.mimo_channel.apply(signal_dict)
            self.chan_true = self.mimo_channel.channel_impulse_response
            return self.rx_signal_dict
        # 0. 可选 TX 端 IQ 不平衡
        signal = self.apply_iq_imbalance(signal_dict, stage="tx")
        # 发射机 IQ 调制器之后、传播信道之前加入功率放大器。
        signal = self.apply_power_amplifier(signal)
        # 1. 多径效应
        signal = self.apply_multipath(signal)
        # 2. 添加AWGN噪声
        signal = self.add_awgn(signal)
        # 3. 应用CFO
        signal = self.add_cfo(signal)
        # 4. 可选 RX 端 IQ 不平衡
        signal = self.apply_iq_imbalance(signal, stage="rx")
        # 保存接收信号
        if self.measured_channel is not None:
            signal["measured_channel_diagnostics"] = dict(
                self.measured_channel.diagnostics
            )
            signal["measured_channel_impulse_response"] = (
                self.measured_channel.channel_impulse_response[0, 0].copy()
            )
        self.rx_signal_dict = signal
        return self.rx_signal_dict

def plot_eye_diagram(signal, symbol_period, num_symbols=100, ax=None):
    """
    绘制眼图
    参数：
        signal: 输入复信号（接收信号）
        symbol_period: 每个符号的采样点数（符号周期）
        num_symbols: 用于绘制眼图的符号数量
        ax: 绘图的坐标轴对象
    """
    if ax is None:
        ax = plt.gca()
    
    # 选择信号段（跳过前导码/训练序列，取稳定部分）
    start_idx = 10000  # 跳过初始过渡部分
    end_idx = start_idx + num_symbols * symbol_period
    signal_segment = signal[start_idx:end_idx]
    
    # 提取实部（也可以绘制虚部或幅度）
    signal_real = np.real(signal_segment)
    
    # 绘制眼图：将每个符号周期的波形叠加
    for i in range(num_symbols):
        start = i * symbol_period
        end = start + symbol_period
        if end <= len(signal_real):
            ax.plot(np.arange(symbol_period), signal_real[start:end], 
                    color='blue', alpha=0.1, linewidth=0.8)
    
    # 美化眼图
    ax.set_title('接收信号眼图（实部）', fontweight='bold')
    ax.set_xlabel('符号周期内采样点')
    ax.set_ylabel('信号幅度（实部）')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, symbol_period)

# 测试
if __name__ == "__main__":
    from transmitter.THzTransmitter import THzTransmitter

    # 初始化参数和发射机
    params = PHYParams()
    params.apply_dict(
    {
        "enable_iq_imbalance": True,
        "iq_imbalance_position": "rx",
        "iq_imbalance_model": "fd",
        "iq_gain_imbalance_db": 2.0,
        "iq_phase_imbalance_deg": 5.0,
        "iq_gI_taps": [1.0, 0.08, -0.03],
        "iq_gQ_taps": [1.0, -0.12, 0.04],
        "iq_power_normalize": False,
    }
    )
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    print(f"发射信号长度：{len(tx_signal_dict['signal_stream'])}, 接收信号长度：{len(rx_signal_dict['signal_stream'])}")
    print(f"采样率：{rx_signal_dict['sample_rate_Hz']} Hz，时长：{rx_signal_dict['duration_seconds']} 秒")
    rx_signal = rx_signal_dict['signal_stream']

    shaped_signal_dict = transmitter.tx_signal_dict
    preamble = transmitter.preamble
    signal_power = np.mean(np.abs(shaped_signal_dict["signal_stream"]) ** 2)
    print(f"成型后信号功率：{signal_power}")
    print(f"成型后信号前100点：{shaped_signal_dict['signal_stream'][:100]}")
    print(f"成型后信号长度：{shaped_signal_dict['signal_length']}")
    print(f"采样率：{shaped_signal_dict['sample_rate_Hz']} Hz，时长：{shaped_signal_dict['duration_seconds']} 秒")

    fig, axes = plt.subplots(1, 3, figsize=(23, 4))
    axes[0].plot(shaped_signal_dict['signal_stream'][len(preamble)*4:len(preamble)*4+500])
    axes[0].set_title("成形滤波信号（前500点）")
    axes[0].set_xlabel("采样点索引")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)
    axes[1].plot(shaped_signal_dict['up'][len(preamble)*4:len(preamble)*4+500], color='orange')
    axes[1].set_title("上采样信号（前500点）")
    axes[1].set_xlabel("采样点索引")
    axes[1].set_ylabel("幅度")
    axes[1].grid(True)
    axes[2].plot(rx_signal[len(preamble)*4:len(preamble)*4+500], color='orange')
    axes[2].set_title("接受信号（前500点）")
    axes[2].set_xlabel("采样点索引")
    axes[2].set_ylabel("幅度")
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()


    if params.get("link_mode") == "sc-fde":
        # 计算符号周期（每个符号的采样点数）
        symbol_period = params.get("oversampling") # 每个符号的采样点数
        
        # 信号可视化（调整布局为2x3，增加眼图子图）
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('太赫兹接收信号分析', fontsize=16, fontweight='bold')
        
        # 子图1: 时域波形（前1000个采样点）
        ax1.plot(np.arange(1000), np.real(rx_signal[:1000]), label='实部', alpha=0.8, linewidth=0.8)
        ax1.plot(np.arange(1000), np.imag(rx_signal[:1000]), label='虚部', alpha=0.8, linewidth=0.8)
        ax1.set_title('信号时域波形（前1000采样点）')
        ax1.set_xlabel('采样点')
        ax1.set_ylabel('幅值')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: 频域频谱
        fft_signal = np.fft.fft(rx_signal)
        freq = np.fft.fftfreq(len(fft_signal), 1/(rx_signal_dict.get("sample_rate_Hz")))
        ax2.plot(freq/1e9, 20*np.log10(np.abs(fft_signal)))
        ax2.set_title('信号频域频谱')
        ax2.set_xlabel('频率 (GHz)')
        ax2.set_ylabel('幅度 (dB)')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 调制符号星座图
        ax3.scatter(np.real(rx_signal[10000:20000]), 
                    np.imag(rx_signal[10000:20000]), 
                    s=5, alpha=0.6, c='orange')
        ax3.set_title('接收星座图')
        ax3.set_xlabel('实部')
        ax3.set_ylabel('虚部')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 子图4: 信号功率分布
        power = np.abs(rx_signal[10000:11000])**2
        ax4.plot(np.arange(1000), power, linewidth=0.8, color='green')
        ax4.set_title('信号功率分布（前1000采样点）')
        ax4.set_xlabel('采样点')
        ax4.set_ylabel('功率 (W)')
        ax4.grid(True, alpha=0.3)
        
        # 子图5: 接收信号眼图（实部）
        plot_eye_diagram(rx_signal, symbol_period, num_symbols=2000, ax=ax5)
        
        # 子图6: 虚部眼图（可选）
        # 重新定义虚部眼图绘制
        start_idx = 10000
        end_idx = start_idx + 2000 * symbol_period
        signal_segment = rx_signal[start_idx:end_idx]
        signal_imag = np.imag(signal_segment)
        for i in range(2000):
            start = i * symbol_period
            end = start + symbol_period
            if end <= len(signal_imag):
                ax6.plot(np.arange(symbol_period), signal_imag[start:end], 
                        color='red', alpha=0.1, linewidth=0.8)
        ax6.set_title('接收信号眼图（虚部）', fontweight='bold')
        ax6.set_xlabel('符号周期内采样点')
        ax6.set_ylabel('信号幅度（虚部）')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, symbol_period)

        plt.tight_layout()
        plt.show()
