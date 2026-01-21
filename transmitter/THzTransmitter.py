import numpy as np
import matplotlib.pyplot as plt
import json
from core.BaseTransmitter import BaseTransmitter
from transmitter.HeaderGenerator import HeaderGenerator
from transmitter.PreambleGenerator import PreambleGenerator
from transmitter.Modulator import THzModulator as Modulator
from transmitter.GIInserter import GIInserter
from transmitter.DataProcesser import BitStreamProcessor
# from utils.Rotator import Rotator 
from transmitter.Encoder import Encoder 
from transmitter.Pulseshaper import TxPulseShaper
from params.PHYParams import PHYParams
from transmitter.Scrambler import Scrambler 

class THzTransmitter(BaseTransmitter):
    """
    THz发射机主类
    """
    def __init__(self, params):
        super().__init__(params)
        # 初始化子模块
        self.preamble_gen = PreambleGenerator(params)
        self.modulator = Modulator(params)
        self.coder = Encoder(params)
        self.gi_inserter = GIInserter(params)
        self.pulse_shaper = TxPulseShaper(params)
        self.scrambler = Scrambler(params)  # 初始化扰码器
        self.assembler = BitStreamProcessor(params)  # 初始化比特流处理器
        # self.rotator = Rotator()  

    def generate_preamble(self):
        """生成前导码（SYNC+SFD+CES）"""
        self.sync, self.sfd, self.ces = self.preamble_gen.generate()
        self.preamble = np.concatenate([self.sync, self.sfd, self.ces])
    
    def assemble_frame(self):
        """组装帧"""
        self.data_bits_dict = self.assembler.run()
        return self.data_bits_dict

    def scramble_data(self):
        """扰码数据"""
        self.data_scrambled_dict = self.scrambler.scramble(self.data_bits_dict)
        return self.data_scrambled_dict

    def channel_encode(self):
        """信道编码（调用RSCoder）"""
        self.coded_bits_dict = self.coder.encode(self.data_scrambled_dict)
        return self.coded_bits_dict

    def modulate(self):
        """调制数据"""
        self.modulated_data_dict = self.modulator.modulate(self.coded_bits_dict)
        return self.modulated_data_dict

    def insert_gi(self):
        """插入GI"""
        self.data_with_gi_dict = self.gi_inserter.insert_gi(self.modulated_data_dict)
        return self.data_with_gi_dict

    def pulse_shaping(self):
        """脉冲成型"""
        shaped_signal_dict = self.pulse_shaper.shape_pulse(self.data_with_gi_dict)
        self.tx_signal_dict = shaped_signal_dict
        return self.tx_signal_dict

    def run(self):
        """执行完整发射流程"""
        self.assemble_frame()
        self.generate_preamble()
        self.scramble_data()
        self.channel_encode()
        self.modulate()
        self.insert_gi()
        self.pulse_shaping()
        return self.tx_signal_dict







def plot_eye_diagram(signal, symbol_period, num_symbols=1000, oversampling=1, ax=None):
    """
    绘制眼图
    参数：
        signal: 输入的复信号（时域波形）
        symbol_period: 符号周期（采样点数）
        num_symbols: 用于绘制眼图的符号数量
        oversampling: 上采样率
        ax: 绘图的坐标轴对象
    """
    if ax is None:
        ax = plt.gca()
    
    # 提取实部和虚部分别绘制眼图
    signal_real = np.real(signal)
    signal_imag = np.imag(signal)
    
    # 计算每个符号的采样点数
    samples_per_symbol = symbol_period * oversampling
    
    # 截取有效信号段（跳过前导码和延迟，取数据段）
    start_idx = int(len(signal) * 0.1)  # 跳过前10%的信号（前导码+延迟）
    end_idx = start_idx + num_symbols * samples_per_symbol
    if end_idx > len(signal):
        end_idx = len(signal)
    signal_segment = signal_real[start_idx:end_idx]
    
    # 重排数据为二维数组：每行一个符号周期的采样点
    num_rows = len(signal_segment) // samples_per_symbol
    eye_data = signal_segment[:num_rows * samples_per_symbol].reshape(-1, samples_per_symbol)
    
    # 绘制实部眼图
    for i in range(num_rows):
        ax.plot(eye_data[i], alpha=0.1, color='blue')
    
    # 美化眼图
    ax.set_title('信号眼图（实部）', fontweight='bold')
    ax.set_xlabel('符号周期内采样点')
    ax.set_ylabel('幅值（实部）')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, samples_per_symbol-1)

# 测试
if __name__ == "__main__":
    # 设置中文字体（避免绘图中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 初始化参数和发射机
    print("="*50)
    print("          太赫兹发射机信号测试程序")
    print("="*50)
    params = PHYParams()
    
    # 打印基础参数信息
    print("\n【1. 基础参数配置】")
    basic_params = {
        "数据时长": params.get("duration"),
        "采样率": params.get("sample_rate"),
        "采样点数": params.get("bit_length"),
        "每符号比特数": params.get("NCBPS"),
        "是否使用CP": params.get("is_cp"),
        "扰码器cinit": params.get("c_init"),
        "子帧长度": params.get("subframe_length"),
        "GI长度": params.get("gi_length"),
        "校验符号数": params.get("rs_nsym"),
        "编码包大小": params.get("rs_packet_size"),
        "上采样率": params.get("oversampling"),
        "滚降系数": params.get("rolloff"),
        "滤波器类型": params.get("filter_type"),
        "滤波器长度": params.get("filter_length"),
    }
    for key, value in basic_params.items():
        print(f"  {key}: {value}")


    # 初始化发射机
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 


    print("\n【2. 组帧输出】")     
    data_bits_dict = transmitter.data_bits_dict
    print(f"前100比特流：{data_bits_dict['bit_stream'][:100] if not data_bits_dict['is_big_bitstream'] else '超大比特流，已保存为文件'}")
    print(f"比特流长度：{data_bits_dict['bit_length']} bit")
    print(f"分帧配置：每帧{data_bits_dict['frame_bit_num']}bit，共{data_bits_dict['frame_num']}帧")
    print(f"采样率：{data_bits_dict['sample_rate_Hz']} Hz，时长：{data_bits_dict['duration_seconds']} 秒")
    print(f"补零数量：{data_bits_dict['padding_bit_num']} bit")
    print(f"是否超大比特流：{data_bits_dict['is_big_bitstream']}")

    print("\n【3. 扰码输出】")
    data_scrambled_dict = transmitter.data_scrambled_dict
    print(f"前100扰码比特流：{data_scrambled_dict['bit_stream'][:100]}")
    print(f"扰码后比特流长度：{data_scrambled_dict['bit_length']} bit")
    print(f"采样率：{data_scrambled_dict['sample_rate_Hz']} Hz，时长：{data_scrambled_dict['duration_seconds']} 秒")
    print(f"补零数量：{data_scrambled_dict['padding_bit_num']} bit")

    print("\n【4. 信道编码输出】")
    coded_bits_dict = transmitter.coded_bits_dict
    print(f"前100编码比特流：{coded_bits_dict['bit_stream'][:100]}")
    print(f"编码后比特流长度：{coded_bits_dict['bit_length']} bit")
    print(f"采样率：{coded_bits_dict['sample_rate_Hz']} Hz，时长：{coded_bits_dict['duration_seconds']} 秒")
    print(f"补零数量：{coded_bits_dict['padding_bit_num']} bit")
    print(f"编码增益：{coded_bits_dict['bit_length']/data_scrambled_dict['bit_length']:.2f}x")

    print("\n【5. 调制输出】")
    modulated_data_dict = transmitter.modulated_data_dict
    print(f"前100调制符号：{modulated_data_dict['symbol_stream'][:100]}")
    print(f"调制后符号长度：{modulated_data_dict['symbol_length']}")
    print(f"采样率：{modulated_data_dict['sample_rate_Hz']} Hz，时长：{modulated_data_dict['duration_seconds']} 秒")
    print(f"补零数量：{modulated_data_dict['padding_bit_num']} bit")
    print(f"每符号比特数：{params.get('NCBPS')} bit")
    transmitter.modulator.plot_constellation(modulated_data_dict["symbol_stream"][:200], use_english=True)

    print("\n【6. GI插入输出】")
    data_with_gi_dict = transmitter.data_with_gi_dict
    print(f"前100带GI符号：{data_with_gi_dict['symbol_stream'][:100]}")
    print(f"带GI符号长度：{data_with_gi_dict['symbol_length']}")
    print(f"采样率：{data_with_gi_dict['sample_rate_Hz']} Hz，时长：{data_with_gi_dict['duration_seconds']} 秒")
    print(f"补零数量：{data_with_gi_dict['padding_bit_num']} bit")

    print("\n【7. 脉冲成型输出】")
    tx_signal_dict = transmitter.tx_signal_dict
    tx_signal = tx_signal_dict['signal_stream']
    print(f"前100成型信号点：{tx_signal[:100]}")
    print(f"成型信号长度：{tx_signal_dict['signal_length']}")
    print(f"采样率：{tx_signal_dict['sample_rate_Hz']} Hz，时长：{tx_signal_dict['duration_seconds']} 秒")
    
    print("\n【8. 信号核心参数】")
    print(f"\n  信号幅值范围: {np.min(np.abs(tx_signal)):.4f} ~ {np.max(np.abs(tx_signal)):.4f}")
    print(f"  信号平均功率: {np.mean(np.abs(tx_signal)**2):.4f} W")
    print(f"  信号峰值功率: {np.max(np.abs(tx_signal)**2):.4f} W")
    print(f"  信号类型: {'复数' if np.iscomplexobj(tx_signal) else '实数'}")

    print("\n【9. 信号可视化】")
    # 创建2行3列的子图布局（最后一列放眼图）
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 子图1: 时域波形（前1000个采样点）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(np.arange(5000), np.imag(tx_signal[5000:10000]), label='虚部', alpha=0.8, linewidth=0.1)
    ax1.plot(np.arange(5000), np.real(tx_signal[5000:10000]), label='实部', alpha=0.6, linewidth=0.8)
    ax1.set_title('信号时域波形（第5000-10000采样点）')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('幅值')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: 频域频谱
    ax2 = fig.add_subplot(gs[0, 1])
    fft_signal = np.fft.fft(tx_signal)
    freq = np.fft.fftfreq(len(fft_signal), 1/(tx_signal_dict.get("sample_rate_Hz")))
    ax2.plot(freq/1e9, 20*np.log10(np.abs(fft_signal)))
    ax2.set_title('信号频域频谱')
    ax2.set_xlabel('频率 (GHz)')
    ax2.set_ylabel('幅度 (dB)')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 信号功率分布
    ax4 = fig.add_subplot(gs[1, 0])
    power = np.abs(tx_signal[5000:10000])**2
    ax4.plot(np.arange(5000), power, linewidth=0.8, color='green')
    ax4.set_title('信号功率分布（第5000-10000采样点）')
    ax4.set_xlabel('采样点')
    ax4.set_ylabel('功率 (W)')
    ax4.grid(True, alpha=0.3)
    
    # 子图4: 眼图
    ax5 = fig.add_subplot(gs[1, 1])  # 占据第2行后两列
    # 计算符号周期（根据采样率和符号速率）
    symbol_period = int(params.get("oversampling"))  # 每个符号的采样点数
    plot_eye_diagram(
        signal=tx_signal,
        symbol_period=6,  # 基础符号周期
        num_symbols=1000,  # 绘制500个符号的眼图
        oversampling=symbol_period,
        ax=ax5
    )
    
    fig.suptitle('太赫兹发射机信号分析', fontsize=16, fontweight='bold')
    plt.show()