import numpy as np
import matplotlib.pyplot as plt
import json
from core.BaseTransmitter import BaseTransmitter
from transmitter.HeaderGenerator import HeaderGenerator
from transmitter.PreambleGenerator import PreambleGenerator
from transmitter.Modulator import THzModulator as Modulator
from transmitter.CPInserter import CPInserter
# from utils.Rotator import Rotator  # 信号旋转工具
from utils.Coder import RSCoder 
from transmitter.Pulseshaper import TxPulseShaper
from params.PHYParams import PHYParams
from utils.Scrambler import Scrambler  # 新增导入扰码器

class THzTransmitter(BaseTransmitter):
    """
    THz发射机主类：调度子模块完成“头部生成→前导码生成→数据调制→CP插入→信号组装”全流程
    """
    def __init__(self, params):
        super().__init__(params)
        # 初始化子模块
        self.header_gen = HeaderGenerator(params)
        self.preamble_gen = PreambleGenerator(params)
        self.modulator = Modulator(params)
        self.coder = RSCoder(params)
        self.cp_inserter = CPInserter(params)
        self.testbits_len = int(73728)  # 测试用数据比特长度
        self.pulse_shaper = TxPulseShaper(params)
        self.scrambler = Scrambler(params)  # 初始化扰码器
        self.data_scrambled = None  # 扰码后数据
        # self.rotator = Rotator()

    def add_header(self, data_bits):
        """生成PHY+MAC头部（调用HeaderGenerator）"""
        self.header_bits = self.header_gen.generate()
        # 添加头部到数据前面
        combined_bits = np.concatenate([self.header_bits, data_bits])
        return combined_bits    

    def generate_preamble(self):
        """生成前导码（SYNC+SFD+CES）"""
        self.sync, self.sfd, self.ces = self.preamble_gen.generate()
        self.preamble = np.concatenate([self.sync, self.sfd, self.ces])

    def scramble_data(self):
        """扰码数据（调用Scrambler）"""
        self.data_scrambled = self.scrambler.scramble(self.data_bits)
        return self.data_scrambled

    def channel_encode(self):
        """信道编码（调用RSCoder）"""
        self.coded_bits = self.coder.encode(self.data_scrambled)
        return self.coded_bits

    def modulate(self):
        """调制数据（调用Modulator）"""
        self.modulated_data = self.modulator.modulate(self.coded_bits)
        return self.modulated_data
    
    def insert_cp(self):
        """插入CP（调用CPInserter）"""
        self.data_with_cp = self.cp_inserter.insert_cp(self.modulated_data)
        return self.data_with_cp

    def assemble_signal(self):
        """组装发射信号：延迟 + 前导码 + 调制头部 + 带CP数据"""
        # 1. 生成随机数据比特
        self.data_bits = self._generate_random_data(self.testbits_len)

        # # 2. 添加头部
        # data_bits = self.add_header(data_bits)

        # 2.5 扰码数据
        self.scramble_data()

        # 3. 信道编码
        self.channel_encode()
        
        # 4. 调制数据并插入CP
        self.modulate()
        self.insert_cp()

        # # 5. 添加前置延迟
        # delay = self.params.get("delay")
        # delay_signal = np.zeros(delay, dtype=np.complex128)
        
        # 5. 组装完整信号
        self.tx_symbols = np.concatenate([
            # delay_signal,
            # self.preamble,
            self.data_with_cp
        ])

    def pulse_shaping(self):
        """脉冲成型（调用TxPulseShaper）"""
        shaped_signal = self.pulse_shaper.shape_pulse(self.tx_symbols)
        self.tx_signal = shaped_signal

    def run(self, data_bits=None):
        """执行完整发射流程（统一调度）"""
        if data_bits is None:
            self.data_bits = self._generate_random_data()
        else:
            self.data_bits = data_bits
        self.generate_preamble()
        self.scramble_data()
        self.channel_encode()
        self.modulate()
        self.insert_cp()
        self.assemble_signal()
        self.pulse_shaping()
        return self.tx_signal

# 新增眼图绘制函数
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
    
    # 1. 初始化参数和发射机
    print("="*50)
    print("          太赫兹发射机信号测试程序")
    print("="*50)
    params = PHYParams()
    
    # 打印基础参数信息
    print("\n【1. 基础参数配置】")
    basic_params = {
        "每符号比特数": params.get("NCBPS"),
        "是否使用CP": params.get("is_cp"),
        "加扰器种子ID": params.get("scrambler_seed_id"),
        "PPRE字段": params.get("ppre"),
        "PW字段": params.get("pw"),
        "前导码类型": params.get("Preamble_type"),
        "调制编码方案(MCS)": params.get("MCS"),
        "系统带宽": params.get("bandwidth"),
        "子帧长度": params.get("subframe_length"),
        "CP长度": params.get("cp_length"),
        "符号速率": params.get("symbol_rate"),
        "上采样率": params.get("oversampling"),
        "滚降系数": params.get("rolloff"),
        "滤波器类型": params.get("filter_type"),
        "滤波器长度": params.get("filter_length"),
    }
    for key, value in basic_params.items():
        print(f"  {key}: {value}")
    
    # 初始化发射机
    transmitter = THzTransmitter(params)
    
    # 2. 执行完整发射流程
    print("\n【2. 执行发射机信号生成流程】")
    # 生成前导码（确保preamble已初始化）
    tx_signal = transmitter.run()
    
    # 3. 详细信号参数统计
    print("\n【3. 信号详细参数统计】")
    # 基础长度信息
    print(f"  最终发射信号总长度: {len(tx_signal)} 采样点")
    print(f"  前导码长度: {len(transmitter.preamble)} 采样点")
    print(f"  调制后数据符号数: {len(transmitter.modulated_data)} 个")
    
    # 计算CP相关长度
    data_with_cp = transmitter.insert_cp(transmitter.modulated_data)
    cp_length = len(data_with_cp) - len(transmitter.modulated_data)
    print(f"  CP长度: {cp_length} 采样点 (比例: {cp_length/len(transmitter.modulated_data):.2f})")
    print(f"  带CP数据总长度: {len(data_with_cp)} 采样点")
    
    # 比特数统计
    if hasattr(transmitter, 'header_bits') and transmitter.header_bits is not None:
        print(f"  头部比特长度: {len(transmitter.header_bits)} bit")
    raw_data_len = transmitter.testbits_len
    print(f"  原始数据比特长度: {raw_data_len} bit")
    total_bits = len(transmitter.header_bits) + raw_data_len if transmitter.header_bits is not None else raw_data_len
    coded_bits = transmitter.channel_encode(np.random.randint(0,2,total_bits, dtype=np.uint8))
    print(f"  信道编码后比特长度: {len(coded_bits)} bit (编码增益: {len(coded_bits)/total_bits:.2f}x)")
    
    # 信号特性分析
    print(f"\n  信号幅值范围: {np.min(np.abs(tx_signal)):.4f} ~ {np.max(np.abs(tx_signal)):.4f}")
    print(f"  信号平均功率: {np.mean(np.abs(tx_signal)**2):.4f} W")
    print(f"  信号峰值功率: {np.max(np.abs(tx_signal)**2):.4f} W")
    print(f"  信号类型: {'复数' if np.iscomplexobj(tx_signal) else '实数'}")
    
    # 4. 信号完整性验证
    print("\n【4. 信号完整性验证】")
    verify_results = []
    # 验证前导码
    verify_results.append(f"前导码生成: {'✓ 正常' if len(transmitter.preamble) > 0 else '✗ 异常'}")
    # 验证调制数据
    verify_results.append(f"调制数据: {'✓ 正常' if len(transmitter.modulated_data) > 0 else '✗ 异常'}")
    # 验证CP插入
    verify_results.append(f"CP插入: {'✓ 正常' if len(data_with_cp) > len(transmitter.modulated_data) else '✗ 异常'}")
    # 验证最终信号
    verify_results.append(f"最终信号: {'✓ 正常' if len(tx_signal) > 0 else '✗ 异常'}")
    # 验证信号类型
    verify_results.append(f"信号类型: {'✓ 复数(符合要求)' if np.iscomplexobj(tx_signal) else '✗ 非复数(不符合要求)'}")
    
    for res in verify_results:
        print(f"  {res}")
    
    # 5. 信号可视化（修改为5个子图，增加眼图）
    print("\n【5. 信号可视化】")
    # 创建2行3列的子图布局（最后一列放眼图）
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
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
    freq = np.fft.fftfreq(len(fft_signal), 1/(params.get("symbol_rate") * params.get("oversampling")))
    ax2.plot(freq/1e9, 20*np.log10(np.abs(fft_signal)))
    ax2.set_title('信号频域频谱')
    ax2.set_xlabel('频率 (GHz)')
    ax2.set_ylabel('幅度 (dB)')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 调制符号星座图
    # 子图3: 调制符号星座图（修复）
    ax3 = fig.add_subplot(gs[0, 2])
    # 取纯调制数据符号（跳过前导码，取前200个调制符号）
    pure_mod_symbols = transmitter.modulated_data[:2000]  # 直接取调制器输出的纯数据符号
    ax3.scatter(pure_mod_symbols.real, 
                pure_mod_symbols.imag,
                s=10, alpha=0.8, c='orange')
    ax3.set_title('16QAM调制符号星座图')
    ax3.set_xlabel('实部')
    ax3.set_ylabel('虚部')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    # 限制坐标范围（16QAM标准范围是±0.9左右）
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    
    # 子图4: 信号功率分布
    ax4 = fig.add_subplot(gs[1, 0])
    power = np.abs(tx_signal[5000:10000])**2
    ax4.plot(np.arange(5000), power, linewidth=0.8, color='green')
    ax4.set_title('信号功率分布（第5000-10000采样点）')
    ax4.set_xlabel('采样点')
    ax4.set_ylabel('功率 (W)')
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 眼图（新增）
    ax5 = fig.add_subplot(gs[1, 1:])  # 占据第2行后两列
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