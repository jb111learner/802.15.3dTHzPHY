import numpy as np
import matplotlib.pyplot as plt
import json
from core.BaseTransmitter import BaseTransmitter
from transmitter.HeaderGenerator import HeaderGenerator
from transmitter.PreambleGenerator import PreambleGenerator
from transmitter.Modulator import QAMModulator as Modulator
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
        self.rscoder = RSCoder(params)
        self.cp_inserter = CPInserter(params)
        self.testbits_len = int(10e5)  # 测试用数据比特长度
        self.pulse_shaper = TxPulseShaper(params)
        self.scrambler = Scrambler(params)  # 初始化扰码器
        # self.rotator = Rotator()

    def add_header(self, data_bits):
        """生成PHY+MAC头部（调用HeaderGenerator）"""
        self.header_bits = self.header_gen.generate()
        # 添加头部到数据前面
        combined_bits = np.concatenate([self.header_bits, data_bits])
        return combined_bits    

    def generate_preamble(self):
        """生成前导码（SYNC+SFD+CES）"""
        sync, sfd, ces = self.preamble_gen.generate()
        self.preamble = np.concatenate([sync, sfd, ces])

    def scramble_data(self, data_bits):
        # 仅对纯数据扰码
        scrambled_data = self.scrambler.scramble(data_bits)
        return scrambled_data

    def channel_encode(self, data_bits):
        """信道编码（调用RSCoder）"""
        coded_bits = self.rscoder.encode(data_bits)
        return coded_bits  

    def modulate(self, data_bits):
        """调制数据（调用Modulator）"""
        return self.modulator.modulate(data_bits)

    def insert_cp(self, data):
        """插入CP（调用CPInserter）"""
        return self.cp_inserter.insert_cp(data)

    def assemble_signal(self):
        """组装发射信号：延迟 + 前导码 + 调制头部 + 带CP数据"""
        # 1. 生成随机数据比特
        data_bits = self._generate_random_data(self.testbits_len)

        # 2. 添加头部
        data_bits = self.add_header(data_bits)

        # 2.5 扰码数据
        data_bits = self.scramble_data(data_bits)

        # 3. 信道编码
        coded_bits = self.channel_encode(data_bits)
        
        # 4. 调制数据并插入CP
        self.modulated_data = self.modulate(coded_bits)
        self.data_with_cp = self.insert_cp(self.modulated_data)

        # # 5. 添加前置延迟
        # delay = self.params.get("delay")
        # delay_signal = np.zeros(delay, dtype=np.complex128)
        
        # 5. 组装完整信号
        self.tx_symbols = np.concatenate([
            # delay_signal,
            self.preamble,
            self.data_with_cp
        ])

    def pulse_shaping(self):
        """脉冲成型（调用TxPulseShaper）"""
        shaped_signal = self.pulse_shaper.shape_pulse(self.tx_symbols)
        self.tx_signal = shaped_signal

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
        "调制阶数": params.get("M"),
        "是否使用CP": params.get("is_cp"),
        "加扰器种子ID": params.get("scrambler_seed_id"),
        "PPRE字段": params.get("ppre"),
        "PW字段": params.get("pw"),
        "前导码类型": params.get("Preamble_type"),
        "调制编码方案(MCS)": params.get("mcs"),
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
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(transmitter.tx_symbols[9000:10000].real, 
                transmitter.tx_symbols[9000:10000].imag,
                s=5, alpha=0.6, c='orange')
    ax3.set_title('调制符号星座图')
    ax3.set_xlabel('实部')
    ax3.set_ylabel('虚部')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
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
        symbol_period=3,  # 基础符号周期
        num_symbols=500,  # 绘制500个符号的眼图
        oversampling=symbol_period,
        ax=ax5
    )
    
    fig.suptitle('太赫兹发射机信号分析', fontsize=16, fontweight='bold')
    plt.show()
    
    # # 6. 数据保存（可选）
    # save_choice = input("\n【6. 数据保存】是否保存测试数据？(y/n): ")
    # if save_choice.lower() == 'y':
    #     # 保存信号数据
    #     np.save('thz_tx_signal.npy', tx_signal)
    #     # 保存参数信息
    #     signal_info = {
    #         "测试时间": np.datetime64('now').astype(str),
    #         "基础参数": basic_params,
    #         "信号长度": len(tx_signal),
    #         "前导码长度": len(transmitter.preamble),
    #         "调制符号数": len(transmitter.modulated_data),
    #         "CP长度": cp_length,
    #         "信号平均功率": float(np.mean(np.abs(tx_signal)**2)),
    #         "信号峰值功率": float(np.max(np.abs(tx_signal)**2))
    #     }
    #     with open('thz_tx_signal_info.json', 'w', encoding='utf-8') as f:
    #         json.dump(signal_info, f, ensure_ascii=False, indent=4)
    #     print("  ✅ 信号数据已保存为: thz_tx_signal.npy")
    #     print("  ✅ 信号信息已保存为: thz_tx_signal_info.json")
    
    # # 7. 测试总结
    # print("\n【7. 测试总结】")
    # print(f"  📊 发射信号总长度: {len(tx_signal)} 采样点")
    # print(f"  📡 信号有效带宽: {np.ptp(freq[np.where(20*np.log10(np.abs(fft_signal)) > np.max(20*np.log10(np.abs(fft_signal))) - 3)]):.2f} Hz")
    # print(f"  ✅ 所有验证项: {'全部通过' if all('✓' in res for res in verify_results) else '部分异常'}")
    # print("\n测试完成！")