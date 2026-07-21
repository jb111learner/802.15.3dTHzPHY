"""
测试 SimulationManager 的核心功能：
- 参数字典输入
- 种子策略（固定/递增/时间）
- 蒙特卡洛多次仿真
- 分阶段 Pipeline 中间结果
- 单参数扫描
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from simulation.SimulationManager import SimulationManager
    from params.PHYParams import PHYParams
    import numpy as np
except ImportError as e:
    print(f"导入失败：{e}。请确保安装依赖：pip install galois numpy matplotlib")
    sys.exit(1)

def test_basic_setup():
    """测试基本设置与单次仿真。"""
    print("=== 测试 1: 基本设置与单次仿真 ===")

    # 创建仿真管理器
    sm = SimulationManager(base_params=PHYParams())

    # 设置控制参数（模拟 UI 映射后的字典）
    control_params = {
        "seed_strategy": "递增种子",
        "SNRdB": 15,
        "bit_source": "PRBS",
    }
    sm.set_control_params(control_params)

    # 单次仿真
    result = sm.run_once()
    print(f"单次仿真完成：BER = {result['metrics']['ber']:.6f}")
    print(f"种子：{result['result']['seed']}")
    print(f"发射信号长度：{len(result['result']['tx_signal']['signal_stream'])}")
    print(f"接收信号长度：{len(result['result']['rx_signal']['signal_stream'])}")
    print(f"解码后比特长度：{len(result['result']['rx_decoded']['signal_stream'])}")
    print()

def test_monte_carlo():
    """测试蒙特卡洛多次仿真。"""
    print("=== 测试 2: 蒙特卡洛多次仿真 ===")

    sm = SimulationManager(base_params=PHYParams())
    sm.set_control_params({
        "seed_strategy": "递增种子",
        "SNRdB": 10,
        "bit_source": "PRBS",
    })

    # 运行 5 次蒙特卡洛
    monte_results = sm.run_monte_carlo()
    bers = [r["metrics"]["ber"] for r in monte_results]
    seeds = [r["result"]["seed"] for r in monte_results]

    print(f"蒙特卡洛 5 次完成：")
    print(f"BER 列表：{[f'{b:.6f}' for b in bers]}")
    print(f"种子列表：{seeds}")
    print(f"平均 BER：{np.mean(bers):.6f}")
    print(f"BER 方差：{np.var(bers):.6e}（递增种子应有变化）")
    print()

def test_seed_strategies():
    """测试不同种子策略。"""
    print("=== 测试 3: 种子策略验证 ===")

    strategies = ["固定种子", "递增种子", "时间种子"]
    for strategy in strategies:
        print(f"测试策略：{strategy}")
        sm = SimulationManager(base_params=PHYParams())
        sm.set_control_params({
            "seed_strategy": strategy,
            "SNRdB": 20,
            "bit_source": "PRBS",
        })

        # 运行两次，观察种子
        r1 = sm.run_once()
        r2 = sm.run_once()
        print(f"  第一次种子：{r1['result']['seed']}")
        print(f"  第二次种子：{r2['result']['seed']}")
        if strategy == "固定种子":
            assert r1['result']['seed'] == r2['result']['seed'], "固定种子应相同"
        elif strategy == "递增种子":
            assert r1['result']['seed'] + 1 == r2['result']['seed'], "递增种子应 +1"
        print("  验证通过")
    print()

def test_param_override():
    """测试参数覆盖（模拟 UI 动态调整）。"""
    print("=== 测试 4: 参数覆盖 ===")

    sm = SimulationManager(base_params=PHYParams())
    sm.set_control_params({
        "seed_strategy": "固定种子",
        "SNRdB": 10,
    })

    # 覆盖 SNRdB
    result = sm.run_once(override_params={"SNRdB": 5})
    print(f"覆盖 SNRdB=5，BER = {result['metrics']['ber']:.6f}")
    print(f"实际参数 SNRdB：{result['params'].get('SNRdB')}")
    print()

def test_scan_single():
    """测试单参数扫描。"""
    print("=== 测试 5: 单参数扫描 ===")

    sm = SimulationManager(base_params=PHYParams())
    sm.set_control_params({
        "seed_strategy": "固定种子",
        "bit_source": "PRBS",
    })

    # 扫描 SNRdB
    snr_values = [5, 10, 15]
    scan_results = sm.scan_single_parameter("SNRdB", snr_values)
    bers = [r["metrics"]["ber"] for r in scan_results]
    print(f"SNRdB 扫描：{snr_values}")
    print(f"对应 BER：{[f'{b:.6f}' for b in bers]}")
    print("图表已保存至 simulation_results/scan_SNRdB.png")
    print()

def test_dict_input():
    """测试字典输入（模拟 backend.map_ui_params_to_phy_params 输出）。"""
    print("=== 测试 6: 字典输入（模拟 UI 映射）===")

    # 模拟 UI 参数字典（从 ParameterConfigPage.get_all_parameters()）
    ui_params = {
        '载频': 300,  # GHz
        '带宽': 38.4,  # GHz
        '采样率': 88200,  # sps
        '数据帧长度': 480,  # symbols
        '单帧数据子帧数量': 51,  # frames
        'GI 长度': 32,  # symbols
        '调制方式': 'QPSK',
        '编码方式': 'RS(255, 192)',
        '波形成形': 'rrc',
        '滚降因子': 0.22,
        '滤波器跨度': 8,
        'SNR': 12,  # dB
        '译码参数': '硬译码',
        '比特源配置': 'PRBS',
        '位深度': '16bits',
        '比特长度': 1000,
    }

    # 模拟 backend 映射
    from thz_sim_ui.services.backend import BackendService
    mapped_params = BackendService.map_ui_params_to_phy_params(ui_params)

    # 创建仿真管理器并应用
    sm = SimulationManager(base_params=PHYParams())
    sm.set_control_params(mapped_params)

    result = sm.run_once()
    print(f"字典输入仿真完成：BER = {result['metrics']['ber']:.6f}")
    print(f"映射参数：{mapped_params}")
    print()

if __name__ == "__main__":
    print("开始 SimulationManager 测试套件\n")

    try:
        test_basic_setup()
        test_monte_carlo()
        test_seed_strategies()
        test_param_override()
        test_scan_single()
        test_dict_input()
    except Exception as e:
        print(f"测试失败：{e}")
        import traceback
        traceback.print_exc()

    print("测试套件完成。如有依赖问题，请安装 galois 等库。")