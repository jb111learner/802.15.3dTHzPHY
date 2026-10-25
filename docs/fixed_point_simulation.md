# 模块级定点仿真

现有浮点链路是 golden reference。只有浮点与定点两种运行模式，开关为
`PHYParams.fixed_point.enable_fixed`，缺省 `False`。范围分析、比较和格式推荐是分析工具。

## 配置与使用

```python
from params.PHYParams import PHYParams
from simulation.SimulationManager import SimulationManager

params = PHYParams()
params.update(
    enable_iq_compensation=False,  # 此示例选择普通Pilot-LS/ZF路径
    fixed_point={
        "enable_fixed": True,
        "default_format": {
            "word_length": 16, "fractional_length": 12,
            "rounding": "nearest_even", "saturation": True,
        },
        "modules": {
            "ofdm.tx.ifft": {"arithmetic_level": "operation"},
            "ofdm.rx.fft": {"arithmetic_level": "operation"},
            "ofdm.rx.channel_estimation": {"arithmetic_level": "operation"},
            "ofdm.rx.equalizer": {"arithmetic_level": "operation"},
        },
    },
    range_analysis={"enabled": True, "reservoir_size": 4096},
)
result = SimulationManager(params, save_plots=False).run_once()
report = result["result"]["fixed_point_report"]
```

顶层关闭优先于所有模块配置。顶层开启后，各模块 `enabled` 决定是否参与量化。
`boundary` 在模块输入/系数/输出量化，内部浮点；`operation` 使用内部整数内核。
没有配置模块等级时使用 `boundary`，不把整个链路宣称为整数实现。

各处理对象提供 `enable_fixed`、`configure_fixed` 和共享 `fixed_context`。
配置用于下一次运行；已构造模块不在计算中途改变格式或状态。

物理信道默认保留浮点运算，最后的接收采样默认量化。要启用已支持的信道内部内核，
除指定 `operation` 外还需明确设置 `quantize_channel_arithmetic=True`：

```python
channel_config = {
    "enable_fixed": True,
    "modules": {
        "channel.multipath": {
            "arithmetic_level": "operation",
            "quantize_channel_arithmetic": True,
        },
        "channel.rx_samples": {"quantize_rx_samples": True},
    },
}
```

静态多径operation要求静态信道和整数延时；实测/MIMO FIR分别配置 `channel.measured` /
`channel.mimo`。关闭 `channel.rx_samples.quantize_rx_samples` 可关闭采样边界量化。
模块关闭可被更精确的子模块开启覆盖；顶层 `enable_fixed=False` 始终关闭所有量化。

模块角色支持 `input_format/coefficient_format/product_format/accumulator_format/output_format`，
也支持 `formats` 中的 `state/phase/noise/denominator/weight/index/matrix/cordic_state`。
节点格式覆盖模块默认，模块默认覆盖链路默认。相关、SC未缩放变换和LLR有专用起始格式；
`coverage.json` 输出每个实际模块解析后的格式，不能仅看全局字长推断硬件配置。

界面“参数配置 → 数值精度”只有浮点/定点选择。旧工程默认浮点，模块高级配置随项目保存。

## 独立比较

```python
from simulation.fixedpoint_comparison import (
    run_float_fixed_comparison, run_comparison_sweep, evaluate_qformat_candidates,
)

comparison = run_float_fixed_comparison(
    params, params.get("fixed_point"),
    {"seed": 2026, "output_dir": "simulation_results/fixedpoint/comparison"},
)
sweep = run_comparison_sweep(
    params, [18, 19, 20, 21, 22], params.get("fixed_point"),
    runs=4, seed=2026, target=1e-3,
    output_dir="simulation_results/fixedpoint/sweep",
)
```

比较共享信息比特、生成信道实现与同一绝对噪声；物理信道使用浮点参考确定噪声方差。
不只共享种子。原全局NumPy随机状态在比较后恢复。阴影衰落没有指定独立种子时，比较入口
使用试次种子；不修改普通浮点运行的随机策略。两条链路独立维护滤波、同步和译码状态。

指标包含节点NMSE、发送数据参考的EVM、有效信息BER、码块BLER和PHY帧FER。
RS码块的GF符号数转换为比特；LDPC按信息块k比特统计。末尾信息填充从BER排除，EVM排除
完全属于填充的符号。缺失比特保守计错，原链路自动补齐/截断事件标记整次试次失败，
不能以短长度截取掩盖丢帧。syndrome是诊断，不能代替真实信息比特正确性。

SNR loss从共同目标BER/BLER交点获得；无交点、零错误无法插值、非单调或统计不足时
输出状态和区间，不给未经支持的精确数值。零错误点提供上界，曲线不把BER=0取对数。
码内比特错误存在相关性；配对多次运行的bootstrap是额外证据，短测试不是低BER确认。

`evaluate_qformat_candidates` 对候选执行相同种子的开发筛选，再用独立种子确认。
未达到性能/溢出预算时返回 `recommended=None`，不会自动部署候选。

## 范围分析与Q格式

统计 `min/max/mean/std/percentile`，复数I/Q、幅度与功率分别记录。
分位数使用独立确定性reservoir，不消耗链路随机数，报告样本数与近似标志。
浮点范围用于候选生成；独立定点运行不从已饱和输出推断推荐格式。

```python
from fixedpoint import recommend_qformats
candidates = recommend_qformats(comparison["float_report"]["ranges"], {
    "word_lengths": [12, 16, 18, 24], "headroom_db": 3,
})
```

范围候选均为 `validated=False`。整数位由原始极值和峰值余量筛选，小数位还需链路比较。
当前默认格式是可运行起点，尚不是最小字长或所有场景的性能推荐。

## 算术语义与支持范围

按 ISO/IEC TR 18037:2008 Clause 4 构建算术模型，不声明C编译器符合性认证。
有符号Qm.n的m包含符号位；报告同时保存W/F。默认最近偶数舍入与饱和是项目选择。
整数加减/乘除、格式转换、FIR、相关、混合基数FFT、CORDIC及满秩小矩阵求解采用显式码值。
宿主int64溢出在计算前检查；不安全字长报错，不以NumPy静默回绕替代。
除法使用从输入格式推导位宽的完整宽整数中间量，报告实际与声明位宽，再转回目标码值；
因此合法的小商不会因计算前的缩放超过64位而被错误拒绝。

SC使用真实480点变换，不补到512；SISO保持ortho，MIMO保持原FFT/IFFT归一化。
各级状态、旋转因子、移位和最终补偿单独记录。CORDIC常数表初始化用浮点，运行迭代为整数，
内部状态使用声明的保护位并逐步处理溢出。矩阵求解采用整数主元消元，量化后奇异矩阵报错，
不隐式加正则项，也不退回浮点pinv。

已支持优先OFDM/SC模块、静态整数延时多径、实测/MIMO FIR、噪声注入和采样量化。
时变/分数延时信道、PA/IQ自适应算法及星座距离内部求解目前以边界量化接入；请求不支持的
operation明确报错。物理环境随机过程可按配置保留浮点，并在覆盖报告中注明。
原默认OFDM可启用宽线性IQ联合估计，此路径对应 `ofdm.rx.iq`，会绕过普通Pilot-LS；
要求普通估计/均衡operation的示例显式关闭IQ补偿，不改变现有工程的默认算法选择。
LDPC本期仅clip后LLR量化，显式软输入标记避免{0,1}误识别，迭代内部保留原浮点。

## PowerShell功能验证

当前依赖完整的解释器为：

```powershell
$thzPython = 'D:\Users\Leezzey\anaconda3\envs\thz\python.exe'
& $thzPython -B -m simulation.fixedpoint_cli run --link ofdm --float --range-analysis
& $thzPython -B -m simulation.fixedpoint_cli compare --link ofdm --operation --snr 30
& $thzPython -B -m simulation.fixedpoint_cli compare --link sc-fde --operation --snr 30
& $thzPython -B -m simulation.fixedpoint_cli compare --link mimo --operation --snr 30
```

CLI使用一帧16QAM与工程LDPC(64,32)演示配置。实际工程请通过PHYParams传入正式参数。
CLI的 `--operation` 开启对应优先收发模块，物理信道仍保留浮点；信道内核通过JSON配置启用。
这些演示不能作为正式低BER或SNR损失结论。

回归测试：

```powershell
& $thzPython -B -m pytest fixedpoint/tests simulation/test_fixedpoint_comparison.py simulation/test_fixedpoint_integration.py simulation/test_fixedpoint_receiver.py simulation/test_fixedpoint_transmitter_channel.py thz_sim_ui/test_fixedpoint_ui.py -q
```

原浮点基线摘要位于 `simulation/fixtures/fixedpoint_float_baseline.json`，包含依赖环境信息。

已执行新增功能与相关回归，覆盖三条浮点摘要、内部定点链路、真实CFO、Qt配置往返及报告导出。
另有既有测试预期与原代码不一致：`test_tdl_channel.py` / `test_measured_channel_resampling.py`
的15项失败、`test_iq_compensation_integration.py` 的2项流程预期失败，均已用HEAD原实现复现。
未为通过这些测试改变浮点算法、默认TDL设置或接收处理顺序。
