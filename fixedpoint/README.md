# 模块级定点仿真基础包

只有两种仿真模式：`fixed_point.enable_fixed=False` 是现有浮点路径，`True` 开启模块配置。范围观察由独立的 `range_analysis.enabled` 控制。比较和格式推荐是辅助工具。

```python
from fixedpoint import FixedPointContext, recommend_qformats

context = FixedPointContext(
    {
        "enable_fixed": True,
        "modules": {
            "ofdm.tx.ifft": {
                "arithmetic_level": "operation",
                "input_format": {"word_length": 16, "fractional_length": 12},
                "output_format": {"word_length": 18, "fractional_length": 14},
                "stage_shifts": [],
            }
        },
    },
    {"enabled": True, "reservoir_size": 4096},
)
time_signal = context.fft(frequency_signal, "ofdm.tx.ifft", inverse=True, norm="ortho")
report = context.report()
candidates = recommend_qformats(report)
```

## 格式与算术语义

- 总字长包含符号位。有符号 `Q4.12` 即 `W=16, F=12`，范围 `[-8, 7.999755859375]`。
- 内部使用 `FixedPointArray` 的整数 I/Q 码值及明确格式。对现有链路返回反量化的 NumPy 数组；多步内部计算可使用 `context.operand()` 保留格式。
- 默认舍入为 `nearest_even`，默认饱和。支持 `nearest_away/toward_zero/floor/ceil`。
- 关闭饱和必须明确选择 `overflow_policy="error"` 或 `"wrap"`。
- 整数乘除保留中间精度再转换。FIR/相关单独配置乘积和累加器，在逐次累加时执行其格式限制。
- 除法使用由输入格式推导位宽的完整 Python 整数中间量，避免合法小输出因共轭乘积或左移超过64位而失败；报告实际及声明最大中间位宽，最终目标格式转换后才返回 int64 码值。
- 负数右移按配置舍入。除零（包括分母量化为零）、NaN/Inf 及不安全宿主整数运算明确报错。
- 本包模拟 ISO/IEC TR 18037:2008 Clause 4 的算术语义，不声明 C 编译器符合性。

`boundary` 是输入/系数/输出的量化实验，内部计算仍浮点。`operation` 使用相应整数内核。覆盖报告逐项列出实际实现；未完成的内部定点能力须明确拒绝。

## 默认角色格式

| 角色 | W/F |
|---|---|
| input/output | 16/12 |
| coefficient | 18/16 |
| product/accumulator | 48/28 |
| state | 24/12 |
| phase | 24/20 |
| matrix | 32/16 |
| index | 32/0 |
| noise | 32/24 |
| denominator | 48/28 |
| weight | 32/20 |

SC 粗/细同步、信道估计及均衡默认输出为 `32/12`；LDPC LLR 默认输出为 `8/2`。这些只是可运行的初始配置，**尚未证明为最优或通过链路性能门限**。可以通过模块的 `formats.{role}` 或 `{role}_format` 覆盖。模块配置按 ID 层级继承，例如 `channel` 的配置适用于 `channel.awgn`。

## FFT/IFFT

整数混合基数内核支持因子仅含 2、3、5 的长度，包括 SC 的 480 点；不改变变换长度。支持轴选择、显式 `n` 以及 `None/backward/forward/ortho` 归一化。其它长度在 `operation` 下报错。

`stage_shifts` 从最外层蝶形向内指定非负右移数。每级乘积移位后累加，最终合并移位补偿和原归一化，保持外部增益。非 `ortho` 归一化使用整数除法，`ortho` 使用量化系数与明确的二进制指数。变换中仍可能因格式不足饱和，报告不会消除该误差。

频域中间结果进入逆变换时应保留其格式，例如：

```python
spectrum = context.fft(samples, module_id, output_role="output")
samples_out = context.fft(spectrum, module_id, inverse=True, input_role="output")
```

若模块时域和频域需要不同格式，可新增专用角色并通过 `input_role/output_role` 指定。

## 范围观察与推荐

`context.observe(node, value)` 记录 I/Q、幅度和功率。基础统计使用可合并的均值/二阶中心矩，报告总体标准差。分位数使用独立确定性的优先级 reservoir，不消耗全局链路 RNG；报告标明近似方法、样本数和尾部解析能力。

报告保留量化前后范围、内部状态、实际角色格式、溢出次数及转换次数。大分位数不能替代极值或理论累加增长。

`recommend_qformats()` 仅从范围与峰值余量生成候选，结果始终为 `validated=false`。必须再做模块误差、EVM、BER/BLER、SNR loss 和独立数据验证。

## 测试

```powershell
& 'D:\Users\Leezzey\anaconda3\envs\thz\python.exe' -m pytest fixedpoint/tests/test_core.py -q
```

测试覆盖舍入边界、饱和/回绕/报错、宿主溢出、复数乘除、FFT 长度与归一化、FIR/相关、范围合并、配置隔离、CORDIC 和矩阵求解。
