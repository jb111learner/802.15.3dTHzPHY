# 半子载波理论推导的独立链路验证

运行程序：`simulation/validate_half_spacing_ofdm_model.py`。回归测试：`simulation/test_half_spacing_ofdm_model.py`。

验证后的传输矩阵、MMSE 正规矩阵及噪声协方差的条件数分析见 [矩阵参数说明](HalfSpacing_OFDM_MatrixParameters.md)。

验证分为无噪声信号和纯噪声两个独立实验，直接调用当前生产代码的 `Waveform`、`FIRLink` 和 `matched_filter`。不经过 LMMSE 或信道译码，避免检测结果掩盖前级模型错误。

## 理论参考独立构造

理论侧使用显式复指数基、信道抽头频响的直接求和，以及文档的解析耦合公式，不调用生产代码的 `coupling_matrix`、`gram_matrix` 和 `fir_frequency_response`。

当前中点采样 tₙ/T=(n+1/2)/N 的精确关系为

$$
(P_N)_{m,k}=\frac{j}{N\sin[\pi(k-m+1/2)/N]},\qquad
C_N=\begin{bmatrix}I&P_N\\P_N^H&I\end{bmatrix}.
$$

原文连续积分对应 P[m,k]=j/[π(k−m+1/2)]。有限采样时 P_N 与 P 不完全相同；测试单独报告两者差异，并用固定子载波数、逐步增加 N 的回归检查确认其收敛，避免把积分离散化误差误判为理论错误。

## 信号实验：不加入接收噪声

链路为发送符号 → 两组 IFFT → 组内 CP 后半频移 → 连续 FIR 线性卷积 → 去 CP → 实际 FFT 匹配滤波。

分别发送双组数据、仅第一组、仅第二组，并在测试数据前发送独立的非零块。FIR 状态跨批次保留，接收结果还与独立的 `numpy.convolve` 串行卷积核对。

比较三个层次：

$$
r=\mu SDa,\qquad
y=\mu T C_NDa,\qquad
z=\frac{y}{\mu T}=C_NDa.
$$

此外，逐列发送完整符号空间的单位基，验证全部 2K 列的有效匹配传输矩阵是否为 C_ND。这样既检查实际随机数据，也检查完整线性算子。

## 噪声实验：发送符号全部置零

全零符号通过实际发送、CP 和 FIR 链路后，在接收端加入复白噪声；生成包括 CP 的接收噪声块，丢弃 CP 后送入同一匹配滤波器。

噪声添加点在传播 FIR 之后，符合原文 r(t)=h*s(t)+n(t) 的模型；不会把接收噪声再用传播信道滤波。

离散时域复噪声方差取 N N₀/T，因此原始匹配输出协方差为

$$
R_n=N_0T C_N.
$$

归一化输出 η=n/(μT) 对应

$$
R_\eta=\nu C_N,\qquad \nu=\frac{N_0}{\mu^2T}.
$$

用 M 个独立块统计均值和无偏样本协方差：

$$
\bar n=\frac1M\sum_b n_b,\qquad
\widehat R=\frac{\sum_b n_bn_b^H-M\bar n\bar n^H}{M-1}.
$$

同时检查两组内部块 N₀TI、两个跨组块 N₀TP_N 和 N₀TP_Nᴴ、零均值，以及 proper 复高斯噪声的零伪协方差 E[nnᵀ]=0。

有限样本的矩阵误差按理论采样尺度判断。复高斯样本协方差的相对 Frobenius RMS 误差为

$$
\frac{\operatorname{tr}(R)}{\sqrt{M-1}\,\|R\|_F}.
$$

全矩阵容差设为该尺度的 3 倍；每个协方差元素还按 √[RᵢᵢRⱼⱼ/(M−1)] 检查，最大误差不超过 8 个采样标准误尺度。没有用实测能量重新拟合理论归一化。

额外以实际匹配滤波器作用于时域单位基得到算子 F，确定性验证 (N N₀/T)FFᴴ=N₀TC_N。该检查应达到浮点精度，可与 Monte Carlo 的抽样误差分开解释。

## 已运行结果

两组合计 512 个子载波，N=512，T=1，默认三抽头归一化多径，CP=97；信号每种组配置 64 块，另验证完整 512 列传输矩阵；纯噪声 M=131072、N₀=0.03。

| 检查 | 数值 |
| --- | --- |
| 无噪声时域与显式理论的相对误差 | 约 4.12×10⁻¹⁴ |
| 完整 C_ND 匹配传输矩阵相对误差 | 3.79×10⁻¹⁵ |
| 实际匹配滤波算子协方差相对误差 | 3.61×10⁻¹⁶ |
| Monte Carlo 协方差相对误差 | 4.414% |
| 理论抽样误差 RMS | 4.427% |
| 协方差元素最大误差/采样标准误尺度 | 3.946 |

上述结果与离散理论一致。Monte Carlo 误差具有预期的有限样本尺度，确定性算子检查达到机器精度。

新增 11 项回归检查还覆盖 AWGN、非单位 T 和块能量、16-QAM、512 子载波的 N=1024 采样、连续积分极限，以及对白噪声假设、错误共轭和错误噪声功率的检出。加上原有测试共 35 项，全部通过。

## 复现与产物

~~~powershell
conda activate communication
python -m simulation.validate_half_spacing_ofdm_model
python -m unittest discover -s simulation -p test_half_spacing_ofdm_model.py -v
~~~

可用 `--noise-trials`、`--n0`、`--subcarriers`、`--samples`、`--symbol-duration`、`--block-energy`、`--channel` 配置验证，使用 `--no-plots` 跳过图像。

默认输出目录 `simulation_results/half_spacing_ofdm_model_validation`：

- `report.json`、`summary.md`：完整指标与说明。
- `signal_checks.csv`、`noise_checks.csv`：两类检查分开记录。
- `matrices.npz`：理论与实测传输矩阵、协方差、均值、伪协方差原始数据。
- `signal_model_comparison.png`、`noise_covariance_comparison.png`：实部和虚部的理论、实测与差值矩阵；差值列使用单独色标。
