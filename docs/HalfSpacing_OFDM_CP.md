# 半子载波双组 OFDM：LMMSE 与 RS/LDPC 编码仿真

当前 `half_spacing_ofdm_sim.py` 保留理论匹配模型、已知多径和实际 CP，比较未编码、RS、LDPC 三种配置。显式求逆只留作小系统公式对照，不参与性能仿真、CSV 曲线数据或绘图。

图例固定为：`LMMSE`、`OFDM`、`LMMSE + RS`、`OFDM + RS`、`LMMSE + LDPC`、`OFDM + LDPC`，不显示 SVD 实现细节。

无噪声信号与纯噪声的独立实际链路验证见 [理论模型验证说明](HalfSpacing_OFDM_ModelValidation.md)，运行 `python -m simulation.validate_half_spacing_ofdm_model`。

## 波形与相同数据速率条件

| 项目 | 半子载波双组系统 | 普通 OFDM |
| --- | --- | --- |
| 子载波数 | 2K，默认 512 | K，默认 256 |
| 调制 | M-QAM，默认 QPSK | M²-QAM，默认 16-QAM |
| 子载波中心 | k/T、(k+1/2)/T | k/T |
| 每个有效块的调制比特 | 2K log₂(M)，默认 1024 | K log₂(M²)，默认 1024 |
| 有效时长、采样数 | T、N，默认 1、512 | 相同 |
| CP 与 FIR | 默认 97 样点、归一化三抽头 | 相同 |
| 有效段期望能量 | Eblock | Eblock |
| 含 CP 总期望能量 | Eblock(1+Lcp/N) | 相同 |

两种系统在同一编码配置下共用相同的信息比特、编码比特、交织排列、FIR 和时域白噪声。块时长均为 T(1+Lcp/N)，编码净数据速率均为

$$
R_{\rm info}=\frac{R\,2K\log_2(M)}{T(1+L_{\rm CP}/N)}.
$$

同一码率下，两种波形净速率相同。编码系统净速率为未编码系统的 R 倍，不声称所有编码与未编码曲线具有相同净速率。名义频带宽度按 K/T 计，未检验模拟频谱掩模或严格占用带宽对齐。

半间隔 CP 保留组内复制后在负时间轴施加半频移的构造；普通 OFDM 直接复制复合波形尾部。接收端仅保留 N 个有效样点。FIR 卷积跨批次保留状态，没有 CP 边界实验。

## 信息比特 Eb/N0 与码率归一化

每种星座满足 E|a|²=1，发射幅度为

$$
\mu_h=\sqrt{\frac{E_{\rm block}}{2KT}},
\qquad \mu_o=\sqrt{\frac{E_{\rm block}}{KT}}.
$$

横轴始终按信息比特定义。令 γ=Eb/N0、b=log₂(M)、R 为码率，则

$$
\nu_h=\frac1{Rb\gamma},\qquad
\nu_o=\frac1{2Rb\gamma},\qquad
\sigma_{\rm time}^2=\mu_h^2N\nu_h=\mu_o^2N\nu_o.
$$

在相同信息 Eb/N0 下，编码噪声方差是未编码噪声方差的 1/R 倍，发射块能量不变。不能遗漏码率因子，否则会高估编码收益。Eb/N0 不计 CP 能量；两种波形 CP 开销一致。

## 理论匹配模型与 LMMSE

中点采样 uₙ=(n+1/2)/N 对应

$$
(P_N)_{m,k}=\frac{j}{N\sin[\pi(k-m+1/2)/N]},
\qquad C_N=\begin{bmatrix}I&P_N\\P_N^H&I\end{bmatrix}.
$$

若 S 为采样基矩阵、D 为已知信道对角阵，归一化匹配模型为

$$
z=C_NDa+\eta,\qquad E[\eta\eta^H]=\nu_hC_N.
$$

`TheoryLMMSE` 仍通过 B=SD/√N 的 SVD 实现完全等价的理论 LMMSE：

$$
B=U\Sigma V^H,\qquad
\hat a=V\operatorname{diag}\left(\frac{\sigma_i}{\sigma_i^2+\nu_h}\right)
\frac{U^Hr}{\mu_h\sqrt N}.
$$

没有截断小奇异值，没有在检测器内部施加有限星座约束，也没有译码反馈。完整耦合 P_N 的信息已经包含在 B 中，匹配输出噪声没有被误当作白噪声。

普通 OFDM 使用 FFT 与单抽头相干 ZF，其硬判决等价于补偿标量 MMSE 输出增益后的 QAM 判决。

## RS：只使用硬译码

默认复用 `utils.Coder.RSCoder` 的 GF(16)、RS(15,11)，每个 GF 符号 4 bit，每码字 60 个编码比特、44 个信息比特，R=11/15，可纠正至多 2 个符号错误。

主脚本直接构造 `utils.Coder.RSCoder` 并调用公开的 `encode`、`decode`，比特顺序保持项目的 GF 符号内部 LSB-first。`simulation/ofdm_channel_coding.py` 仅保留码字分帧和交织，不包含编译码器。

接收数据只来自最近 QAM 点硬判决，不向 RS 提供 LLR 可靠性、不提供擦除位置，也不执行 Chase。声明无法纠正时保留收到的系统信息部分，失败码字仍计入 BER。超过纠错半径时可能误译码，实际信息错误率也会统计这些情况。

## LDPC：软信息与归一化 min-sum

默认使用项目 `LDPCCoder` 的 1440 bit 矩阵模式，选择 11/15 码率，即 LDPC(1440,1056)，使其与默认 RS 同码率。可用 `--ldpc-rate 14/15` 切换为 LDPC(1440,1344)。

沿用项目的 `minus_literal` 矩阵方向、系统信息位置和编码器。译码使用归一化 min-sum，默认 α=0.8、最多 30 轮、LLR 截幅 ±20，以校验综合全零作为停止条件。

主脚本直接构造 `utils.Coder.LDPCCoder` 并调用公开的 `encode`、`decode`。已删除独立的 min-sum 实现、加速类及编译码封装；公共 `utils/Coder.py` 未修改。回归测试检查实际仿真确实调用了这两个公共编译码器。

输入遵循项目译码器的浮点 LLR 接口，正值对应比特 0。仿真侧只计算并解交织软信息，不实现 LDPC 消息传递。

### LMMSE 软信息校准

设 G=WB，其中 W 为 LMMSE 线性映射。逐子载波有效增益为

$$
g_i=G_{ii}=\sum_j|V_{ij}|^2\frac{\sigma_j^2}{\sigma_j^2+\nu_h}.
$$

单位输入协方差模型下，估计可写为 âᵢ=gᵢaᵢ+eᵢ，残余复方差为

$$
v_i=E|e_i|^2=g_i(1-g_i).
$$

max-log QAM 解映射使用 â、g、v，而不是直接把 â 当作无偏符号并代入原始 AWGN 方差。对每个比特：

$$
L_b\approx
\min_{s:b(s)=1}\frac{|\hat a-gs|^2}{v}
-
\min_{s:b(s)=0}\frac{|\hat a-gs|^2}{v}.
$$

正 LLR 对应比特 0。实轴距离除以复方差 v，避免额外的 2 倍因子。方形 QAM 按 I/Q 两个 PAM 轴分别计算。

半间隔残余干扰采用标量高斯近似，跨子载波相关性未传递给 LDPC；这不是完整联合检测译码，也不保证 LDPC 一定改善 BER。普通 OFDM 软信息使用 ZF 后复噪声方差 ν_o/|H|²。

## 完整码字、交织与误码统计

为避免补零和尾部截断，每个编码单元的长度取码长与 OFDM 调制比特数的最小公倍数。

| 默认配置 | 每编码单元 OFDM 块数 | 每单元码字数 | 单元编码比特 |
| --- | --- | --- | --- |
| RS(15,11)，每 GF 符号 4 bit | 15 | 256 | 15360 |
| LDPC(1440,1056) | 45 | 32 | 46080 |

默认在每个编码单元内随机交织，两种波形使用相同排列，译码前解交织；可用 `--no-interleave` 关闭。随机种子可复现，统计与公式检查使用独立随机流。

`--frames 2000` 表示期望发送块数，实际取完整编码单元：未编码 2000 块、RS 2010 块、LDPC 2025 块。每个 SNR 的信息比特数分别为 2048000、1509376、1520640。

可用 `--coding-frames` 单独设置编码对照的块数。当前扩展 SNR 运行使用未编码 2000 块、编码 45 块，编码结果主要用于链路对照，采样精度低于此前的大样本结果；各自的信息比特数完整保存在 CSV。未设置该选项时仍与 `--frames` 相同。

- `ber`：译码后信息 BER，分母为 `bit_count`。
- `raw_ber`：译码前编码比特 BER，分母为 `coded_bit_count`。
- `information_codeword_fer`：根据真实信息比特计算，包含未检出的误译码。
- `decoder_failed_codewords`：LDPC 译码器声明失败的码字数，与真实 FER 不完全相同；项目 RS 接口未返回该计数，RS 行留空。
- BER 标准误：未编码按独立 OFDM 块统计，编码按独立编码/交织单元统计。
- MSE：译码前符号误差；未编码核对理论公式，编码预测依赖单位输入协方差假设，不作强制通过检查。

零误码表示有限样本未观察到错误。图中使用曲线对应的空心标记和向下箭头放在 1/Nbits 处，仅为显示标记，不是置信上界。

## 历史大样本多径结果（0–30 dB）

N=512，半间隔 512 个 QPSK 子载波，普通 256 个 16-QAM 子载波，CP=97，归一化抽头延时 (0,41,97)，seed=2026。默认两种码率均为 11/15。

| Eb/N0 (dB) | LMMSE | LMMSE + RS | LMMSE + LDPC | OFDM | OFDM + RS | OFDM + LDPC |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.185614 | 0.205394 | 0.192662 | 0.0187896 | 0.017659 | 本次零误码 |
| 15 | 0.180756 | 0.198758 | 0.185178 | 0.00296289 | 0.000245797 | 本次零误码 |
| 20 | 0.179346 | 0.197673 | 0.181643 | 0.000108887 | 本次零误码 | 本次零误码 |
| 30 | 0.177467 | 0.196198 | 0.180352 | 本次零误码 | 本次零误码 | 本次零误码 |

当前默认半间隔系统的残余误码较高，两种编码没有改善最终 BER；不能假定在强 ICI 下添加高码率编码就会获得增益。RS 超纠错半径的误译码、LDPC 未收敛和软信息的相关残余近似均需要结合实际统计解读。该结果不排除其他码率或联合检测译码的收益。

该表来自此前的大样本运行，不是当前编码 45 块的新数据。当前完整数据以输出目录为准。

## 当前曲线与高信噪比验证

六条曲线分别采用不同颜色、线型和空心标记，图例移到绘图区外；主图增加 LMMSE 局部放大，图例仍仅显示 LMMSE、OFDM 及编码名称。

默认时域 SNR 范围扩展为 0、5、10、15、20、25、30、40、60、80、100、120 dB。512 子载波的未编码 BER 从 30 dB 的 0.177467 降至 120 dB 的 0.167076，不能把这段仍很高的 BER 说成已经趋零。

固定有限规模、准确模型、非零信道、充分 CP 和精确算术下，LMMSE 没有正的渐近误码平台。满秩证明、完整 512 子载波的正特征值下界，以及最高 540 dB 的小规模高精度参考见 [高信噪比说明](HalfSpacing_OFDM_HighSNR.md)。24 项回归测试及各实际运行的公式/MSE 检查通过。

## 运行

~~~powershell
conda activate communication
python half_spacing_ofdm_sim.py
python half_spacing_ofdm_sim.py --codes rs
python half_spacing_ofdm_sim.py --codes ldpc --ldpc-rate 14/15
python half_spacing_ofdm_sim.py --codes none
python half_spacing_ofdm_sim.py --no-interleave
python half_spacing_ofdm_sim.py --channel awgn --output-dir simulation_results/half_spacing_ofdm_coded_awgn
python half_spacing_ofdm_sim.py --frames 2000 --coding-frames 45 --high-snr-analysis
python -m simulation.ofdm_lmmse_high_snr --report simulation_results/half_spacing_ofdm_comparison/report.json
python -m unittest discover -s simulation -p "test_half_spacing_ofdm_*.py" -v
~~~

默认结果仍写入 `simulation_results/half_spacing_ofdm_comparison`：`ber_comparison.png` 显示六条 BER 曲线，`performance.png` 同时显示未编码 MSE，扩展 SNR 后 `metrics.csv` 共 72 行，`report.json` 保存编译码和归一化说明，`summary.md` 汇总当前结果。高精度参考保存在 `high_snr_analysis` 子目录。

galois 内部的 JIT 缓存默认写入工作区 `.cache/numba`；仿真不再实现任何自定义编译码 JIT。高精度分析额外使用 mpmath，当前 communication 环境已具备。
