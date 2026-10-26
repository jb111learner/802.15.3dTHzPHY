# LMMSE 无误码平台：证明、数值验证与适用范围

本说明对应 `half_spacing_ofdm_sim.py` 和 `simulation/ofdm_lmmse_high_snr.py`。分析对象是未编码、固定有限规模的理论 LMMSE；RS/LDPC 编码不是无平台证明的前提。

## 数学结论

设有效样点数 N≥d=2K，半间隔频率按 q/2 排序，q=0,…,d−1。采样矩阵为

$$
S_{n,q}=\exp\left[j\frac{2\pi(n+1/2)q}{2N}\right].
$$

取前 d 行，消除单位模列相位后是节点 z_q=exp(jπq/N) 的 Vandermonde 矩阵。节点互不相同，所以该子矩阵行列式非零，S 满列秩。

若所有子载波频响 H_q 非零，D=diag(H_q) 可逆。因此

$$
B=\frac{SD}{\sqrt N},\qquad A=B^HB=D^HC_ND\succ0.
$$

记 ρ_i>0 为 A 的特征值，ν 为归一化噪声功率。独立零均值单位协方差输入下，理论 LMMSE 的误差协方差与平均符号 MSE 为

$$
R_e=\nu(A+\nu I)^{-1},\qquad
\operatorname{MSE}(\nu)=\frac1d\sum_i\frac{\nu}{\rho_i+\nu}.
$$

固定有限 d 时，各 ρ_i 为固定正数，故

$$
\lim_{\nu\to0}\operatorname{MSE}(\nu)=0.
$$

固定有限 QAM 星座的最近点判决若发生符号错误，则估计误差幅度至少为最小点距的一半。由 Markov 不等式，

$$
\operatorname{BER}\le\operatorname{SER}
\le\min\left(1,\frac{4\operatorname{MSE}}{d_{\min}^2}\right),
\qquad d_{\min}^2=\frac6{M-1}.
$$

所以理想模型的 BER 趋于零，不存在正的渐近误码平台。有限次 Monte Carlo 出现零误码不是这个结论的证明。

前提包括：充分 CP、准确同步与信道、固定有限子载波数、非零子载波信道和精确算术。若某个 H_q=0、模型失配、实现截断或精度有限，上述结论不能直接套用。

## 完整 512 子载波的正特征值下界

使用前 d 行的 Vandermonde 子式和 Cauchy–Binet 公式，

$$
\det A\ge
N^{-d}\prod_q|H_q|^2
\prod_{r=1}^{d-1}
\left[2\sin\left(\frac{\pi r}{2N}\right)\right]^{2(d-r)}
\equiv\Delta>0.
$$

又因为每个特征值不超过 tr(A)=Σ|H_q|²，

$$
\rho_{\min}\ge\frac{\Delta}{[\operatorname{tr}(A)]^{d-1}}>0.
$$

该公式适用于当前完整 512 子载波，不需要计算完整高精度谱。默认信道下，80 位十进制精度计算得到 log10(该下界)≈−49820.3210。

这是极松的充分下界，不是实际最小特征值，不可将由它得到的约 498263 dB 充分 SNR 解读为实际所需 SNR。它的用途是给出完整规模的正定性和极限保证。

后续通过高精度 Vandermonde 结构逆得到完整 512 子载波的极端参数：ρmin≈2.18×10⁻⁵¹⁵，ν≈ρmin 的最弱模式参考约 5143.61 dB。此计算获得极端奇异值而非完整谱，详见 [传输与检测矩阵参数](HalfSpacing_OFDM_MatrixParameters.md)。

## 实际时域测试：0–120 dB

当前完整系统仍使用 N=512、512 个 QPSK 子载波、三抽头归一化多径、CP=97，每个 SNR 2000 块。

| Eb/N0 (dB) | LMMSE BER |
| --- | --- |
| 30 | 0.177467 |
| 60 | 0.173819 |
| 80 | 0.172081 |
| 100 | 0.169169 |
| 120 | 0.167076 |

完整系统在这段范围内缓慢下降，仍然存在很高 BER。不能把该曲线当作已实测收敛到零的证据；很多弱模式在这段 SNR 内还没有充分恢复。

相同接收代码用于总计 8 子载波的时域参考时，20 dB BER=0.0435625，40 dB BER=0.00359375，60–120 dB 未观察到误码。每个 SNR 为 32000 bit，实测 MSE 与理论一致。结果在 `high_snr_analysis/waveform_8_tones`。

时域实现仍限制到 120 dB。继续机械地提高浮点波形 SNR，会遇到弱奇异值、运算舍入及噪声叠加精度的问题；不把这些结果作为精确模型的极限证明。

## 高精度参考：最高 540 dB

在相同 N 和 FIR 下，对总计 8、16、32、64 子载波计算高精度谱，并核对高精度无噪声恢复。通过相位变换将 Hermitian 正规矩阵转成实对称矩阵，使用 mpmath eigsy，精度分别为 80、82、114、178 位。

谱的正性来自模型满秩；高精度数值进一步检查分解残差、无噪声恢复和与当前 SVD 实现的 0–120 dB MSE 一致性。

设 A=V diag(ρ_i)Vᴴ，w 为独立单位复白噪声。等价 LMMSE 误差为

$$
e=-V\operatorname{diag}\left(\frac\nu{\rho_i+\nu}\right)V^Ha
+V\operatorname{diag}\left(\frac{\sqrt{\nu\rho_i}}{\rho_i+\nu}\right)w.
$$

采用特征值对数和稳定 logistic 权重计算该式，直接从 e 统计 MSE，避免用病态浮点时域矩阵恢复极弱模式或从近似相等的数中相减得到小误差。

| 子载波数 | log10(ρmin) | ν≈ρmin 的 SNR (dB) | 最后测试 SNR (dB) | 最后理论 BER 上界 |
| --- | --- | --- | --- | --- |
| 8 | −4.405385 | 41.04 | 140 | 3.224e−11 |
| 16 | −10.439614 | 101.39 | 180 | 1.732e−9 |
| 32 | −23.320580 | 230.20 | 300 | 6.544e−9 |
| 64 | −47.136271 | 468.35 | 540 | 2.142e−9 |

这些最终点均未观察到误码，并且对应 MSE/BER 上界很小。它们是等价白化特征信道参考，不是当前双精度时域程序在 540 dB 的直接实验。完整 512 子载波的高精度谱没有计算；不会将小系统结果标成 512 子载波结果。

## 实现检查与复现

24 项回归测试覆盖当前编译码 API 的直接调用、信息比特统计、LMMSE 方程等价、软信息校准、高精度谱及复多径相位变换、特征信道公式、正下界、零频响例外，以及 Monte Carlo MSE 与理论一致性。

编译码仅调用 `utils/Coder.py` 的公开 `RSCoder`、`LDPCCoder`，没有独立 RS/LDPC 实现。扩展范围编码对照使用 45 块/点，每条编码曲线 33792 信息 bit/点；这是小样本链路对照，不能与此前大样本结果混用。

~~~powershell
conda activate communication
python half_spacing_ofdm_sim.py --frames 2000 --coding-frames 45 --high-snr-analysis
python -m simulation.ofdm_lmmse_high_snr --report simulation_results/half_spacing_ofdm_comparison/report.json
python half_spacing_ofdm_sim.py --k 4 --codes none --frames 2000 --ebn0-db 0 20 40 60 80 100 120 --no-plots --output-dir simulation_results/half_spacing_ofdm_comparison/high_snr_analysis/waveform_8_tones
python -m unittest discover -s simulation -p "test_half_spacing_ofdm_*.py" -v
~~~

主曲线仍为 `ber_comparison.png`；高精度参考图为 `high_snr_analysis/lmmse_high_snr.png`，对应目录保存 metrics.csv、report.json 和 summary.md。
