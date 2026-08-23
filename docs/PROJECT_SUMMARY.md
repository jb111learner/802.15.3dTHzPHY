# 802.15.3dTHzPHY — 项目技术总结

> 太赫兹通信物理层浮点仿真平台 | IEEE 802.15.3d SC-PHY | SC-FDE / OFDM / MIMO-OFDM

---

## 1. 项目概述

基于 Python 的太赫兹（THz）通信链路级仿真器，实现 **IEEE 802.15.3d SC-PHY** 物理层规范，支持双模（SC-FDE / OFDM）收发信机及独立的 2×2 MIMO-OFDM 分支。配备 PySide6 桌面 UI，可进行交互式参数配置、批量对比仿真和结果分析。

- **语言**: Python 3.12
- **核心依赖**: NumPy, SciPy, Matplotlib, galois (GF运算), PySide6
- **入口**: `main.py` → `thz_sim_ui.app.run()`
- **环境**: Conda env `thz`

## 2. 项目架构

```
802.15.3dTHzPHY/
├── main.py                    # 程序入口
├── params/PHYParams.py        # 参数定义与校验
├── transmitter/               # 发射机模块
├── channel/                   # 信道模块
├── receiver/                  # 接收机模块
├── simulation/                # 仿真管理器、测试、脚本
├── utils/                     # 编码、扰码、序列生成
├── thz_sim_ui/                # PySide6 桌面UI
├── core/                      # 基类接口
├── docs/                      # 文档
├── projects/                  # 工程项目文件夹
│   ├── single/                #   单方案工程
│   └── batch/                 #   批量对比工程
└── receiver/simulation_results/ # 仿真输出图表
```

### 数据流架构

仿真采用 **signal_dict 管道** 模式——每个模块接收并返回包含 `signal_stream`, `sample_rate_Hz`, `duration_seconds`, `signal_length`, `padding_bit_num` 的字典，而非裸数组。三大编排器负责调度：

```
THzTransmitter.run() → THzChannel.run() → THzReceiver.run()
```

每个阶段内部包含多个子模块的链式调用。

## 3. 参考标准与规范

| 标准 | 适用范围 | 实现位置 |
|------|---------|---------|
| **IEEE 802.15.3d (THz-SC-PHY)** | LDPC(1440,1344) 14/15 和 LDPC(1440,1056) 11/15 码率 | `utils/LDPCMatrix.py` |
| **3GPP TS 38.211** | Gold 序列扰码 (31阶 m-sequence, Nc=1600) | `utils/Seq_gen.py`, `transmitter/Scrambler.py` |
| **3GPP TR 38.901 V19.4.0** | TDL-A~E 信道模型 | `channel/MultipathChannel.py` |
| **Modified Rapp PA** | 300GHz 功率放大器非线性 | `channel/npa/PowerAmplifier.py` |
| **Golay 互补对** | 前导码 (SYNC+SFD+CES) 生成与信道估计 | `utils/Seq_gen.py`, `receiver/ChannelEstimator.py` |

### IEEE 802.15.3d LDPC 码

- **14/15 码率**: n=1440, k=1344, m=96。H 矩阵由 Table 12-17 + Equation 12-6 构建
- **11/15 码率**: n=1440, k=1056, m=384。H 矩阵由 Table 13-10 + Equation 13-1 构建
- 系统编码 `c = [i, p]`，校验位 `p = (H_info^T · H_parity^{-1})^T` (GF(2) 满秩校验)
- 译码: **Min-Sum** 或 **Normalized Min-Sum** (α=0.8)，最大 30 次迭代，LLR 截断 20.0

## 4. 全链路技术流程

### 4.1 发射端 (TX)

三路分支：**MIMO** → `MIMOOFDMProcessor`；**SC-FDE** → 编码→调制→GI→前导码→脉冲成型；**OFDM** → 编码→调制→OFDM(IFFT+导频)→GI→前导码→脉冲成型。

#### 4.1.1 比特流生成 (`DataProcesser.py`)

- **源**: PRBS (`numpy.random.default_rng`) 或文件输入
- **种子策略**: 固定种子 / 递增种子 / 时间种子
- **帧结构**:
  - SC-FDE: 51 块 × (480 数据 + 32 CP) = 26,112 符号/帧
  - OFDM: 48 符号 × (512 子载波 + 32 CP) = 26,112 符号/帧
- **帧比特数**: `NCBPS × 子帧长度 × 子帧数 × 码率`
- 零填充对齐帧边界

#### 4.1.2 信道编码 (`Encoder.py`, `utils/Coder.py`)

**Reed-Solomon**:
- Galois GF(2^c_exp) 有限域运算 (默认 GF(16))
- 默认 RS(15,11), nsym=4
- LSB-first 位↔符号映射
- 译码模式: 硬译码 / auto-erase(2擦除) / Chase-II 软判决 (p=3 测试模式)

**LDPC**:
- 工程模式: n=672, k=336, 列重 3 稀疏矩阵, seed=2024
- 802.15.3d 模式: 见 §3
- 译码: Min-Sum / Normalized Min-Sum, 校验节点 min1/min2 + 符号积, 伴随式早停

#### 4.1.3 调制 (`Modulator.py`)

| 调制 | NCBPS | 缩放因子 | 业界常用 |
|------|-------|---------|---------|
| BPSK | 1 | 1.0 | 低速率鲁棒通信 |
| QPSK | 2 | √2 | 802.11, LTE |
| 8PSK | 3 | 1.0 | EDGE, DVB-S2 |
| 16QAM | 4 | √10 | 802.11a/g, LTE |
| 64QAM | 6 | √42 | 802.11n/ac |
| 256QAM | 8 | √170 | 802.11ax, 5G NR |
| APSK | 可变 | - | DVB-S2/S2X |

- **π/2 递增旋转**: 所有调制乘以 `exp(jπn/2)` 降低 PAPR
- Gray 映射, Es=1 归一化

#### 4.1.4 OFDM 处理 (`TxOFDMProcesser.py`)

- 512 子载波, 48 OFDM 符号/帧
- **块状导频**: 位置 [0, 16, 32], 相位编码 [0°, 90°, 0°]
- IFFT `norm='ortho'`
- 业界常用算法: **导频辅助信道估计** (802.11a/g/n/ac 标准做法)

#### 4.1.5 MIMO-OFDM 发射 (`MIMOOFDMProcessor.py`)

- 帧结构: [2 同步符号, Nt 正交训练符号, 数据块]
- `tx_scale = 1/√Nt` 功率归一化
- **层映射**: Round-robin 轮转 (`layers[i] = symbols[i::num_streams]`)
- 每天线独立 IFFT + CP

#### 4.1.6 GI 插入 (`GIInserter.py`)

- **CP 模式**: 每块前附加最后 `gi_length` 个符号 (OFDM/SC-FDE 标准做法)
- **Golay 模式**: Golay 序列作为保护间隔 (802.15.3d SC-PHY)

#### 4.1.7 前导码 (`PreambleInsertor.py`)

- **Golay 互补对** 递归构造:
  - a128 / b128 为基序列
  - a256=[b128,a128], b256=[-b128,a128]
  - 类推到 a512, b512
- **短前导码** (3328 符号): SYNC (a128×14) + SFD (-a128) + CES (b128+a512+b512+a256)
- **长前导码** (5120 符号): SYNC (a128×28)

#### 4.1.8 脉冲成型 (`Pulseshaper.py`)

- **SC-FDE**: RRC / RC / Rect 滤波器
- **OFDM**: `firwin` 低通 (Hamming 窗), DC 增益补偿
- **RRC 公式**: 标准 root-raised-cosine, 滚降因子默认 0.22

### 4.2 信道 (CHANNEL)

#### 4.2.1 信道编排 (`THzChannel.run()`)

```
TX IQ不平衡 → PA非线性 → 多径 → AWGN → CFO/相位噪声 → RX IQ不平衡
```

MIMO 分支走独立的 `MIMOChannel.apply()`。

#### 4.2.2 多径信道 (`MultipathChannel.py`)

- **3GPP TR 38.901 TDL-A~E** 抽头模型
- **Jakes 和之正弦波** 时变衰落: 48 正弦波, `fd_max = v·fc/c`
- 分数延迟 FIR 滤波 (Hann 窗, half_len=12)
- 大尺度衰落: FSPL / 对数距离 + 对数正态阴影

#### 4.2.3 MIMO 信道 (`MIMOChannel.py`)

- 模型: Identity / IID Rayleigh / Rician (K 因子)
- 空间相关性: Kronecker `rho^{|i-j|}` 矩阵
- 可加载实测 CIR (`full_mimo=True`)

#### 4.2.4 实测 THz MIMO 信道 (`MeasuredChannel.py`)

三组 2×2 MIMO 实测数据 (300GHz频段):

| 场景 | 文件 | 有效时延 | RMS时延扩展 | 交叉链路 |
|------|------|---------|-----------|---------|
| 8 cm | `THz_MIMO_08cm_2.npz` | 1.6 ns | 0.151 ns | -22.1 dB |
| 12 cm | `THz_MIMO_12cm_2.npz` | 1.2 ns | 0.083 ns | -9.8 dB |
| 50 cm | `THz_MIMO_50cm_2.npz` | 0.5 ns | 0.064 ns | -2.75 dB |

- 原始采样率 30 GHz, 抽头网格 33.33 ps
- 处理: 主径对齐 → 时延截断 → 噪声底扣除 → 累计功率 95% 主径选择
- 模式: 确定性回放 / PDP-Rayleigh 随机化
- 重采样: `resample_poly` + Kaiser 窗 (β=8.6), 功率守恒

#### 4.2.5 AWGN (`AWGN.py`)

- 模式1 热噪声: `Pn = k·T_sys·B`, `T_sys = T_noise · 10^(NF/10)`
- 模式2 SNR: `Pn = Ps/10^(SNR/10)`
- 每维标准差 `√(Pn/2)`

#### 4.2.6 频偏与相位噪声 (`CFO.py`)

- CFO: `Δf = ppm × 1e-6 × fc`
- 相位噪声: 一阶 IIR 滤波高斯噪声

#### 4.2.7 IQ 不平衡 (`IQImbalance.py`)

- **模型**: `y = μ·x + ν·conj(x)`
- μ,ν 由增益不平衡 dB 和相位不平衡 deg 导出
- **FID** (频率无关): 单组 μ,ν
- **FD** (频率相关): I/Q 支路不同 FIR 滤波器
- 可配置位置 (TX/RX/Both) 和功率归一化

#### 4.2.8 功率放大器 (`channel/npa/PowerAmplifier.py`)

- **Modified Rapp**:
  - AM-AM: `G·ρ/(1+|Gρ/Vsat|^(2p))^(1/(2p))`
  - AM-PM: `A·ρ^q1/(1+|ρ/B|^q2)`
- 300 GHz 实测参数集 (M260-003 dataset)
- 自动输入功率缩放

### 4.3 接收端 (RX)

#### 4.3.1 接收链完整流程

```
前端IQ补偿(已知FID/盲白化)
  → 匹配滤波
  → 粗同步 (SYNC FFT互相关)
  → 细同步 (SFD 互相关)
  → 粗CFO估计与补偿
  → 下采样
  → 细CFO估计与补偿
  → CES IQ补偿 (SC-FDE)
  → 噪声估计
  → [OFDM: 导频LS+相位跟踪 | SC-FDE: CES信道估计+均衡]
  → 判决导向IQ补偿 (SC-FDE)
  → LLR软解调
  → 译码 (RS/LDPC)
  → 解扰
```

#### 4.3.2 同步 (`sync/CoarseSync.py`, `sync/FineSync.py`)

- **粗同步**: `scipy.signal.correlate(method="fft")` — **Schmidl-Cox** 风格的 SYNC 互相关
- **细同步**: SFD 相关 + `find_peaks(height=0.5·max)` 峰值检测
- 合并 coarse+fine 偏移后统一截取

#### 4.3.3 CFO 估计 (`CFOEstimator.py`)

- **延迟自相关**: `res = Σ conj(cx)·sx` 过两段重复符号
- `phase = angle(res)`, `cfo = phase/(2π)·fs/L`
- 两级: 粗 (过采样, L=128×sps) + 细 (符号率, L=128)
- 补偿: `exp(-j2π·f·n/fs)`

#### 4.3.4 信道估计 (`ChannelEstimator.py`)

**SC-FDE — Golay 互补对时域相关**:
- 提取 CES 中 a512/b512 → 卷积 `c = (ra + rb)/(2·N_ces)`
- 实测信道: 零时延附近最大能量滑动窗口 + 带符号循环抽头放置
- 非实测: `find_peaks(height=0.1·max)` 稀疏化

**OFDM — 扩展窗口频域 LS**:
- 窗口 = N_ces + Lh - 1 = 543
- `H = Y/X`, a/b 平均, 阈值掩码 1e-3

**模式**: per_frame (逐帧) / global (首帧共享)

#### 4.3.5 均衡器 (`Equalizer.py`)

- **ZF**: `W = 1/H`，|H| 低于 5% 峰值时裁剪保护 (防止 NaN)
- **MMSE**: `W = conj(H)/(|H|² + N0)`
- CP 去除 → FFT shift → 频域均衡 → IFFT → F-order 拉平
- **后均衡噪声**: `N0·mean(|W|²) + residual_ISI_var`

#### 4.3.6 噪声估计 (`NoiseEstimator.py`)

- **无偏**: SYNC 重塑为 (128, Nblks)，相邻列差分: `noise_var = Σ|diff|²/(128·(Nblks-1)·2)`

#### 4.3.7 IQ 补偿

**SC-FDE — 实值并行 FIR 补偿** (`IQCompensator.py` + `IQRealParallelCalibrator`):
- 损伤模型: `rI = e1·yI + e2·yQ + dI`, `rQ = e3·yI + e4·yQ + dQ`
- LS/岭回归估计 e1~e4 → 设计 4-FIR 后补偿器 f1~f4 → 应用 DC 偏移
- 逐帧 `per_frame` / 首帧复用 `first_frame`

**SC-FDE — 判决导向残差补偿** (`DecisionDirectedIQCompensator`):
- 均衡后符号硬判决 → 以判决为参考 → 迭代 LS 估计残差 → 补偿
- 默认 3 次迭代, filter_len=5

**OFDM — 宽线性 IQ 补偿** (`IQCompensatorOFDM.py`):
- 子载波配对模型: `Y[k] = A[k]·X[k] + B[k]·conj(X[-k])`
- 3 个块状导频 (需满秩宽线性设计) → LS求解 A,B
- CIR 窗平滑 → 逐镜像对 2×2 MMSE → 条件数退化回标量 ZF

**前端盲白化** (`BlindRXIQWhiteningCompensator`):
- 二阶统计量: I/Q 协方差矩阵特征分解 → 白化, CFO 前执行

**已知参数 FID 求逆** (`ConfiguredRXIQCompensator`):
- 精确宽线性逆: `x̂ = (μ*·y - ν·y*)/(|μ|² - |ν|²)`

#### 4.3.8 OFDM 解调 (`RxOFDMProcesser.py`)

- 去掉前导码 + CP → FFT grid → 导频 LS / 宽线性联合均衡
- 公共相位跟踪: 解卷导频相位 → 线性拟合 → 全部符号去旋转
- 提取 45 个数据符号/块 → F-order 流

#### 4.3.9 LLR 软解调 (`DeModulator.py`)

- **Max-Log LLR**: 所有解调器先去 π/2 旋转
- BPSK: `-4·Re(s)/σ²`; QPSK: `-2√2·Re/Im(s)/σ²`
- 16QAM/64QAM: 分段线性; 256QAM: Gray掩码最小距离
- APSK: 缓存 max-log S0/S1 子集

#### 4.3.10 MIMO 接收 (`MIMOReceiverProcessor.py`)

- 已知 RX IQ 宽线性逆 (逐天线) → 滑动模板同步 → CFO (两重复符号)
- LS 信道估计 (对角训练) → **逐子载波 ZF/MMSE 检测**
- 层解映射 (逆 round-robin)

## 5. 关键参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fc` | 1000 GHz | 载频 |
| `bandwidth` | 30 GHz | 信号带宽 |
| `sample_rate` | 30 GBd | 符号率 |
| `link_mode` | `"ofdm"` | 链路模式 (sc-fde/ofdm) |
| `subwave_num` | 512 | OFDM 子载波数 |
| `subframe_length` | 480 | SC-FDE 块长 |
| `subframe_num` | 51 | 每帧块数 |
| `subframe_ofdm_num` | 48 | OFDM 每符号数 |
| `gi_length` | 32 | CP 长度 |
| `oversampling` | 4 | 过采样率 |
| `NCBPS` | 6 | 每符号比特数 |
| `code_type` | `"LDPC"` | RS / LDPC |
| `ldpc_standard_rate` | `"14/15"` | LDPC 码率 |
| `Preamble_type` | `"short"` | 短/长前导 |
| `SNRdB` | 20 | SNR (dB) |
| `equalizer_method` | `"zf"` | ZF / MMSE |
| `enable_mimo` | False | MIMO 开关 |
| `mimo_channel_model` | `"iid_rayleigh"` | MIMO 信道模型 |
| `multipath_source` | `"simulated"` | 仿真/实测信道 |
| `measured_channel_scenario` | `"8cm"` | 实测场景 |

## 6. 业界常用算法对照

| 环节 | 本项目算法 | 业界常用 | 标准/文献 |
|------|----------|---------|----------|
| 扰码 | Gold 序列 (31阶) | Gold / PN9 / PN15 | 3GPP TS 38.211 §5.2.1 |
| 信道编码 | RS(15,11)/RS(255,192) + LDPC(1440,k) | LDPC (5G NR BG1/2) | IEEE 802.15.3d §12-13 |
| 调制 | QPSK/16QAM/64QAM + π/2旋转 | π/2-BPSK/QPSK (SC) | 802.15.3d, 3GPP |
| 前导码 | Golay 互补对 a512/b512 | Zadoff-Chu / Golay | 802.15.3d, 802.11ad |
| 同步 | Schmidl-Cox (延迟自相关) | Schmidl-Cox / Minn / Park | Schmidl & Cox 1997 |
| CFO估计 | 两段重复符号相位差 | 同左 + CP相关 | Moose 1994 |
| 信道估计 | Golay时域相关 (SC) / 导频LS (OFDM) | LS / MMSE / DFT窗 | 802.11a/g/n |
| 均衡 | ZF/MMSE 频域单抽头 (SC-FDE) | ZF/MMSE/MLSE | Sari 1995 (SC-FDE) |
| IQ补偿 | 宽线性 (OFDM) / 实值FIR (SC-FDE) | Widely Linear / Adaptive | Valkama 2001 |
| 多径信道 | TDL-A~E + Jakes | CDL/TDL (3GPP), Rayleigh/Rician | 3GPP TR 38.901 |
| PA非线性 | Modified Rapp | Rapp / Saleh / Volterra | Rapp 1991 |
| MIMO检测 | 逐子载波 ZF/MMSE | ZF/MMSE/MLD/SD | 802.11n/ac |

## 7. 仿真能力

- **单方案仿真**: 参数配置 → 蒙特卡洛运行 → PNG 图表 + metrics.json
- **批量对比**: 多方案 SNR 扫描 → BER 叠加图 + 谱效叠加图
- **MIMO 2×2**: 独立 MIMO-OFDM 链路全流程
- **实测信道**: 3 组 300GHz MIMO 实测 CIR 回放与随机化
- **IQ 不平衡**: FID/FD 注入 + 多种补偿方案对比
- **功率放大器**: Modified Rapp 非线性 AM-AM/AM-PM

## 8. 文件清单

### 发射机 (transmitter/)
| 文件 | 行数 | 功能 |
|------|------|------|
| `THzTransmitter.py` | 336 | TX 编排器 |
| `DataProcesser.py` | 246 | 比特流生成与组帧 |
| `Encoder.py` | 107 | RS/LDPC 编码封装 |
| `Modulator.py` | 406 | QAM/APSK 调制 |
| `Scrambler.py` | 89 | Gold 序列扰码 |
| `TxOFDMProcesser.py` | 268 | OFDM IFFT+导频 |
| `MIMOOFDMProcessor.py` | 93 | MIMO-OFDM 帧构建 |
| `MIMOLayerMapper.py` | 18 | 空间流层映射 |
| `GIInserter.py` | 103 | CP/Golay 插入 |
| `PreambleInsertor.py` | 251 | 前导码生成与插入 |
| `Pulseshaper.py` | 349 | RRC/RC/Rect 脉冲成型 |
| `HeaderGenerator.py` | 141 | 37-bit PHY Header |

### 信道 (channel/)
| 文件 | 行数 | 功能 |
|------|------|------|
| `THzChannel.py` | 334 | 信道编排器 |
| `MultipathChannel.py` | 1588 | 3GPP TDL + Jakes 衰落 |
| `MeasuredChannel.py` | 464 | 实测 MIMO CIR |
| `MIMOChannel.py` | 201 | MIMO 频率选择性信道 |
| `AWGN.py` | 142 | 热噪声 / SNR 模式 |
| `CFO.py` | 232 | 频偏 + 相位噪声 |
| `IQImbalance.py` | 298 | FID/FD IQ 不平衡 |
| `npa/PowerAmplifier.py` | 121 | Modified Rapp PA |

### 接收机 (receiver/)
| 文件 | 行数 | 功能 |
|------|------|------|
| `THzReceiver.py` | 466 | RX 编排器 |
| `MatchedFilter.py` | 194 | 匹配滤波 |
| `sync/CoarseSync.py` | 139 | SYNC FFT 互相关 |
| `sync/FineSync.py` | 324 | SFD 相关细同步 |
| `CFOEstimator.py` | 430 | 两级 CFO 估计 |
| `Downsampler.py` | 193 | 符号率恢复 |
| `ChannelEstimator.py` | 441 | Golay/LS 信道估计 |
| `Equalizer.py` | 423 | ZF/MMSE 频域均衡 |
| `NoiseEstimator.py` | 416 | 差分噪声估计 |
| `RxOFDMProcesser.py` | 561 | OFDM 解调 |
| `DeModulator.py` | 660 | Max-Log LLR 软解调 |
| `Decoder.py` | 229 | RS/LDPC 译码封装 |
| `IQCompensator.py` | 1079 | CES/DD IQ 补偿 |
| `IQCompensatorOFDM.py` | 407 | 宽线性 OFDM IQ |
| `MIMOReceiverProcessor.py` | 194 | MIMO RX 流水线 |
| `MIMOChannelEstimator.py` | 30 | MIMO LS 估计 |
| `MIMODetector.py` | 54 | ZF/MMSE 检测 |
| `MIMOLayerDemapper.py` | 24 | 逆层映射 |

### 工具 (utils/)
| 文件 | 功能 |
|------|------|
| `Coder.py` (667) | RS + LDPC 编解码 |
| `LDPCMatrix.py` (414) | 802.15.3d LDPC 矩阵 |
| `Seq_gen.py` (102) | Gold 序列 + Golay 对 |
| `MIMOUtils.py` (39) | MIMO 工具 |
