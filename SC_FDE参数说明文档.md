# SC-FDE通信仿真流程参数说明

## 一、发射端模块（transmitter）

### 1. BitStreamProcessor（比特流处理器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 比特源类型 | `bit_source` | str | "PRBS" | 比特源类型："PRBS"（伪随机序列）或"文件输入" |
| 文件路径 | `file_path` | str | None | 文件输入时的路径 |
| 数据时长 | `duration` | float | 1e-5 | 数据时长（秒） |
| 采样率 | `sample_rate` | int | 10e9 | 采样率（Bd） |
| 位深度 | `bit_depth` | int | 8 | 每采样点比特深度 |
| 比特流长度 | `bit_length` | int | None | 比特流总长度（bit），None时由duration计算 |
| 固定种子 | `random_seed` | int | None | 固定随机种子 |
| 种子策略 | `seed_strategy` | str | "递增种子" | 种子策略："固定种子"、"递增种子"、"时间种子" |

---

### 2. Scrambler（扰码器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 启用扰码 | `scramble` | bool | True | 是否启用扰码 |
| 扰码初始化 | `c_init` | int | 0x12345678 | 扰码器初始化值（3GPP TS 38.211 Gold序列） |

---

### 3. Encoder（编码器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 编码类型 | `code_type` | str | "RS" | 编码类型："RS"（里德-所罗门码）或"LDPC" |
| RS包大小 | `rs_packet_size` | int | 192 | RS编码包大小 |
| RS校验符号数 | `rs_nsym` | int | 63 | RS校验符号数量 |
| RS有限域指数 | `rs_c_exp` | int | 8 | RS有限域指数 |

---

### 4. Modulator（调制器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| MCS方案 | `MCS` | int | 1 | 调制编码方案：0=DBPSK，1=QAM系列，2=APSK系列 |
| 每符号比特数 | `NCBPS` | int | 4 | 每符号比特数：1=BPSK，2=QPSK，3=8PSK，4=16QAM，6=64QAM，8=256QAM |
| 编码率 | `Rate` | float | 1.0 | 编码率（用于LLR缩放） |
| APSK环参数 | `APSK_RINGS` | list | [(4,1.0),(12,2.85)] | APSK环参数，格式[(点数,半径),...] |
| APSK相位偏移 | `APSK_PHASE_OFFSETS` | list | [π/4, π/12] | APSK各环相位偏移（弧度） |

---

### 5. GIInserter（保护间隔插入器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| GI类型 | `gi_type` | str | "cp" | 保护间隔类型："cp"（循环前缀）或"golay"（Golay序列） |
| GI长度 | `gi_length` | int | 64 | 保护间隔长度（符号数） |
| 子帧长度 | `subframe_length` | int | 480 | 数据子帧长度（符号数） |
| 子帧数量 | `subframe_num` | int | 51 | 单数据帧子帧数量 |

---

### 6. PreambleInsertor（前导码插入器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 前导码类型 | `Preamble_type` | str | "long" | 前导码类型："short"（14次a128重复）或"long"（28次a128重复） |
| 相位旋转 | `phase_rotation` | float | π/4 | 前导码相位旋转角度（弧度） |
| PPRE字段 | `ppre` | int | 1 | 物理层前导码配置（0-3） |
| PW字段 | `pw` | int | 0 | PW字段 |

**前导码结构：**
- SYNC：a128序列重复14/28次，用于帧检测和粗同步
- SFD：起始帧定界符，用于细同步
- CES：信道估计序列（b128+a512+b512），用于信道估计

---

### 7. TxPulseShaper（脉冲成型器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 过采样率 | `oversampling` | int | 4 | 过采样率 |
| 滚降系数 | `rolloff` | float | 0.22 | 滚降系数（0~1） |
| 滤波器类型 | `filter_type` | str | "rrc" | 滤波器类型："rc"（升余弦）、"rrc"（根升余弦）、"rect"（矩形） |
| 滤波器长度 | `filter_length` | int | 32 | 滤波器长度（符号数） |

---

## 二、信道模块（channel）

### 1. MultipathChannel（多径信道）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 启用多径 | `enable_multipath` | bool | True | 是否启用多径效应 |
| 多径路径 | `multipath_paths` | list | 见下方 | 多径路径列表，每个路径包含delay_samples、gain_type、gain |
| 衰落模型 | `fading_model` | str | "static" | 衰落模型："static"（静态）、"frame"（帧级变化）、"block"（块级变化） |
| 使用分数延迟 | `use_frac_delay` | bool | False | 是否启用分数延迟滤波 |
| 分数延迟滤波器半长 | `frac_filter_half_len` | int | 12 | 分数延迟FIR滤波器半长 |
| 分数延迟窗函数 | `frac_filter_window` | str | "hann" | 窗函数："hann"、"hamming"、"rect" |
| 静态信道开关 | `static_channel` | bool | True | True时强制所有路径按静态增益处理 |
| 归一化信道功率 | `normalize_channel_power` | bool | False | 是否按sum(|gain|²)对路径增益归一化 |
| 启用帧间记忆 | `enable_multipath_memory` | bool | False | 是否启用静态多径的帧间输入历史缓冲区 |
| 衰落随机种子 | `fading_seed` | int | None | 衰落随机种子 |
| 块衰落长度 | `block_length` | int | 256 | 块衰落时每多少采样点更新一次路径增益 |

**默认多径路径配置：**
```python
[
    {"delay_samples": 0.0,  "gain_type": "static", "gain": 1.0 + 0.0j},
    {"delay_samples": 8.0,  "gain_type": "static", "gain": 0.45 * exp(-jπ/6)},
    {"delay_samples": 16.0, "gain_type": "static", "gain": 0.25 * exp(-jπ/3)},
    {"delay_samples": 32.0, "gain_type": "static", "gain": 0.25 * exp(-jπ/2)},
    {"delay_samples": 64.0, "gain_type": "static", "gain": 0.25 * exp(-jπ/12)},
]
```

---

#### 1.1 PDP功率时延谱参数

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 使用PDP | `use_pdp` | bool | False | 是否使用PDP自动生成multipath_paths |
| PDP模型 | `pdp_model` | str | "manual" | PDP模型："manual"、"uniform"、"exponential"、"custom" |
| PDP路径数量 | `pdp_num_paths` | int | 6 | PDP自动生成的路径数量 |
| PDP总功率 | `pdp_total_power` | float | 1.0 | PDP路径平均功率总和 |
| PDP最大时延(采样点) | `pdp_max_delay_samples` | float | 32.0 | 最大路径时延（采样点） |
| PDP最大时延(秒) | `pdp_max_tau` | float | None | 最大路径时延（秒） |
| PDP随机时延 | `pdp_random_delays` | bool | False | 是否随机生成路径时延 |
| PDP随机种子 | `pdp_seed` | int | None | PDP随机时延种子 |
| PDP RMS时延(采样点) | `pdp_rms_delay_samples` | float | 8.0 | 指数PDP衰减尺度（采样点） |
| PDP RMS时延(秒) | `pdp_rms_tau` | float | None | 指数PDP衰减尺度（秒） |
| PDP增益类型 | `pdp_gain_type` | str | "rayleigh" | 增益类型："static"、"rayleigh"、"rician" |
| PDP包含LOS | `pdp_include_los` | bool | False | 是否将第0条路径作为LOS路径 |
| PDP LOS增益类型 | `pdp_los_gain_type` | str | "rician" | LOS路径增益类型 |
| PDP Rician K因子 | `pdp_rician_K` | float | 10.0 | LOS Rician路径的K因子 |
| PDP LOS初始相位 | `pdp_los_phase` | float | 0.0 | LOS初始相位 |
| PDP LOS多普勒 | `pdp_f_los` | float | 0.0 | LOS多普勒频率 |
| PDP功率归一化 | `pdp_normalize_power` | bool | False | 是否把PDP平均功率归一化 |

---

#### 1.2 大尺度衰落参数

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 启用大尺度衰落 | `enable_large_scale_fading` | bool | False | 是否启用大尺度衰落 |
| 大尺度模型 | `large_scale_model` | str | "none" | 模型："none"、"fspl"、"log_distance" |
| 链路距离 | `link_distance_m` | float | 1.0 | 发射机到接收机距离（米） |
| 载频 | `carrier_frequency_Hz` | float | 300e9 | 载波频率（Hz） |
| 参考距离 | `reference_distance_m` | float | 1.0 | 对数距离模型参考距离d0 |
| 参考路径损耗 | `reference_path_loss_db` | float | None | 参考损耗（dB） |
| 路径损耗指数 | `path_loss_exponent` | float | 2.0 | 对数距离模型路径损耗指数n |
| 阴影衰落标准差 | `shadowing_std_db` | float | 0.0 | 阴影衰落标准差（dB） |
| 大尺度种子 | `large_scale_seed` | int | None | 阴影衰落随机种子 |

---

### 2. AWGN（加性高斯白噪声）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 启用AWGN | `enable_awgn` | bool | True | 是否启用AWGN噪声 |
| 信噪比 | `SNRdB` | float | 25 | 信噪比（dB） |
| 噪声温度 | `noise_temperature` | float | None | 噪声温度（K），与SNRdB二选一 |
| 噪声系数 | `noise_figure_db` | float | None | 噪声系数（dB） |
| 载波频率 | `fc` | float | 1000e9 | 载波频率（Hz） |
| 信号带宽 | `bandwidth` | float | 30e9 | 信号带宽（Hz） |

---

### 3. CFO（载波频率偏移）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 启用CFO | `enable_cfo` | bool | True | 是否启用载波频率偏移 |
| 频偏PPM | `ppm` | float | 1 | 频率偏移（ppm） |
| 载波频率 | `fc` | float | 1000e9 | 载波频率（Hz） |
| 启用相位噪声 | `enable_phase_noise` | bool | True | 是否启用相位噪声 |
| 相位噪声标准差 | `phase_noise_std` | float | 0.01 | 相位噪声标准差（弧度） |
| 相位噪声带宽 | `phase_noise_bw` | float | 100e3 | 相位噪声低通带宽（Hz） |

---

### 4. IQImbalance（IQ不平衡）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 启用IQ不平衡 | `enable_iq_imbalance` | bool | False | 是否启用IQ不平衡 |
| IQ幅度不平衡 | `iq_gain_imbalance_db` | float | 0.0 | IQ幅度不平衡（dB） |
| IQ相位不平衡 | `iq_phase_imbalance_deg` | float | 0.0 | IQ相位不平衡（度） |
| 不平衡位置 | `iq_imbalance_position` | str | "rx" | 不平衡位置："tx"、"rx"、"both" |

---

## 三、接收端模块（receiver）

### 1. RxMatchedFilter（匹配滤波器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| （从发射端继承） | - | - | - | 自动使用发射端的脉冲成型滤波器系数 |

---

### 2. CoarseSync（粗同步）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 相关阈值 | `sync_threshold` | float | 0.1 | 自相关检测阈值（0~1） |
| 平台长度因子 | `scaling_factor` | int | 30 | 连续高相关区间最小长度因子 |

**检测方法：**
- `auto`：自相关检测（基于重复SYNC序列）
- `cross`：互相关检测（基于本地SYNC序列）

---

### 3. FineSync（细同步）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 搜索窗口 | `search_window` | int | 50 | 峰值搜索窗口大小（采样点数） |
| 最小峰值比例 | `min_peak_ratio` | float | 0.5 | 最小峰值比例阈值 |

**功能：** 基于SFD序列互相关，找到最佳采样点位置

---

### 4. CFOEstimator（频偏估计器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| （从发射端继承） | - | - | - | 使用发射端的SYNC序列参数 |

**估计模式：**
- 粗估计：基于过采样信号的SYNC段
- 细估计：基于符号率信号的SYNC段

---

### 5. Downsampler（下采样器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 过采样率 | `oversampling` | int | 4 | 过采样率（与发射端一致） |

---

### 6. ChannelEstimator（信道估计器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 估计模式 | `estimation_mode` | str | "per_frame" | 估计模式："per_frame"（每帧独立）或"global"（全局共享） |
| 信道长度 | `Lh` | int | 64 | 信道长度（与GI长度一致） |
| FFT大小 | `nfft` | int | 480 | FFT大小（与子帧长度一致） |

**估计方法：** 基于CES序列互相关，使用a512/b512序列进行信道冲激响应估计

---

### 7. NoiseEstimator（噪声估计器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 估计模式 | `estimation_mode` | str | "per_frame" | 估计模式："per_frame"或"global" |

**估计原理：** 利用SYNC序列的重复块结构，通过块间差分消去信号，仅保留噪声

---

### 8. FreqDomainEqualizer（频域均衡器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 均衡方法 | `equalizer_method` | str | "zf" | 均衡方法："zf"（迫零）或"mmse"（最小均方误差） |
| GI类型 | `gi_type` | str | "cp" | GI类型（与发射端一致） |
| GI长度 | `gi_length` | int | 64 | GI长度（与发射端一致） |
| 子帧长度 | `subframe_length` | int | 480 | 子帧长度（与发射端一致） |

---

### 9. THzDemodulator（解调器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| MCS方案 | `MCS` | int | 1 | 调制方案（与发射端一致） |
| 每符号比特数 | `NCBPS` | int | 4 | 每符号比特数（与发射端一致） |
| 编码率 | `Rate` | float | 1.0 | 编码率（与发射端一致） |

**输出：** LLR（对数似然比）值，通过`llr_to_bits()`转换为比特

---

### 10. Decoder（解码器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 编码类型 | `code_type` | str | "RS" | 编码类型（与发射端一致） |
| 解码模式 | `decode_mode` | str | "chase" | 解码模式："hard"（硬判决）或"chase"（Chase合并） |
| Chase候选数 | `chase_num_per_packet` | int | 2 | Chase合并每包试探数量 |

---

### 11. DeScrambler（解扰器）

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 扰码初始化 | `c_init` | int | 0x12345678 | 扰码器初始化值（与发射端一致） |

---

## 四、仿真控制参数

| 参数名 | PHYParams键名 | 类型 | 默认值 | 说明 |
|--------|-------------|------|--------|------|
| 链路模式 | `link_mode` | str | "sc-fde" | 链路模式："sc-fde"或"ofdm" |
| 运行次数 | `run_times` | int | 1 | 仿真运行次数 |
| 保存中间结果 | `save_intermediate` | bool | False | 是否保存中间仿真结果 |

---

## 五、SC-FDE仿真流程总览

```
发射端：
  BitStreamProcessor → Scrambler → Encoder → Modulator → GIInserter → PreambleInsertor → TxPulseShaper
  
信道：
  IQImbalance(tx) → MultipathChannel → AWGN → CFO → IQImbalance(rx)
  
接收端：
  RxMatchedFilter → FineSync → CFOEstimator(coarse) → Downsampler → CFOEstimator(fine) → 
  ChannelEstimator → NoiseEstimator → FreqDomainEqualizer → THzDemodulator → Decoder → DeScrambler
```

---

## 六、关键参数约束关系

| 参数关系 | 说明 |
|----------|------|
| `gi_length ≥ 最大多径延迟` | GI长度必须大于信道最大延迟，否则无法完全消除ISI |
| `subframe_length ≥ gi_length` | 子帧长度应大于GI长度 |
| `filter_length` 为 `oversampling` 的倍数 | 滤波器长度应与过采样率匹配 |
| `NCBPS` 与 `MCS` 匹配 | MCS=1时NCBPS支持1/2/3/4/6/8，MCS=2时支持APSK |
| `oversampling` 发射/接收端一致 | 发射端上采样率与接收端下采样率必须相同 |
| `c_init` 发射/接收端一致 | 扰码器初始化值必须一致才能正确解扰 |
| `SNRdB` 与 `noise_temperature` 二选一 | 设置SNRdB后自动计算噪声温度，反之亦然 |

---

## 七、参数分类索引

### 7.1 按数据类型分类

| 类型 | 参数列表 |
|------|----------|
| bool | scramble, enable_awgn, enable_phase_noise, enable_cfo, enable_iq_imbalance, enable_multipath, use_frac_delay, static_channel, normalize_channel_power, enable_multipath_memory, use_pdp, pdp_random_delays, pdp_include_los, pdp_normalize_power, enable_large_scale_fading, save_intermediate |
| int | bit_depth, bit_length, random_seed, c_init, rs_nsym, rs_c_exp, rs_packet_size, NCBPS, MCS, subframe_num, subframe_length, gi_length, ppre, pw, oversampling, filter_length, pdp_num_paths, frac_filter_half_len, block_length, jakes_num_sinusoids, chase_num_per_packet, run_times |
| float | duration, sample_rate, fc, bandwidth, rolloff, phase_rotation, Rate, SNRdB, noise_temperature, noise_figure_db, phase_noise_std, phase_noise_bw, ppm, iq_gain_imbalance_db, iq_phase_imbalance_deg, pdp_total_power, pdp_max_delay_samples, pdp_max_tau, pdp_rms_delay_samples, pdp_rms_tau, pdp_rician_K, pdp_los_phase, pdp_f_los, link_distance_m, carrier_frequency_Hz, reference_distance_m, reference_path_loss_db, path_loss_exponent, shadowing_std_db |
| str | bit_source, file_path, seed_strategy, link_mode, gi_type, code_type, filter_type, Preamble_type, iq_imbalance_position, fading_model, pdp_model, frac_filter_window, pdp_gain_type, pdp_los_gain_type, large_scale_model, equalizer_method, decode_mode |
| list | APSK_RINGS, APSK_PHASE_OFFSETS, multipath_paths, pdp_custom_delay_samples, pdp_custom_taus, pdp_custom_powers |

### 7.2 按功能模块分类

| 模块 | 参数列表 |
|------|----------|
| 数据生成 | bit_source, file_path, duration, sample_rate, bit_depth, bit_length, random_seed, seed_strategy |
| 调制编码 | MCS, NCBPS, Rate, code_type, rs_nsym, rs_c_exp, rs_packet_size, scramble, c_init, APSK_RINGS, APSK_PHASE_OFFSETS |
| 帧结构 | subframe_num, subframe_length, gi_length, gi_type, ppre, pw, Preamble_type, phase_rotation |
| 脉冲成型 | oversampling, rolloff, filter_type, filter_length |
| 信道 | enable_multipath, multipath_paths, fading_model, use_frac_delay, frac_filter_half_len, frac_filter_window, static_channel, normalize_channel_power, enable_multipath_memory, fading_seed, block_length, use_pdp, pdp_model, pdp_num_paths, pdp_total_power, pdp_max_delay_samples, pdp_max_tau, pdp_random_delays, pdp_seed, pdp_rms_delay_samples, pdp_rms_tau, pdp_gain_type, pdp_include_los, pdp_los_gain_type, pdp_rician_K, pdp_los_phase, pdp_f_los, pdp_normalize_power, enable_large_scale_fading, large_scale_model, link_distance_m, carrier_frequency_Hz, reference_distance_m, reference_path_loss_db, path_loss_exponent, shadowing_std_db, large_scale_seed, enable_awgn, SNRdB, noise_temperature, noise_figure_db, fc, bandwidth, enable_cfo, ppm, enable_phase_noise, phase_noise_std, phase_noise_bw, enable_iq_imbalance, iq_gain_imbalance_db, iq_phase_imbalance_deg, iq_imbalance_position |
| 接收处理 | equalizer_method, decode_mode, chase_num_per_packet |
| 仿真控制 | link_mode, run_times, save_intermediate |
