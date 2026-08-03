# SISO-OFDM IQ 不平衡补偿

## 处理结构

OFDM 中的 IQ 不平衡会把子载波 `k` 与镜像子载波 `(-k) mod N` 耦合：

```text
Y[k] = A[k] X[k] + B[k] conj(X[(-k) mod N])
```

`A[k]` 和 `B[k]` 分别表示有效直通与镜像响应，其中已经包含传播信道、TX IQ 和 RX IQ 的共同作用。接收端不需要把这些物理参数分别辨识。

`receiver/IQCompensatorOFDM.py` 对每个镜像子载波对构造增广矩阵：

```text
[Y[k]       ]   [A[k]          B[k]       ] [X[k]        ]
[conj(Y[-k])] = [conj(B[-k])   conj(A[-k])] [conj(X[-k])]
```

然后使用正则化逆矩阵恢复 `X[k]` 和 `X[-k]`。DC 与 Nyquist 子载波使用同一增广形式单独求解。

## 导频

导频块位置和数量保持不变，默认配置为：

```python
pilot_block_indexes = [0, 16, 32]
ofdm_iq_pilot_phase_codes_deg = [0.0, 90.0, 0.0]
```

第二个块导频乘以 `j`，使直通列和镜像列形成满秩设计矩阵。发送端和接收端均使用每个导频块对应的独立参考序列。

## 配置方法

```python
enable_iq_compensation = True
iq_compensation_method = "auto"
```

可选方法：

- `auto`：OFDM 使用 `ofdm_widely_linear`，SC-FDE 使用 `decision_directed`；
- `ofdm_widely_linear`：未知参数联合估计；
- `configured_inverse`：已知 RX FID 参数精确逆补偿；如果还存在 TX IQ、both 或 FD 损伤，会继续使用 OFDM 宽线性均衡；
- `decision_directed`：仅用于 SC-FDE；在 OFDM 下自动重定向；
- `ces`：保留旧接口，不再附加执行 DD。

数值稳定性参数：

```python
iq_ofdm_estimation_ridge = 1e-8
iq_ofdm_equalizer_ridge = 1e-8
iq_ofdm_condition_limit = 1e8
iq_ofdm_response_taps = None  # None 使用 GI 长度
```

有效响应的时域截断采用最大能量循环窗口，能够保留跨 FFT 边界的匹配滤波前后游标。矩阵条件数超过限制时，该镜像子载波对回退为直通标量均衡。

## CFO 联合处理

信道模型在 CFO 后注入 RX IQ，因此已知 RX FID 参数必须在 CFO 补偿前反演。未知 RX IQ 与 CFO 同时启用时，接收机先根据复包络二阶统计量进行盲 I/Q 白化，再进行同步和 CFO 补偿，最终由 OFDM 宽线性均衡去除频率相关残差。

盲白化不读取 `rx_iq_gain_imbalance_db` 或 `rx_iq_phase_imbalance_deg`。默认跳过输入起始 20% 的前导码和滤波瞬态来估计协方差：

```python
iq_blind_covariance_trim_fraction = 0.2
```

## 诊断输出

OFDM 解调结果包含：

- `iq_compensation_method`；
- `iq_compensation_stages`；
- `iq_frontend_diagnostics`；
- `iq_ofdm_diagnostics`；
- `ofdm_freq_grid` 和 `ofdm_equalized_grid`。

每帧诊断记录导频设计条件数、导频残差功率、镜像对子矩阵条件数、回退数量和估计 IRR。

## 限制

- PA 非线性不满足宽线性模型，开启强非线性 PA 时不能保证完全恢复；
- 有效信道时变速度必须低于三个块导频能够跟踪的范围；
- 导频相位编码必须保持满秩，参数校验会拒绝不可辨识的配置。
