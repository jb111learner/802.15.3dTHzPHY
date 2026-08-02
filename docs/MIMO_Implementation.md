# MIMO-OFDM 实现说明

## 当前能力

项目在保留原 SISO 分支的基础上增加了独立 MIMO-OFDM 数据路径，当前支持：

- 2×2 及一般 `Nt×Nr` 空间复用（当前要求空间流数等于发射天线数）；
- 每根发射天线独立的 OFDM IFFT 和循环前缀；
- 两个重复同步符号、公共 CFO 估计与多接收天线联合定时；
- 按发射天线时分正交的训练符号；
- 频率选择性 Rayleigh、Rician、Identity MIMO 信道；
- 收发空间相关性；
- 理想 CSI 或 LS 估计 CSI；
- ZF、MMSE 逐子载波检测；
- AWGN、公共接收本振 CFO/相位噪声、逐天线 IQ 不平衡；
- BER、有效吞吐率、信道估计 NMSE 和信道条件数指标。

## 启用示例

```python
params.update(
    enable_mimo=True,
    link_mode="ofdm",
    num_tx=2,
    num_rx=2,
    num_spatial_streams=2,
    mimo_scheme="spatial_multiplexing",
    mimo_detector="mmse",
    mimo_channel_model="iid_rayleigh",
    mimo_csi_mode="estimated",
    enable_multipath=True,
    mimo_num_taps=4,
)
```

发射信号采用 `(num_tx, num_samples)`，接收信号采用
`(num_rx, num_samples)`，信道频响采用
`(num_rx, num_tx, num_subcarriers)`。

## 帧结构

```text
零保护间隔
→ 重复同步符号 × 2
→ TX0 正交训练
→ TX1 正交训练
→ 多空间流数据 OFDM 符号
→ 尾部保护间隔
```

数据资源网格按总发射功率固定原则乘以 `1/sqrt(num_tx)`，因此与 SISO
比较时不会凭空增加总发射功率。

## 已知边界

- MIMO 当前仅接入 OFDM，SC-FDE 继续走原 SISO 分支；
- 当前仅实现空间复用，不包含 STBC、SVD 预编码和混合波束成形；
- MIMO 分支暂不接入 PA 非线性，启用 `enable_pa` 会给出明确错误；
- RX IQ 补偿使用配置参数执行宽线性逆变换，尚未实现盲估计或判决导向 MIMO IQ 校准；
- 相位噪声已可注入，但尚无专用的 MIMO 公共相位误差跟踪器。

## 测试

推荐在项目环境中运行：

```powershell
$env:NUMBA_DISABLE_JIT='1'
conda run -n thz python -m pytest simulation/test_mimo_core.py simulation/test_mimo_integration.py -q
```

`NUMBA_DISABLE_JIT=1` 仅用于规避 Windows 上 `galois/numba` 首次创建缓存较慢，
不会改变 MIMO 算法代码。
