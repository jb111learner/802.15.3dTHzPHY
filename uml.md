@startuml THz 802.15.3d PHY 仿真系统UML类图
' 样式配置
skinparam classAttributeIconSize 0
skinparam classBorderThickness 1.2
skinparam arrowThickness 1.1
skinparam linetype ortho

' 1. 核心基类（core/目录）
abstract class BaseParams {
  - _params: dict
  + __init__()
  + validate(): void  // 抽象方法：参数校验
  + update(**kwargs): void  // 动态更新参数
}

abstract class BaseTransmitter {
  - params: BaseParams
  - tx_signal: np.ndarray
  - header_bits: np.ndarray
  - preamble: np.ndarray
  - modulated_data: np.ndarray
  + __init__(params: BaseParams)
  + generate_header(): void  // 抽象：生成头部
  + generate_preamble(): void  // 抽象：生成前导码
  + modulate(data_bits: np.ndarray): np.ndarray  // 抽象：调制
  + insert_cp(data: np.ndarray): np.ndarray  // 抽象：插入CP
  + assemble_signal(data_with_cp: np.ndarray): void  // 抽象：组装信号
  + run(): np.ndarray  // 执行发射流程
  - _generate_random_data(): np.ndarray  // 辅助：生成随机数据
}

abstract class BaseChannel {
  - params: BaseParams
  - rx_signal: np.ndarray
  + __init__(params: BaseParams)
  + apply_multipath(signal: np.ndarray): np.ndarray  // 抽象：多径效应
  + add_awgn(signal: np.ndarray): np.ndarray  // 抽象：加噪声
  + add_cfo(signal: np.ndarray): np.ndarray  // 抽象：加频偏
  + run(tx_signal: np.ndarray): np.ndarray  // 执行信道流程
}

abstract class BaseReceiver {
  - params: BaseParams
  - preamble: np.ndarray
  - rx_signal: np.ndarray
  - sync_offset: int
  - chan_est: np.ndarray
  - equalized_data: np.ndarray
  + __init__(params: BaseParams, preamble: np.ndarray)
  + coarse_sync(): void  // 抽象：粗同步
  + estimate_and_compensate_cfo(): void  // 抽象：频偏估计与补偿
  + fine_sync(): void  // 抽象：细同步
  + estimate_channel(): void  // 抽象：信道估计
  + estimate_noise_var(): void  // 抽象：噪声方差估计
  + equalize(): np.ndarray  // 抽象：均衡
  + run(rx_signal: np.ndarray): np.ndarray  // 执行接收流程
}

' 2. 参数类（params/目录）
class PHYParams extends BaseParams {
  + __init__()  // 初始化THz SC-PHY具体参数
  + validate(): void  // 重写：校验THz参数合法性
}

' 3. 发射机子模块（transmitter/目录）
class HeaderGenerator {
  - params: PHYParams
  + __init__(params: PHYParams)
  + generate_phy_header(): np.ndarray  // 生成PHY头部比特
  + generate_mac_header(): np.ndarray  // 生成MAC头部（可选）
}

class PreambleGenerator {
  - params: PHYParams
  + __init__(params: PHYParams)
  + generate(): tuple[np.ndarray, np.ndarray, np.ndarray]  // 生成SYNC/SFD/CES
}

class Modulator {
  - params: PHYParams
  + __init__(params: PHYParams)
  + qam_mod(data_bits: np.ndarray): np.ndarray  // 4-QAM调制
}

class CPInserter {
  - params: PHYParams
  + __init__(params: PHYParams)
  + insert(data: np.ndarray): np.ndarray  // 插入循环前缀
}

class THzTransmitter extends BaseTransmitter {
  - header_gen: HeaderGenerator
  - preamble_gen: PreambleGenerator
  - modulator: Modulator
  - cp_inserter: CPInserter
  - rotator: Rotator
  + __init__(params: PHYParams)
  + generate_header(): void  // 重写：调用HeaderGenerator
  + generate_preamble(): void  // 重写：调用PreambleGenerator
  + modulate(data_bits: np.ndarray): np.ndarray  // 重写：调用Modulator
  + insert_cp(data: np.ndarray): np.ndarray  // 重写：调用CPInserter
  + assemble_signal(data_with_cp: np.ndarray): void  // 重写：组装发射信号
}

' 4. 信道子模块（channel/目录）
class MultipathChannel {
  - params: PHYParams
  - chan_impulse: np.ndarray  // 信道冲激响应
  + __init__(params: PHYParams)
  + apply(signal: np.ndarray): np.ndarray  // 应用多径卷积
}

class AWGN {
  - params: PHYParams
  + __init__(params: PHYParams)
  + add(signal: np.ndarray): np.ndarray  // 添加高斯白噪声
}

class CFO {
  - params: PHYParams
  + __init__(params: PHYParams)
  + add(signal: np.ndarray): np.ndarray  // 添加频率偏移
}

class THzChannel extends BaseChannel {
  - multipath: MultipathChannel
  - awgn: AWGN
  - cfo: CFO
  - chan_true_fft: np.ndarray  // 真实信道频域响应（用于校验）
  + __init__(params: PHYParams)
  + apply_multipath(signal: np.ndarray): np.ndarray  // 重写：调用MultipathChannel
  + add_awgn(signal: np.ndarray): np.ndarray  // 重写：调用AWGN
  + add_cfo(signal: np.ndarray): np.ndarray  // 重写：调用CFO
}

' 5. 接收机子模块（receiver/目录）
' 同步子模块
class CoarseSync {
  - params: PHYParams
  - sync_seq: np.ndarray  // 已知SYNC序列
  - corr_mag: np.ndarray  // 相关性结果
  + __init__(params: PHYParams, sync_seq: np.ndarray)
  + detect(rx_signal: np.ndarray): int  // 检测粗同步偏移
}

class FineSync {
  - params: PHYParams
  - sfd_seq: np.ndarray  // 已知SFD序列
  + __init__(params: PHYParams, sfd_seq: np.ndarray)
  + detect(sfd_field: np.ndarray): int  // 检测细同步偏移
}

class CFOEstimator {
  - params: PHYParams
  + __init__(params: PHYParams)
  + estimate(sync_field: np.ndarray): float  // 估计频偏
  + compensate(signal: np.ndarray, freq_offset: float): np.ndarray  // 补偿频偏
}

' 其他接收子模块
class ChannelEstimator {
  - params: PHYParams
  - ces_seq: np.ndarray  // 已知CES序列
  + __init__(params: PHYParams, ces_seq: np.ndarray)
  + estimate(ces_field: np.ndarray): np.ndarray  // 估计信道响应
}

class NoiseEstimator {
  - params: PHYParams
  + __init__(params: PHYParams)
  + estimate(sync_field: np.ndarray): float  // 估计噪声方差
}

class Equalizer {
  - params: PHYParams
  + __init__(params: PHYParams)
  + freq_equalize(rx_data: np.ndarray, chan_est: np.ndarray, noise_var: float): np.ndarray  // 频域均衡
}

class THzReceiver extends BaseReceiver {
  - sync: np.ndarray
  - sfd: np.ndarray
  - ces: np.ndarray
  - coarse_sync_module: CoarseSync
  - fine_sync_module: FineSync
  - cfo_estimator: CFOEstimator
  - chan_estimator: ChannelEstimator
  - noise_estimator: NoiseEstimator
  - equalizer: Equalizer
  - rotator: Rotator
  + __init__(params: PHYParams, preamble: np.ndarray)
  + coarse_sync(): void  // 重写：调用CoarseSync
  + estimate_and_compensate_cfo(): void  // 重写：调用CFOEstimator
  + fine_sync(): void  // 重写：调用FineSync
  + estimate_channel(): void  // 重写：调用ChannelEstimator
  + estimate_noise_var(): void  // 重写：调用NoiseEstimator
  + equalize(): np.ndarray  // 重写：调用Equalizer
}

' 6. 工具类（utils/目录）
class Rotator {
  + rotate(signal: np.ndarray): np.ndarray  // 信号旋转
  + de_rotate(signal: np.ndarray): np.ndarray  // 信号解旋转
}

class Plotter {
  - figures: list[plt.Figure]
  + __init__()
  + plot_coarse_sync(corr_mag: np.ndarray, sync_offset: int, threshold: float): void
  + plot_fine_sync(corr_mag: np.ndarray, timing_offset: int): void
  + plot_channel_est(h_true: np.ndarray, h_est: np.ndarray): void
  + plot_equalization(orig_data: np.ndarray, eq_data: np.ndarray): void
  + save_all(save_path: str): void  // 保存所有图表
}

class Metrics {
  + calc_nmse(true: np.ndarray, est: np.ndarray): float  // 计算NMSE（dB）
  + calc_ber(orig_bits: np.ndarray, recv_bits: np.ndarray): float  // 计算误码率
}

' 7. 仿真平台模块（simulation/目录）
class ResultAnalyzer {
  + export_results(results: list[dict], path: str): void  // 导出仿真结果（CSV/Excel）
  + generate_report(results: list[dict]): void  // 生成仿真报告
}

class SimulationManager {
  - params: PHYParams
  - transmitter: THzTransmitter
  - channel: THzChannel
  - receiver: THzReceiver
  - plotter: Plotter
  - metrics: Metrics
  + __init__()
  + configure_params(**kwargs): void  // 配置仿真参数
  + run_single_simulation(): dict  // 执行单次仿真
  + run_batch_simulation(param_grid: list[dict]): list[dict]  // 执行批量仿真
}

' 关系定义（继承/组合/依赖）
' 继承关系
PHYParams -|> BaseParams
THzTransmitter -|> BaseTransmitter
THzChannel -|> BaseChannel
THzReceiver -|> BaseReceiver

' 组合关系（整体-部分，部分生命周期与整体一致）
THzTransmitter *-- HeaderGenerator
THzTransmitter *-- PreambleGenerator
THzTransmitter *-- Modulator
THzTransmitter *-- CPInserter
THzTransmitter *-- Rotator

THzChannel *-- MultipathChannel
THzChannel *-- AWGN
THzChannel *-- CFO

THzReceiver *-- CoarseSync
THzReceiver *-- FineSync
THzReceiver *-- CFOEstimator
THzReceiver *-- ChannelEstimator
THzReceiver *-- NoiseEstimator
THzReceiver *-- Equalizer
THzReceiver *-- Rotator

SimulationManager *-- PHYParams
SimulationManager *-- THzTransmitter
SimulationManager *-- THzChannel
SimulationManager *-- Plotter
SimulationManager *-- Metrics
SimulationManager *-- ResultAnalyzer

' 依赖关系（使用对方的方法/属性，无生命周期绑定）
THzReceiver ..> THzTransmitter : 依赖发射机生成的前导码
SimulationManager ..> THzReceiver : 动态初始化接收机
BaseTransmitter ..> PHYParams : 使用参数配置发射逻辑
BaseChannel ..> PHYParams : 使用参数配置信道特性
BaseReceiver ..> PHYParams : 使用参数配置接收逻辑

@enduml