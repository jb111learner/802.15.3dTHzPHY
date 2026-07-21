from typing import Any, Dict, List

import numpy as np
from core.BaseParams import BaseParams


class PHYParams(BaseParams):
    def apply_dict(self, param_dict: Dict[str, Any]) -> List[str]:
        """安全更新参数，忽略 BaseParams 不支持的字段，返回未知键列表。"""
        unknown_keys = []
        for k, v in param_dict.items():
            if k in self._params:
                self.update(**{k: v})
            else:
                unknown_keys.append(k)
        return unknown_keys

    def _init_params(self):
        self._params = {
            # 仿真样本参数
            "data_source": "PRBS",                                                                  # 比特源类型（'PRBS', '文件输入'）
            "file_path": None,                                                                     # 文件路径 
            "duration": 1e-5,                                                                      # 数据时长（秒）
            "sample_rate": 30e9,                                                                   # 采样率（Bd）
            "sample_length": None,                                                                 # 符号总长度（Sa)
            "random_seed": None,                                                                   # 固定/手动种子
            "seed_strategy": "递增种子",                                                            # 随机种子策略("固定种子", "递增种子", "时间种子")
            "link_mode":"sc-fde",                                                                  # 链路模式（"SC-FDE", "OFDM"）


            # 基础参数
            "fc": 1000e9,                                                                           # 载波频率（1 THz）
            "bandwidth": 30e9,                                                                      # 信号带宽（30 GHz）
            "subframe_num": 51,                                                                     # 数据载荷子帧数量
            "subframe_length": 480,                                                                 # 数据子帧长度(symbols)
            "gi_length": 32,                                                                        # GI长度(symbols)
            "gi_type": "cp",                                                                        # GI类型（循环前缀"cp"或格雷序列"golay"）
            "c_init": 0x12345678,                                                                   # 扰码器参数
            # "frame_bit_num": 73728,                                                               # 单帧比特数

            # OFDM相关参数
            "subwave_num":512,                                                                       # 子载波数
            "subframe_ofdm_num": 48,                                                                 # 单数据帧OFDM子帧数量
            "pilot_block_indexes":[0, 16, 32],                                                       # 块状导频索引
            # "enable_window_filter":False,                                                            # 是否启用加窗与频谱成型
            # "rolling_width":64,                                                                      # 过渡带宽度


            # 调制相关
            "NCBPS": 6,                                                                             # 每符号比特数（1=BPSK,2=QPSK,3=8PSK,4=16QAM,6=64QAM）
            "APSK_RINGS": [(4, 1.0), (12, 2.85)],                                                   # (每环点数, 半径)
            "APSK_PHASE_OFFSETS": [np.pi/4, np.pi/12],                                              # 每环相位偏置（可设全0）
            "code_type": "LDPC",                                                                      # 信道编码类型(RS编码/LDPC)

            # RS码相关设置 (RS(15,11) shortened over GF(256): n=15, k=11, t=2)
            "rs_nsym": 63,                                                                           # Reed-Solomon校验符号数量 (n-k)
            "rs_c_exp": 8,                                                                           # Reed-Solomon有限域指数 (GF(2^8)=GF(256))
            "rs_packet_size": 192,                                                                   # Reed-Solomon编码包大小 (k)

            # LDPC码相关设置
            "ldpc_n": 672,                                                                          # Engineering LDPC codeword length
            "ldpc_k": 336,                                                                          # Engineering LDPC information length
            "ldpc_matrix_type": "ieee802153d_1440",                                                 # engineering / ieee802153d_1440
            "ldpc_standard_rate": "14/15",                                                          # IEEE 802.15.3d LDPC rate: 14/15 / 11/15
            "ldpc_rate11_direction": "minus_literal",                                               # 11/15: minus_literal follows Eq. 13-1 but cannot encode c=[i,p]; plus_systematic_candidate is pending confirmation
            "ldpc_max_iter": 30,                                                                    # LDPC min-sum max iterations
            "ldpc_col_weight": 3,                                                                   # LDPC sparse A row weight
            "ldpc_dv": 3,                                                                           # Legacy alias for ldpc_col_weight
            "ldpc_seed": 2024,                                                                      # LDPC sparse matrix seed
            "ldpc_decode_algorithm": "min_sum",                                                     # min_sum / normalized_min_sum
            "ldpc_llr_clip": 20.0,                                                                  # LDPC LLR clipping threshold
            "ldpc_hard_llr_value": 12.0,                                                            # Hard-decision fallback LLR magnitude
            "ldpc_min_sum_alpha": 0.8,                                                              # Normalized min-sum scale

            # 波形成型相关配置
            "oversampling": 4,                                                                      # 上采样率
            "rolloff": 0.22,                                                                        # 滚降系数
            "filter_type": "rrc",                                                                   # 滤波器类型
            "filter_length": 32,                                                                    # 滤波器长度 

            # 前导码相关
            "Preamble_type": "long",                                                                # 前导码类型
            "phase_rotation": np.pi/4,                                                              # 前导码相位旋转角度（弧度）

            # 其他参数
            "scramble": True,                                                                       # 是否使用扰码
            "ppre": 1,                                                                              # PPRE字段（物理层前导码配置，0-3）
            "pw": 0,                                                                                # PW字段
            "MCS": 1,                                                                               # 调制编码方案("0": DBPSK, "1":QAM, "2":APSK)

            # 信道相关
            "enable_awgn": True,                                                                    # 是否启用AWGN噪声
            "enable_phase_noise": True,                                                             # 是否启用相位噪声
            "enable_cfo": True,                                                                     # 是否启用载波频率偏移
            "noise_temperature": None,                                                              # 信道噪声温度（开尔文）
            "noise_figure_db": None,                                                                # 信道噪声系数（dB）
            "SNRdB": 24,                                                                            # 信噪比（dB），与温度+带宽二选一
            "phase_noise_std": 0.01,                                                                # 相位噪声标准差（弧度）
            "phase_noise_bw": 100e3,                                                                # 相位噪声低通带宽（Hz）
            "ppm": 1,                                                                             # 频偏ppm值 

            "enable_pa": False,
            "pa_model": "modified_rapp",
            "pa_fc_Hz": 300e9,
            "pa_input_power_dbm": -13.2,
            # 这里使用归一化复基带功率域：
            # P = mean(|x|^2) / pa_load_ohm
            # 默认 1.0 更适合你的链路级复包络仿真
            "pa_load_ohm": 1.0,
            "pa_auto_input_scaling": True,
            # 论文图中 AM-PM 是角度量，代码内部会转 rad
            "pa_phase_unit": "deg",
            # 是否从论文开源参数 CSV 读取
            "pa_load_params_from_dataset": True,
            "pa_rapp_dataset_dir": "M260003_Rapp_dataset",
            "pa_use_nearest_fc": False,
            # 300 GHz 默认参数；即使不读 CSV，也能直接运行
            "pa_G": 11.2616,
            "pa_Vsat": 0.0628,
            "pa_p": 1.0013,
            "pa_A": -6.2038e4,
            "pa_B": 0.0160,
            "pa_q1": 1.7344,
            "pa_q2": 1.8972,

            # IQ 不平衡参数
            "enable_iq_imbalance": True,           # 是否启用 I/Q 不平衡
            "iq_imbalance_position": "rx",         # tx / rx / both，默认模拟接收机 IQ 不平衡
            "iq_imbalance_model": "fid",           # 当前仅实现 frequency-independent IQ imbalance
            "iq_power_normalize": True,           # 默认不对 IQ 不平衡输出做功率归一化
            "tx_iq_gain_imbalance_db": 0.0,        # TX: 20log10(Q branch amplitude / I branch amplitude)
            "tx_iq_phase_imbalance_deg": 0.0,      # TX: theta in degrees
            "rx_iq_gain_imbalance_db": 0.0,        # RX: 20log10(Q branch amplitude / I branch amplitude)
            "rx_iq_phase_imbalance_deg": 0.0,      # RX: theta in degrees
            "tx_iq_gI_taps": [1.0],                # TX FD IQ: I branch FIR taps
            "tx_iq_gQ_taps": [1.0],                # TX FD IQ: Q branch FIR taps
            "rx_iq_gI_taps": [1.0],                # RX FD IQ: I branch FIR taps
            "rx_iq_gQ_taps": [1.0],                # RX FD IQ: Q branch FIR taps
            "iq_gI_taps": [1.0],                   # Legacy/global FD IQ: I branch FIR taps
            "iq_gQ_taps": [1.0],                   # Legacy/global FD IQ: Q branch FIR taps
            "iq_gain_imbalance_db": 0.0,           # Legacy I/Q 幅度不平衡，单位 dB
            "iq_phase_imbalance_deg": 0.0,         # Legacy I/Q 相位不平衡，单位 degree
            
            # 多径信道相关参数（第一版：静态多径 + 可选分数延迟，默认关闭以保持原 AWGN 链路不变）
            "enable_multipath": True,                                                                # 是否启用多径信道
            "multipath_paths": [                                                                     # 多径路径列表；delay_samples 优先于 tau
                {"delay_samples": 0.0,  "gain_type": "static", "gain": 1.0 + 0.0j},
                {"delay_samples": 8.0,  "gain_type": "static", "gain": 0.45 * np.exp(-1j * np.pi / 6)},
                {"delay_samples": 16.0, "gain_type": "static", "gain": 0.25 * np.exp(-1j * np.pi / 3)},
                # {"delay_samples": 32.0,  "gain_type": "static", "gain": 0.25 * np.exp(-1j * np.pi / 2)},
            ],
            # ==================PDP 相关参数==================
            # PDP 功率时延谱路径生成参数
            "use_pdp": False,                          # 是否使用 PDP 自动生成 multipath_paths
            "pdp_model": "manual",                     # manual / uniform / exponential / custom
            "pdp_num_paths": 6,                        # PDP 自动生成的路径数量
            "pdp_total_power": 1.0,                    # PDP 路径平均功率总和

            # PDP 时延配置。优先使用 pdp_max_delay_samples；若为 None，则可使用 pdp_max_tau。
            "pdp_max_delay_samples": 32.0,             # 最大路径时延，单位：采样点
            "pdp_max_tau": None,                       # 最大路径时延，单位：秒，后续由 tau*fs 转换
            "pdp_random_delays": False,                # 是否随机生成路径时延；False 时使用均匀时延网格
            "pdp_seed": None,                          # PDP 随机时延种子；None 时不固定随机几何

            # exponential PDP 参数
            "pdp_rms_delay_samples": 8.0,              # 指数 PDP 的衰减尺度，单位：采样点
            "pdp_rms_tau": None,                       # 指数 PDP 的衰减尺度，单位：秒

            # custom PDP 参数
            "pdp_custom_delay_samples": None,          # 自定义 PDP 路径时延，单位：采样点
            "pdp_custom_taus": None,                   # 自定义 PDP 路径时延，单位：秒
            "pdp_custom_powers": None,                 # 自定义 PDP 路径平均功率

            # PDP 生成路径的增益类型
            "pdp_gain_type": "rayleigh",               # static / rayleigh / rician
            "pdp_include_los": False,                   # 是否将第 0 条路径作为 LOS 路径
            "pdp_los_gain_type": "rician",             # LOS 路径增益类型：static / rician / rayleigh
            "pdp_rician_K": 10.0,                      # PDP 中 LOS Rician 路径的 K 因子
            "pdp_los_phase": 0.0,                      # LOS 初始相位
            "pdp_f_los": 0.0,                          # LOS 多普勒，当前 frame/block 模型下用于相位推进

            "pdp_normalize_power": False,               # 是否把 PDP 平均功率归一化到 pdp_total_power
            # ============================================

            # =========大尺度衰落 / 路径损耗参数===================
            "enable_large_scale_fading": False,        # 是否启用大尺度衰落
            "large_scale_model": "none",               # none / fspl / log_distance
            "link_distance_m": 1.0,                    # 发射机到接收机距离，单位 m
            "carrier_frequency_Hz": 300e9,             # 载频，单位 Hz；FSPL 需要用到
            "reference_distance_m": 1.0,               # 对数距离模型参考距离 d0
            "reference_path_loss_db": None,            # 若为 None，则用 d0 处 FSPL 作为参考损耗
            "path_loss_exponent": 2.0,                 # 对数距离模型路径损耗指数 n
            "shadowing_std_db": 0.0,                   # 阴影衰落标准差，单位 dB；0 表示关闭
            "large_scale_seed": None,                  # 阴影衰落随机种子
            # ================================================

            "use_frac_delay": False,                                                                  # 是否启用分数延迟滤波
            "frac_filter_half_len": 12,                                                              # 分数延迟 FIR 半长，滤波器长度为 2L+1
            "frac_filter_window": "hann",                                                            # 分数延迟滤波器窗函数：hann / hamming / rect
            "static_channel": True,                                                                 # 调试开关：True 时强制所有路径按静态增益处理
            "normalize_channel_power": False,                                                        # 是否按 sum(|gain|^2) 对路径增益归一化
            "enable_multipath_memory": False,                                                        # 是否启用静态多径的帧间输入历史缓冲区
            "fading_model": "static",          # static / frame / block / sample；当前实现 static 和 frame
            "fading_seed": None,              # 衰落随机种子；None 表示每次随机
            "block_length": 256,             # 块衰落，fading_model="block" 时，每多少个采样点更新一次路径增益
            "jakes_num_sinusoids": 48,        # 后续 Jakes 模型使用，当前预留

            # 接收机相关
            "equalizer_method": "zf",                                                               # 均衡方法（ZF/MMSE）

            # RS 解码器相关参数
            "decode_mode": "hard",                                                                 # 译码算法（hard / chase）
            "chase_num_per_packet": 2,                                                              # Chase 译码每包试探数量（仅 decode_mode=chase 有效）

            

            # 仿真控制参数
            "run_times": 1,                                                                         # 运行次数
            "save_intermediate": False,                                                             # 是否保存中间结果


            # 其他参数
            "Rate": 1,                                                                        
        }

    def validate(self):
        # """校验参数合法性"""
        # # assert self.get("bandwidth") == 8.64, "当前仅支持8.64 GHz带宽"
        # assert self.get("ppre") >= 0 and self.get("ppre") <= 3, "PPRE字段必须为0到3之间的整数"
        # assert self.get("pw") in [0, 1], "PW字段必须为0或1"
        # assert self.get("subframe_length") > 0, "数据子帧长度必须为正整数"
        # # assert self.get("N") == 64, "当前仅支持64点FFT" 
        # assert len(self.get("chan_delays")) == len(self.get("chan_gains")), "多径时延和增益长度必须一致"
        # assert self.get("Preamble_type") in ["short", "long"], "前导码类型必须为'short'或'long'"
        # assert self.get("fc") > 0, "载波频率必须为正数"
        # assert self.get("ppm") >= 0, "频偏ppm值不能为负"
        # # assert isinstance(self.get("is_cp"), bool), "is_cp必须为布尔值"
        # # assert isinstance(self.get("isRotated"), bool), "isRotated必须为布尔值"
        # assert self.get("gi_length") >= 0, "循环前缀长度不能为负"
        # assert self.get("gi_length") < self.get("subframe_length"), "循环前缀长度必须小于数据块长度N"
        # print("PHY参数校验通过")
        pass

if __name__ == "__main__":
    params = PHYParams()
    print(params)
    params.update(SNRdB=15)
    print(params)