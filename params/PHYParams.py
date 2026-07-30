from typing import Any, Dict, List

import numpy as np
from core.BaseParams import BaseParams


class PHYParams(BaseParams):
    _LEGACY_MULTIPATH_KEYS = {
        "multipath_paths", "chan_delays", "chan_gains", "use_pdp",
        "use_frac_delay", "frac_filter_half_len", "frac_filter_window",
        "static_channel", "normalize_channel_power", "enable_multipath_memory",
        "fading_model", "fading_seed", "block_length", "jakes_num_sinusoids",
        "tdl_jakes_num_sinusoids",
    }

    @classmethod
    def _is_legacy_multipath_key(cls, key: str) -> bool:
        return key in cls._LEGACY_MULTIPATH_KEYS or key.startswith("pdp_")

    def apply_dict(self, param_dict: Dict[str, Any]) -> List[str]:
        """安全更新参数，忽略 BaseParams 不支持的字段，返回未知键列表。"""
        unknown_keys = []
        for k, v in param_dict.items():
            if k in self._params or self._is_legacy_multipath_key(k):
                self.update(**{k: v})
            else:
                unknown_keys.append(k)
        return unknown_keys

    def update(self, **kwargs):
        """更新现代参数，并按需接收旧多径键而不将其放入默认参数表。"""
        unknown_keys = [
            key for key in kwargs
            if key not in self._params and not self._is_legacy_multipath_key(key)
        ]
        if unknown_keys:
            raise KeyError(f"参数 {unknown_keys[0]} 不存在")
        self._params.update(kwargs)
        self.validate()

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
            "link_mode":"ofdm",                                                                  # 链路模式（"SC-FDE", "OFDM"）


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
            # "enable_window_filter":False,                                                          # 是否启用加窗与频谱成型
            # "rolling_width":64,                                                                    # 过渡带宽度


            # 调制相关
            "NCBPS": 6,                                                                             # 每符号比特数（1=BPSK,2=QPSK,3=8PSK,4=16QAM,6=64QAM）
            "APSK_RINGS": [(4, 1.0), (12, 2.85)],                                                   # (每环点数, 半径)
            "APSK_PHASE_OFFSETS": [np.pi/4, np.pi/12],                                              # 每环相位偏置（可设全0）
            "code_type": "LDPC",                                                                      # 信道编码类型(RS编码/LDPC)

            # RS码相关设置 (OFDM模式建议使用RS(15,11) ， SC-FDE模式建议使用RS(255,192))
            "rs_nsym": 4,                                                                           # Reed-Solomon校验符号数量 (n-k)
            "rs_c_exp": 4,                                                                           # Reed-Solomon有限域指数 (GF(2^8)=GF(256))
            "rs_packet_size": 11,                                                                   # Reed-Solomon编码包大小 (k)

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
            "Preamble_type": "short",                                                                # 前导码类型
            "phase_rotation": np.pi/4,                                                              # 前导码相位旋转角度（弧度）

            # 其他参数
            "scramble": True,                                                                       # 是否使用扰码
            "ppre": 1,                                                                              # PPRE字段（物理层前导码配置，0-3）
            "pw": 0,                                                                                # PW字段
            "MCS": 1,                                                                               # 调制编码方案("0": DBPSK, "1":QAM, "2":APSK)

            # 信道相关
            "enable_awgn": True,                                                                    # 是否启用AWGN噪声
            "enable_phase_noise": False,                                                             # 是否启用相位噪声
            "enable_cfo": False,                                                                     # 是否启用载波频率偏移
            "enable_cfo_compensation": False,                                                        # 是否启用CFO补偿
            "noise_temperature": None,                                                              # 信道噪声温度（开尔文）
            "noise_figure_db": None,                                                                # 信道噪声系数（dB）
            "SNRdB": 20,                                                                            # 信噪比（dB），与温度+带宽二选一
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
            "enable_iq_imbalance": False,           # 是否启用 I/Q 不平衡
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

            # RX IQ 不平衡补偿参数（独立于信道损伤注入开关）
            "enable_iq_compensation": True,        # 是否在接收机细 CFO 后启用 IQ 补偿
            "iq_compensation_method": "decision_directed",       # IQ补偿方法: ces / decision_directed
            "iq_comp_filter_len": 5,               # IQ 损伤/补偿 FIR 长度
            "iq_comp_ridge_lambda": 0.0,           # LS 岭回归系数，0 表示使用伪逆
            "iq_compensation_mode": "per_frame",  # per_frame / first_frame
            "iq_comp_dd_iterations": 3,            # 判决导向IQ补偿迭代次数
            
            # 3GPP TR 38.901 TDL 多径模式。TDL 模式启用时，仅需选择模型和 DS。
            # 10/30/100/300/1000 ns 均可作为 tdl_delay_spread_ns。
            "enable_multipath": False,
            "tdl_model": None,                        # None=旧 multipath/PDP 回退；TDL-A~TDL-E=标准 TDL
            "tdl_delay_spread_ns": 100.0,              # 目标 RMS delay spread，单位 ns
            "tdl_velocity_mps": 0.0,                    # 移动速度；0=固定随机信道，>0=连续 Jakes 衰落

            # 旧 multipath_paths/PDP 配置不再作为现代 PHY 参数暴露；
            # MultipathChannel 仍保留兼容解析，传入 tdl_model=None 时使用旧路径。

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

            # 接收机相关
            "equalizer_method": "zf",                                                               # 均衡方法（ZF/MMSE）
            "enable_channel_equalization": True,                                                      # 是否启用信道估计与补偿

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