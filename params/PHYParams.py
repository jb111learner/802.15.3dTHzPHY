import numpy as np
from core.BaseParams import BaseParams

# 使用示例（子类实现）
class PHYParams(BaseParams):
    def _init_params(self):
        self._params = {
            "NCBPS": 2,                                                                             # 每符号比特数（1=BPSK,2=QPSK,3=8PSK,4=16QAM,6=64QAM）
            "is_cp": True,                                                                          # 是否使用循环前缀
            # "isRotated": True,                                                                    # 是否进行相位旋转          
            "scrambler_seed_id": 3,                                                                 # 加扰器种子ID（0-15，用于生成加扰序列）
            "ppre": 1,                                                                              # PPRE字段（物理层前导码配置，0-3）
            "pw": 0,                                                                                # PW字段（功率控制相关，0或1） 
            "Preamble_type": "short",                                                               # 前导码类型
            "phase_rotation": np.pi/4,                                                              # 前导码相位旋转角度（弧度）
            "MCS": 1,                                                                               # 调制编码方案
            "bandwidth": 8.64,                                                                      # 系统带宽（8.64 GHz，THz频段典型值）
            "subframe_length": 480,                                                                 # 数据子帧长度
            "cp_length": 32,                                                                        # 循环前缀长度
            "Rate": 1,   

            "scrambler_init": np.array([1,0,1,1,0,1,0], dtype=int),                                 # 扰码器初始状态（7位）

            "rs_nsym": 63,                                                                          # Reed-Solomon校验符号数量
            "rs_c_exp": 8,                                                                          # Reed-Solomon有限域指数
            "rs_packet_size": 192,                                                                  # Reed-Solomon编码包大小

            "symbol_rate": 30e9,                                                                    # 符号率
            "oversampling": 2,                                                                      # 采样率
            "rolloff": 0.22,                                                                        # 滚降系数
            "filter_type": "rrc",                                                                   # 滤波器类型（根升余弦）
            "filter_length": 8,                                                                     # 滤波器长度（采样点数）    

            "delay": 0,                                                                          # 前置延迟（采样点数）
            "chan_delays": np.array([0, 16, 32, 64]),                                              # 多径时延（采样点数）
            "chan_gains": np.array([0.9, 0.7, 0.5*np.exp(-1j*np.pi/6), 0.25*np.exp(-1j*np.pi/3)]),  # 多径增益
            "SNRdB": 5,                                                                            # 信噪比
            "fc": 1000e9,                                                                           # 载波频率（1 THz）
            "ppm": 10,                                                                              # 频偏ppm值                                                                           
        }

    def validate(self):
        """校验参数合法性"""
        assert self.get("bandwidth") == 8.64, "当前仅支持8.64 GHz带宽"
        assert self.get("scrambler_seed_id") >= 0 and self.get("scrambler_seed_id") <= 15, "加扰器种子ID必须为0到15之间的整数"
        assert self.get("ppre") >= 0 and self.get("ppre") <= 3, "PPRE字段必须为0到3之间的整数"
        assert self.get("pw") in [0, 1], "PW字段必须为0或1"
        assert self.get("subframe_length") > 0, "数据子帧长度必须为正整数"
        # assert self.get("N") == 64, "当前仅支持64点FFT" 
        assert len(self.get("chan_delays")) == len(self.get("chan_gains")), "多径时延和增益长度必须一致"
        assert self.get("Preamble_type") in ["short", "long"], "前导码类型必须为'short'或'long'"
        assert self.get("fc") > 0, "载波频率必须为正数"
        assert self.get("ppm") >= 0, "频偏ppm值不能为负"
        assert isinstance(self.get("is_cp"), bool), "is_cp必须为布尔值"
        # assert isinstance(self.get("isRotated"), bool), "isRotated必须为布尔值"
        assert self.get("cp_length") >= 0, "循环前缀长度不能为负"
        assert self.get("cp_length") < self.get("subframe_length"), "循环前缀长度必须小于数据块长度N"
        print("PHY参数校验通过")

if __name__ == "__main__":
    params = PHYParams()
    print(params)
    params.update(SNRdB=15)