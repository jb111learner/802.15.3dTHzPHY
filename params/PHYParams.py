import numpy as np
from core.BaseParams import BaseParams

class PHYParams(BaseParams):
    def _init_params(self):
        self._params = {
            # 仿真样本参数
            "file_path": None,
            "duration": None,                                                                       # 数据时长（秒）
            "sample_rate": 441000.0,  
            "bit_length": 140000,                                                                   # 比特长度

            # 基础参数
            "fc": 1000e9,                                                                           # 载波频率（1 THz）
            "bandwidth": 100e9,                                                                      # 信号带宽（10 GHz）
            "subframe_length": 480,                                                                 # 数据子帧长度
            "gi_length": 32,                                                                        # 循环前缀长度
            "is_cp": False,                                                                         # 是否使用循环前缀
            "c_init": 0x12345678, 
            "frame_bit_num": 73728,                                                                 # 单帧比特数


            # 发射机相关
            "NCBPS": 4,                                                                             # 每符号比特数（1=BPSK,2=QPSK,3=8PSK,4=16QAM,6=64QAM）
            "APSK_RINGS": [(4, 1.0), (12, 2.85)],                                                   # (每环点数, 半径)
            "APSK_PHASE_OFFSETS": [np.pi/4, np.pi/12],                                              # 每环相位偏置（可设全0）
            "code_type": "RSC",                                                                     # 信道编码类型(RS编码/LDPC)
            "rs_nsym": 63,                                                                          # Reed-Solomon校验符号数量
            "rs_c_exp": 8,                                                                          # Reed-Solomon有限域指数
            "rs_packet_size": 192,                                                                  # Reed-Solomon编码包大小
            "oversampling": 4,                                                                      # 上采样率
            "rolloff": 0.22,                                                                        # 滚降系数
            "filter_type": "rrc",                                                                   # 滤波器类型
            "filter_length": 8,                                                                     # 滤波器长度  


            # 前导码相关
            "ppre": 1,                                                                              # PPRE字段（物理层前导码配置，0-3）
            "pw": 0,                                                                                # PW字段
            "Preamble_type": "short",                                                               # 前导码类型
            "phase_rotation": np.pi/4,                                                              # 前导码相位旋转角度（弧度）
            "MCS": 2,                                                                               # 调制编码方案("0": DBPSK, "1":QAM, "2":APSK)
            "bandwidth": 8.64,                                                                      # 系统带宽 (GHz)


            # 信道相关
            "noise_temperature": None,                                                              # 信道噪声温度（开尔文）
            "noise_figure_db": None,                                                                # 信道噪声系数（dB）
            "SNRdB": 20,                                                                            # 信噪比（dB），与温度+带宽二选一
            "delay": 0,                                                                             # 前置延迟（采样点数）
            "chan_delays": np.array([0, 16, 32, 64]),                                               # 多径时延（采样点数）
            "chan_gains": np.array([0.9, 0.7, 0.5*np.exp(-1j*np.pi/6), 0.25*np.exp(-1j*np.pi/3)]),  # 多径增益
            "ppm": 10,                                                                              # 频偏ppm值   



            # 其他参数
            "Rate": 1,                                                                        
        }

    def validate(self):
        """校验参数合法性"""
        assert self.get("bandwidth") == 8.64, "当前仅支持8.64 GHz带宽"
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
        assert self.get("gi_length") >= 0, "循环前缀长度不能为负"
        assert self.get("gi_length") < self.get("subframe_length"), "循环前缀长度必须小于数据块长度N"
        print("PHY参数校验通过")

if __name__ == "__main__":
    params = PHYParams()
    print(params)
    params.update(SNRdB=15)