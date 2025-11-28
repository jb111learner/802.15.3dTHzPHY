import numpy as np
from core.BaseParams import BaseParams

# 使用示例（子类实现）
class PHYParams(BaseParams):
    def _init_params(self):
        self._params = {
            """系统基础参数"""
            "fs": 1760e6,                                                                           # 码片速率
            "M": 4,                                                                                 # 调制阶数（4-QAM）
            "is_cp": True,                                                                          # 是否使用循环前缀
            "isRotated": True,                                                                      # 是否进行相位旋转
            "N": 64,                                                                                # FFT点数            
            "scrambler_seed_id": 3,                                                                 # 加扰器种子ID（0-15，用于生成加扰序列）
            "ppre": 1,                                                                              # PPRE字段（物理层前导码配置，0-3）
            "pw": 0,                                                                                # PW字段（功率控制相关，0或1） 
            "Preamble_type": "short",                                                               # 前导码类型           
            """PHY层协议信息参数"""  
            "mcs": 5,                                                                               # 调制编码方案（0-15，对应不同码率和调制方式）
            "bandwidth": 8.64,                                                                      # 系统带宽（8.64 GHz，THz频段典型值）
            "frame_length": 1500,                                                                   # 数据帧长度（1500字节，以太网标准帧长）
            """信道参数"""                                  
            "chan_delays": np.array([0, 4, 8, 12]),                                                 # 多径时延
            "chan_gains": np.array([0.9, 0.7, 0.5*np.exp(-1j*np.pi/6), 0.25*np.exp(-1j*np.pi/3)]),  # 多径增益
            "SNRdB": 10,                                                                            # 信噪比
            """频偏参数"""
            "fc": 100e9,                                                                            # 载波频率（100 GHz）
            "ppm": 30,                                                                               # 频偏ppm值（典型值30 ppm）
        }

    def validate(self):
        """校验参数合法性"""
        assert self.get("mcs") >= 0 and self.get("mcs") <= 15, "MCS必须为0到15之间的整数"
        assert self.get("bandwidth") == 8.64, "当前仅支持8.64 GHz带宽"
        assert self.get("scrambler_seed_id") >= 0 and self.get("scrambler_seed_id") <= 15, "加扰器种子ID必须为0到15之间的整数"
        assert self.get("ppre") >= 0 and self.get("ppre") <= 3, "PPRE字段必须为0到3之间的整数"
        assert self.get("pw") in [0, 1], "PW字段必须为0或1"
        assert self.get("frame_length") > 0, "数据帧长度必须为正整数"
        assert self.get("SNRdB") >= 0, "SNR不能为负"
        assert self.get("N") == 64, "当前仅支持64点FFT" 
        assert len(self.get("chan_delays")) == len(self.get("chan_gains")), "多径时延和增益长度必须一致"
        assert self.get("Preamble_type") in ["short", "long"], "前导码类型必须为'short'或'long'"
        assert self.get("M") in [4, 16, 64, 256], "当前仅支持4-QAM, 16-QAM, 64-QAM和256-QAM"
        assert self.get("fs") > 0, "码片速率必须为正数"
        assert self.get("fc") > 0, "载波频率必须为正数"
        assert self.get("ppm") >= 0, "频偏ppm值不能为负"
        assert isinstance(self.get("is_cp"), bool), "is_cp必须为布尔值"
        assert isinstance(self.get("isRotated"), bool), "isRotated必须为布尔值"
        print("PHY参数校验通过")