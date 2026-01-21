import numpy as np
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler 
from utils.Coder import RSCoder

class Encoder:
    """
    支持按字节分包编码/解码。
    """
    def __init__(self, params):
        self.params = params
        self.code_type = self.params.get("code_type")   # 编码类型
        if self.code_type not in ["RSC", "LDPC"]:
            raise ValueError("当前仅支持RSC编码, LDPC编码")
        if self.code_type == "RSC":
            # 初始化RS编解码器
            self.encoder = RSCoder(self.params)
            # 获取RS编码参数
            self.nsym = self.encoder.nsym               # 校验符号数
            self.packet_size = self.encoder.packet_size   # 原始包大小
            self.efficiency = self.packet_size / (self.packet_size + self.nsym)  # 编码效率
            # 编码后每个包的大小 = 原始包大小 + 校验符号数
            self.encoded_packet_size = self.packet_size + self.nsym
        
        # 输出参数      
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.bit_length = None  # 比特长度
        self.padding_bit_num = 0   # 补零的比特数

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "bit_stream": 比特流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "bit_length": 比特长度,
            "padding_bit_num": 补零比特数,
        """
        # 校验输入字典完整性
        required_keys = [
            "bit_stream", "sample_rate_Hz", "duration_seconds", "bit_length","padding_bit_num"
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["bit_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
        self.sample_rate = data_dict["sample_rate_Hz"] / self.efficiency
        self.duration = data_dict["duration_seconds"]
        self.bit_length = data_dict["bit_length"] / self.efficiency
        self.padding_bit_num = data_dict["padding_bit_num"]   

    def encode(self, data_dict):
        """
        输入：数据字典
        输出：编码后数据字典
        """
        self._verification_data(data_dict)
        bits = data_dict["bit_stream"]
        bits_encoded = self.encoder.encode(bits)

        if len(bits_encoded) != round(self.bit_length):
            raise ValueError(f"编码后比特长度不匹配：期望 {round(self.bit_length)}，实际 {len(bits_encoded)}")

        result_dict = {
            "bit_stream": bits_encoded,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "bit_length": round(self.bit_length),
            "padding_bit_num": self.padding_bit_num,
        }   
        # 转换为比特流返回
        return result_dict
    
if __name__ == "__main__":
    params = PHYParams()
    p1 = BitStreamProcessor(params)
    res1 = p1.run()
    scrambler = Scrambler(params)
    scrambled_data_dict = scrambler.scramble(res1)
    coder = Encoder(params)
    encoded_data_dict = coder.encode(scrambled_data_dict)
    print(f"比特流长度：{encoded_data_dict['bit_length']} bit")
    print(f"采样率：{encoded_data_dict['sample_rate_Hz']} Hz，时长：{encoded_data_dict['duration_seconds']} 秒")
    print(f"补零数量：{encoded_data_dict['padding_bit_num']} bit")