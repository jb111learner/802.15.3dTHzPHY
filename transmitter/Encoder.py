import numpy as np
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from transmitter.Scrambler import Scrambler
from utils.Coder import RSCoder, LDPCCoder

class Encoder:
    """
    支持 RS / LDPC 编码。
    """
    def __init__(self, params):
        self.params = params
        self.code_type = self.params.get("code_type")
        if self.code_type not in ["RS", "LDPC"]:
            raise ValueError("当前仅支持RS编码, LDPC编码")
        if self.code_type == "RS":
            self.encoder = RSCoder(self.params)
            self.nsym = self.encoder.nsym
            self.packet_size = self.encoder.packet_size
            self.efficiency = self.packet_size / (self.packet_size + self.nsym)
            self.encoded_packet_size = self.packet_size + self.nsym
        elif self.code_type == "LDPC":
            self.encoder = LDPCCoder(self.params)
            self.nsym = self.encoder.r          # parity bits per codeword
            self.packet_size = self.encoder.k   # info bits per codeword
            self.efficiency = self.encoder.efficiency
            self.encoded_packet_size = self.encoder.n
        
        # 输出参数      
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.bit_length = None  # 比特长度
        self.padding_bit_num = 0   # 补零的比特数
        self.frame_bit_num = None # 单帧比特数
        self.frame_num = None # 帧数

    def _verification_data(self, data_dict):
        """
        校验输入数据字典合法性
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 比特流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 比特长度,
            "padding_bit_num": 补零比特数,
            "frame_bit_num": 单帧比特数,
            "frame_num": 帧数,
        """
        # 校验输入字典完整性
        required_keys = [
            "signal_stream", "sample_rate_Hz", "duration_seconds", "signal_length","padding_bit_num","frame_bit_num","frame_num"    
        ]
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"输入数据字典缺少必要键值：{key}")
            
        # 校验分帧信息一致性
        if round(data_dict["sample_rate_Hz"] * data_dict["duration_seconds"]) != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的采样率与时长不匹配")  
        
        if data_dict["frame_bit_num"] * data_dict["frame_num"] != data_dict["signal_length"]:
            raise ValueError("输入数据字典中的分帧信息不匹配")
        
        # 编码增加比特数量和发送时长，不改变物理比特时钟。旧实现通过
        # 提高 sample_rate 保持时长不变，会间接抬高最终波形采样率。
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.bit_length = data_dict["signal_length"] / self.efficiency
        self.duration = self.bit_length / self.sample_rate
        self.frame_bit_num = data_dict["frame_bit_num"] * (self.packet_size + self.nsym) / self.packet_size
        self.frame_num = data_dict["frame_num"] 
        self.padding_bit_num = data_dict["padding_bit_num"]   

    def encode(self, data_dict):
        """
        输入：数据字典
        输出：编码后数据字典
        """
        self._verification_data(data_dict)
        bits = data_dict["signal_stream"]
        bits_encoded = self.encoder.encode(bits)

        if len(bits_encoded) != round(self.bit_length):
            raise ValueError(f"编码后比特长度不匹配：期望 {round(self.bit_length)}，实际 {len(bits_encoded)}")

        result_dict = {
            "signal_stream": bits_encoded,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": round(self.bit_length),
            "padding_bit_num": self.padding_bit_num,
            "frame_bit_num": self.frame_bit_num,
            "frame_num": self.frame_num,
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
    print(f"比特流长度：{encoded_data_dict['signal_length']} bit")
    print(f"采样率：{encoded_data_dict['sample_rate_Hz']} Hz，时长：{encoded_data_dict['duration_seconds']} 秒")
    print(f"补零数量：{encoded_data_dict['padding_bit_num']} bit")
