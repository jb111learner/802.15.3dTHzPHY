import numpy as np
from params.PHYParams import PHYParams
from transmitter.DataProcesser import BitStreamProcessor
from utils.Seq_gen import Generator


class Scrambler:
    """
    扰码器/解扰码器：严格遵循3GPP TS 38.211标准实现
    核心：两个31阶m序列异或生成Gold序列，完全匹配5G NR物理层规范
    """
    def __init__(self, params):
        self.params = params
        self.generator = Generator(params)
        # 输出参数      
        self.sample_rate = None  # 采样率
        self.duration = None  # 时长
        self.bit_length = None  # 比特长度
        self.frame_bit_num = None # 单帧比特数
        self.frame_num = None # 帧数
        self.padding_bit_num = 0   # 补零的比特数
    
    def _verification_data(self, data_dict):
        """
        验证并封装返回结果字典
        :param data_dict: 输入数据字典，包含以下键值：
            "signal_stream": 比特流,
            "sample_rate_Hz": 采样率,
            "duration_seconds": 时长,
            "signal_length": 比特长度,
            "frame_num": 帧数,
            "padding_bit_num": 补零比特数,
            "frame_bit_num": 单帧比特数
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
        
        self.sample_rate = data_dict["sample_rate_Hz"]
        self.duration = data_dict["duration_seconds"]
        self.bit_length = data_dict["signal_length"]
        self.padding_bit_num = data_dict["padding_bit_num"]
        self.frame_bit_num = data_dict["frame_bit_num"]
        self.frame_num = data_dict["frame_num"]

    def scramble(self, data_dict):
        """
        3GPP标准扰码：输入比特流与Gold序列逐位异或
        :param data_bits: 输入二进制比特流（np.array，0/1）
        :return: 扰码后的比特流
        """
        self._verification_data(data_dict)
        data_bits = data_dict["signal_stream"]
        seq_length = len(data_bits)
        # 生成Gold序列
        self.c_seq = self.generator.generate_gold_sequence(seq_length)
        # 逐位异或
        scrambled_bits = (data_bits ^ self.c_seq) % 2
        result_dict = {
            "signal_stream": scrambled_bits,
            "sample_rate_Hz": self.sample_rate,
            "duration_seconds": self.duration,
            "signal_length": self.bit_length,
            "padding_bit_num": self.padding_bit_num,
            "frame_bit_num": self.frame_bit_num,
            "frame_num": self.frame_num
        }   
        return result_dict


if __name__ == "__main__":
    params = PHYParams()
    p1 = BitStreamProcessor(params)
    res1 = p1.run()
    scrambler = Scrambler(params)
    scrambled_data = scrambler.scramble(res1)
    print(f"比特流长度：{scrambled_data['signal_length']} bit")
    print(f"采样率：{scrambled_data['sample_rate_Hz']} Hz，时长：{scrambled_data['duration_seconds']} 秒")
    print(f"补零数量：{scrambled_data['padding_bit_num']} bit")