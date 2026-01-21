import numpy as np
from transmitter.THzTransmitter import THzTransmitter
from params.PHYParams import PHYParams
from channel.THzChannel import THzChannel
from receiver.MatchedFilter import RxMatchedFilter
from receiver.GIremover import GIRemover
from receiver.DeModulator import THzDemodulator
from utils.Coder import RSCoder

class Decoder:
    """
    支持按字节分包编码/解码。
    """
    def __init__(self, params):
        self.params = params
        self.code_type = self.params.get("code_type")   # 编码类型
        if self.code_type not in ["RSC", "LDPC"]:
            raise ValueError("当前仅支持RSC解码, LDPC解码")
        if self.code_type == "RSC":
            # 初始化RS编解码器
            self.decoder = RSCoder(self.params)
            # 获取RS解码参数
            self.nsym = self.decoder.nsym               # 校验符号数
            self.packet_size = self.decoder.packet_size   # 原始包大小
            self.efficiency = self.packet_size / (self.packet_size + self.nsym)  # 编码效率
        
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
        
        self.sample_rate = data_dict["sample_rate_Hz"] * self.efficiency
        self.duration = data_dict["duration_seconds"]
        self.bit_length = data_dict["bit_length"] * self.efficiency
        self.padding_bit_num = data_dict["padding_bit_num"]   

    def decode(self, data_dict):
        """
        输入：数据字典
        输出：解码后数据字典
        """
        self._verification_data(data_dict)
        bits = data_dict["bit_stream"]
        bits_encoded = self.decoder.decode(bits, int(self.bit_length))
        if len(bits_encoded) != round(self.bit_length):
            raise ValueError(f"解码后比特长度不匹配：期望 {round(self.bit_length)}，实际 {len(bits_encoded)}")

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
    # 初始化参数和发射机
    params = PHYParams()
    transmitter = THzTransmitter(params)    
    # 执行完整发射流程  
    tx_signal_dict = transmitter.run() 
    # 初始化信道并生成接收信号
    channel = THzChannel(params)
    rx_signal_dict = channel.run(tx_signal_dict)
    # 初始化接收端匹配滤波器
    rx_matched_filter = RxMatchedFilter(transmitter.pulse_shaper)
    # 进行匹配滤波并获得中间信号
    rx_matched_dict = rx_matched_filter.recover_symbols(rx_signal_dict)
    # 初始化GI移除器
    gi_remover = GIRemover(params)
    # 进行GI移除
    rx_data_no_gi_dict = gi_remover.remove_gi(rx_matched_dict)
    # 解调
    snr_db = params.get("SNRdB")    
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.00 / (2 * snr_linear))
    demodulator = THzDemodulator(params)
    llr_dict = demodulator.demodulate(rx_data_no_gi_dict, sigma)
    # LLR转比特
    rec_bits_dict = demodulator.llr_to_bits(llr_dict)
    # 初始化解码器
    decoder = Decoder(params)
    # 进行解码
    decoded_bits_dict = decoder.decode(rec_bits_dict)
    data_scrambled_dict = transmitter.data_scrambled_dict
    print(f"\n===== 解码结果分析 =====")
    print(f"解码比特长度：{len(decoded_bits_dict['bit_stream'])}, 原始比特长度：{len(data_scrambled_dict['bit_stream'])}")
    print(f"原始比特（前10）：{data_scrambled_dict['bit_stream'][0:10]}")
    print(f"解码比特（前10）：{decoded_bits_dict['bit_stream'][0:10]}")
    print(f"比特错误数：{np.sum(data_scrambled_dict['bit_stream'] != decoded_bits_dict['bit_stream'])}")
    print(f"误码率：{np.sum(data_scrambled_dict['bit_stream'] != decoded_bits_dict['bit_stream'])/len(data_scrambled_dict['bit_stream']):.6f}")
    